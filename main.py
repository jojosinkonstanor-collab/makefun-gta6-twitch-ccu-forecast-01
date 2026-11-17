"""GTA VI Twitch CCU forecast (make.fun gta6-twitch-ccu-forecast-01).

Reads each <id>.in case file (the Twitch viewer count of Grand Theft Auto VI
up to the end of the first hour of its release day, 2026-11-19 UTC, plus the
launch-day curves of past big game launches shipped in the case) and writes
<id>.out: five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of concurrent Twitch
viewers for each of hours 2..24 of release day.

Method (see README.md for the full write-up and validation)
-----------------------------------------------------------
Launch days fall into regimes set by *when* the game unlocks, and the
quantiles are forecast per hour in log space, anchored on the scale the
launch has already shown:

1. Regime. Hour 1's print and the case notes decide the regime:
   - hour-1 viewers < 5,500: nothing big had unlocked by 00:00 UTC, the
     game unlocks later in the day (a "daytime-global" launch: tiny hour 1,
     then a 100x+ rise to the evening peak).
   - otherwise a rolling-midnight launch (the GTA VI case: rolling midnight
     local, New Zealand first 2026-11-18T11:00Z): hour 1 already holds the
     regions that unlocked overnight, then a dip in the small hours and a
     rise to an evening peak around 19:00-21:00 UTC.
   - a simultaneous global unlock at ~00:00 UTC is a third, rare regime
     (peak in hour 1-2, then decay).
2. Training set. For the rolling regime, the comparables that share its
   shape: drop daytime-global launches, early-access-contaminated launches
   (their "release day" is not the real peak day) and weak analogues, and
   drop early-peak simultaneous launches. Console-only launches get double
   weight: GTA VI launches on PS5/Xbox only, and PC-simultaneous unlocks
   inflate hour 1 and flatten the day's curve.
3. Anchor. A = max(hour-1 print, last pre-release hourly mean). Hour 1
   (00:00-01:00 UTC) is a dead hour - deep night in Europe - while the hour
   before it is the CET-midnight unlock; the anchor is the truer scale of
   the launch. In all four practice launches the pre-release hour exceeded
   hour 1.
4. Model. Per hour h = 2..24, log(viewers_h / A) ~ Normal(mu_h, sigma_h)
   with mu_h = mean_h + b_h * x, x = log(A) - mean log(A) over training.
   The slope b_h is the per-hour regression of the log-ratio on launch size
   (bigger launches rise less, relative to their start: a 400k launch peaks
   ~1.2-1.4x its anchor, a 10k one ~6-9x). The size effect is capped at the
   largest training launch, so a record-size launch is forecast like the
   biggest launches seen, never by extrapolating a line past the data.
   sigma_h is the residual standard deviation (floor 0.10 log units),
   widened/narrowed by a 0.9 spread factor tuned by cross-validation.
   Quantiles use the normal z-scores at the five scored levels.

Everything is standard library only, deterministic (no randomness, no
network, no clock), and runs in well under a second per case.

Validation (offline, on the kit's own data only)
-----------------------------------------------
- Leave-one-out over the 37 kit comparables: loss 0.28 on rolling launches
  vs 0.40 for the house persistence baseline.
- On the four practice launches (ground truth read once from SullyGnome,
  the public source the kit itself cites - validation only, never shipped):
  mean pinball loss 0.148 vs 0.387 for the house baseline. The scorer was
  verified by reproducing the house baseline's reported practice loss
  (0.387) exactly.
- The final case will train on 42 comparables (the four practice launches
  included), so the final fit has more, and more relevant, data than these
  validation numbers reflect.

Run:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py   (reads /input, writes /output)
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (harmless here,
# this file is self-contained, but keeps the layout identical to the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# --- tuned constants (see README.md; chosen by leave-one-out CV + the
# practice launches, never by looking at the final case, which does not
# exist yet) ---------------------------------------------------------------
DG_V1_THRESHOLD = 5500      # hour-1 viewers below this: daytime-global regime
CONSOLE_WEIGHT = 2.0        # weight of console-only launches in the fit
SPREAD_MULT = 0.9           # multiplies sigma_h
SIGMA_FLOOR = 0.10          # log units
ADJ_CLAMP = 1.0             # cap on |b_h * x| (log units)


def round2(x: float):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
# Regime classification and training-set construction
# --------------------------------------------------------------------------

def _tier(unlock_text):
    """'early-access-contaminated' / 'weak' / 'comparable' / 'target-history'
    from the 'Tier: ...' note in a comparable's unlock field, if present."""
    m = re.search(r"Tier:\s*([a-z\-]+)", unlock_text or "")
    return m.group(1) if m else None


def _day_curve(comp):
    """Release-day hourly viewers [h1..h24] from a comparable's hourly points."""
    ds = parse_ts(comp.get("day_start"))
    if ds is None:
        return None
    by = {}
    for h in comp.get("hourly") or []:
        t = parse_ts(h.get("from"))
        if t is not None and h.get("viewers") is not None:
            by[t] = h["viewers"]
    v = [by.get(ds + timedelta(hours=i)) for i in range(24)]
    if any(x is None for x in v):
        return None
    return v


def _pre_release_peak_ratio(comp, day_curve):
    """Max hourly viewers in [-14d, -36h) over the release-day peak. Used only
    when a comparable carries no Tier label: a launch whose category already
    peaked days before release day was early-access contaminated."""
    ds = parse_ts(comp.get("day_start"))
    if ds is None or not day_curve:
        return 0.0
    peak = max(day_curve)
    pre = [h["viewers"] for h in (comp.get("hourly") or [])
           if parse_ts(h.get("from")) is not None
           and parse_ts(h.get("from")) < ds - timedelta(hours=36)
           and h.get("viewers") is not None]
    if not pre or peak <= 0:
        return 0.0
    return max(pre) / peak


def _is_contaminated(comp, day_curve):
    tier = _tier(comp.get("unlock", ""))
    if tier is not None:
        return tier.startswith("early-access")
    return _pre_release_peak_ratio(comp, day_curve) > 1.5


def _is_weak(comp):
    return _tier(comp.get("unlock", "")) == "weak"


def _is_console_only(comp):
    plats = comp.get("platforms") or ""
    return "PC" not in plats


def classify_target(case, p1):
    """'rolling' | 'daytime' | 'earlypeak' for the case being forecast."""
    notes = (case.get("notes") or "").lower()
    if p1 < DG_V1_THRESHOLD:
        return "daytime"
    if "rolling midnight" in notes or "midnight local" in notes:
        return "rolling"
    if "simultaneous" in notes or "single global unlock" in notes:
        # A simultaneous unlock at/before ~06:00 UTC peaks in hour 1-2;
        # one at midday or later behaves like a daytime launch.
        m = re.search(r"(\d{1,2}):00\s*z", notes)
        if m and int(m.group(1)) <= 6:
            return "earlypeak"
        if "14:00" in notes or "15:00" in notes or "18:00" in notes:
            return "daytime"
        return "earlypeak"
    return "rolling"  # default: most big launches roll at midnight local


def training_set(case, regime):
    """(curves, anchors, weights) for the regime, from the case's comparables."""
    items = []
    for comp in case.get("comparables") or []:
        name = (comp.get("name") or comp.get("game") or "")
        if "Grand Theft Auto VI" in name and "before release" in name:
            continue  # target-history, not a launch
        v = _day_curve(comp)
        if v is None or v[0] is None or v[0] <= 0:
            continue
        peak = max(v)
        ph = v.index(peak) + 1
        ds = parse_ts(comp.get("day_start"))
        pre_last = None
        if ds is not None:
            pre = [(parse_ts(h.get("from")), h.get("viewers"))
                   for h in (comp.get("hourly") or [])]
            pre = [(t, vv) for t, vv in pre
                   if t is not None and t < ds and vv is not None
                   and t >= ds - timedelta(hours=6)]
            if pre:
                pre_last = sorted(pre)[-1][1]
        anchor = max(v[0], pre_last or 0)
        items.append(dict(v=v, v1=v[0], anchor=anchor, peak=peak, ph=ph,
                          contaminated=_is_contaminated(comp, v),
                          weak=_is_weak(comp),
                          weight=CONSOLE_WEIGHT if _is_console_only(comp) else 1.0))
    if regime == "daytime":
        pool = [it for it in items if it["v1"] < DG_V1_THRESHOLD]
    elif regime == "earlypeak":
        pool = [it for it in items if it["ph"] <= 6 and it["v1"] >= DG_V1_THRESHOLD]
    else:  # rolling
        pool = [it for it in items
                if it["v1"] >= DG_V1_THRESHOLD
                and not it["contaminated"] and not it["weak"] and it["ph"] > 6]
    if len(pool) < 3:  # too little: fall back to every usable launch
        pool = items
    return pool


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def fit_forecast(p1, anchor, pool):
    """Per-hour lognormal quantiles. Returns {hour: [q05..q95]} for 2..24."""
    p1 = max(1.0, float(p1))
    anchor = max(1.0, float(anchor))
    xs = [math.log(max(it["anchor"], 1.0)) for it in pool]
    w = [it["weight"] for it in pool]
    sw = sum(w)
    xbar = sum(wi * x for wi, x in zip(w, xs)) / sw
    lo, hi = min(xs) - xbar, max(xs) - xbar
    # Size effect, capped at the largest/smallest training launch: a
    # record-size launch is forecast like the biggest launches seen.
    xg = max(lo, min(hi, math.log(anchor) - xbar))
    # log-ratio matrix R[h][i] = log(viewers_h / anchor)
    R = {h: [] for h in range(2, 25)}
    for it in pool:
        a = max(it["anchor"], 1.0)
        for h in range(2, 25):
            vh = it["v"][h - 1]
            R[h].append(math.log(max(vh, 1.0) / a) if vh and vh > 0 else None)
    out = {}
    for h in range(2, 25):
        pairs = [(w[i], xs[i] - xbar, r)
                 for i, r in enumerate(R[h]) if r is not None]
        swp = sum(wi for wi, _, _ in pairs)
        mh = sum(wi * r for wi, _, r in pairs) / swp
        num = sum(wi * dx * (r - mh) for wi, dx, r in pairs)
        den = sum(wi * dx * dx for wi, dx, _ in pairs)
        b = num / den if den > 0 else 0.0
        resid = [r - (mh + b * dx) for _, dx, r in pairs]
        sig = math.sqrt(sum(wi * e * e for (wi, _, _), e in zip(pairs, resid))
                        / max(swp - 2, 1))
        sig = max(sig, SIGMA_FLOOR)
        mu = mh + max(-ADJ_CLAMP, min(ADJ_CLAMP, b * xg))
        row = []
        prev = 0.0
        for z in Z:
            q = anchor * math.exp(mu + z * sig * SPREAD_MULT)
            q = max(1.0, min(1e9, q))
            q = max(q, prev)  # keep the row non-decreasing
            row.append(round2(q))
            prev = row[-1]
        out[h] = row
    return out


def baseline_forecast(p1):
    """House persistence baseline (the kit starter): every hour stays at the
    hour-1 print, spread widening like a random walk in log space."""
    p1 = max(1, p1)
    out = {}
    for h in range(2, 25):
        row = []
        prev = 0.0
        for z in Z:
            q = round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)))
            q = max(1.0, min(1e9, q))
            q = max(q, prev)
            row.append(q)
            prev = q
        out[h] = row
    return out


# --------------------------------------------------------------------------
# Case handling
# --------------------------------------------------------------------------

def last_pre_release_hour(case):
    """Latest hourly mean strictly before day_start, within 6 hours of it
    (the CET-midnight unlock hour for a rolling launch). None if unavailable
    or too stale to trust as a scale anchor."""
    ds = parse_ts(case.get("day_start"))
    if ds is None:
        return None
    pts = []
    for row in (case.get("history_hourly") or []):
        t = parse_ts(row.get("from"))
        if t is not None and row.get("viewers") is not None:
            pts.append((t, row["viewers"]))
    if not pts:
        for row in (case.get("history") or []):
            t = parse_ts(row.get("at"))
            if t is not None and row.get("viewers") is not None:
                pts.append((t, row["viewers"]))
    pre = [(t, v) for t, v in pts if t < ds and t >= ds - timedelta(hours=6)]
    if not pre:
        return None
    return sorted(pre)[-1][1]


def forecast_case(case):
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), known[0] if known else None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first.get("viewers") or 1)
    try:
        regime = classify_target(case, p1)
        pool = training_set(case, regime)
        if len(pool) < 3:
            raise ValueError("training set too small")
        pre_last = last_pre_release_hour(case)
        anchor = max(p1, pre_last or 0)
        quantiles = fit_forecast(p1, anchor, pool)
    except Exception:
        quantiles = baseline_forecast(p1)
    hours = [{"hour": h, "q": quantiles[h]} for h in range(2, 25)]
    return {"hours": hours}


def validate_output(obj):
    """The grader's shape rules; True only if the output is a valid entry."""
    if set(obj.keys()) != {"hours"}:
        return False
    rows = obj["hours"]
    if len(rows) != 23 or [r.get("hour") for r in rows] != list(range(2, 25)):
        return False
    for r in rows:
        if set(r.keys()) != {"hour", "q"}:
            return False
        q = r["q"]
        if len(q) != 5:
            return False
        if any((not isinstance(x, (int, float))) or x != x or x in (float("inf"), float("-inf"))
               or x <= 0 or x > 1e9 for x in q):
            return False
        if any(q[i] > q[i + 1] for i in range(4)):
            return False
    return True


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
            if not validate_output(forecast):
                raise ValueError("invalid forecast shape")
        except Exception:
            # Never leave a case without a valid file: fall back to the
            # persistence baseline anchored on hour 1 (or 1 viewer).
            known = case.get("known") or []
            p1 = 1
            if known:
                p1 = max(1, known[0].get("viewers") or 1)
            forecast = {"hours": [{"hour": h, "q": baseline_forecast(p1)[h]}
                                for h in range(2, 25)]}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:  # pragma: no cover - cannot happen at 23 rows
            forecast = {"hours": [{"hour": h, "q": [round2(x) for x in row["q"]]}
                                  for h, row in ((r["hour"], r) for r in forecast["hours"])]}
            text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
