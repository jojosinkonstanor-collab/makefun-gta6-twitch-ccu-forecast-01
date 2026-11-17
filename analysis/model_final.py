"""Candidate final model.

- Regime: rolling-midnight launches (R1) only for training: comparables that
  are not daytime-global (hour-1 < 5500), not early-access-contaminated, not
  weak tier, and not early-peak (peak hour <= 6).
- Anchor A = max(hour-1 print, last pre-release hourly mean). Ratios are
  log(viewers_h / A); size x = log(A).
- Per-hour regression of the log-ratio on x, slope shrunk toward the pooled
  (all-hours) slope; the size effect x - xbar is capped at the largest
  training size so a record-size launch is forecast like the biggest
  launches seen, not by extrapolating a line past the data.
- sigma_h = residual std (floor 0.10); quantiles via normal z-scores.
- Console-only launches get weight 2 (GTA VI is console-only).
"""
import math

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

def _clamp(v, c):
    return max(-c, min(c, v))

def fit_final(p1, anchor, train_v, train_anchor, train_weight=None,
              shrink=0.5, sigma_floor=0.10, spread_mult=1.0, adj_clamp=1.0):
    p1 = max(1.0, float(p1))
    anchor = max(1.0, float(anchor))
    n = len(train_v)
    w = train_weight or [1.0] * n
    xs = [math.log(max(a, 1.0)) for a in train_anchor]
    sw = sum(w)
    xbar = sum(wi * x for wi, x in zip(w, xs)) / sw
    xmax = max(xs)
    xg = max(min(xs) - xbar, min(max(xs) - xbar, math.log(anchor) - xbar))
    # log-ratio matrix vs anchor
    R = {h: [] for h in range(2, 25)}
    for i, v in enumerate(train_v):
        a = max(train_anchor[i], 1.0)
        for h in range(2, 25):
            vh = v[h - 1]
            R[h].append(math.log(max(vh, 1.0) / a) if vh and vh > 0 else None)
    # pooled slope
    num = den = 0.0
    for h in range(2, 25):
        pairs = [(w[i], xs[i] - xbar, r) for i, r in enumerate(R[h]) if r is not None]
        mh = sum(wi * r for wi, _, r in pairs) / sum(wi for wi, _, _ in pairs)
        for wi, dx, r in pairs:
            num += wi * dx * (r - mh)
            den += wi * dx * dx
    b_pool = num / den if den > 0 else 0.0
    out = {}
    for h in range(2, 25):
        pairs = [(w[i], xs[i] - xbar, r) for i, r in enumerate(R[h]) if r is not None]
        swp = sum(wi for wi, _, _ in pairs)
        mh = sum(wi * r for wi, _, r in pairs) / swp
        n2 = sum(wi * dx * (r - mh) for wi, dx, r in pairs)
        d2 = sum(wi * dx * dx for wi, dx, r in pairs)
        b_h = n2 / d2 if d2 > 0 else 0.0
        b = (1 - shrink) * b_pool + shrink * b_h
        resid = [r - (mh + b * dx) for _, dx, r in pairs]
        sig = math.sqrt(sum(wi * e * e for (wi, _, _), e in zip(pairs, resid)) / max(swp - 2, 1))
        sig = max(sig, sigma_floor)
        mu = mh + _clamp(b * xg, adj_clamp)
        out[h] = [max(1.0, anchor * math.exp(mu + z * sig * spread_mult)) for z in Z]
    return out
