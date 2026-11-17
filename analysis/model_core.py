"""Core model (shared by LOO validation and the shipped main.py logic).

Forecast for hours 2..24 of a launch day, anchored on the hour-1 print p1:
  log(v_h / p1) ~ Normal(mu_h, sigma_h)
  mu_h = mean_h + b_h * clamp(x - xbar, +/-CLAMP),  x = ln(p1)
  sigma_h = residual std of the per-hour regression (floor 0.10)
Quantiles via the normal z-scores at levels [0.05, 0.25, 0.5, 0.75, 0.95].

Training set = clean rolling-midnight launches: comparables that are NOT
daytime-global (hour-1 < 5500 viewers: nothing big had unlocked yet), NOT
early-access-contaminated, NOT weak tier. The target's regime is decided
separately (see regime.py logic in main.py); this module only fits the
rolling regime.
"""
import math

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

def _clamp(v, c):
    return max(-c, min(c, v))

def fit_predict(p1, train_curves, train_v1s, clamp=1.0, sigma_floor=0.10,
                spread_mult=1.0):
    """train_curves: list of lists v[0..23] (hourly viewers, release day).
    train_v1s: list of hour-1 viewers aligned with train_curves.
    Returns {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    p1 = max(1.0, float(p1))
    xs = [math.log(max(v, 1.0)) for v in train_v1s]
    xbar = sum(xs) / len(xs)
    xg = math.log(p1) - xbar
    # log-ratio matrix R[h][i]
    R = {h: [] for h in range(2, 25)}
    for v in train_curves:
        v1 = max(v[0], 1.0)
        for h in range(2, 25):
            vh = v[h - 1]
            R[h].append(math.log(max(vh, 1.0) / v1) if vh and vh > 0 else None)
    out = {}
    for h in range(2, 25):
        pairs = [(xs[i] - xbar, r) for i, r in enumerate(R[h]) if r is not None]
        mh = sum(r for _, r in pairs) / len(pairs)
        num = sum(dx * (r - mh) for dx, r in pairs)
        den = sum(dx * dx for dx, _ in pairs)
        b = num / den if den > 0 else 0.0
        resid = [r - (mh + b * dx) for dx, r in pairs]
        sig = math.sqrt(sum(e * e for e in resid) / max(len(resid) - 2, 1))
        sig = max(sig, sigma_floor)
        mu = mh + _clamp(b * xg, clamp)
        out[h] = [max(1.0, p1 * math.exp(mu + z * sig * spread_mult)) for z in Z]
    return out
