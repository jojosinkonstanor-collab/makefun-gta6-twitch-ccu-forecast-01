"""Leave-one-out cross-validation over the kit comparables.

Each comparable is treated as a pseudo-case: known = hour-1 viewers of its
release day, truth = hours 2..24. Score = mean pinball loss on log scale,
exactly as the bounty scorer defines it. Baseline = house persistence
(sigma 0.15). All data comes from the kit's input/*.in files only.
"""
import json, math
from datetime import datetime, timedelta
import statistics as st

KIT = "/home/hatch/workspace/makefun/kit-gta6-twitch-ccu-forecast-01"
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

def parse(t): return datetime.fromisoformat(t.replace("Z","+00:00"))

def load_games():
    c = json.load(open(f"{KIT}/input/p00.in"))
    games = []
    for comp in c["comparables"]:
        name = comp.get("name") or comp.get("game")
        if "Grand Theft Auto VI" in name:  # target-history, not a launch
            continue
        ds = parse(comp["day_start"])
        by = {parse(h["from"]): h["viewers"] for h in comp["hourly"]}
        v = [by.get(ds + timedelta(hours=h)) for h in range(24)]
        if any(x is None for x in v):
            continue
        unlock = comp.get("unlock", "")
        tier = "?"
        if "Tier:" in unlock:
            tier = unlock.split("Tier:")[1].split(".")[0].strip()
        games.append(dict(name=name, tier=tier, v=v, v1=v[0],
                          peak=max(v), ph=v.index(max(v))+1))
    return games

def pinball(y, q, tau):
    y = max(y, 1); q = max(q, 1e-9)
    d = math.log(y) - math.log(q)
    return 2 * max(tau * d, (tau - 1) * d)

def score_game(v, forecast_q):
    """forecast_q: dict hour(2..24) -> list of 5 quantiles. Returns mean pinball."""
    tot, n = 0.0, 0
    for h in range(2, 25):
        y = max(v[h-1], 1)
        for tau, q in zip(LEVELS, forecast_q[h]):
            tot += pinball(y, q, tau); n += 1
    return tot / n

def typical_miss(v, forecast_q):
    errs = [abs(math.log(max(v[h-1],1)) - math.log(max(forecast_q[h][2],1e-9))) for h in range(2,25)]
    return 100 * (math.exp(sum(errs)/len(errs)) - 1)

def coverage(v, forecast_q):
    ok = sum(1 for h in range(2,25) if forecast_q[h][0] <= max(v[h-1],1) <= forecast_q[h][4])
    return ok / 23

# ---------------- models ----------------
def baseline_forecast(v1):
    p1 = max(1, v1)
    return {h: [round(p1*math.exp(z*0.15*math.sqrt(h-1)), 2) for z in Z] for h in range(2,25)}

def log_ratios(games):
    """Per hour h=2..24: list of ln(v_h / v_1) over games (v1>0, v_h>0)."""
    out = {h: [] for h in range(2,25)}
    for g in games:
        v1 = max(g["v"][0], 1)
        for h in range(2,25):
            vh = g["v"][h-1]
            if vh > 0:
                out[h].append(math.log(vh / v1))
    return out

def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 0: return 0.0
    if n == 1: return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos)); hi = min(lo+1, n-1)
    frac = pos - lo
    return sorted_vals[lo]*(1-frac) + sorted_vals[hi]*frac

def empirical_forecast(v1, ratios, smooth=0):
    p1 = max(1, v1)
    med = {}
    qs = {}
    for h in range(2,25):
        s = sorted(ratios[h])
        qs[h] = [quantile(s, t) for t in LEVELS]
    if smooth:
        # moving average across hours for each level (window 2*smooth+1)
        for tau_i in range(5):
            vals = [qs[h][tau_i] for h in range(2,25)]
            sm = []
            for i in range(23):
                lo = max(0, i-smooth); hi = min(23, i+smooth+1)
                sm.append(sum(vals[lo:hi])/(hi-lo))
            for i, h in enumerate(range(2,25)):
                qs[h][tau_i] = sm[i]
    return {h: [max(1.0, p1*math.exp(q)) for q in qs[h]] for h in range(2,25)}

def lognormal_forecast(v1, ratios, spread_mult=1.0, smooth=0, weights=None):
    """Per hour: mu = weighted mean of log-ratios, sigma = weighted std.
    Quantiles via normal z. Optionally smooth mu/sigma across hours."""
    p1 = max(1, v1)
    mu, sig = {}, {}
    for h in range(2,25):
        vals = ratios[h]
        w = weights[h] if weights else [1.0]*len(vals)
        sw = sum(w)
        m = sum(wi*vi for wi, vi in zip(w, vals))/sw
        var = sum(wi*(vi-m)**2 for wi, vi in zip(w, vals))/sw
        mu[h] = m; sig[h] = math.sqrt(max(var, 1e-6))
    if smooth:
        for d in (mu, sig):
            vals = [d[h] for h in range(2,25)]
            sm = []
            for i in range(23):
                lo = max(0, i-smooth); hi = min(23, i+smooth+1)
                sm.append(sum(vals[lo:hi])/(hi-lo))
            for i, h in enumerate(range(2,25)):
                d[h] = sm[i]
    out = {}
    for h in range(2,25):
        s = sig[h]*spread_mult
        out[h] = [max(1.0, p1*math.exp(mu[h] + z*s)) for z in Z]
    return out

# regime helpers
def is_daytime_global(g):   # tiny hour 1: nothing big unlocked yet
    return g["v1"] < 5500
def is_contaminated(g):
    return g["tier"].startswith("early-access")
def is_weak(g):
    return g["tier"] == "weak"

def clean_rolling(games):
    return [g for g in games if not is_daytime_global(g) and not is_contaminated(g) and not is_weak(g)]

def run_loo(games, model_fn, label, subset=None):
    tot, n = 0.0, 0
    tm, cov = [], []
    per = []
    for i, g in enumerate(games):
        train = games[:i] + games[i+1:]
        fc = model_fn(g, train)
        if fc is None:
            continue
        l = score_game(g["v"], fc)
        per.append((g["name"], l))
        if subset is None or subset(g):
            tot += l; n += 1
            tm.append(typical_miss(g["v"], fc)); cov.append(coverage(g["v"], fc))
    print(f"{label:55s} loss={tot/max(n,1):.4f} (n={n}) typ_miss={100*(st.mean([math.log(1+t/100) for t in tm]) if tm else 0):.1f}% cov90={st.mean(cov) if cov else 0:.3f}")
    return tot/max(n,1), per

if __name__ == "__main__":
    games = load_games()
    print(f"{len(games)} launch games")
    print("daytime-global:", [g['name'] for g in games if is_daytime_global(g)])
    print("contaminated:", [g['name'] for g in games if is_contaminated(g)])
    print("weak:", [g['name'] for g in games if is_weak(g)])
    print("clean rolling n =", len(clean_rolling(games)))
    print()
    # M0 baseline
    run_loo(games, lambda g, tr: baseline_forecast(g["v1"]), "M0 baseline persistence")
    run_loo(games, lambda g, tr: baseline_forecast(g["v1"]), "M0 baseline (clean-rolling subset)",
            subset=lambda g: g in clean_rolling(games))
    # M1 pooled empirical
    run_loo(games, lambda g, tr: empirical_forecast(g["v1"], log_ratios(tr)), "M1 pooled empirical quantiles")
    # M2 clean-rolling empirical (train filtered; target any)
    run_loo(games, lambda g, tr: empirical_forecast(g["v1"], log_ratios(clean_rolling(tr))), "M2 clean-rolling empirical")
    run_loo(games, lambda g, tr: empirical_forecast(g["v1"], log_ratios(clean_rolling(tr))), "M2 clean-rolling (clean-rolling subset)",
            subset=lambda g: g in clean_rolling(games))
    # M3 clean-rolling lognormal
    for sm in (1.0, 1.2, 1.4):
        run_loo(games, lambda g, tr, sm=sm: lognormal_forecast(g["v1"], log_ratios(clean_rolling(tr)), spread_mult=sm),
                f"M3 clean-rolling lognormal x{sm} (clean-rolling subset)",
                subset=lambda g: g in clean_rolling(games))
    # M3b smoothed lognormal
    for sm in (1.0, 1.2):
        run_loo(games, lambda g, tr, sm=sm: lognormal_forecast(g["v1"], log_ratios(clean_rolling(tr)), spread_mult=sm, smooth=1),
                f"M3b clean-rolling lognormal smooth1 x{sm} (clean-rolling subset)",
                subset=lambda g: g in clean_rolling(games))

# ---------------- size-conditioned variants ----------------
def size_lognormal_forecast(v1, train, spread_mult=1.0, clamp=0.6, shared_slope=True, shrink=0.5):
    """mu_h = a_h + b_h * x, x = ln(v1) - mean(ln v1_train), b_h shrunk toward pooled b.
    sigma_h = residual std (floor 0.10). x clamped to +/-clamp in effect."""
    import statistics as st
    p1 = max(1, v1)
    xs = [math.log(max(g["v1"],1)) for g in train]
    xbar = sum(xs)/len(xs)
    xg = math.log(p1) - xbar
    # per-hour ratios
    ratios = log_ratios(train)
    # pooled slope: regress (r_hi - mean_h) on (x_i - xbar)
    num = den = 0.0
    for h in range(2,25):
        vals = ratios[h]
        mh = sum(vals)/len(vals)
        for i, g in enumerate(train):
            # need per-game ratio; rebuild
            pass
    # rebuild per-game ratio matrix aligned with train order
    R = {h: [] for h in range(2,25)}
    for g in train:
        v1g = max(g["v"][0],1)
        for h in range(2,25):
            vh = g["v"][h-1]
            R[h].append(math.log(max(vh,1)/v1g) if vh>0 else None)
    num = den = 0.0
    for h in range(2,25):
        vals = [r for r in R[h] if r is not None]
        mh = sum(vals)/len(vals)
        for i, r in enumerate(R[h]):
            if r is None: continue
            dx = xs[i]-xbar
            num += dx*(r-mh); den += dx*dx
    b_pool = num/den if den>0 else 0.0
    out = {}
    for h in range(2,25):
        vals = [(xs[i]-xbar, r) for i, r in enumerate(R[h]) if r is not None]
        mh = sum(r for _, r in vals)/len(vals)
        # per-hour OLS slope
        n2 = d2 = 0.0
        for dx, r in vals:
            n2 += dx*(r-mh); d2 += dx*dx
        b_h = n2/d2 if d2>0 else 0.0
        b = (1-shrink)*b_pool + shrink*b_h if shared_slope else b_h
        a = mh  # intercept at xbar
        resid = [r - (a + b*dx) for dx, r in vals]
        sig = math.sqrt(sum(e*e for e in resid)/max(len(resid)-2,1))
        sig = max(sig, 0.10)
        eff = max(-clamp, min(clamp, b*xg))
        mu = a + eff
        out[h] = [max(1.0, p1*math.exp(mu + z*sig*spread_mult)) for z in Z]
    return out

def kernel_lognormal_forecast(v1, train, tau=1.2, spread_mult=1.0):
    p1 = max(1, v1)
    lx = math.log(p1)
    R = {h: [] for h in range(2,25)}
    W = []
    for g in train:
        w = math.exp(-((math.log(max(g["v1"],1))-lx)**2)/(2*tau*tau))
        W.append(w)
        v1g = max(g["v"][0],1)
        for h in range(2,25):
            vh = g["v"][h-1]
            R[h].append(math.log(max(vh,1)/v1g) if vh>0 else None)
    out = {}
    for h in range(2,25):
        pairs = [(W[i], r) for i, r in enumerate(R[h]) if r is not None]
        sw = sum(w for w,_ in pairs)
        if sw <= 1e-9:
            pairs = [(1.0, r) for _, r in pairs]; sw = len(pairs)
        mu = sum(w*r for w,r in pairs)/sw
        var = sum(w*(r-mu)**2 for w,r in pairs)/sw
        sig = max(math.sqrt(var), 0.10)
        out[h] = [max(1.0, p1*math.exp(mu + z*sig*spread_mult)) for z in Z]
    return out

def run_more():
    games = load_games()
    cr = clean_rolling(games)
    sub = lambda g: g in cr
    for shrink in (0.0, 0.5, 1.0):
        run_loo(games, lambda g, tr, s=shrink: size_lognormal_forecast(g["v1"], clean_rolling(tr), shrink=s),
                f"S1 size-lognormal shrink={shrink} (clean-rolling)", subset=sub)
    for tau in (0.8, 1.2, 1.8):
        run_loo(games, lambda g, tr, t=tau: kernel_lognormal_forecast(g["v1"], clean_rolling(tr), tau=t),
                f"S3 kernel tau={tau} (clean-rolling)", subset=sub)
    # size model on ALL games (for reference)
    run_loo(games, lambda g, tr: size_lognormal_forecast(g["v1"], clean_rolling(tr), shrink=0.5),
            "S1 size-lognormal shrink=0.5 (ALL games)")
