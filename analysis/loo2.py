import sys, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import *

def size_model(v1, train, spread_mult=1.0, clamp=0.6, shrink=1.0,
               sigma_smooth=0, sigma_floor=0.10, use_empirical_resid=False,
               second_feature=None, time_weight=0.0):
    """General size-conditioned lognormal. second_feature: callable(game)->float (log-scale),
    pooled slope for it. time_weight: per-year decay weight (0 = none)."""
    p1 = max(1, v1)
    xs = [math.log(max(g["v1"],1)) for g in train]
    xbar = sum(xs)/len(xs)
    # per-game second feature
    f2 = None
    if second_feature:
        f2 = [second_feature(g) for g in train]
        f2bar = sum(f2)/len(f2)
    # weights for recency
    import datetime as dt
    yrs = []
    for g in train:
        ds = g.get("day_start")
        yrs.append(2026.0)  # placeholder; set below
    # rebuild with dates: need day_start on game dicts
    R = {h: [] for h in range(2,25)}
    for g in train:
        v1g = max(g["v"][0],1)
        for h in range(2,25):
            vh = g["v"][h-1]
            R[h].append(math.log(max(vh,1)/v1g) if vh>0 else None)
    # pooled slopes
    def pooled_slope(feat):
        num = den = 0.0
        for h in range(2,25):
            vals = [(feat[i], r) for i, r in enumerate(R[h]) if r is not None]
            mh = sum(r for _, r in vals)/len(vals)
            mf = sum(f for f, _ in vals)/len(vals)
            for f, r in vals:
                num += (f-mf)*(r-mh); den += (f-mf)**2
        return num/den if den>0 else 0.0
    b_pool = pooled_slope([x-xbar for x in xs])
    c_pool = pooled_slope([f-f2bar for f in f2]) if f2 is not None else 0.0
    xg = math.log(p1) - xbar
    fg = None
    if second_feature:
        fg = second_feature({"v1": v1, "v": [v1]+[0]*23})  # target game: need pre stats; handled by caller via dict
    out = {}
    resid_std = {}
    for h in range(2,25):
        vals = [(xs[i]-xbar, (f2[i]-f2bar) if f2 is not None else 0.0, r)
                for i, r in enumerate(R[h]) if r is not None]
        mh = sum(r for _,_,r in vals)/len(vals)
        n2 = d2 = 0.0
        for dx, _, r in vals:
            n2 += dx*(r-mh); d2 += dx*dx
        b_h = n2/d2 if d2>0 else 0.0
        b = (1-shrink)*b_pool + shrink*b_h
        # second-feature slope: use pooled only
        c = c_pool
        resid = [r - (mh + b*dx + c*df) for dx, df, r in vals]
        sig = math.sqrt(sum(e*e for e in resid)/max(len(resid)-3,1))
        resid_std[h] = max(sig, sigma_floor)
        eff = max(-clamp, min(clamp, b*xg))
        if fg is not None:
            eff += max(-clamp, min(clamp, c*(fg-f2bar)))
        mu = mh + eff
        out[h] = (mu, resid_std[h])
    if sigma_smooth:
        vals = [resid_std[h] for h in range(2,25)]
        sm = []
        for i in range(23):
            lo = max(0, i-sigma_smooth); hi = min(23, i+sigma_smooth+1)
            sm.append(sum(vals[lo:hi])/(hi-lo))
        for i, h in enumerate(range(2,25)):
            resid_std[h] = sm[i]
            out[h] = (out[h][0], sm[i])
    if use_empirical_resid:
        # pool standardized residuals across hours
        std_res = []
        for h in range(2,25):
            mu, sg = out[h]
            vals = [(xs[i]-xbar, (f2[i]-f2bar) if f2 is not None else 0.0, r)
                    for i, r in enumerate(R[h]) if r is not None]
            mh = sum(r for _,_,r in vals)/len(vals)
            n2 = d2 = 0.0
            for dx, _, r in vals:
                n2 += dx*(r-mh); d2 += dx*dx
            b_h = n2/d2 if d2>0 else 0.0
            b = (1-shrink)*b_pool + shrink*b_h
            for dx, df, r in vals:
                std_res.append((r - (mh + b*dx + c_pool*df))/max(resid_std[h], 1e-6))
        std_res.sort()
        eq = [quantile(std_res, t) for t in LEVELS]
        return {h: [max(1.0, p1*math.exp(out[h][0] + resid_std[h]*e*spread_mult)) for e in eq]
                for h in range(2,25)}
    return {h: [max(1.0, p1*math.exp(out[h][0] + z*resid_std[h]*spread_mult)) for z in Z]
            for h in range(2,25)}

def run_round2():
    games = load_games()
    cr = clean_rolling(games)
    sub = lambda g: g in cr
    big = lambda g: g in cr and g["v1"] >= 100000
    print("== clamp tuning (shrink=1.0) ==")
    for cl in (0.3, 0.45, 0.6, 0.8, 1.2, 99):
        run_loo(games, lambda g, tr, cl=cl: size_model(g["v1"], clean_rolling(tr), clamp=cl, shrink=1.0),
                f"clamp={cl} (clean-rolling)", subset=sub)
    print("== sigma smoothing / floor ==")
    for ss, sf in ((1, 0.10), (2, 0.10), (1, 0.15), (0, 0.15)):
        run_loo(games, lambda g, tr, ss=ss, sf=sf: size_model(g["v1"], clean_rolling(tr), clamp=0.6, shrink=1.0, sigma_smooth=ss, sigma_floor=sf),
                f"sigma_smooth={ss} floor={sf} (clean-rolling)", subset=sub)
    print("== empirical standardized residuals ==")
    run_loo(games, lambda g, tr: size_model(g["v1"], clean_rolling(tr), clamp=0.6, shrink=1.0, use_empirical_resid=True),
            "empirical resid (clean-rolling)", subset=sub)
    print("== per-game losses, S1 shrink=1.0 clamp=0.6 ==")
    _, per = run_loo(games, lambda g, tr: size_model(g["v1"], clean_rolling(tr), clamp=0.6, shrink=1.0),
                     "S1 final-ish (clean-rolling)", subset=sub)
    for name, l in sorted(per, key=lambda x: -x[1]):
        g = next(x for x in games if x["name"]==name)
        mark = "CR" if g in cr else ("DG" if is_daytime_global(g) else ("EA" if is_contaminated(g) else "wk"))
        print(f"  {mark} {name:42s} v1={g['v1']:8d} loss={l:.4f}")
    print("== big clean-rolling only (v1>=100k) ==")
    run_loo(games, lambda g, tr: size_model(g["v1"], clean_rolling(tr), clamp=0.6, shrink=1.0),
            "S1 (big clean-rolling)", subset=big)
    run_loo(games, lambda g, tr: baseline_forecast(g["v1"]),
            "M0 baseline (big clean-rolling)", subset=big)

if __name__ == "__main__":
    run_round2()
