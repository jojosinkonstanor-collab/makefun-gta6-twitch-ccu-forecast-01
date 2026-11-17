import sys, json, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import score_game, typical_miss, coverage, baseline_forecast, LEVELS, Z
from loo5 import load_full, rolling, is_dg, is_ea, is_weak, is_early_peak, fit_weighted
from model_core import fit_predict
import statistics as st
from datetime import datetime, timedelta
def parse(t): return datetime.fromisoformat(t.replace("Z","+00:00"))
KIT = "/home/hatch/workspace/makefun/kit-gta6-twitch-ccu-forecast-01"

def load_full2():
    """Adds pre_last (last hourly viewers before day_start) to each game."""
    c = json.load(open(f"{KIT}/input/p00.in"))
    games = load_full()
    byname = {}
    for comp in c["comparables"]:
        name = comp.get("name") or comp.get("game")
        if "Grand Theft Auto VI" in name: continue
        ds = parse(comp["day_start"])
        pts = sorted(comp["hourly"], key=lambda h: h["from"])
        pre = [(parse(h["from"]), h["viewers"]) for h in pts if parse(h["from"]) < ds]
        pre_last = pre[-1][1] if pre else None
        # latest within 6h before day_start
        pre6 = [v for t, v in pre if t >= ds - timedelta(hours=6)]
        byname[name] = (pre_last, pre6[-1] if pre6 else None)
    for g in games:
        g["pre_last"], g["pre6_last"] = byname[g["name"]]
    return games

def fit_anchor(p1, anchor, train, train_anchors, clamp=1.0, sigma_floor=0.10, spread_mult=1.0, weights=None):
    """Ratios vs anchor; size variable x = ln(anchor)."""
    p1 = max(1.0, float(p1)); anchor = max(1.0, float(anchor))
    xs = [math.log(max(a, 1.0)) for a in train_anchors]
    w = weights or [1.0]*len(train)
    sw = sum(w)
    xbar = sum(wi*x for wi, x in zip(w, xs))/sw
    xg = math.log(anchor) - xbar
    out = {}
    for h in range(2, 25):
        pairs = []
        for i, g in enumerate(train):
            vh = g["v"][h-1]; a = max(train_anchors[i], 1.0)
            if vh and vh > 0:
                pairs.append((w[i], xs[i]-xbar, math.log(max(vh,1.0)/a)))
        swp = sum(wi for wi,_,_ in pairs)
        mh = sum(wi*r for wi,_,r in pairs)/swp
        num = sum(wi*dx*(r-mh) for wi,dx,r in pairs); den = sum(wi*dx*dx for wi,dx,r in pairs)
        b = num/den if den>0 else 0.0
        resid = [r-(mh+b*dx) for _,dx,r in pairs]
        sig = math.sqrt(sum(wi*e*e for (wi,_,_),e in zip(pairs,resid))/max(swp-2,1))
        sig = max(sig, sigma_floor)
        mu = mh + max(-clamp, min(clamp, b*xg))
        out[h] = [max(1.0, anchor*math.exp(mu + z*sig*spread_mult)) for z in Z]
    return out

def loo_eval(games, predict, label, subset):
    tot, n = 0.0, 0; tm, cov = [], []
    for i, g in enumerate(games):
        if not subset(g): continue
        train = games[:i]+games[i+1:]
        fc = predict(g, train)
        tot += score_game(g["v"], fc); n += 1
        tm.append(typical_miss(g["v"], fc)); cov.append(coverage(g["v"], fc))
    print(f"{label:50s} loss={tot/n:.4f} (n={n}) typ_miss={st.mean(tm):.1f}% cov90={st.mean(cov):.3f}")
    return tot/n

def practice_eval(predict, label):
    truth = json.load(open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis/validation/practice_truth.json"))
    # pre_last for practice games from case history_hourly
    tot = 0.0; per = []
    for pid in ("p00","p01","p02","p03"):
        t = truth[pid]; v = [float(x) for x in t["viewers"]]
        case = json.load(open(f"{KIT}/input/{pid}.in"))
        hh = sorted(case["history_hourly"], key=lambda r: r["from"])
        ds = parse(case["day_start"])
        pre = [r for r in hh if parse(r["from"]) < ds]
        pre_last = pre[-1]["viewers"] if pre else v[0]
        fc = predict(v[0], pre_last)
        l = score_game(v, fc); tot += l; per.append((pid, round(l,4)))
    print(f"{label:50s} PRACTICE loss={tot/4:.4f}  per-case={per}")
    return tot/4

if __name__ == "__main__":
    games = load_full2()
    R1 = rolling(games)
    from loo import clean_rolling
    CR19 = [g for g in games if not is_dg(g) and not is_ea(g) and not is_weak(g)]
    sub_r1 = lambda g: g in R1
    print("pre_last vs v1 (R1):", [(g["name"], g["pre_last"], g["v1"]) for g in R1])
    print()
    # fair comparison on R1 subset:
    loo_eval(games, lambda g, tr: fit_predict(g["v1"], [x["v"] for x in CR19 if x in tr], [x["v1"] for x in CR19 if x in tr], clamp=1.0),
             "19-game clean-rolling (LOO R1)", sub_r1)
    loo_eval(games, lambda g, tr: fit_predict(g["v1"], [x["v"] for x in rolling(tr)], [x["v1"] for x in rolling(tr)], clamp=1.0),
             "A: R1-only (LOO R1)", sub_r1)
    # F: anchor = max(v1, pre_last); size x = ln(anchor); ratios vs anchor
    def pred_F(g, tr):
        r = rolling(tr)
        return fit_anchor(g["v1"], max(g["v1"], g["pre_last"] or 0), r,
                          [max(x["v1"], x["pre_last"] or 0) for x in r])
    loo_eval(games, pred_F, "F: anchor=max(v1,pre_last) (LOO R1)", sub_r1)
    practice_eval(lambda v1, pl: fit_anchor(v1, max(v1, pl or 0), R1, [max(x["v1"], x["pre_last"] or 0) for x in R1]),
                  "F: anchor=max(v1,pre_last)")
    # G: ratios vs v1, size x = ln(max(v1,pre_last))
    def fit_G(p1, anchor_size, train, train_sizes, clamp=1.0, weights=None):
        p1 = max(1.0, float(p1))
        xs = [math.log(max(a,1.0)) for a in train_sizes]
        w = weights or [1.0]*len(train)
        sw = sum(w); xbar = sum(wi*x for wi,x in zip(w,xs))/sw
        xg = math.log(max(anchor_size,1.0)) - xbar
        out = {}
        for h in range(2,25):
            pairs = []
            for i, g in enumerate(train):
                vh = g["v"][h-1]
                if vh and vh>0:
                    pairs.append((w[i], xs[i]-xbar, math.log(max(vh,1.0)/max(g["v"][0],1.0))))
            swp = sum(wi for wi,_,_ in pairs)
            mh = sum(wi*r for wi,_,r in pairs)/swp
            num = sum(wi*dx*(r-mh) for wi,dx,r in pairs); den = sum(wi*dx*dx for wi,dx,r in pairs)
            b = num/den if den>0 else 0.0
            resid = [r-(mh+b*dx) for _,dx,r in pairs]
            sig = max(math.sqrt(sum(wi*e*e for (wi,_,_),e in zip(pairs,resid))/max(swp-2,1)), 0.10)
            mu = mh + max(-clamp, min(clamp, b*xg))
            out[h] = [max(1.0, p1*math.exp(mu + z*sig)) for z in Z]
        return out
    def pred_G(g, tr):
        r = rolling(tr)
        return fit_G(g["v1"], max(g["v1"], g["pre_last"] or 0), r, [max(x["v1"], x["pre_last"] or 0) for x in r])
    loo_eval(games, pred_G, "G: ratios vs v1, size=max(v1,pre) (LOO R1)", sub_r1)
    practice_eval(lambda v1, pl: fit_G(v1, max(v1, pl or 0), R1, [max(x["v1"], x["pre_last"] or 0) for x in R1]),
                  "G: ratios vs v1, size=max(v1,pre)")
    # G2: G + console x2
    def pred_G2(g, tr):
        r = rolling(tr)
        w = [2.0 if x["console_only"] else 1.0 for x in r]
        return fit_G(g["v1"], max(g["v1"], g["pre_last"] or 0), r, [max(x["v1"], x["pre_last"] or 0) for x in r], weights=w)
    loo_eval(games, pred_G2, "G2: G + console x2 (LOO R1)", sub_r1)
    practice_eval(lambda v1, pl: fit_G(v1, max(v1, pl or 0), R1, [max(x["v1"], x["pre_last"] or 0) for x in R1],
                                       weights=[2.0 if x["console_only"] else 1.0 for x in R1]),
                  "G2: G + console x2")
    # baseline practice for reference
    practice_eval(lambda v1, pl: baseline_forecast(v1), "M0 baseline")
