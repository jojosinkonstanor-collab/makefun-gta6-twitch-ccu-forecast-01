import sys, json, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import score_game, typical_miss, coverage, baseline_forecast, LEVELS, Z
from model_core import fit_predict
import statistics as st
from datetime import datetime, timedelta
def parse(t): return datetime.fromisoformat(t.replace("Z","+00:00"))

KIT = "/home/hatch/workspace/makefun/kit-gta6-twitch-ccu-forecast-01"

def load_full():
    c = json.load(open(f"{KIT}/input/p00.in"))
    games = []
    for comp in c["comparables"]:
        name = comp.get("name") or comp.get("game")
        if "Grand Theft Auto VI" in name: continue
        ds = parse(comp["day_start"])
        by = {parse(h["from"]): h["viewers"] for h in comp["hourly"]}
        v = [by.get(ds + timedelta(hours=h)) for h in range(24)]
        if any(x is None for x in v): continue
        unlock = comp.get("unlock", "")
        tier = "?"
        if "Tier:" in unlock: tier = unlock.split("Tier:")[1].split(".")[0].strip()
        plats = comp.get("platforms", "")
        games.append(dict(name=name, tier=tier, v=v, v1=v[0], platforms=plats,
                          peak=max(v), ph=v.index(max(v))+1,
                          console_only=("PC" not in plats)))
    return games

def is_dg(g): return g["v1"] < 5500
def is_ea(g): return g["tier"].startswith("early-access")
def is_weak(g): return g["tier"] == "weak"
def is_early_peak(g): return g["ph"] <= 6
def rolling(games):  # R1: rolling midnight shape
    return [g for g in games if not is_dg(g) and not is_ea(g) and not is_weak(g) and not is_early_peak(g)]

def fit_weighted(p1, train, weights, clamp=1.0, sigma_floor=0.10, spread_mult=1.0):
    """Weighted version of model_core.fit_predict."""
    p1 = max(1.0, float(p1))
    xs = [math.log(max(g["v1"], 1.0)) for g in train]
    # weighted mean of x
    sw = sum(weights)
    xbar = sum(w*x for w, x in zip(weights, xs)) / sw
    xg = math.log(p1) - xbar
    out = {}
    for h in range(2, 25):
        pairs = []
        for i, g in enumerate(train):
            vh = g["v"][h-1]
            if vh and vh > 0:
                pairs.append((weights[i], xs[i]-xbar, math.log(max(vh,1.0)/max(g["v"][0],1.0))))
        swp = sum(w for w,_,_ in pairs)
        mh = sum(w*r for w,_,r in pairs) / swp
        num = sum(w*dx*(r-mh) for w,dx,r in pairs)
        den = sum(w*dx*dx for w,dx,r in pairs)
        b = num/den if den > 0 else 0.0
        resid = [r-(mh+b*dx) for _,dx,r in pairs]
        wres = [w for w,_,_ in pairs]
        sig = math.sqrt(sum(w*e*e for w,e in zip(wres,resid))/max(swp-2,1))
        sig = max(sig, sigma_floor)
        mu = mh + max(-clamp, min(clamp, b*xg))
        out[h] = [max(1.0, p1*math.exp(mu + z*sig*spread_mult)) for z in Z]
    return out

def loo_eval(games, predict, label, subset):
    tot, n = 0.0, 0; tm, cov = [], []
    for i, g in enumerate(games):
        if not subset(g): continue
        train = games[:i]+games[i+1:]
        fc = predict(g, train)
        tot += score_game(g["v"], fc); n += 1
        tm.append(typical_miss(g["v"], fc)); cov.append(coverage(g["v"], fc))
    print(f"{label:46s} loss={tot/n:.4f} (n={n}) typ_miss={st.mean(tm):.1f}% cov90={st.mean(cov):.3f}")
    return tot/n

def practice_eval(predict, label):
    truth = json.load(open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis/validation/practice_truth.json"))
    tot = 0.0
    per = []
    for pid in ("p00","p01","p02","p03"):
        t = truth[pid]; v = [float(x) for x in t["viewers"]]
        fc = predict(v[0])
        l = score_game(v, fc); tot += l; per.append((pid, round(l,4)))
    print(f"{label:46s} PRACTICE loss={tot/4:.4f}  per-case={per}")
    return tot/4

if __name__ == "__main__":
    games = load_full()
    R1 = rolling(games)
    print(f"R1 rolling-only training n={len(R1)}: {[g['name'] for g in R1]}")
    print(f"console-only in R1: {[g['name'] for g in R1 if g['console_only']]}")
    print(f"early-peak excluded: {[g['name'] for g in games if is_early_peak(g) and not is_dg(g) and not is_ea(g)]}")
    print()
    sub_r1 = lambda g: g in R1
    # A: R1-only, unweighted
    loo_eval(games, lambda g, tr: fit_predict(g["v1"], [x["v"] for x in rolling(tr)], [x["v1"] for x in rolling(tr)], clamp=1.0),
             "A: R1-only unweighted (LOO R1)", sub_r1)
    practice_eval(lambda v1: fit_predict(v1, [x["v"] for x in R1], [x["v1"] for x in R1], clamp=1.0),
                  "A: R1-only unweighted")
    # B: R1-only, console-only x2
    def pred_B(g, tr):
        r = rolling(tr); w = [2.0 if x["console_only"] else 1.0 for x in r]
        return fit_weighted(g["v1"], r, w)
    loo_eval(games, pred_B, "B: R1 console x2 (LOO R1)", sub_r1)
    practice_eval(lambda v1: fit_weighted(v1, R1, [2.0 if x["console_only"] else 1.0 for x in R1]),
                  "B: R1 console x2")
    # C: R1-only, console-only x3
    def pred_C(g, tr):
        r = rolling(tr); w = [3.0 if x["console_only"] else 1.0 for x in r]
        return fit_weighted(g["v1"], r, w)
    loo_eval(games, pred_C, "C: R1 console x3 (LOO R1)", sub_r1)
    practice_eval(lambda v1: fit_weighted(v1, R1, [3.0 if x["console_only"] else 1.0 for x in R1]),
                  "C: R1 console x3")
    # D: console-only only
    def pred_D(g, tr):
        r = [x for x in rolling(tr) if x["console_only"]]
        return fit_predict(g["v1"], [x["v"] for x in r], [x["v1"] for x in r], clamp=1.0)
    loo_eval(games, pred_D, "D: console-only only (LOO R1)", sub_r1)
    practice_eval(lambda v1: fit_predict(v1, [x["v"] for x in R1 if x["console_only"]],
                                          [x["v1"] for x in R1 if x["console_only"]], clamp=1.0),
                  "D: console-only only")
    # E: R1 + console x2 + spread 0.9
    def pred_E(g, tr):
        r = rolling(tr); w = [2.0 if x["console_only"] else 1.0 for x in r]
        return fit_weighted(g["v1"], r, w, spread_mult=0.9)
    loo_eval(games, pred_E, "E: R1 console x2 spread0.9 (LOO R1)", sub_r1)
    practice_eval(lambda v1: fit_weighted(v1, R1, [2.0 if x["console_only"] else 1.0 for x in R1], spread_mult=0.9),
                  "E: R1 console x2 spread0.9")
