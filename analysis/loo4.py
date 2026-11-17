import sys, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import load_games, clean_rolling, score_game, typical_miss, coverage, baseline_forecast, LEVELS, Z, pinball
from model_core import fit_predict
import statistics as st

def run_loo2(games, predict, label, subset=None):
    tot, n = 0.0, 0; tm, cov = [], []
    for i, g in enumerate(games):
        train = games[:i] + games[i+1:]
        fc = predict(g, train)
        if fc is None: continue
        if subset is None or subset(g):
            tot += score_game(g["v"], fc); n += 1
            tm.append(typical_miss(g["v"], fc)); cov.append(coverage(g["v"], fc))
    print(f"{label:52s} loss={tot/max(n,1):.4f} (n={n}) typ_miss={st.mean(tm):.1f}% cov90={st.mean(cov):.3f}")
    return tot/max(n,1)

def clean_train(train):
    return clean_rolling(train)

if __name__ == "__main__":
    games = load_games()
    cr = clean_rolling(games)
    sub = lambda g: g in cr
    big = lambda g: g in cr and g["v1"] >= 100000
    for cl in (0.8, 1.0, 1.2, 1.5, 99):
        run_loo2(games, lambda g, tr, cl=cl: fit_predict(g["v1"],
                    [x["v"] for x in clean_train(tr)], [x["v1"] for x in clean_train(tr)], clamp=cl),
                 f"core clamp={cl} (clean-rolling)", subset=sub)
    for sm in (0.9, 1.0, 1.1):
        run_loo2(games, lambda g, tr, sm=sm: fit_predict(g["v1"],
                    [x["v"] for x in clean_train(tr)], [x["v1"] for x in clean_train(tr)], clamp=1.0, spread_mult=sm),
                 f"core clamp=1.0 spread x{sm} (clean-rolling)", subset=sub)
    print("--- big clean-rolling (v1>=100k) ---")
    run_loo2(games, lambda g, tr: fit_predict(g["v1"],
                [x["v"] for x in clean_train(tr)], [x["v1"] for x in clean_train(tr)], clamp=1.0),
             "core clamp=1.0 (big)", subset=big)
    run_loo2(games, lambda g, tr: fit_predict(g["v1"],
                [x["v"] for x in clean_train(tr)], [x["v1"] for x in clean_train(tr)], clamp=1.0, spread_mult=0.9),
             "core clamp=1.0 spread x0.9 (big)", subset=big)
    run_loo2(games, lambda g, tr: baseline_forecast(g["v1"]), "M0 baseline (big)", subset=big)
    print("--- all games with regime switch (daytime-global -> pooled lognormal on DG only) ---")
    def regime_predict(g, tr):
        if g["v1"] < 5500:
            dg = [x for x in tr if x["v1"] < 5500]
            if len(dg) >= 3:
                return fit_predict(g["v1"], [x["v"] for x in dg], [x["v1"] for x in dg], clamp=1.0)
        return fit_predict(g["v1"], [x["v"] for x in clean_train(tr)], [x["v1"] for x in clean_train(tr)], clamp=1.0)
    run_loo2(games, regime_predict, "regime-switched core (ALL)")
    run_loo2(games, lambda g, tr: baseline_forecast(g["v1"]), "M0 baseline (ALL)")
