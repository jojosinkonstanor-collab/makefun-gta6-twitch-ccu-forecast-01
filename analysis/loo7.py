import sys, json, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import score_game, typical_miss, coverage, LEVELS, Z
from loo5 import rolling, is_dg, is_ea, is_weak, is_early_peak
from loo6 import load_full2, fit_anchor, loo_eval, practice_eval
from model_core import fit_predict

def anchor_of(g):
    return max(g["v1"], g["pre_last"] or 0)

def ens(fc_list, weights=None):
    """Geometric-mean ensemble of forecasts (average in log space)."""
    w = weights or [1.0]*len(fc_list)
    sw = sum(w)
    out = {}
    for h in range(2, 25):
        row = []
        for t in range(5):
            s = sum(wi*math.log(max(fc[h][t], 1e-9)) for wi, fc in zip(w, fc_list))/sw
            row.append(max(1.0, math.exp(s)))
        out[h] = row
    return out

if __name__ == "__main__":
    games = load_full2()
    R1 = rolling(games)
    CR19 = [g for g in games if not is_dg(g) and not is_ea(g) and not is_weak(g)]
    sub_r1 = lambda g: g in R1

    # H: 19-game training + anchor F
    def pred_H(g, tr):
        r = [x for x in tr if x in CR19]
        return fit_anchor(g["v1"], anchor_of(g), r, [anchor_of(x) for x in r])
    loo_eval(games, pred_H, "H: 19-game + anchorF (LOO R1)", sub_r1)
    practice_eval(lambda v1, pl: fit_anchor(v1, max(v1, pl or 0), CR19, [anchor_of(x) for x in CR19]),
                  "H: 19-game + anchorF")
    # I: R1 + anchorF + console x2
    def pred_I(g, tr):
        r = rolling(tr)
        w = [2.0 if x["console_only"] else 1.0 for x in r]
        return fit_anchor(g["v1"], anchor_of(g), r, [anchor_of(x) for x in r], weights=w)
    loo_eval(games, pred_I, "I: R1 + anchorF + console x2 (LOO R1)", sub_r1)
    practice_eval(lambda v1, pl: fit_anchor(v1, max(v1, pl or 0), R1, [anchor_of(x) for x in R1],
                                            weights=[2.0 if x["console_only"] else 1.0 for x in R1]),
                  "I: R1 + anchorF + console x2")
    # J: ensemble(19-game v1-anchored, R1 anchorF)
    def pred_J(g, tr):
        r19 = [x for x in tr if x in CR19]
        f1 = fit_predict(g["v1"], [x["v"] for x in r19], [x["v1"] for x in r19], clamp=1.0)
        r1 = rolling(tr)
        f2 = fit_anchor(g["v1"], anchor_of(g), r1, [anchor_of(x) for x in r1])
        return ens([f1, f2])
    loo_eval(games, pred_J, "J: ens(19game-v1, R1-anchorF) (LOO R1)", sub_r1)
    practice_eval(lambda v1, pl: ens([
        fit_predict(v1, [x["v"] for x in CR19], [x["v1"] for x in CR19], clamp=1.0),
        fit_anchor(v1, max(v1, pl or 0), R1, [anchor_of(x) for x in R1])]),
        "J: ens(19game-v1, R1-anchorF)")
    # K: ensemble(19-game v1, 19-game anchorF)
    def pred_K(g, tr):
        r19 = [x for x in tr if x in CR19]
        f1 = fit_predict(g["v1"], [x["v"] for x in r19], [x["v1"] for x in r19], clamp=1.0)
        f2 = fit_anchor(g["v1"], anchor_of(g), r19, [anchor_of(x) for x in r19])
        return ens([f1, f2])
    loo_eval(games, pred_K, "K: ens(19game-v1, 19game-anchorF) (LOO R1)", sub_r1)
    practice_eval(lambda v1, pl: ens([
        fit_predict(v1, [x["v"] for x in CR19], [x["v1"] for x in CR19], clamp=1.0),
        fit_anchor(v1, max(v1, pl or 0), CR19, [anchor_of(x) for x in CR19])]),
        "K: ens(19game-v1, 19game-anchorF)")
    # L: 19-game v1-anchored, spread 0.9 (reference best-simple)
    loo_eval(games, lambda g, tr: fit_predict(g["v1"], [x["v"] for x in CR19 if x in tr], [x["v1"] for x in CR19 if x in tr], clamp=1.0, spread_mult=0.9),
             "L: 19-game v1 spread0.9 (LOO R1)", sub_r1)
    practice_eval(lambda v1, pl: fit_predict(v1, [x["v"] for x in CR19], [x["v1"] for x in CR19], clamp=1.0, spread_mult=0.9),
                  "L: 19-game v1 spread0.9")
