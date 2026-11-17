import sys, json, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import score_game, typical_miss, coverage, LEVELS, Z
from loo5 import rolling, is_dg, is_ea, is_weak
from loo6 import load_full2, fit_anchor, loo_eval, practice_eval
from model_core import fit_predict

def anchor_of(g): return max(g["v1"], g["pre_last"] or 0)

def ens(fc_list, weights=None):
    w = weights or [1.0]*len(fc_list); sw = sum(w); out = {}
    for h in range(2, 25):
        out[h] = [max(1.0, math.exp(sum(wi*math.log(max(fc[h][t],1e-9)) for wi,fc in zip(w,fc_list))/sw)) for t in range(5)]
    return out

if __name__ == "__main__":
    games = load_full2()
    R1 = rolling(games)
    CR19 = [g for g in games if not is_dg(g) and not is_ea(g) and not is_weak(g)]
    sub_r1 = lambda g: g in R1
    big_r1 = lambda g: g in R1 and g["v1"] >= 100000

    # M: 19-game + anchorF + console x2
    def pred_M(g, tr):
        r = [x for x in tr if x in CR19]
        w = [2.0 if x["console_only"] else 1.0 for x in r]
        return fit_anchor(g["v1"], anchor_of(g), r, [anchor_of(x) for x in r], weights=w)
    loo_eval(games, pred_M, "M: 19game + anchorF + console x2 (LOO R1)", sub_r1)
    loo_eval(games, pred_M, "M: 19game + anchorF + console x2 (LOO bigR1)", big_r1)
    practice_eval(lambda v1, pl: fit_anchor(v1, max(v1, pl or 0), CR19, [anchor_of(x) for x in CR19],
                                            weights=[2.0 if x["console_only"] else 1.0 for x in CR19]),
                  "M: 19game + anchorF + console x2")
    # I on big subset
    def pred_I(g, tr):
        r = rolling(tr)
        w = [2.0 if x["console_only"] else 1.0 for x in r]
        return fit_anchor(g["v1"], anchor_of(g), r, [anchor_of(x) for x in r], weights=w)
    loo_eval(games, pred_I, "I: R1 + anchorF + console x2 (LOO bigR1)", big_r1)
    # K on big subset
    def pred_K(g, tr):
        r19 = [x for x in tr if x in CR19]
        f1 = fit_predict(g["v1"], [x["v"] for x in r19], [x["v1"] for x in r19], clamp=1.0)
        f2 = fit_anchor(g["v1"], anchor_of(g), r19, [anchor_of(x) for x in r19])
        return ens([f1, f2])
    loo_eval(games, pred_K, "K: ens (LOO bigR1)", big_r1)
    # I spread 0.9 / 1.1
    practice_eval(lambda v1, pl: fit_anchor(v1, max(v1, pl or 0), R1, [anchor_of(x) for x in R1],
                                            weights=[2.0 if x["console_only"] else 1.0 for x in R1], spread_mult=0.9),
                  "I spread0.9")
    practice_eval(lambda v1, pl: fit_anchor(v1, max(v1, pl or 0), R1, [anchor_of(x) for x in R1],
                                            weights=[2.0 if x["console_only"] else 1.0 for x in R1], spread_mult=1.1),
                  "I spread1.1")
    print()
    print("=== Hypothetical GTA VI forecasts (median q50 curve) ===")
    # pretend target: v1 given, pre_last = 1.3*v1 (typical CET-midnight > hour1)
    for v1 in (200_000, 500_000, 1_000_000):
        pl = 1.3*v1
        fI = fit_anchor(v1, max(v1, pl), R1, [anchor_of(x) for x in R1],
                        weights=[2.0 if x["console_only"] else 1.0 for x in R1])
        fK = ens([fit_predict(v1, [x["v"] for x in CR19], [x["v1"] for x in CR19], clamp=1.0),
                  fit_anchor(v1, max(v1, pl), CR19, [anchor_of(x) for x in CR19])])
        for tag, f in (("I", fI), ("K", fK)):
            q50 = [round(f[h][2]) for h in range(2,25)]
            pk = max(q50); ph = q50.index(pk)+2
            print(f"v1={v1:>7,} [{tag}] q50 peak={pk:>8,} (x{pk/v1:.2f} @h{ph}) dip_h3={q50[1]:>8,} "
                  f"h18={q50[16]:>8,} h24={q50[22]:>8,} q05_h20={f[20][0]:.0f} q95_h20={f[20][4]:.0f}")
