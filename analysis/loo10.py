import sys, json, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import score_game
from loo5 import rolling, is_dg, is_ea, is_weak
from loo6 import load_full2, loo_eval, practice_eval
from model_final import fit_final

def anchor_of(g): return max(g["v1"], g["pre_last"] or 0)

def mk_pred(shrink, spread):
    def pred(g, tr):
        r = rolling(tr)
        w = [2.0 if x["console_only"] else 1.0 for x in r]
        return fit_final(g["v1"], anchor_of(g), [x["v"] for x in r], [anchor_of(x) for x in r],
                         train_weight=w, shrink=shrink, spread_mult=spread)
    return pred

def mk_practice(shrink, spread):
    def f(v1, pl):
        w = [2.0 if x["console_only"] else 1.0 for x in R1]
        return fit_final(v1, max(v1, pl or 0), [x["v"] for x in R1], [anchor_of(x) for x in R1],
                         train_weight=w, shrink=shrink, spread_mult=spread)
    return f

if __name__ == "__main__":
    games = load_full2()
    R1 = rolling(games)
    sub_r1 = lambda g: g in R1
    big_r1 = lambda g: g in R1 and g["v1"] >= 100000
    for shrink, spread in ((1.0, 1.0), (1.0, 0.9), (1.0, 0.85), (0.5, 0.9)):
        loo_eval(games, mk_pred(shrink, spread), f"FINAL shrink={shrink} spread={spread} (LOO R1)", sub_r1)
        loo_eval(games, mk_pred(shrink, spread), f"FINAL shrink={shrink} spread={spread} (LOO bigR1)", big_r1)
        practice_eval(mk_practice(shrink, spread), f"FINAL shrink={shrink} spread={spread}")
        print()
    print("=== Hypothetical GTA VI q50 (FINAL shrink=1.0 spread=0.9) ===")
    for v1 in (200_000, 500_000, 1_000_000, 1_500_000):
        pl = 1.3*v1
        w = [2.0 if x["console_only"] else 1.0 for x in R1]
        f = fit_final(v1, max(v1, pl), [x["v"] for x in R1], [anchor_of(x) for x in R1],
                      train_weight=w, shrink=1.0, spread_mult=0.9)
        q50 = [round(f[h][2]) for h in range(2,25)]
        pk = max(q50); ph = q50.index(pk)+2
        print(f"v1={v1:>7,} anchor={max(v1,pl):>7,} peak={pk:>8,} (x{pk/v1:.2f} of v1, x{pk/max(v1,pl):.2f} of anchor @h{ph}) "
              f"dip_h3={q50[1]:>8,} h18={q50[16]:>8,} h24={q50[22]:>8,} q05_h20={f[20][0]:.0f} q95_h20={f[20][4]:.0f}")
