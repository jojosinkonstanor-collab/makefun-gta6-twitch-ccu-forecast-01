import sys, json, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import score_game, typical_miss, coverage
from loo5 import rolling, is_dg, is_ea, is_weak
from loo6 import load_full2, fit_anchor, loo_eval, practice_eval
from model_final import fit_final

def anchor_of(g): return max(g["v1"], g["pre_last"] or 0)

if __name__ == "__main__":
    games = load_full2()
    R1 = rolling(games)
    CR19 = [g for g in games if not is_dg(g) and not is_ea(g) and not is_weak(g)]
    sub_r1 = lambda g: g in R1
    big_r1 = lambda g: g in R1 and g["v1"] >= 100000

    def mk(train_fn, **kw):
        def pred(g, tr):
            r = train_fn(tr)
            w = [2.0 if x["console_only"] else 1.0 for x in r]
            return fit_final(g["v1"], anchor_of(g), [x["v"] for x in r],
                             [anchor_of(x) for x in r], train_weight=w, **kw)
        return pred
    def mk_practice(train, **kw):
        w = [2.0 if x["console_only"] else 1.0 for x in train]
        return lambda v1, pl: fit_final(v1, max(v1, pl or 0), [x["v"] for x in train],
                                        [anchor_of(x) for x in train], train_weight=w, **kw)

    for shrink in (0.0, 0.5, 1.0):
        loo_eval(games, mk(rolling, shrink=shrink), f"FINAL R1 shrink={shrink} (LOO R1)", sub_r1)
        loo_eval(games, mk(rolling, shrink=shrink), f"FINAL R1 shrink={shrink} (LOO bigR1)", big_r1)
        practice_eval(mk_practice(R1, shrink=shrink), f"FINAL R1 shrink={shrink}")
        print()
    for sm in (0.9, 1.0):
        loo_eval(games, mk(rolling, shrink=0.5, spread_mult=sm), f"FINAL R1 shrink=0.5 spread{sm} (LOO R1)", sub_r1)
        practice_eval(mk_practice(R1, shrink=0.5, spread_mult=sm), f"FINAL R1 shrink=0.5 spread{sm}")
    print()
    print("=== Hypothetical GTA VI q50 curves (FINAL R1 shrink=0.5) ===")
    for v1 in (200_000, 500_000, 1_000_000):
        pl = 1.3*v1
        w = [2.0 if x["console_only"] else 1.0 for x in R1]
        f = fit_final(v1, max(v1, pl), [x["v"] for x in R1], [anchor_of(x) for x in R1], train_weight=w, shrink=0.5)
        q50 = [round(f[h][2]) for h in range(2,25)]
        pk = max(q50); ph = q50.index(pk)+2
        print(f"v1={v1:>7,} peak={pk:>8,} (x{pk/v1:.2f} @h{ph}) dip_h3={q50[1]:>8,} h18={q50[16]:>8,} h24={q50[22]:>8,} "
              f"q05_h20={f[20][0]:.0f} q95_h20={f[20][4]:.0f}")
