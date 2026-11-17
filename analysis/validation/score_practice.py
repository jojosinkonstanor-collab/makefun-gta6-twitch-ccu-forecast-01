"""Score model vs house baseline on the 4 practice cases, using the
validation-only ground truth extracted from SullyGnome (see
extract_practice_truth.py). This reproduces the dev-phase grading."""
import sys, json, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import load_games, clean_rolling, score_game, typical_miss, coverage, baseline_forecast
from model_core import fit_predict

truth = json.load(open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis/validation/practice_truth.json"))
games = load_games()
cr = clean_rolling(games)
print(f"training on {len(cr)} clean-rolling comparables (as the practice cases see them)")
print()

tot_m, tot_b = 0.0, 0.0
for pid in ("p00","p01","p02","p03"):
    t = truth[pid]
    v = [float(x) for x in t["viewers"]]
    fc_m = fit_predict(v[0], [x["v"] for x in cr], [x["v1"] for x in cr], clamp=1.0)
    fc_b = baseline_forecast(v[0])
    lm, lb = score_game(v, fc_m), score_game(v, fc_b)
    tot_m += lm; tot_b += lb
    print(f"{pid} {t['name']:28s} v1={v[0]:8.0f} peak={max(v):8.0f} (x{max(v)/v[0]:.1f} @h{v.index(max(v))+1})")
    print(f"    model  loss={lm:.4f} typ_miss={typical_miss(v, fc_m):.1f}% cov90={coverage(v, fc_m):.3f}")
    print(f"    base   loss={lb:.4f} typ_miss={typical_miss(v, fc_b):.1f}% cov90={coverage(v, fc_b):.3f}")
    print(f"    model q50 by hour: {[round(fc_m[h][2]) for h in range(2,25)]}")
    print(f"    truth           : {[round(x) for x in v[1:]]}")
print()
print(f"MEAN practice loss: model={tot_m/4:.4f}  baseline={tot_b/4:.4f}  (house reports baseline ~0.387)")
