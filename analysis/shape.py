import sys, json, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import score_game, typical_miss, coverage, LEVELS, Z
from loo5 import rolling, is_dg, is_ea, is_weak
from loo6 import load_full2, loo_eval, practice_eval
from loo import quantile
import statistics as st

def anchor_of(g): return max(g["v1"], g["pre_last"] or 0)

def shape_amp_predict(p1, anchor, train, weights=None, spread_mult=1.0, sigma_floor=0.10):
    """Amplitude-shape decomposition.
    P = peak/anchor; n_h = v_h/peak. log P regressed on log anchor (shrunk slope:
    use OLS slope, it is a single regression on n games). Shape: per-hour median
    of log n_h and its std. Quantiles combine amplitude + shape variance."""
    p1 = max(1.0, float(p1)); anchor = max(1.0, float(anchor))
    w = weights or [1.0]*len(train)
    # amplitude
    xs, lps = [], []
    for i, g in enumerate(train):
        a = max(anchor_of(g), 1.0)
        xs.append(math.log(a)); lps.append(math.log(max(g["peak"],1.0)/a))
    sw = sum(w); xbar = sum(wi*x for wi,x in zip(w,xs))/sw
    mp = sum(wi*lp for wi,lp in zip(w,lps))/sw
    num = sum(wi*(x-xbar)*(lp-mp) for wi,x,lp in zip(w,xs,lps))
    den = sum(wi*(x-xbar)**2 for wi,x in zip(w,xs))
    b = num/den if den>0 else 0.0
    resid_p = [lp-(mp+b*(x-xbar)) for x,lp in zip(xs,lps)]
    sig_p = math.sqrt(sum(wi*e*e for wi,e in zip(w,resid_p))/max(sw-2,1))
    # cap size extrapolation at training range
    xg = max(min(xs)-xbar, min(max(xs)-xbar, math.log(anchor)-xbar))
    mu_p = mp + b*xg
    # shape: per-hour log(v_h/peak)
    out = {}
    for h in range(2,25):
        vals = []
        for i, g in enumerate(train):
            vh = g["v"][h-1]
            if vh and vh>0:
                vals.append((w[i], math.log(max(vh,1.0)/max(g["peak"],1.0))))
        swp = sum(wi for wi,_ in vals)
        med = quantile(sorted(v for _,v in vals), 0.5)
        # std around median (use mean for variance)
        m = sum(wi*v for wi,v in vals)/swp
        var = sum(wi*(v-m)**2 for wi,v in vals)/swp
        sig_s = math.sqrt(max(var, 1e-6))
        sig = math.sqrt(sig_p**2 + sig_s**2)
        sig = max(sig, sigma_floor)
        mu = mu_p + med
        out[h] = [max(1.0, anchor*math.exp(mu + z*sig*spread_mult)) for z in Z]
    return out

if __name__ == "__main__":
    games = load_full2()
    R1 = rolling(games)
    CR19 = [g for g in games if not is_dg(g) and not is_ea(g) and not is_weak(g)]
    sub_r1 = lambda g: g in R1
    big_r1 = lambda g: g in R1 and g["v1"] >= 100000

    def pred_SA(g, tr, train_fn=rolling):
        r = train_fn(tr)
        w = [2.0 if x["console_only"] else 1.0 for x in r]
        return shape_amp_predict(g["v1"], anchor_of(g), r, weights=w)
    loo_eval(games, lambda g, tr: pred_SA(g, tr), "SA R1 console x2 (LOO R1)", sub_r1)
    loo_eval(games, lambda g, tr: pred_SA(g, tr), "SA R1 console x2 (LOO bigR1)", big_r1)
    practice_eval(lambda v1, pl: shape_amp_predict(v1, max(v1, pl or 0), R1,
                  weights=[2.0 if x["console_only"] else 1.0 for x in R1]),
                  "SA R1 console x2")
    # shape stability: median normalized curve for R1
    print()
    print("median normalized shape n_h (R1):")
    for h in range(1, 25):
        vals = sorted(math.log(g["v"][h-1]/g["peak"]) for g in R1 if g["v"][h-1] > 0)
        print(f"  h{h:2d}: median={math.exp(quantile(vals,0.5)):.3f}  q25={math.exp(quantile(vals,0.25)):.3f}  q75={math.exp(quantile(vals,0.75)):.3f}")
    # amplitude vs size
    print()
    print("log(peak/anchor) vs log(anchor) for R1:")
    for g in sorted(R1, key=lambda x: anchor_of(x)):
        print(f"  {g['name']:42s} anchor={anchor_of(g):8.0f} peak/anchor={g['peak']/anchor_of(g):6.2f}")
