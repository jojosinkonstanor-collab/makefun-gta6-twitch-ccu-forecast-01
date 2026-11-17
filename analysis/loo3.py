import sys, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis")
from loo import *
from loo2 import size_model
import json
from datetime import datetime, timedelta
def parse(t): return datetime.fromisoformat(t.replace("Z","+00:00"))

KIT = "/home/hatch/workspace/makefun/kit-gta6-twitch-ccu-forecast-01"

def load_games_full():
    """Games with pre-release stats attached."""
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
        pre24 = [vv for t, vv in sorted(by.items()) if ds - timedelta(hours=24) <= t < ds]
        pre24 = [vv for vv in pre24 if vv is not None]
        import statistics as st
        pre24_med = st.median(pre24) if pre24 else 0
        games.append(dict(name=name, tier=tier, v=v, v1=v[0], day_start=comp["day_start"],
                          pre24_med=pre24_med, peak=max(v), ph=v.index(max(v))+1))
    return games

def run_round3():
    games = load_games_full()
    cr = [g for g in games if g["v1"] >= 5500 and not g["tier"].startswith("early-access") and g["tier"] != "weak"]
    sub = lambda g: g in cr
    print("clean rolling n =", len(cr))
    # baseline S1 with clamp 0.8/0.9
    for cl in (0.8, 0.9, 1.0):
        run_loo(games, lambda g, tr, cl=cl: size_model(g["v1"], [x for x in tr if x in cr], clamp=cl, shrink=1.0),
                f"S1 clamp={cl} (clean-rolling)", subset=sub)
    # second feature: ln(v1/pre24_med)
    def f_hype(g):
        return math.log(max(g["v1"],1)/max(g.get("pre24_med",1),1))
    run_loo(games, lambda g, tr: size_model(g["v1"], [x for x in tr if x in cr], clamp=0.9, shrink=1.0,
                                            second_feature=f_hype),
            "S1 + hype feature (clean-rolling)", subset=sub)
    # second feature: ln(pre24_med) itself
    def f_pre(g):
        return math.log(max(g.get("pre24_med",1),1))
    run_loo(games, lambda g, tr: size_model(g["v1"], [x for x in tr if x in cr], clamp=0.9, shrink=1.0,
                                            second_feature=f_pre),
            "S1 + pre-level feature (clean-rolling)", subset=sub)

if __name__ == "__main__":
    run_round3()
