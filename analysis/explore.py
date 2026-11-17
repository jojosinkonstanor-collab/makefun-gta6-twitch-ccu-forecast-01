import json, math
from datetime import datetime, timedelta
def parse(t): return datetime.fromisoformat(t.replace("Z","+00:00"))

KIT = "/home/hatch/workspace/makefun/kit-gta6-twitch-ccu-forecast-01"
c = json.load(open(f"{KIT}/input/p00.in"))

def day_curve(comp):
    ds = parse(comp["day_start"])
    by = {parse(h["from"]): h["viewers"] for h in comp["hourly"]}
    v = []
    for h in range(24):
        v.append(by.get(ds + timedelta(hours=h)))
    return v

def pre_stats(comp):
    ds = parse(comp["day_start"])
    pts = sorted(comp["hourly"], key=lambda h: h["from"])
    pre = [(parse(h["from"]), h["viewers"]) for h in pts if parse(h["from"]) < ds]
    pre24 = [v for t, v in pre if t >= ds - timedelta(hours=24)]
    pre6  = [v for t, v in pre if t >= ds - timedelta(hours=6)]
    pre14x = [v for t, v in pre if t < ds - timedelta(hours=36)]  # exclude last 36h
    import statistics as st
    return {
        "pre24_med": st.median(pre24) if pre24 else None,
        "pre6_mean": sum(pre6)/len(pre6) if pre6 else None,
        "pre6_last": pre6[-1] if pre6 else None,
        "pre14_max": max(pre14x) if pre14x else None,
        "pre14_med": st.median(pre14x) if pre14x else None,
    }

rows = []
for comp in c["comparables"]:
    name = comp.get("name") or comp.get("game")
    if "Grand Theft Auto VI" in name: continue
    unlock = comp.get("unlock","")
    tier = "?"
    if "Tier:" in unlock: tier = unlock.split("Tier:")[1].split(".")[0].strip()
    v = day_curve(comp)
    if any(x is None for x in v): print("MISSING day hours:", name, v); continue
    v1 = v[0]
    peak = max(v); ph = v.index(peak)+1
    ps = pre_stats(comp)
    rows.append(dict(name=name, tier=tier, v1=v1, peak=peak, ph=ph,
                     peak_v1=peak/max(v1,1), v1_pre24=v1/max(ps["pre24_med"] or 1,1),
                     v1_pre6=v1/max(ps["pre6_mean"] or 1,1),
                     pre14max_v1=(ps["pre14_max"] or 0)/max(v1,1),
                     pre14max_peak=(ps["pre14_max"] or 0)/max(peak,1),
                     pre24_med=ps["pre24_med"], pre6_last=ps["pre6_last"]))

print(f"{'game':42s} {'tier':24s} {'v1':>8s} {'peak':>8s} {'ph':>3s} {'pk/v1':>7s} {'v1/pre24':>8s} {'v1/pre6':>7s} {'pre14mx/v1':>10s} {'pre14mx/pk':>10s}")
for r in sorted(rows, key=lambda r: -r["peak_v1"]):
    print(f"{r['name'][:42]:42s} {r['tier'][:24]:24s} {r['v1']:8d} {r['peak']:8d} {r['ph']:3d} {r['peak_v1']:7.1f} {r['v1_pre24']:8.2f} {r['v1_pre6']:7.2f} {r['pre14max_v1']:10.2f} {r['pre14max_peak']:10.2f}")
