"""Extract the 4 practice launches' release-day hourly 'Average viewers' from
SullyGnome (the same public source the kit cites in input/NOTICE.txt).

Fetched once, 2026-11-17, 4 requests total (>=2 s apart), 'Average viewers'
dataset only. Used ONLY for offline validation of the model against the
practice cases -- never read by the shipped program (main.py only reads the
case file it is given at run time) and never shipped as entry data.

Files: /tmp/{wolverine_sep2026,rdr2_oct2018,gow_nov2022,sm2_oct2023}.json
"""
import json
from datetime import datetime, timedelta

SPECS = [
    ("p00", "Marvel's Wolverine", "/tmp/wolverine_sep2026.json", "2026-09-15", 23357),
    ("p01", "Red Dead Redemption 2", "/tmp/rdr2_oct2018.json", "2018-10-26", 116550),
    ("p02", "God of War Ragnarök", "/tmp/gow_nov2022.json", "2022-11-09", 269517),
    ("p03", "Marvel's Spider-Man 2", "/tmp/sm2_oct2023.json", "2023-10-20", 119087),
]

out = {}
for pid, name, path, day, known_v1 in SPECS:
    d = json.load(open(path))
    labels = d["data"]["labels"]
    avg = next(ds for ds in d["data"]["datasets"] if ds["label"] == "Average viewers")
    series = {labels[i]: avg["data"][i] for i in range(len(labels))}
    day_start = datetime.fromisoformat(day)
    v = []
    for h in range(24):
        t = (day_start + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M")
        v.append(series.get(t))
    print(f"{pid} {name}: hour1 extracted={v[0]} (kit known={known_v1}) match={v[0]==known_v1} "
          f"missing={[i+1 for i,x in enumerate(v) if x is None]}")
    print("   day curve:", v)
    out[pid] = {"name": name, "day_start": day, "viewers": v}

json.dump(out, open("/home/hatch/workspace/makefun/gta6-twitch-ccu-forecast-01/analysis/validation/practice_truth.json","w"), indent=1)
print("saved analysis/validation/practice_truth.json")
