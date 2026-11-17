#!/usr/bin/env bash
# Full check for the gta6-twitch-ccu-forecast-01 entry.
# Usage: bash tests/check.sh [/path/to/unpacked-kit]
#   - runs main.py on the kit's practice cases
#   - validates every output file's shape (grader rules)
#   - checks determinism (two runs byte-identical)
#   - scores against the practice truth if analysis/validation/practice_truth.json exists
#   - runs edge cases (daytime launch, no history, no comparables, huge hour 1)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KIT="${1:-$HOME/workspace/makefun/kit-gta6-twitch-ccu-forecast-01}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail=0
say() { echo "== $*"; }

say "run on kit practice cases"
mkdir -p "$TMP/out" "$TMP/out2"
MAKEFUN_INPUT_DIR="$KIT/input" MAKEFUN_OUTPUT_DIR="$TMP/out" python3 -P -s "$ROOT/main.py" || fail=1
MAKEFUN_INPUT_DIR="$KIT/input" MAKEFUN_OUTPUT_DIR="$TMP/out2" python3 -P -s "$ROOT/main.py" || fail=1

say "validate output shape"
python3 - "$TMP/out" <<'EOF' || fail=1
import json, sys
d = sys.argv[1]
for pid in ("p00","p01","p02","p03"):
    obj = json.load(open(f"{d}/{pid}.out"))
    assert set(obj) == {"hours"}, pid
    assert len(obj["hours"]) == 23, pid
    assert [r["hour"] for r in obj["hours"]] == list(range(2,25)), pid
    for r in obj["hours"]:
        assert set(r) == {"hour","q"}, pid
        assert len(r["q"]) == 5, pid
        assert all(isinstance(x,(int,float)) and 0 < x <= 1e9 for x in r["q"]), pid
        assert all(r["q"][i] <= r["q"][i+1] for i in range(4)), pid
    import os
    assert os.path.getsize(f"{d}/{pid}.out") <= 65536, pid
print("all 4 outputs valid")
EOF

say "determinism"
diff -r "$TMP/out" "$TMP/out2" >/dev/null && echo "two runs byte-identical" || { echo "NON-DETERMINISTIC"; fail=1; }

if [ -f "$ROOT/analysis/validation/practice_truth.json" ]; then
    say "score vs practice truth"
    (cd "$ROOT" && OUT_DIR="$TMP/out" python3 - <<'EOF') || fail=1
import json, math, os, sys
sys.path.insert(0, "analysis")
from loo import score_game, baseline_forecast
truth = json.load(open("analysis/validation/practice_truth.json"))
out = os.environ["OUT_DIR"]
tot_m = tot_b = 0.0
for pid in ("p00","p01","p02","p03"):
    v = [float(x) for x in truth[pid]["viewers"]]
    fc = {r["hour"]: r["q"] for r in json.load(open(f"{out}/{pid}.out"))["hours"]}
    lm, lb = score_game(v, fc), score_game(v, baseline_forecast(v[0]))
    tot_m += lm; tot_b += lb
    print(f"  {pid}: model={lm:.4f} baseline={lb:.4f}")
print(f"  MEAN: model={tot_m/4:.4f} baseline={tot_b/4:.4f}")
assert tot_m/4 < tot_b/4, "model must beat baseline on practice"
EOF
else
    echo "(no practice truth file; scoring skipped)"
fi

say "edge cases"
python3 - "$ROOT" "$TMP" <<'EOF' || fail=1
import json, copy, os, subprocess, sys
root, tmp = sys.argv[1], sys.argv[2]
base = json.load(open(os.path.expanduser("~/workspace/makefun/kit-gta6-twitch-ccu-forecast-01/input/p00.in")))
from datetime import datetime, timedelta, timezone
def iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
ds = datetime(2026,11,19,tzinfo=timezone.utc)
def mk(cid, v1, history=True, comps=True, notes="Rolling midnight local."):
    c = copy.deepcopy(base); c["id"] = cid
    c["day_start"] = iso(ds); c["checkpoint"] = iso(ds+timedelta(hours=1))
    c["known"] = [{"hour":1,"from":iso(ds),"until":iso(ds+timedelta(hours=1)),"viewers":v1,"channels":100,"reads":60}]
    c["notes"] = notes
    c["history_hourly"] = ([{"from":iso(ds-timedelta(hours=1)),"viewers":int(v1*1.2),"channels":90,"reads":60}] if history else [])
    c["history"] = []
    if not comps: c["comparables"] = []
    return c
cases = {
  "edge_dg": mk("edge_dg", 100, notes="Single simultaneous global unlock 2026-11-19T14:00Z."),
  "edge_nohist": mk("edge_nohist", 120000, history=False),
  "edge_nocomp": mk("edge_nocomp", 120000, comps=False),
  "edge_huge": mk("edge_huge", 2000000),
  "edge_tiny": mk("edge_tiny", 5),
}
for cid, c in cases.items():
    d = f"{tmp}/{cid}"; os.makedirs(f"{d}/input", exist_ok=True)
    json.dump(c, open(f"{d}/input/{cid}.in","w"))
    env = dict(os.environ, MAKEFUN_INPUT_DIR=f"{d}/input", MAKEFUN_OUTPUT_DIR=f"{d}/out")
    r = subprocess.run([sys.executable, "-P", "-s", f"{root}/main.py"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (cid, r.stderr)
    f = json.load(open(f"{d}/out/{cid}.out"))
    assert set(f)=={"hours"} and len(f["hours"])==23
    assert [x["hour"] for x in f["hours"]]==list(range(2,25))
    for row in f["hours"]:
        assert len(row["q"])==5 and all(0<x<=1e9 for x in row["q"])
        assert all(row["q"][i]<=row["q"][i+1] for i in range(4))
    print(f"  {cid}: ok")
EOF

if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "CHECKS FAILED"; exit 1; fi
