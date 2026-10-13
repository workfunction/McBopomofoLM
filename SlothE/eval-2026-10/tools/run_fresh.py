"""Frozen fresh-set run (see plan.md pre-registration). usage: run_fresh.py labeled.jsonl out.json"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import sys, json, time; sys.path.insert(0, evalcfg.WORK)
import evalpipe, pipeline
W = evalcfg.WORK
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
items = [(r["id"], r["readings"], r["han"], "") for r in rows]
DATA = evalcfg.DATA_TXT; TW = W + "/data_twins.txt"
out = {"ids": [r["id"] for r in rows]}
t0 = time.time()
pipeline.CLI[1] = DATA; evalpipe._first.clear()
fw = evalpipe.first_walks(items)
out["A"] = {i: pipeline.text(fw[i]["stock"]) for i, *_ in items}
out["inwalk"] = {i: pipeline.text(fw[i]["inwalk"]) for i, *_ in items}
out["nodes_inwalk"] = {i: fw[i]["inwalk"] for i, *_ in items}
out["nreq"] = {i: len(fw[i]["requests"]) for i, *_ in items}
print("A done", time.time() - t0, flush=True)
out["B"] = evalpipe.run(items, "q35", 2, .5, False, False); print("B done", time.time() - t0, flush=True)
out["C"] = evalpipe.run(items, "q35", 1, .5, True, True); print("C done", time.time() - t0, flush=True)
pipeline.CLI[1] = TW; evalpipe._first.clear()
out["D"] = evalpipe.run(items, "q35", 1, .5, True, True); print("D done", time.time() - t0, flush=True)
out["E"] = evalpipe.run(items, ("both", .25), 1, .5, True, True); print("E done", time.time() - t0, flush=True)
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
