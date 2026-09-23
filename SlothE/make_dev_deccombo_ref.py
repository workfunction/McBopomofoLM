"""Read-only: dev walks of walk2's frozen 12M dec-combo config (beta 0.3, lambda 1.5, tau 0.9), made with
walk2's own functions (walk2/dec_combo.py get_nodes / contexts / decide / finals, walk2_cli --repin).

Decoder scores (--scores):
  live  : walk2 dec_combo.Dec(cache=False).live(ctx, top-3) per node, exactly as dec_combo.test_pass does
          for the test sets (sloth/build/dec_score, read-only). This is the reference used by the app test.
  cache : walk2's dev decoder cache results/walk2/dec_cache.jsonl. NOTE: that cache scored all 497
          empty-context (first-node) candidates in one dec_score call; 396 of them came back as exactly
          0.0, so first-node decisions on dev are corrupted there. Kept only to document the difference.
Writes only to the path given in --out.
"""
import argparse
import json
import sys
from pathlib import Path

W = Path("/Users/chingh/Workspace/agent/notes/scratch/ime-lm-poc")
sys.path.insert(0, str(W / "walk2"))
sys.path.insert(0, str(W / "sloth"))
import dec_combo as dc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", choices=["live", "cache"], default="live")
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = dc.get_nodes("12m", "dev", [0.3])
if a.scores == "cache":
    mem = {}
    for l in open(W / "results" / "walk2" / "dec_cache.jsonl", encoding="utf-8"):
        c, v, lp = json.loads(l)
        mem[(c, v)] = lp
    dec_of = lambda sid, k, ctx, top: [mem[(ctx, v)] for v, _ in top]
else:
    dec = dc.Dec(cache=False)
    live = {}
    for (b, sid), r in rows.items():
        for k, (n, c) in enumerate(zip(r["nodes"], dc.contexts(r["nodes"]))):
            if len(n[4]) >= 2:
                live[(sid, k)] = dec.live(c, [v for v, _ in n[4]])[0]
    dec_of = lambda sid, k, ctx, top: live[(sid, k)]
res = dc.finals("12m", "dev", rows, 0.3, dec_of, [(1.5, 0.9)])[(1.5, 0.9)]
with open(a.out, "w", encoding="utf-8") as f:
    for sid in sorted(res):
        f.write(json.dumps({"sent_id": sid, "cfg": ["12m", 0.3, 1.5, 0.9], "walk_nodes": res[sid]}, ensure_ascii=False) + "\n")
print(f"dev sentences {len(res)} scores={a.scores}")
