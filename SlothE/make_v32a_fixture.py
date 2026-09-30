"""v3.2a parity fixture: the full-sentence decoder gate of reports/w9.md (fix 1, config A: lambda 2, tau 0.5, top-3)
on whole dev sentences (plan F, no carried context), replayed read-only from walk2/w9.py.

Every sentence is one chunk: the encoder cache line, the W0 rows (walk2_cli --nodes), the gate's decisions through
w9.gate() with the decoder scores of results/w9 and earlier caches (a call that no cache holds is an error: this
script never starts a decoder), and the committed walk after the real re-walk (w9.repin: engine override + guard).
Writes McBopomofoTests/SlothEFixtures/v32a_dev_full_gate.jsonl, one row per sentence:
  sid, readings, line (encoder cache line, UNK split undone: the app applies it itself),
  dec [[context, candidate texts, scores]] (candidate text = candidate + the text right of the node),
  pins [[start, len, value]], final [[start, end, value]] (the committed walk)
The replay of the gate must reproduce results/w9/dev.F.s0.dev.chars.json for config A (checked below).
usage: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 15 $W/sloth/.venv/bin/python SlothE/make_v32a_fixture.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / "walk2"))
import w9  # noqa: E402
import w7  # noqa: E402

CFG = ("F", 2.0, 0.5, 0)
tmp = Path(tempfile.mkdtemp(prefix="v32afx."))
w9.TMP = tmp


class RecDec:
    """w9.Dec without a decoder process: the same caches, every call must be found; calls are recorded."""
    def __init__(self):
        self.mem, self.calls = {}, []
        src = [*(W / "results" / "walk2").glob("dec_cache.v2.*.jsonl")] + \
              [p for d in ("window3", "window3b", "unk", "w7", "w8", "w9") for p in (W / "results" / d).glob("dec_cache.*.jsonl")]
        for p in src:
            for l in open(p, encoding="utf-8"):
                try:
                    c, vs, sc = json.loads(l)
                except ValueError:
                    continue
                self.mem.setdefault((c, tuple(vs)), sc)

    def get(self, ctx, vals):
        sc = self.mem[(ctx, tuple(vals))]      # KeyError = a call no cache holds
        self.calls.append([ctx, list(vals), sc])
        return sc


def unsplit(line):
    t = line.rstrip("\n").split("\t")
    for i in range(3, len(t)):
        x = t[i].split(" ")
        if x[0] != "x":
            x[0] = f"{float(x[0]) + w7.UNK_ADJ:.6f}"
        t[i] = " ".join(x)
    return "\t".join(t)


D, sents = w9.load_sents("dev")
ch = w9.plan_chunks(sents, "F", 0)
enc = w7.Enc()
name = "w9.dev.F.s0.v32afx"
W0 = w9.w0_rows(name, "dev", ch, sents, enc)
dec = RecDec()
rows, reqs = {}, {}
for sid in sorted(ch):
    cid = f"{sid}#0"
    model, nodes = W0[cid]
    dec.calls = []
    pins = w9.gate(nodes, "", CFG, dec, {"req": 0, "chars": 0, "pins": 0}) if model else []
    rows[sid] = {"sid": sid, "readings": sents[sid]["readings"], "line": unsplit(enc.line(cid, [], sents[sid]["readings"])),
                 "dec": dec.calls, "pins": [list(p) for p in pins]}
    if pins:
        reqs[sid] = (cid, pins)
final = w9.repin(name, {sid: v for sid, v in reqs.items()})
for sid in rows:
    _, nodes = W0[f"{sid}#0"]
    rows[sid]["final"] = final[sid] if sid in final else [[n[0], n[1], n[2]] for n in nodes]
# the replay must equal what W9 measured for this config
chars = {sid: w7.chars_of_chunks([(0, len(sents[sid]["readings"]), rows[sid]["final"])], D.gold[sid]) for sid in D.gold}
want = json.loads((W / "results" / "w9" / "dev.F.s0.dev.chars.json").read_text())[w9.key(CFG)]
assert chars == want, "replay differs from results/w9/dev.F.s0.dev"
fx = Path(__file__).resolve().parents[1] / "McBopomofoTests" / "SlothEFixtures" / "v32a_dev_full_gate.jsonl"
with open(fx, "w", encoding="utf-8") as f:
    for sid in sorted(rows):
        f.write(json.dumps(rows[sid], ensure_ascii=False) + "\n")
print("sentences", len(rows), "with pins", sum(1 for r in rows.values() if r["pins"]), "decoder calls",
      sum(len(r["dec"]) for r in rows.values()), "replay == results/w9 config A ->", fx)
shutil.rmtree(tmp)
