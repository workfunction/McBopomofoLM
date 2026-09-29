"""v3.1a parity fixture: chunked typing with carried context (reports/w7.md), dev, seed 1, random cuts.

Replays walk2/w7.py conditions N and Ce1 read-only: the chunking (seed 1), the per-round encoder cache lines w7
wrote (results/w7/tmp/dev.s1.<cond>.r<r>.tsv) and the decoder scores it used (its caches), through w7's own
walk_batch (walk2_cli + dec_combo). The replay must reproduce results/w7/dev.s1.chars.json exactly. Scratch
files go to a temp dir. Writes McBopomofoTests/SlothEFixtures/v31a_dev_chunks.jsonl, one row per sentence:
  sid, readings, chunks [[s, e]], and per condition N / Ce1 one entry per chunk:
    line   cache line of the chunk positions (UNK split undone: the app applies it itself)
    enc    encoder context w7 used (Ce1: the sentence's readings before the chunk; N: none)
    ctx    decoder text context w7 prepended (Ce1: this sentence's committed text, last 64 chars)
    dec    the decoder calls of this chunk [[context, candidates, scores]]
    final  the committed walk [[s, e, value]] (chunk-relative)
usage: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 15 $W/sloth/.venv/bin/python SlothE/make_v31a_fixture.py
"""
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / "walk2"))
import w7  # noqa: E402
import dec_combo as dc  # noqa: E402

SET, SEED = "dev", 1
tmp = Path(tempfile.mkdtemp(prefix="v31afx."))
w7.TMP = tmp
SRC = W / "results" / "w7" / "tmp"


class RecDec:
    """w7.Dec without a decoder process: its caches (same load order), fill() must find every call."""
    def __init__(self):
        self.mem = {}
        for p in [*(W / "results" / "walk2").glob("dec_cache.v2.*.jsonl"), *(W / "results" / "unk").glob("dec_cache.*.jsonl"),
                  *(W / "results" / "w7").glob("dec_cache.*.jsonl")]:
            for l in open(p, encoding="utf-8"):
                c, vs, sc = json.loads(l)
                self.mem.setdefault((c, tuple(vs)), sc)

    def fill(self, calls):
        miss = {(c, tuple(v)) for c, v in calls} - set(self.mem)
        assert not miss, f"{len(miss)} decoder calls not cached"


calls = {}                                           # cid -> [[ctx, cands, scores]]
_finals, PREFIX = dc.finals, {}
def finals(size, name, rows, beta, dec_of, configs):  # noqa: E302
    def rec(sid, k, ctx, top):
        sc = dec_of(sid, k, ctx, top)
        calls.setdefault(sid, []).append([PREFIX[sid] + ctx, [v for v, _ in top], sc])
        return sc
    return _finals(size, name, rows, beta, rec, configs)
dc.finals = finals


def unsplit(line):
    t = line.rstrip("\n").split("\t")
    for i in range(3, len(t)):
        x = t[i].split(" ")
        if x[0] != "x":
            x[0] = f"{float(x[0]) + w7.UNK_ADJ:.6f}"
        t[i] = " ".join(x)
    return "\t".join(t)


D = w7.cv.Data(SET)
sents = {json.loads(l)["sent_id"]: json.loads(l) for l in open(D.snap / "sentences.jsonl", encoding="utf-8")}
sents = {k: v for k, v in sents.items() if k in D.gold}
rng = random.Random(SEED)
ch = {sid: w7.chunking(len(s["readings"]), rng) for sid, s in sorted(sents.items())}
saved = json.loads((W / "results" / "w7" / f"{SET}.s{SEED}.chunks.json").read_text())
assert {k: [list(x) for x in v] for k, v in ch.items()} == saved, "chunking differs from w7"
rd = {sid: s["readings"] for sid, s in sents.items()}
want_chars = json.loads((W / "results" / "w7" / f"{SET}.s{SEED}.chars.json").read_text())
dec = RecDec()
out = {sid: {"sid": sid, "readings": rd[sid], "chunks": ch[sid]} for sid in sorted(ch)}
for cond in ("N", "Ce1"):
    committed = {sid: [] for sid in sents}
    for r in range(max(len(c) for c in ch.values())):
        src = {l.split("\t", 1)[0]: l for l in open(SRC / f"{SET}.s{SEED}.{cond}.r{r}.tsv", encoding="utf-8")}
        lines, model, PREFIX = {}, {}, {}
        for sid, cs in ch.items():
            if r >= len(cs):
                continue
            s, e = cs[r]
            cid = f"{sid}#{r}"
            ctx_text = "".join(w7.text(n) for *_, n in committed[sid]) if cond == "Ce1" else ""
            lines[cid] = src[cid]
            assert src[cid].split("\t")[2].split(" ") == rd[sid][s:e]
            model[cid] = e - s >= 2 or (cond == "Ce1" and s > 0)
            PREFIX[cid] = ctx_text[-64:]
        got = w7.walk_batch(f"{SET}.s{SEED}.{cond}.r{r}", lines, model, PREFIX, dec)
        for cid, nodes in got.items():
            sid = cid.rsplit("#", 1)[0]
            s, e = ch[sid][r]
            committed[sid].append((s, e, nodes))
            out[sid].setdefault(cond, []).append({
                "line": unsplit(lines[cid]), "enc": rd[sid][:s] if cond == "Ce1" else [], "ctx": PREFIX[cid],
                "dec": calls.get(cid, []), "final": nodes})
        calls.clear()
    got_chars = {sid: w7.chars_of_chunks(committed[sid], D.gold[sid]) for sid in D.gold}
    assert got_chars == want_chars[cond], f"replay of {cond} differs from w7"
    print(cond, "replay == results/w7", len(got_chars), "sentences")
fx = Path(__file__).resolve().parents[1] / "McBopomofoTests" / "SlothEFixtures" / "v31a_dev_chunks.jsonl"
with open(fx, "w", encoding="utf-8") as f:
    for sid in sorted(out):
        f.write(json.dumps(out[sid], ensure_ascii=False) + "\n")
print("chunks", sum(len(v) for v in ch.values()), "decoder calls", sum(len(c["dec"]) for o in out.values() for c in o["Ce1"]),
      "->", fx)
shutil.rmtree(tmp)
