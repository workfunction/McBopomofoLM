"""v3.0.2 parity fixtures: UNK variant 1 (reports/unk.md) + 1-syllable LM key h (reports/w5.md).

Runs walk2's own pipeline read-only (walk2/unk.py run() with walk2/w5.py's cache-only decoder scores and w5_cli,
variant 1): the decoder pins, the app walk, the windows and the v2 order. Scratch caches go to a temp dir, so
walk2/ and results/ are not written. Writes to McBopomofoTests/SlothEFixtures/:
  v302_dev_walks.jsonl   500 dev sentences: cache line (unshifted 25M log p), A' decoder pins, unk_cli v1 walk
  v302_dev_units.jsonl   143 dev units (== results/w5/dev.v1.units.jsonl) + cache line + want = w5.order(u, "h")
  v302_single.jsonl      1,032 single-syllable readings: cache line, pins, the A' walk, window, b2, and (v3.0.2b: a
                         1-reading buffer shows the stock walk) stock = McBopomofo's first candidate (the stock walk
                         of a 1-reading grid), want = w5.order(h, single) with the stock value shown
usage: PYTHONDONTWRITEBYTECODE=1 nice -n 15 $W/sloth/.venv/bin/python SlothE/make_v302_fixtures.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / "walk2"))
import w5  # noqa: E402  (imports unk, window3, window3b)
import unk  # noqa: E402
import window3 as w3  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="v302fx."))
unk.UC = tmp
unk.OUT = w3.OUT = w5.OUT = tmp
shutil.copyfile(W / "walk2" / "w5_cache" / "single_25m.tsv", tmp / "single_25m.tsv")
req = {}
_sh = w3.sh
def cap(cmd, **kw):                                   # noqa: E302
    if any(c.endswith("/unk_cli") for c in cmd):
        req["last"] = kw["input"]
    return _sh(cmd, **kw)
w3.sh = cap

mem = w5.Mem()
nr, nr_path = unk.nr_table()
assert unk.N_UNK == 12276
FX = Path(__file__).resolve().parents[1] / "McBopomofoTests" / "SlothEFixtures"


def cache_lines(path):
    return {l.split("\t", 1)[0]: l.rstrip("\n") for l in open(path, encoding="utf-8")}


def pins_of(text):
    out = {}
    for line in text.strip("\n").split("\n"):
        f = line.split("\t")
        out[f[0]] = [[int(f[k]), int(f[k + 1]), f[k + 2]] for k in range(1, len(f) - 2, 3)]
    return out


# dev: walks + units
out, info = unk.run("dev", 1, mem, nr, nr_path)
assert info["app_walk_equals_walk2"] == "500/500", info
pins, cache = pins_of(req["last"]), cache_lines(unk.base_cache("dev"))
with open(FX / "v302_dev_walks.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps({"sid": r["sid"], "cache": cache[r["sid"]], "pins": pins[r["sid"]], "walk": r["walk"]},
                           ensure_ascii=False) + "\n")
nodes = {(r["sid"], n["s"], n["e"]): n for r in out for n in r["nodes"]}
units = [json.loads(l) for l in open(W / "results" / "w5" / "dev.v1.units.jsonl", encoding="utf-8")]
jumps = 0
with open(FX / "v302_dev_units.jsonl", "w", encoding="utf-8") as f:
    for u in units:
        n = nodes[(u["sid"], u["s"], u["e"])]
        assert n["win"] == u["win"] and n["b2"] == u["b2"] and n["v"] == u["v"]
        want = w5.order(u, "h")
        jumps += want[0] != u["b2"][0]
        f.write(json.dumps({**u, "cache": cache[u["sid"]], "want": want}, ensure_ascii=False) + "\n")
print(f"dev: walks={len(out)} units={len(units)} jumps={jumps} {info}")

# single-syllable readings
out, info = unk.run("single", 1, mem, nr, nr_path)
pins, cache = pins_of(req["last"]), cache_lines(unk.base_cache("single"))
ex = {}
with open(FX / "v302_single.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        n = r["nodes"][0]
        assert len(r["nodes"]) == 1
        stock = n["win"][0][3]
        u = {"s": n["s"], "e": n["e"], "v": stock, "win": n["win"], "b2": n["b2"], "shown": stock}
        want = w5.order(u, "h", single=True)
        top = next(e[3] for e in n["win"] if len(e[3]) == 1)
        f.write(json.dumps({"sid": r["sid"], "cache": cache[r["sid"]], "pins": pins[r["sid"]], "inwalk": r["walk"],
                            **u, "stock": stock, "want": want, "top": top}, ensure_ascii=False) + "\n")
        if r["sid"] in ("ㄧㄚˊ", "ㄉㄢˋ"):
            ex[r["sid"]] = (stock, top, 1 + next(i for i, j in enumerate(want) if n["win"][j][3] == top),
                            [n["win"][j][3] for j in want[:3]])
print(f"single: readings={len(out)} {info['app_walk_equals_walk2']} examples={ex}")
shutil.rmtree(tmp)
