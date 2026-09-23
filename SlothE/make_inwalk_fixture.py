"""Reference for the in-walk parity test (McBopomofoTests/SlothEInWalkTests.mm): walk2's offline
12M in-walk rescoring (beta=0.3, gamma=0, pen=-15, variant guard on), as produced by
walk2/walk2_cli over walk2's per-position cache, joined with each sentence's readings and the stock walk.

usage (stdlib only):
  walk2/walk2_cli --data data/data.txt --cache <copy of walk2/cache/sloth_12m.tsv> \
      --unk walk2/cache/unk_chars.txt --variants walk2/variants.tsv --pen -15 --betas 0.3 --gammas 0 > ref.jsonl
  python3 make_inwalk_fixture.py --ref ref.jsonl --cache <same cache copy> --walks data/walks.jsonl
Output: McBopomofoTests/SlothEFixtures/inwalk_parity.jsonl
  {"sid", "readings": [...], "nodes": [[s,e,v],..] (guarded walk), "text", "stock_text", "reverted"}
"""
import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "McBopomofoTests" / "SlothEFixtures" / "inwalk_parity.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--walks", required=True)
    a = ap.parse_args()
    readings = {}
    for line in open(a.cache, encoding="utf-8"):
        f = line.split("\t", 3)
        readings[f[0]] = f[2].split()
    stock = {}
    for line in open(a.walks, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            stock[d["sent_id"]] = d["walk_text"]
    n = diff = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for line in open(a.ref, encoding="utf-8"):
            d = json.loads(line)
            assert d["b"] == 0.3 and d["g"] == 0
            nodes = d["guard"] if d["guard"] is not None else d["walk"]
            text = "".join(v for _, _, v in nodes)
            sid = d["sid"]
            out.write(json.dumps({"sid": sid, "readings": readings[sid], "nodes": nodes, "text": text,
                                  "stock_text": stock[sid], "reverted": d["rev"]}, ensure_ascii=False) + "\n")
            n += 1
            diff += text != stock[sid]
    print(f"inwalk_parity: {n} sentences, {diff} differ from the stock walk -> {OUT}")


if __name__ == "__main__":
    main()
