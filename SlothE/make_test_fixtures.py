"""Reference fixtures for McBopomofoTests/SlothETests.mm, produced by the offline PoC's own Python
code (scratch/ime-lm-poc/sloth: zhuyin_fmt.py, slothe_rt.py, run_sloth_rerank.py), imported read-only.

Outputs (McBopomofoTests/SlothEFixtures/):
  syllable_map.jsonl    {"in", "token", "id", "how"} for every syllable in data.txt + edge cases
  rerank_parity.jsonl   {"id", "readings", "span", "candidates", "span_candidates", "walk_text",
                         "ranking", "n_scored"}: run_sloth_rerank.predict() with SlothE-T 12M (ggml)
  logprob_probe.jsonl   {"readings", "pos", "char", "logp"}: per-position legal-masked log-probs

usage (PYTHONDONTWRITEBYTECODE=1 keeps the read-only sloth dir clean):
  PYTHONDONTWRITEBYTECODE=1 <sloth venv python> make_test_fixtures.py --sloth <sloth dir> \
      --items <items.jsonl> --data-txt <McBopomofo data.txt>
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "McBopomofoTests" / "SlothEFixtures"

EDGE = ["˙ㄉㄜ", "ㄉㄜ˙", "ㄉㄜˉ", " ㄇㄚ ", "ㄇㄚ·", "ㄇㄚ・", "ㄇㄚ•", "", " ", "　ㄇㄚ　",
        "_punctuation_list", "_punctuation_，", "ㄅ", "abc", "ㄇㄛ˙", "ㄈㄨ˙", "ㄧㄞˊ", "˙", "ˊ",
        "ㄋㄧˇ", "ㄏㄠˇ", "ㄕˋ", "ㄓ", "é", "<unk>", "<pad>", "ㄌㄩˋ", "ㄦˊ"]

PROBES = [
    ("ㄨㄛˇ ㄐㄧㄣ ㄊㄧㄢ ㄑㄩˋ ㄕˋ ㄔㄤˇ ㄇㄞˇ ㄘㄞˋ", [(4, "市"), (4, "是"), (4, "世"), (5, "場"), (0, "我")]),
    ("ㄓㄜˋ ㄍㄜ˙ ㄍㄨㄥ ㄕˋ ㄏㄣˇ ㄋㄢˊ", [(2, "公"), (2, "工"), (3, "式"), (3, "勢"), (1, "個")]),
    ("ㄐㄧㄣ ㄊㄧㄢ ㄊㄧㄢ ㄑㄧˋ ㄏㄣˇ ㄏㄠˇ", [(0, "今"), (3, "氣"), (3, "器"), (5, "好")]),
    ("ㄊㄞˊ ㄨㄢ", [(0, "台"), (0, "臺"), (1, "灣")]),
    ("ㄕㄜˊ ㄇㄛ˙", [(0, "什"), (1, "麼"), (1, "么")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sloth", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--data-txt", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.sloth)
    import numpy as np
    import zhuyin_fmt
    from run_sloth_rerank import predict
    from slothe_rt import SlothE

    OUT.mkdir(parents=True, exist_ok=True)
    eng = SlothE("12m", "ggml", 4)

    syls = set()
    for line in open(a.data_txt, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        key = line.split(" ", 1)[0]
        syls.update(zhuyin_fmt.split_reading(key))
    rows = sorted(syls) + EDGE
    with open(OUT / "syllable_map.jsonl", "w", encoding="utf-8") as f:
        for s in rows:
            tok, sid, how = zhuyin_fmt.to_slothe(s, eng.syl_vocab)
            f.write(json.dumps({"in": s, "token": tok, "id": int(sid), "how": how}, ensure_ascii=False) + "\n")
    print(f"syllable_map: {len(rows)} rows ({len(syls)} data.txt syllables)")

    items = [json.loads(l) for l in open(a.items, encoding="utf-8") if l.strip()]
    n = 0
    with open(OUT / "rerank_parity.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            p = predict(eng, it)
            f.write(json.dumps({"id": it["id"], "readings": it["readings"], "span": it["span"],
                                "candidates": it["candidates"], "span_candidates": it["span_candidates"],
                                "walk_text": it["walk_text"], "ranking": p["ranking"],
                                "n_scored": p["n_scored"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"rerank_parity: {n} items")

    with open(OUT / "logprob_probe.jsonl", "w", encoding="utf-8") as f:
        for readings, probes in PROBES:
            r = readings.split()
            _, ids, _ = eng.ids(r)
            lp = eng.logprobs(ids, eng.logits(ids))
            for pos, ch in probes:
                cid = eng.char2id.get(ch)
                v = float(lp[pos, cid]) if cid is not None else float("-inf")
                f.write(json.dumps({"readings": r, "pos": pos, "char": ch,
                                    "logp": v if np.isfinite(v) else None}, ensure_ascii=False) + "\n")
    print("logprob_probe: written")


if __name__ == "__main__":
    main()
