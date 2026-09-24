"""v2 parity reference: walk2's config A' algorithm (walk2/dec_combo.py, imported read-only) driven by Core ML
scores from the SAME two compiled models the app bundles (SlothE/Bundle/SlothE/{enc25m,dec60m}.mlmodelc),
run with coremltools on CPU_AND_NE.

  1. encoder: enc25m.mlmodelc, function = smallest L in {8,16,32,64,256} >= T, caller-side fp16 embedding rows
     (enc25m_embed_f16.bin), mask; legal-masked log-softmax in float32 (ane/enc/common.logprobs) ->
     walk2-format cache TSV (6 decimals, as walk2/make_cache.py / ane/enc/make_cache_coreml.py);
  2. walk2_cli --nodes (beta 0.3, pen -15, guard) on that cache -> in-walk nodes + encoder top-3 per node;
  3. decoder: every (left context, top-3) call, <bos> + tokenizer(ctx + cand), function = smallest T in
     {16,32,64,96} >= longest sequence, B = 3 padded with <pad>, score = sum of lp over the sequence
     (ane/dec/common.sums);
  4. dec_combo.finals(lambda 2, tau 0.5) -> walk2_cli --repin -> final guarded walks.
Outputs (McBopomofoTests/SlothEFixtures/):
  v2_parity.jsonl     {"sid","readings","inwalk":[[s,e,v]],"final":[[s,e,v]],"calls":[[ctx,[cands],[scores]]],"pins":[[s,l,v]]}
  v2_logprob_probe.jsonl  per sentence (first 40): positions x a few char ids -> log-prob (encoder check)
  v2_tokenizer.jsonl  {"text","ids"} for every decoder text + edge cases (tokenizers library = ground truth)
and SlothE/v2/work/: the cache TSV and a summary JSON.
usage: PYTHONDONTWRITEBYTECODE=1 $W/ane/dec/.venv/bin/python make_v2_reference.py
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent
W = APP.parent
B = APP / "SlothE" / "Bundle" / "SlothE"
FIX = APP / "McBopomofoTests" / "SlothEFixtures"
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)
sys.dont_write_bytecode = True
sys.path.insert(0, str(W / "ane" / "enc"))
sys.path.insert(0, str(W / "walk2"))
import common as enc_common  # noqa: E402  (ane/enc: zhuyin_fmt ids, legal mask, float32 logprobs)
import dec_combo as dc  # noqa: E402
import cv  # noqa: E402

BETA, LAM, TAU = 0.3, 2.0, 0.5
ENC_L = [8, 16, 32, 64, 256]
DEC_T = [16, 32, 64, 96]
CU = ct.ComputeUnit.CPU_AND_NE
EMB = np.fromfile(B / "enc25m_embed_f16.bin", dtype=np.float16).reshape(1539, 352)
TOK = Tokenizer.from_file(str(B / "dec_tokenizer.json"))
BOS, PAD = TOK.token_to_id("<bos>"), TOK.token_to_id("<pad>")
enc_fn, dec_fn = {}, {}


def enc_model(L):
    if L not in enc_fn:
        t0 = time.time()
        enc_fn[L] = ct.models.CompiledMLModel(str(B / "enc25m.mlmodelc"), compute_units=CU, function_name=f"L{L}")
        print(f"enc L{L} loaded in {time.time() - t0:.1f}s", flush=True)
    return enc_fn[L]


def dec_model(T):
    if T not in dec_fn:
        t0 = time.time()
        dec_fn[T] = ct.models.CompiledMLModel(str(B / "dec60m.mlmodelc"), compute_units=CU, function_name=f"t{T}")
        print(f"dec t{T} loaded in {time.time() - t0:.1f}s", flush=True)
    return dec_fn[T]


def enc_logprobs(ids):
    T = len(ids)
    L = next(x for x in ENC_L if x >= T)
    x = np.zeros((1, L, 352), np.float16)
    x[0, :T] = EMB[ids]
    mk = np.zeros((1, L), np.float16)
    mk[0, :T] = 1
    lg = enc_model(L).predict({"ids": x, "mask": mk})["logits"][0, :T].astype(np.float32)
    return enc_common.logprobs(ids, lg)


def dec_scores(ctx, cands):
    seqs = [[BOS] + TOK.encode(ctx + c).ids for c in cands]
    n = max(len(s) for s in seqs)
    T = next(t for t in DEC_T if t >= n)
    x = np.full((3, T), PAD, dtype=np.int32)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
    lp = dec_model(T).predict({"ids": x})["lp"]
    return [float(np.asarray(lp[i, :len(s) - 1], dtype=np.float64).sum()) for i, s in enumerate(seqs)], T


def main():
    rows = [json.loads(l) for l in open(FIX / "inwalk_parity.jsonl", encoding="utf-8")]
    char2id = json.load(open(W / "sloth/sloth-zhuyin-linux/model/slothe_4m_onnx/char2id.json", encoding="utf-8"))
    UNK = enc_common.UNK_CHAR_ID
    id2char = {i: c for c, i in char2id.items() if i != UNK}
    lines, probe = [], []
    for k, r in enumerate(rows):
        sid, rd = r["sid"], r["readings"]
        ids = enc_common.ids_of(rd)
        lp, legal = enc_logprobs(ids)
        fields = []
        for i in range(len(ids)):
            row = lp[i]
            toks = [f"{row[UNK]:.6f}" if np.isfinite(row[UNK]) else "x"]
            for cid in np.nonzero(np.isfinite(row))[0]:
                if cid == UNK or int(cid) not in id2char:
                    continue
                toks += [id2char[int(cid)], f"{row[cid]:.6f}"]
            fields.append(" ".join(toks))
        lines.append("\t".join([sid, str(len(ids)), " ".join(rd)] + fields) + "\n")
        if k < 40:
            pts = []
            for i in range(len(ids)):
                fin = np.nonzero(np.isfinite(lp[i]))[0]
                for cid in list(fin[:2]) + list(fin[-1:]) + [int(np.argmax(np.where(legal[i], lp[i], -np.inf)))]:
                    pts.append([i, int(cid), float(lp[i][cid])])
            probe.append({"sid": sid, "readings": rd, "points": pts})
    cache = WORK / "sloth_25m_coreml_ane.tsv"
    cache.write_text("".join(lines), encoding="utf-8")
    real_cli = dc.cli

    def cli(size, set_name, args, stdin=None):   # walk2_cli with the Core ML cache (walk2 itself untouched)
        import subprocess
        p = subprocess.run([str(W / "walk2" / "walk2_cli"), "--data", str(W / "data" / "data.txt"),
                            "--cache", str(cache), "--unk", str(W / "walk2" / "cache" / "unk_chars.txt"),
                            "--variants", str(W / "walk2" / "variants.tsv"), "--pen", str(cv.PEN), *args],
                           input=stdin, capture_output=True, text=True, check=True)
        return [json.loads(l) for l in p.stdout.splitlines()]
    dc.cli = cli
    nodes = dc.get_nodes("25m", "dev", [BETA])
    calls, texts, buckets = {}, set(), {}
    for (b, sid), r in nodes.items():
        for n, ctx in zip(r["nodes"], dc.contexts(r["nodes"])):
            if len(n[4]) >= 2:
                cands = tuple(v for v, _ in n[4])
                if (ctx, cands) not in calls:
                    calls[(ctx, cands)], T = dec_scores(ctx, cands)
                    buckets[T] = buckets.get(T, 0) + 1
                    texts.update(ctx + c for c in cands)
    fin = dc.finals("25m", "dev", nodes, BETA, lambda sid, k, ctx, top: calls[(ctx, tuple(v for v, _ in top))], [(LAM, TAU)])[(LAM, TAU)]
    dc.cli = real_cli
    out = []
    for r in rows:
        sid = r["sid"]
        nd = nodes[(BETA, sid)]["nodes"]
        ctxs = dc.contexts(nd)
        cl, pins = [], []
        for k, n in enumerate(nd):
            if len(n[4]) >= 2:
                cands = tuple(v for v, _ in n[4])
                sc = calls[(ctxs[k], cands)]
                cl.append([ctxs[k], list(cands), sc])
                x = dc.decide(n, sc, LAM, TAU)
                if x is not None:
                    pins.append(list(x))
        out.append({"sid": sid, "readings": r["readings"], "inwalk": [[n[0], n[1], n[2]] for n in nd],
                    "final": fin[sid], "calls": cl, "pins": pins})
    with open(FIX / "v2_parity.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    with open(FIX / "v2_logprob_probe.jsonl", "w", encoding="utf-8") as f:
        for o in probe:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    extra = ["", "的", "我們", "，", "。", "123", "１２３", "abc", "Hello world", "  兩個空白", "a  b", "你好 世界", "tab\there",
             "換行\n了", "😀笑", "😀😁", "é", "ㄅㄆㄇ", "…", "「引號」", "100%", "it's", "don't", "¥€$", "　全形空白",
             "ヶ", "Ⅻ", "½", "𠀀", "é", "<bos>", "a<pad>b", "🤔🤔", "  ", " x", "x ", "\n\n", "A.B", "x—y"]
    with open(FIX / "v2_tokenizer.jsonl", "w", encoding="utf-8") as f:
        for t in sorted(texts) + extra:
            f.write(json.dumps({"text": t, "ids": TOK.encode(t).ids}, ensure_ascii=False) + "\n")
    # vs walk2's ggml/llama A' walks
    ref = {json.loads(l)["sent_id"]: json.loads(l)["walk_nodes"] for l in open(W / "results/walk2/dev_final2_Aprime.walks.jsonl", encoding="utf-8")}
    D = cv.Data("dev")
    diff = [sid for sid in fin if [list(x) for x in fin[sid]] != [list(x) for x in ref[sid]]]
    def correct(walks):
        tot = 0
        for sid, n in walks.items():
            got = cv.chars_of(n, len(D.gold[sid]))
            tot += sum(1 for a, b in zip(got, D.gold[sid]) if a == b)
        return tot
    summary = {"sentences": len(out), "calls": len(calls), "buckets": buckets, "pins": sum(len(o["pins"]) for o in out),
               "inwalk_differs_from_final": sum(1 for o in out if [x[2] for x in o["inwalk"]] != [x[2] for x in o["final"]]),
               "vs_ggml_llama_Aprime_differing": len(diff), "differing_sids": diff,
               "dev_correct_chars_coreml": correct(fin), "dev_correct_chars_ggml_llama": correct(ref),
               "dev_total_chars": sum(len(D.gold[s]) for s in fin), "tokenizer_texts": len(texts) + len(extra)}
    (WORK / "v2_reference_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
