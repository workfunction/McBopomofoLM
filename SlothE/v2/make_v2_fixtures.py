"""Extra v2 fixtures (Core ML 25M encoder / decoder, the app's bundled .mlmodelc, coremltools CPU_AND_NE):
  rerank_parity.jsonl  the phase-1 items re-ranked by run_sloth_rerank.predict() (sloth, read-only) with
                       Core ML 25M log-probs (same items, "ranking"/"n_scored" recomputed)
  logprob_probe.jsonl  the phase-1 probes, "logp" recomputed with Core ML 25M
  v2_dec_long.jsonl    decoder calls with long left contexts (previous dev sentences + the in-walk context),
                       one per function t16/t32/t64/t96 region: {"ctx","cands","scores","T","seqs"}
usage: PYTHONDONTWRITEBYTECODE=1 $W/ane/dec/.venv/bin/python make_v2_fixtures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import make_v2_reference as ref  # noqa: E402  (same models, embedding, tokenizer, bucket rules)

W = ref.W
sys.path.insert(0, str(W / "sloth"))
from run_sloth_rerank import predict  # noqa: E402
from slothe_rt import SlothE  # noqa: E402
import zhuyin_fmt  # noqa: E402


class CoreMLSlothE(SlothE):
    """sloth's SlothE with the forward pass replaced by the app's Core ML encoder."""
    def __init__(self):   # no ggml / torch load
        self.syl_vocab = ref.enc_common.SYL
        self.char2id = json.load(open(W / "sloth/sloth-zhuyin-linux/model/slothe_4m_onnx/char2id.json", encoding="utf-8"))
        self.id2char = {i: c for c, i in self.char2id.items() if i != ref.enc_common.UNK_CHAR_ID}
        self.mask = ref.enc_common.MASK
        self.norm_mode = "none"

    def logits(self, ids):
        T = len(ids)
        L = next(x for x in ref.ENC_L if x >= T)
        x = np.zeros((1, L, 352), np.float16)
        x[0, :T] = ref.EMB[ids]
        mk = np.zeros((1, L), np.float16)
        mk[0, :T] = 1
        return ref.enc_model(L).predict({"ids": x, "mask": mk})["logits"][0, :T].astype(np.float32)


def main():
    eng = CoreMLSlothE()
    rows = [json.loads(l) for l in open(ref.FIX / "rerank_parity.jsonl", encoding="utf-8")]
    changed = 0
    for r in rows:
        p = predict(eng, r)
        changed += p["ranking"] != r["ranking"]
        r["ranking"], r["n_scored"] = p["ranking"], p["n_scored"]
    with open(ref.FIX / "rerank_parity.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rerank_parity: {len(rows)} items, {changed} rankings differ from the 12M ggml ones")
    probes = [json.loads(l) for l in open(ref.FIX / "logprob_probe.jsonl", encoding="utf-8")]
    for pr in probes:
        _, ids, _ = eng.ids(pr["readings"])
        lp = eng.logprobs(ids, eng.logits(ids))
        cid = eng.char2id.get(pr["char"])
        v = float(lp[pr["pos"], cid]) if cid is not None else float("-inf")
        pr["logp"] = v if np.isfinite(v) else None
    with open(ref.FIX / "logprob_probe.jsonl", "w", encoding="utf-8") as f:
        for pr in probes:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    print(f"logprob_probe: {len(probes)} probes")
    # long contexts: previous dev sentences' gold-free final text (reference walks) + candidates
    par = [json.loads(l) for l in open(ref.FIX / "v2_parity.jsonl", encoding="utf-8")]
    texts = ["".join(v for _, _, v in p["final"]) for p in par]
    out, seen = [], {}
    k = 0
    for p in par:
        for ctx, cands, _ in p["calls"]:
            prev = "".join(texts[max(0, k - 12):k])
            k += 1
            for take in (8, 24, 48, 64, 80):
                longctx = (prev + ctx)[-take:] if take <= 64 else prev[-(take - len(ctx)):] + ctx
                sc, T = ref.dec_scores(longctx[-64:], cands)
                seqs = [[ref.BOS] + ref.TOK.encode(longctx[-64:] + c).ids for c in cands]
                if seen.get(T, 0) < 40:
                    seen[T] = seen.get(T, 0) + 1
                    out.append({"ctx": longctx, "cands": cands, "scores": sc, "T": T, "seqs": seqs})
        if all(seen.get(t, 0) >= 40 for t in (16, 32, 64)) or k > 900:
            break
    for ctx in ("😀" * 64, "😀😁" * 40, "a1" * 40):   # token-dense contexts reach the t96 function
        cands = ["的", "得", "地"]
        sc, T = ref.dec_scores(ctx[-64:], cands)
        seqs = [[ref.BOS] + ref.TOK.encode(ctx[-64:] + c).ids for c in cands]
        seen[T] = seen.get(T, 0) + 1
        out.append({"ctx": ctx, "cands": cands, "scores": sc, "T": T, "seqs": seqs})
    with open(ref.FIX / "v2_dec_long.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"v2_dec_long: {len(out)} calls by function {seen}")


if __name__ == "__main__":
    main()
