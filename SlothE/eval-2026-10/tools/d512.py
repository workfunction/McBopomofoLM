"""D512 reranker scoring, following the Funmula/0c-sequence-d512 model card example."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import json, numpy as np, onnxruntime as ort
P = evalcfg.D512_DIR
vocab = json.load(open(P + "/vocab.json", encoding="utf-8"))
CHARS = {c: i + 8 for i, c in enumerate(vocab["characters"])}
BASES = {b: i + 2 for i, b in enumerate(vocab["bases"])}
TONES = {"ˊ": 2, "ˇ": 3, "ˋ": 4, "˙": 5}
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(P + "/model.onnx", so, providers=["CPUExecutionProvider"])

def syllable(s):
    base = s[:-1] if s[-1] in TONES else s
    return BASES[base], TONES.get(s[-1], 1)

def log_prob(context, reading, candidate):
    """None when the reading or a candidate char is outside the vocabulary (model-card fallback)."""
    try:
        sounds = [syllable(s) for s in reading]
        cand = [CHARS[c] for c in candidate]
    except KeyError:
        return None
    head = [1] + [CHARS.get(c, 4) for c in context[-64:]] + [2]
    n, out = len(cand), len(head) + len(sounds)
    ids = head + [0] * n + [3] + cand[:-1]
    bases = [0] * len(head) + [b for b, _ in sounds] + [0] * n
    tones = [0] * len(head) + [t for _, t in sounds] + [0] * n
    types = [0] * len(head) + [1] * n + [0] + [2] * (n - 1)
    L = len(ids)
    row = lambda x: np.array([x], dtype=np.int64)
    feeds = {"ids": row(ids), "bases": row(bases), "tones": row(tones), "types": row(types),
             "positions": row(list(range(L))), "mask": np.tril(np.ones((L, L), dtype=bool))[None, None],
             "proj_pos": np.arange(out, L, dtype=np.int64), "target_node": row(list(range(n))),
             "target_ids": row([x - 6 for x in cand])}
    return float(sess.run(["scores"], feeds)[0][0])
