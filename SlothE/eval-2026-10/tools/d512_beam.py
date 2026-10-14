"""D512 as a standalone Zhuyin->text decoder (no encoder, no dictionary walk): constrained beam search.
The ONNX graph's internal `log_softmax` (next-char distribution over the output inventory at proj_pos) is exposed
as a second output; at each syllable only chars that McBopomofo's data.txt lists as single-char unigrams for that
reading are allowed. usage: d512_beam.py labeled.jsonl out.json [--beam 5] [--pool general]"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import json, os, sys
import numpy as np, onnx, onnxruntime as ort
SRC = os.path.join(evalcfg.D512_DIR, "model.onnx"); DST = os.path.join(evalcfg.WORK, "d512_logits.onnx")
if not os.path.exists(DST):
    m = onnx.load(SRC)
    m.graph.output.append(onnx.helper.make_tensor_value_info("log_softmax", onnx.TensorProto.FLOAT, None))
    onnx.save(m, DST)
so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("D512_THREADS", "4"))
sess = ort.InferenceSession(DST, so, providers=["CPUExecutionProvider"])
vocab = json.load(open(os.path.join(evalcfg.D512_DIR, "vocab.json"), encoding="utf-8"))
CHARS = {c: i + 8 for i, c in enumerate(vocab["characters"])}
BASES = {b: i + 2 for i, b in enumerate(vocab["bases"])}
TONES = {"ˊ": 2, "ˇ": 3, "ˋ": 4, "˙": 5}
LEGAL = {}
for l in open(evalcfg.DATA_TXT, encoding="utf-8"):
    f = l.split()
    if len(f) == 3 and len(f[1]) == 1 and "-" not in f[0] and not f[0].startswith(("_", "#")) and f[1] in CHARS:
        LEGAL.setdefault(f[0], []).append(f[1])

def syl(s):
    b = s[:-1] if s[-1] in TONES else s
    return BASES[b], TONES.get(s[-1], 1)

def next_lp(context, reading, prefix):
    head = [1] + [CHARS.get(c, 4) for c in context[-64:]] + [2]
    sounds = [syl(s) for s in reading]; n = len(sounds)
    ids = head + [0] * n + [3] + [CHARS[c] for c in prefix]
    bases = [0] * len(head) + [b for b, _ in sounds] + [0] * (len(prefix) + 1)
    tones = [0] * len(head) + [t for _, t in sounds] + [0] * (len(prefix) + 1)
    types = [0] * len(head) + [1] * n + [0] + [2] * len(prefix)
    L = len(ids); row = lambda x: np.array([x], dtype=np.int64)
    feeds = {"ids": row(ids), "bases": row(bases), "tones": row(tones), "types": row(types), "positions": row(list(range(L))),
             "mask": np.tril(np.ones((L, L), dtype=bool))[None, None], "proj_pos": np.array([L - 1], dtype=np.int64),
             "target_node": np.zeros((1, 1), dtype=np.int64), "target_ids": np.zeros((1, 1), dtype=np.int64)}
    return sess.run(["log_softmax"], feeds)[0][0]   # index j <-> char id j + 6

def next_lp_batch(context, reading, prefixes):
    """All beam prefixes in one forward pass: each prefix is a branch after [OUT] (tree/ancestor mask)."""
    head = [1] + [CHARS.get(c, 4) for c in context[-64:]] + [2]
    sounds = [syl(s) for s in reading]; n = len(sounds)
    base_ids = head + [0] * n + [3]; P = len(base_ids)
    ids = list(base_ids); bases = [0] * len(head) + [b for b, _ in sounds] + [0]
    tones = [0] * len(head) + [t for _, t in sounds] + [0]; types = [0] * len(head) + [1] * n + [0]
    pos = list(range(P)); parent = list(range(P)); last = []
    for pre in prefixes:
        prev = P - 1
        for j, c in enumerate(pre):
            ids.append(CHARS[c]); bases.append(0); tones.append(0); types.append(2); pos.append(P + j)
            parent.append(prev); prev = len(ids) - 1
        last.append(prev)
    L = len(ids); mask = np.zeros((L, L), dtype=bool)
    mask[:P, :P] = np.tril(np.ones((P, P), dtype=bool))
    for i in range(P, L):
        k = i
        while k >= P:
            mask[i, k] = True; k = parent[k]
        mask[i, :P] = True
    row = lambda x: np.array([x], dtype=np.int64)
    feeds = {"ids": row(ids), "bases": row(bases), "tones": row(tones), "types": row(types), "positions": row(pos),
             "mask": mask[None, None], "proj_pos": np.array(last, dtype=np.int64),
             "target_node": np.zeros((1, 1), dtype=np.int64), "target_ids": np.zeros((1, 1), dtype=np.int64)}
    return sess.run(["log_softmax"], feeds)[0]

def decode(reading, context="", beam=5, batched=True):
    try:
        [syl(s) for s in reading]
    except KeyError:
        return None
    beams = [("", 0.0)]
    for t, r in enumerate(reading):
        cands = LEGAL.get(r)
        if not cands: return None
        nxt = []
        lps = next_lp_batch(context, reading, [p for p, _ in beams]) if batched else [next_lp(context, reading, p) for p, _ in beams]
        for (pre, sc), lp in zip(beams, lps):
            for c in cands: nxt.append((pre + c, sc + float(lp[CHARS[c] - 6])))
        nxt.sort(key=lambda x: -x[1]); beams = nxt[:beam]
    return beams[0][0]

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    beam = int(sys.argv[sys.argv.index("--beam") + 1]) if "--beam" in sys.argv else 5
    pool = sys.argv[sys.argv.index("--pool") + 1] if "--pool" in sys.argv else None
    if pool: rows = [r for r in rows if pool in r["pools"]]
    # incremental + resumable: one JSON line per unit in <out>.partial.jsonl, merged into <out> at the end
    part = sys.argv[2] + ".partial.jsonl"; out = {}
    if os.path.exists(part):
        for l in open(part, encoding="utf-8"):
            try: k, v = json.loads(l); out[k] = v
            except ValueError: pass   # torn last line from an interrupted run
    with open(part, "a", encoding="utf-8") as f:
        for k, r in enumerate(rows):
            if r["id"] in out: continue
            out[r["id"]] = decode(r["readings"], beam=beam)
            f.write(json.dumps([r["id"], out[r["id"]]], ensure_ascii=False) + "\n"); f.flush()
            if k % 200 == 0: print(k, len(rows), flush=True)
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
