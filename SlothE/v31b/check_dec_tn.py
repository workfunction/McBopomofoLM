"""v3.1b: the merged decoder's t16/t32/t64/t96 give the shipped outputs bit for bit.
Old = the v2.x bundle's dec60m.mlmodelc (copied to build/v31b/old_bundle before re-staging), new = the re-staged
SlothE/Bundle/SlothE/dec60m.mlmodelc (t* + n*). Inputs: the decoder calls of McBopomofoTests/SlothEFixtures/
v2_parity.jsonl (<bos> + context + candidate, B=3, padded as the app does), per function the calls that fit.
Compares the full lp tensors (float32 bytes) on CPU_ONLY and on CPU_AND_NE, sequentially (1 process).
usage: nice -n 15 $W/ane/dec/.venv/bin/python SlothE/v31b/check_dec_tn.py [calls_per_function=40]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1]
sys.path.insert(0, str(APP.parent / "ane" / "dec"))
import coremltools as ct  # noqa: E402
from common import TOK, BOS  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OLD = APP / "build" / "v31b" / "old_bundle" / "dec60m.mlmodelc"
NEW = APP / "SlothE" / "Bundle" / "SlothE" / "dec60m.mlmodelc"
calls = [c for l in open(APP / "McBopomofoTests" / "SlothEFixtures" / "v2_parity.jsonl", encoding="utf-8")
         for c in json.loads(l).get("calls", [])]
rows = {16: [], 32: [], 64: [], 96: []}
for ctx, cands, *_ in calls:
    seqs = [[BOS] + TOK.encode(ctx[-64:] + c).ids for c in cands]
    T = next((t for t in rows if max(map(len, seqs)) <= t), None)
    if T and len(rows[T]) < N:
        x = np.zeros((3, T), dtype=np.int32)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = s
        rows[T].append(x)
for T in rows:   # long functions: pad short real inputs into them (same causal model)
    k = 0
    while len(rows[T]) < N and k < len(rows[16]):
        x = np.zeros((3, T), dtype=np.int32)
        x[:, :16] = rows[16][k]
        rows[T].append(x)
        k += 1
ok = True
for unit, cu in (("cpu", ct.ComputeUnit.CPU_ONLY), ("ane", ct.ComputeUnit.CPU_AND_NE)):
    for T, xs in rows.items():
        h = {}
        for name, path in (("old", OLD), ("new", NEW)):
            m = ct.models.CompiledMLModel(str(path), compute_units=cu, function_name=f"t{T}")
            d = hashlib.sha256()
            for x in xs:
                d.update(np.ascontiguousarray(m.predict({"ids": x})["lp"], dtype=np.float32).tobytes())
            h[name] = d.hexdigest()
            del m
        same = h["old"] == h["new"]
        ok &= same
        print(f"{unit} t{T}: {len(xs)} calls, sha256(lp) old {h['old'][:16]} new {h['new'][:16]} identical={same}", flush=True)
print("ALL IDENTICAL" if ok else "DIFFERENT")
