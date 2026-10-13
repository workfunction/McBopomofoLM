"""Run the SlothE-T 25M encoder (HF fp32 master, full ternary) on CPU; legal-masked log-probs per position."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import json, sys, numpy as np, torch
M = evalcfg.SLOTH_DIR
REPO = evalcfg.REPO
sys.path.insert(0, M)
from gate_slothe_ternary import build  # noqa: E402
torch.set_num_threads(4)
ck = torch.load(M + "/slothe.pt", map_location="cpu")
model = build(ck)
SV = json.load(open(M + "/syl_vocab.json", encoding="utf-8"))
LEGAL = np.load(M + "/syl2legal.npz")
LEGAL = LEGAL[LEGAL.files[0]].astype(bool)
ID2C = {}; C2ID = {}
for l in open(REPO + "/SlothE/Bundle/SlothE/char2id.tsv", encoding="utf-8"):
    c, i = l.rstrip("\n").split("\t"); i = int(i); C2ID[c] = i
    if i != 2: ID2C.setdefault(i, c)

def forward(readings):
    sid = torch.tensor([[SV.get(r, 1) for r in readings]])
    with torch.no_grad():
        lg = model(sid, torch.ones_like(sid, dtype=torch.bool))[0].double().numpy()
    out = []
    for t, r in enumerate(readings):
        m = LEGAL[SV.get(r, 1)]
        x = np.where(m, lg[t], -np.inf); z = np.logaddexp.reduce(x[m]) if m.any() else 0.0
        out.append({ID2C.get(i, f"#{i}"): float(x[i] - z) for i in np.nonzero(m)[0]})
    return out

if __name__ == "__main__":
    print(ck.get("config"), LEGAL.shape)
    for d in map(json.loads, open(REPO + "/McBopomofoTests/SlothEFixtures/logprob_probe.jsonl", encoding="utf-8")):
        lp = forward(d["readings"])[d["pos"]]
        print(d["char"], f"fixture={d['logp']:.3f} torch={lp.get(d['char'], float('nan')):.3f}")
