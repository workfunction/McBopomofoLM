"""pred_q35_60m decoder scoring as SlothEDecoder: log P(tokenize(ctx[-64:] + cand) | <bos>)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
P = evalcfg.Q35_DIR
torch.set_num_threads(4)
tok = PreTrainedTokenizerFast(tokenizer_file=P + "/tokenizer.json")
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float32).eval()

def ids(text):
    return [1] + tok.encode(text, add_special_tokens=False)

def score(ctx, cands):
    ctx = ctx[-64:]
    seqs = [ids(ctx + c) for c in cands]
    while max(len(s) for s in seqs) > 96 and ctx:  # frozen app rule: keep buffer, trim left context to fit t96
        ctx = ctx[1:]; seqs = [ids(ctx + c) for c in cands]
    if max(len(s) for s in seqs) > 96: return None
    L = max(len(s) for s in seqs)
    x = torch.tensor([s + [0] * (L - len(s)) for s in seqs])
    m = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs])
    with torch.no_grad():
        lp = model(input_ids=x, attention_mask=m).logits.float().log_softmax(-1)
    out = []
    for b, s in enumerate(seqs):
        out.append(float(sum(lp[b, t - 1, s[t]] for t in range(1, len(s)))))
    return out
