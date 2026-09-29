"""v3.1b decoder package: the shipped t16/t32/t64/t96 (B=3 scorers) + the next-token n16/n32/n64/n96 (B=1,
reports/ane-dec.md "next-token function") in ONE multifunction package with shared weights, from the same parts.
  1. the parts are the shipped ones: a t-only merge of ane/dec/mf/parts/*_fp16 has the shipped
     ane/dec/mf/dec_mf_fp16.mlpackage's weight.bin byte for byte and the same t16..t96 program functions;
  2. t + n merge -> McBopomofoLM/build/v31b/dec_mf_tn_fp16.mlpackage (default function t16, as shipped): its t16..t96
     program functions equal the shipped ones and its weight.bin holds the shipped blobs at the same offsets (only
     the header's blob count differs), so every weight a t function reads is the shipped bytes;
  3. SlothE/v31b/dec_next_first.tsv: char \t token ids whose decoded text starts with that char (partial-byte and
     whitespace-initial tokens dropped) = typo/t2.py _dinit / typo/t2c.py, the study's first-char grouping.
usage: nice -n 15 $W/ane/dec/.venv/bin/python SlothE/v31b/build_dec_tn.py
"""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE.parents[2]
DEC = W / "ane" / "dec"
sys.path.insert(0, str(DEC))
import coremltools as ct  # noqa: E402
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction  # noqa: E402
from tokenizers import decoders  # noqa: E402
from common import TOK  # noqa: E402

TS = (16, 32, 64, 96)
OUT = W / "McBopomofoLM" / "build" / "v31b"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def merge(fns, dst):
    d = MultiFunctionDescriptor()
    for name, p in fns.items():
        d.add_function(str(p), src_function_name="main", target_function_name=name)
    d.default_function_name = "t16"
    shutil.rmtree(dst, ignore_errors=True)
    save_multifunction(d, str(dst))


tparts = {f"t{T}": DEC / "mf" / "parts" / f"dec_b3_t{T}_fp16.mlpackage" for T in TS}
nparts = {f"n{T}": DEC / "mf_next" / "parts" / f"dec_b1_n{T}_fp16.mlpackage" for T in TS}
SHIPPED = DEC / "mf" / "dec_mf_fp16.mlpackage"
shipped_spec = ct.utils.load_spec(str(SHIPPED))
shipped_w = (SHIPPED / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").read_bytes()


def same_t(pkg):
    spec = ct.utils.load_spec(str(pkg))
    w = (Path(pkg) / "Data" / "com.apple.CoreML" / "weights" / "weight.bin").read_bytes()
    fns = all(spec.mlProgram.functions[f"t{T}"] == shipped_spec.mlProgram.functions[f"t{T}"] for T in TS)
    # the 64-byte storage header starts with the blob count; every blob after it must be the shipped bytes
    return fns, w[64:len(shipped_w)] == shipped_w[64:], len(w)


with tempfile.TemporaryDirectory() as tmp:
    t_only = Path(tmp) / "t_only.mlpackage"
    merge(tparts, t_only)
    fns, prefix, n = same_t(t_only)
    print(f"t-only merge vs shipped: t functions equal {fns}, weight blobs equal {prefix and n == len(shipped_w)}")
    assert fns and prefix and n == len(shipped_w), "the t parts are not the shipped ones"
OUT.mkdir(parents=True, exist_ok=True)
dst = OUT / "dec_mf_tn_fp16.mlpackage"
merge(tparts | nparts, dst)
fns, prefix, n = same_t(dst)
size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
print(f"merged t16..t96 + n16..n96 -> {dst} {size:.1f} MB; t functions equal shipped {fns}; "
      f"weight.bin = the shipped blobs at the same offsets {prefix} (+{n - len(shipped_w)} B of n-only blobs appended)")
assert fns and prefix
if TOK.decoder is None:
    TOK.decoder = decoders.ByteLevel()
first = {}
for t in range(TOK.get_vocab_size()):
    s = TOK.decode([t])
    if s and s[0] != "�" and not s[0].isspace():
        first.setdefault(s[0], []).append(t)
with open(HERE / "dec_next_first.tsv", "w", encoding="utf-8") as f:
    for c in sorted(first):
        f.write(c + "\t" + " ".join(map(str, first[c])) + "\n")
print(f"first-char table: {len(first)} chars, {sum(map(len, first.values()))} tokens")
