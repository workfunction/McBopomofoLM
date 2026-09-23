"""Stage the SlothE-T 12M runtime files that McBopomofoLM bundles (Contents/Resources/SlothE/).

Inputs (read-only):
  * HF cache, Luigi/sloth-ime-models @ REV: slothe-t-12m-256x12.gguf, syl_vocab.json, syl2legal.npz
  * sloth-zhuyin-linux model/slothe_4m_onnx/char2id.json (same char ids the web demo uses with the 12M GGUF)
  * ThirdParty/opencc/{TWVariants,HKVariants}.txt (Apache-2.0 OpenCC data)
Outputs (SlothE/Bundle/SlothE/, copied into the app as Contents/Resources/SlothE/):
  slothe-t-12m-256x12.gguf   byte copy
  syl_vocab.tsv              "<syllable>\t<id>" per line
  char2id.tsv                "<char>\t<id>" per line
  syl2legal.bin              b"SLM1" + u32 n_syl + u32 n_char + row-major bit mask
                             (bit c of row s = byte[s*row_bytes + c//8] >> (c%8) & 1)
  variants.tsv               one orthographic-variant class per line, chars space separated
                             (same union-find rule as walk2/make_variants.py)
  MANIFEST.txt               provenance + sha256 of every staged file
  runtime-manifest.txt       "<sha256> <bytes> <file>" per runtime input; the app checks size + sha256 of
                             every listed file before loading a model and treats any mismatch as a load
                             failure (SlothEManifest.cpp)
  NOTICE.txt, OpenCC-LICENSE.txt  third-party notices
usage: <python with numpy> prep_resources.py --sloth-upstream <dir of sloth-zhuyin-linux>
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "Bundle" / "SlothE"   # folder reference -> Contents/Resources/SlothE/
REPO = "Luigi/sloth-ime-models"
REV = "e7d13c9c451dc01ab9546be53c574f4dbe0fdd54"
GGUF = "slothe-t-12m-256x12.gguf"
DEC_GGUF = "pred_q35_60m-q4.gguf"   # SlothE decoder (phase 3), llama.cpp qwen35


def hf_path(name):
    snap = Path.home() / ".cache" / "huggingface" / "hub" / "models--Luigi--sloth-ime-models" / "snapshots" / REV
    p = snap / name
    if not p.exists():
        raise SystemExit(f"missing in HF cache: {p}")
    return p


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_tsv(path, mapping):
    rows = []
    for k, v in sorted(mapping.items(), key=lambda kv: (kv[1], kv[0])):
        if not k or any(c in k for c in "\t\n\r"):
            raise SystemExit(f"unencodable key {k!r} in {path.name}")
        rows.append(f"{k}\t{int(v)}\n")
    path.write_text("".join(rows), encoding="utf-8")


def variant_classes(srcs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in srcs:
        for line in p.read_text(encoding="utf-8").splitlines():
            src, _, tgts = line.partition("\t")
            for t in tgts.split():
                if len(src) == 1 and len(t) == 1 and t != src:
                    parent[find(src)] = find(t)
    classes = {}
    for c in list(parent):
        classes.setdefault(find(c), set()).add(c)
    return sorted(sorted(v) for v in classes.values() if len(v) > 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sloth-upstream", required=True)
    a = ap.parse_args()
    up = Path(a.sloth_upstream)
    OUT.mkdir(parents=True, exist_ok=True)

    gguf_src = hf_path(GGUF)
    (OUT / GGUF).write_bytes(gguf_src.read_bytes())
    dec_src = hf_path(DEC_GGUF)
    (OUT / DEC_GGUF).write_bytes(dec_src.read_bytes())

    syl = json.load(open(hf_path("syl_vocab.json"), encoding="utf-8"))
    write_tsv(OUT / "syl_vocab.tsv", syl)
    char2id = json.load(open(up / "model" / "slothe_4m_onnx" / "char2id.json", encoding="utf-8"))
    write_tsv(OUT / "char2id.tsv", char2id)

    mask = np.load(hf_path("syl2legal.npz"))["mask"].astype(bool)
    n_syl, n_char = mask.shape
    assert n_syl == len(syl), (n_syl, len(syl))
    assert max(char2id.values()) < n_char
    packed = np.packbits(mask, axis=1, bitorder="little")
    with open(OUT / "syl2legal.bin", "wb") as f:
        f.write(b"SLM1" + struct.pack("<II", n_syl, n_char))
        f.write(packed.tobytes(order="C"))
    # round-trip check of the packing
    back = np.unpackbits(np.frombuffer((OUT / "syl2legal.bin").read_bytes()[12:], dtype=np.uint8)
                         .reshape(n_syl, -1), axis=1, bitorder="little")[:, :n_char].astype(bool)
    assert (back == mask).all()

    cls = variant_classes([HERE / "ThirdParty" / "opencc" / "TWVariants.txt",
                           HERE / "ThirdParty" / "opencc" / "HKVariants.txt"])
    (OUT / "variants.tsv").write_text("".join(" ".join(c) + "\n" for c in cls), encoding="utf-8")

    (OUT / "OpenCC-LICENSE.txt").write_bytes((HERE / "ThirdParty" / "opencc" / "LICENSE.txt").read_bytes())
    (OUT / "llama.cpp-LICENSE.txt").write_bytes((HERE / "ThirdParty" / "llama.cpp" / "LICENSE").read_bytes())
    (OUT / "ggml-LICENSE.txt").write_bytes((HERE / "ThirdParty" / "ggml" / "LICENSE").read_bytes())
    (OUT / "NOTICE.txt").write_text(
        "McBopomofoLM bundles third-party components (side-by-side test build, not for redistribution):\n"
        "- SlothE-T 12M model files (slothe-t-12m-256x12.gguf, syl_vocab, syl2legal): "
        "huggingface.co/Luigi/sloth-ime-models, Apache-2.0.\n"
        "- char2id table and libslothe (slothe.cpp): github.com/vieenrose/sloth-zhuyin-linux. "
        "That repository has no LICENSE file; license status is unresolved. libslothe carries a local patch "
        "(per-length compute-graph cache, SlothE/ThirdParty/slothe/slothe-graph-cache.patch; load errors return "
        "instead of exit(), slothe-load-errors.patch).\n"
        "- SlothE decoder pred_q35_60m-q4.gguf (Qwen3.5 hybrid, 59.8M params): huggingface.co/Luigi/sloth-ime-models, "
        "Apache-2.0.\n"
        "- ggml (github.com/ggml-org/ggml @ 456172e), MIT (ggml-LICENSE.txt), statically linked for the encoder.\n"
        "- llama.cpp (github.com/ggml-org/llama.cpp @ 7ab4ee7) with its bundled ggml, MIT (llama.cpp-LICENSE.txt), "
        "statically linked for the decoder.\n"
        "- Orthographic variant classes derived from OpenCC TWVariants.txt / HKVariants.txt, "
        "Apache-2.0 (OpenCC-LICENSE.txt).\n", encoding="utf-8")
    files = [GGUF, DEC_GGUF, "syl_vocab.tsv", "char2id.tsv", "syl2legal.bin", "variants.tsv"]
    lines = [
        f"SlothE-T 12M runtime files for McBopomofoLM",
        f"model: huggingface.co/{REPO} @ {REV} ({GGUF}, syl_vocab.json, syl2legal.npz), Apache-2.0",
        f"char2id: github.com/vieenrose/sloth-zhuyin-linux model/slothe_4m_onnx/char2id.json",
        f"variants: OpenCC TWVariants.txt + HKVariants.txt (Apache-2.0), char-level union-find, {len(cls)} classes",
        f"source sha256 {GGUF}: {sha256(gguf_src)}",
        f"source sha256 {DEC_GGUF}: {sha256(dec_src)}",
        f"n_syl={n_syl} n_char={n_char} n_char2id={len(char2id)}",
        "",
    ] + [f"{sha256(OUT / n)}  {n}" for n in files]
    (OUT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "runtime-manifest.txt").write_text(
        "# McBopomofoLM checks the byte size and sha256 of every file below before loading a model.\n"
        "# Any mismatch or missing file = that model is not loaded (encoder -> stock McBopomofo,\n"
        "# decoder -> encoder-only in-walk). Format: <sha256> <bytes> <file>\n"
        + "".join(f"{sha256(OUT / n)} {(OUT / n).stat().st_size} {n}\n" for n in files), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
