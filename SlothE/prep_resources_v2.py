"""Stage the McBopomofoLM v2 runtime files (Core ML only, ANE) into SlothE/Bundle/SlothE/
(folder reference -> Contents/Resources/SlothE/).

Inputs (read-only):
  * $W/ane/enc/models/enc25m_multi_pal2_emb.mlpackage   SlothE-T 25M encoder, 2-bit ternary, functions L8/L16/L32/L64/L256,
                                                        caller-side embedding (ids = fp16 rows [1, L, 352], mask [1, L])
  * $W/ane/enc/work/embed_f16_25m.bin                   the fp16 embedding table [1539, 352] (== w25m.npz["embed"])
  * $W/ane/dec/mf/dec_mf_fp16.mlpackage                 SlothE decoder pred_q35_60m, fp16, functions t16/t32/t64/t96, B=3
  * HF cache Luigi/sloth-ime-models @ e7d13c9 pred_q35_60m/tokenizer.json (the decoder's byte-level BPE)
  * the v1 staging's vocabulary files (syl_vocab.tsv, char2id.tsv, syl2legal.bin, variants.tsv; same vocab for 12M/25M)
Outputs (SlothE/Bundle/SlothE/):
  enc25m.mlmodelc/        xcrun coremlcompiler compile of the encoder package
  dec60m.mlmodelc/        xcrun coremlcompiler compile of the decoder package
  enc25m_embed_f16.bin    embedding table
  dec_tokenizer.json      tokenizer.json, byte copy
  syl_vocab.tsv char2id.tsv syl2legal.bin variants.tsv
  runtime-manifest.txt    "<sha256> <bytes> <path>" for every runtime file, including every file inside the
                          two .mlmodelc directories; the app verifies it before loading (SlothEEngine.cpp)
  MANIFEST.txt NOTICE.txt OpenCC-LICENSE.txt
usage: python3 prep_resources_v2.py   (needs only the standard library + xcrun)
"""
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = HERE.parent.parent
OUT = HERE / "Bundle" / "SlothE"
ENC_PKG = W / "ane" / "enc" / "models" / "enc25m_multi_pal2_emb.mlpackage"
EMBED = W / "ane" / "enc" / "work" / "embed_f16_25m.bin"
DEC_PKG = W / "ane" / "dec" / "mf" / "dec_mf_fp16.mlpackage"
HF = Path.home() / ".cache/huggingface/hub/models--Luigi--sloth-ime-models/snapshots/e7d13c9c451dc01ab9546be53c574f4dbe0fdd54"
TOKENIZER = HF / "pred_q35_60m" / "tokenizer.json"
KEEP = ["syl_vocab.tsv", "char2id.tsv", "syl2legal.bin", "variants.tsv", "OpenCC-LICENSE.txt"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_sha(d):
    h = hashlib.sha256()
    for p in sorted(x for x in Path(d).rglob("*") if x.is_file()):
        h.update(str(p.relative_to(d)).encode() + b"\0" + sha256(p).encode() + b"\n")
    return h.hexdigest()


def compile_pkg(pkg, name, tmp):
    src = Path(tmp) / f"{name}.mlpackage"
    shutil.copytree(pkg, src)
    subprocess.run(["xcrun", "coremlcompiler", "compile", str(src), str(tmp)], check=True, capture_output=True)
    dst = OUT / f"{name}.mlmodelc"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(Path(tmp) / f"{name}.mlmodelc"), dst)
    return dst


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keep = {n: (OUT / n).read_bytes() for n in KEEP}
    for p in OUT.iterdir():   # v2 ships no GGUF, no ggml/llama.cpp licence files
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for n, b in keep.items():
        (OUT / n).write_bytes(b)
    with tempfile.TemporaryDirectory() as tmp:
        compile_pkg(ENC_PKG, "enc25m", tmp)
        compile_pkg(DEC_PKG, "dec60m", tmp)
    shutil.copyfile(EMBED, OUT / "enc25m_embed_f16.bin")
    shutil.copyfile(TOKENIZER, OUT / "dec_tokenizer.json")
    assert (OUT / "enc25m_embed_f16.bin").stat().st_size == 1539 * 352 * 2
    (OUT / "NOTICE.txt").write_text(
        "McBopomofoLM v2 bundles third-party parts (side-by-side test build, not for redistribution):\n"
        "- SlothE-T 25M encoder (enc25m.mlmodelc, enc25m_embed_f16.bin; converted to Core ML from the published "
        "weights) and its syl_vocab / syl2legal tables: huggingface.co/Luigi/sloth-ime-models, Apache-2.0.\n"
        "- SlothE decoder pred_q35_60m (dec60m.mlmodelc, dec_tokenizer.json; converted to Core ML from the published "
        "weights): huggingface.co/Luigi/sloth-ime-models, Apache-2.0.\n"
        "- char2id table: github.com/vieenrose/sloth-zhuyin-linux, which has no LICENSE file; license status unresolved.\n"
        "- Orthographic variant classes derived from OpenCC TWVariants.txt / HKVariants.txt, Apache-2.0 "
        "(OpenCC-LICENSE.txt).\n"
        "v2 runs the models with Core ML on the Apple Neural Engine only. It contains no ggml, llama.cpp or "
        "libslothe code.\n", encoding="utf-8")
    files = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*")
                   if p.is_file() and p.name not in ("runtime-manifest.txt", "MANIFEST.txt", "NOTICE.txt", "OpenCC-LICENSE.txt"))
    (OUT / "runtime-manifest.txt").write_text(
        "# McBopomofoLM v2 checks the byte size and sha256 of every file below before loading a model.\n"
        "# Any mismatch or missing file = that model is not loaded (encoder -> stock McBopomofo,\n"
        "# decoder -> encoder-only in-walk). Format: <sha256> <bytes> <path>\n"
        + "".join(f"{sha256(OUT / n)} {(OUT / n).stat().st_size} {n}\n" for n in files), encoding="utf-8")
    lines = [
        "SlothE runtime files for McBopomofoLM v2 (Core ML, ANE only)",
        f"encoder source: {ENC_PKG.relative_to(W)} tree sha256 {tree_sha(ENC_PKG)}",
        f"decoder source: {DEC_PKG.relative_to(W)} tree sha256 {tree_sha(DEC_PKG)}",
        f"embedding: {EMBED.relative_to(W)} sha256 {sha256(EMBED)}",
        f"tokenizer: HF pred_q35_60m/tokenizer.json sha256 {sha256(TOKENIZER)}",
        "compiled with: xcrun coremlcompiler compile",
        "config A' (walk2/frozen_final2.json): 25M, beta 0.3, gamma 0, pen -15, variant guard; decoder lambda 2, tau 0.5, top-3",
        "",
    ]
    (OUT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(len(files), "files in runtime-manifest.txt")


if __name__ == "__main__":
    main()
