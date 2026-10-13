#!/bin/bash
# Build walk_cli (stock walk / in-walk walk / decoder requests / pinned walk) from the repo's C++ sources.
# Linux: SlothEEngine.cpp is copied and patched (NFC normalize -> identity, CommonCrypto -> OpenSSL); needs g++ and libssl-dev.
# macOS: the original sources build as-is with -framework CoreFoundation.
set -euo pipefail
TOOLS=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$TOOLS/../../.." && pwd)
WORK=${EVAL_WORK:-$HOME/.cache/mcbpmflm-eval}; mkdir -p "$WORK/build"
E=$REPO/Source/Engine; SL=$REPO/Source/SlothE
ENGINE_SRC=$SL/SlothEEngine.cpp; LIBS=""
if [ "$(uname -s)" = "Darwin" ]; then
  LIBS="-framework CoreFoundation"
else
  ENGINE_SRC=$WORK/build/SlothEEngine.cpp
  python3 - "$SL/SlothEEngine.cpp" "$ENGINE_SRC" <<'PY'
import sys
s = open(sys.argv[1], encoding="utf-8").read()
s = s.replace("#include <CommonCrypto/CommonDigest.h>", "#include <openssl/sha.h>").replace("#include <CoreFoundation/CoreFoundation.h>", "")
a = s.index("std::string NfcNormalize(const std::string& s) {"); b = s.index("\n}\n", a) + 3
s = s[:a] + "std::string NfcNormalize(const std::string& s) { return s; }  // inputs are NFC already\n" + s[b:]
for k, v in [("CC_SHA256_CTX", "SHA256_CTX"), ("CC_SHA256_Init", "SHA256_Init"), ("CC_SHA256_Update", "SHA256_Update"),
             ("CC_SHA256_Final", "SHA256_Final"), ("CC_SHA256_DIGEST_LENGTH", "SHA256_DIGEST_LENGTH"), ("static_cast<CC_LONG>", "static_cast<size_t>")]:
    s = s.replace(k, v)
open(sys.argv[2], "w", encoding="utf-8").write(s)
PY
  LIBS="-lcrypto"
fi
g++ -std=c++17 -O2 -w -I"$E" -I"$E/gramambular2" -I"$SL" -o "$WORK/walk_cli" "$TOOLS/walk_cli.cpp" "$ENGINE_SRC" "$SL/SlothEWalk.cpp" \
  "$E/gramambular2/reading_grid.cpp" "$E/McBopomofoLM.cpp" "$E/ParselessLM.cpp" "$E/ParselessPhraseDB.cpp" "$E/UserPhrasesLM.cpp" \
  "$E/PhraseReplacementMap.cpp" "$E/AssociatedPhrasesV2.cpp" "$E/ByteBlockBackedDictionary.cpp" "$E/MemoryMappedFile.cpp" \
  "$E/UTF8Helper.cpp" "$E/VariantAnnotator.cpp" "$E/UserOverrideModel.cpp" $LIBS
echo "built $WORK/walk_cli"
