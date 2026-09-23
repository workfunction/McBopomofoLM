#!/bin/bash
# Build the SlothE-T runtime dependencies for McBopomofoLM as arm64 static libraries:
#   build/slothe-deps/lib/{libslothe,libggml,libggml-base,libggml-cpu}.a
# CMake: any cmake >= 3.20 on PATH, or CMAKE=/path/to/cmake.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="$ROOT/build/slothe-deps"
CMAKE="${CMAKE:-cmake}"
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

"$CMAKE" -S "$HERE" -B "$OUT/cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DCMAKE_C_COMPILER="$(xcrun --find clang)" \
  -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)" \
  -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)"
"$CMAKE" --build "$OUT/cmake" -j8 --target slothe ggml ggml-base ggml-cpu

mkdir -p "$OUT/lib"
for lib in slothe ggml ggml-base ggml-cpu; do
  f="$(/usr/bin/find "$OUT/cmake" -name "lib$lib.a" -print -quit)"
  [ -n "$f" ] || { echo "missing lib$lib.a" >&2; exit 1; }
  cp "$f" "$OUT/lib/"
done
lipo -info "$OUT"/lib/*.a

# ---- SlothE decoder: llama.cpp (@7ab4ee7, vendored in ThirdParty/llama.cpp) with its own ggml ----
# Built static, then pre-linked into ONE relocatable object that exports only the llama_* C API:
# every symbol of llama.cpp's bundled ggml becomes local, so it cannot clash with the encoder's
# ggml (ThirdParty/ggml @456172e) that libslothe uses. Output: build/slothe-deps/lib/libllama_isolated.a
LB="$OUT/llama-cmake"
"$CMAKE" -S "$HERE/ThirdParty/llama.cpp" -B "$LB" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
  -DCMAKE_C_COMPILER="$(xcrun --find clang)" -DCMAKE_CXX_COMPILER="$(xcrun --find clang++)" \
  -DCMAKE_OSX_SYSROOT="$(xcrun --sdk macosx --show-sdk-path)" \
  -DCMAKE_C_FLAGS=-fno-common -DCMAKE_CXX_FLAGS=-fno-common \
  -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
"$CMAKE" --build "$LB" -j8 --target llama
LIBS=()
for lib in llama ggml ggml-base ggml-cpu; do
  f="$(/usr/bin/find "$LB" -name "lib$lib.a" -print -quit)"
  [ -n "$f" ] || { echo "missing llama.cpp lib$lib.a" >&2; exit 1; }
  LIBS+=(-force_load "$f")
done
xcrun ld -r -arch arm64 -platform_version macos 13.0 "$(xcrun --sdk macosx --show-sdk-version)" -exported_symbols_list "$HERE/llama_exports.txt" "${LIBS[@]}" -o "$OUT/llama_isolated.o"
rm -f "$OUT/lib/libllama_isolated.a"
xcrun libtool -static -o "$OUT/lib/libllama_isolated.a" "$OUT/llama_isolated.o"
echo "global symbols exported by libllama_isolated.a (non-llama_* must be 0):"
nm -gU "$OUT/lib/libllama_isolated.a" 2>/dev/null | /usr/bin/awk 'NF>=3 && $3 !~ /^_llama_/' | wc -l
