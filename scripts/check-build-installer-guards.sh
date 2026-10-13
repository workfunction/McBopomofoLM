#!/bin/bash
# Self-check for the cleanup rules of build-installer.sh. It runs the script end to end in a
# throwaway repository layout (a temp folder; the real repository is never touched) with stubs
# for xcodebuild, codesign and lipo on PATH, so the run reaches the whole way through the build,
# the cleanup and the output stage. Checked:
#   - nothing in the script deletes recursively except the one `rm -rf -- "$TMP"` of its own
#     mktemp folder, and nothing uses find -delete / -exec;
#   - the temp folder is made in the per-user temp folder (`getconf DARWIN_USER_TEMP_DIR`, here a
#     stub pointing into the layout), the environment's TMPDIR is ignored, and the parent must be
#     a real directory owned by the user with mode 0700 (a wrong mode and a symlink are refused;
#     a parent swapped for a symlink during the build keeps the cleanup from deleting anything);
#   - an ordinary run leaves no temp folder behind, only .gitkeep in NotarizedArchives, and the
#     output zip; a DERIVED_DATA / BUILD_LOG in the environment is ignored; no repo folder named
#     build is created or touched;
#   - a stray file in NotarizedArchives, a symlinked NotarizedArchives, and an existing output
#     file each make it refuse, without deleting anything;
#   - a failing build is cleaned up; a build during which the folder holding NotarizedArchives is
#     swapped for a symlink to another folder (the race a check-then-delete would lose) deletes
#     at most the one zip file by its exact name, and nothing else there.
# Prints OK at the end; exits 1 if any check failed.
#
#   scripts/check-build-installer-guards.sh

set -euo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/bi-guards.XXXXXX")"
trap 'rm -rf -- "$WORK"' EXIT

failures=0
check() { # check <name> <exit status of the condition: 0 = ok>
    if [ "$2" -eq 0 ]; then echo "ok   $1"; else echo "FAIL $1"; failures=$((failures + 1)); fi
}
yes_if() { if "$@"; then echo 0; else echo 1; fi; } # yes_if <test...> -> 0 / 1

# ---- stubs -----------------------------------------------------------------------------------
mkdir -p "$WORK/bin"
cat >"$WORK/bin/xcodebuild" <<'STUB'
#!/bin/bash
# Stub: -version works; a build makes a fake product the way Xcode would (Info.plist, a binary,
# the NotarizedArchives folder copied without .DS_Store), unless STUB_XCODEBUILD says otherwise.
[ "${1:-}" = "-version" ] && { echo "Xcode (stub)"; exit 0; }
echo "$*" >"$STUB_REPO/xcodebuild-args"
dd=""
while [ $# -gt 0 ]; do [ "$1" = "-derivedDataPath" ] && dd="$2"; shift; done
echo "$dd" >"$STUB_REPO/dd-path"
case "${STUB_XCODEBUILD:-ok}" in
swapparent)
    # While "building", the per-user temp folder is swapped for a symlink to another folder that
    # holds a folder with the very name of this run's temp folder (what a delete would follow).
    tmpdir="$(dirname "$dd")"
    parent="$(dirname "$tmpdir")"
    mkdir -p "$STUB_VICTIM/$(basename "$tmpdir")"
    : >"$STUB_VICTIM/$(basename "$tmpdir")/sentinel"
    mv "$parent" "$parent.real"
    ln -s "$STUB_VICTIM" "$parent"
    ;;
fail) echo "stub: build failed" >&2; exit 1 ;;
swap)
    # While "building", the folder that holds NotarizedArchives is replaced by a symlink to
    # somebody else's folder (what a check-then-delete would have followed).
    mv "$STUB_REPO/Source/Installer" "$STUB_REPO/Source/Installer.real"
    ln -s "$STUB_VICTIM" "$STUB_REPO/Source/Installer"
    ;;
esac
app="$dd/Build/Products/Release/McBopomofoInstaller.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources/NotarizedArchives"
cp "$STUB_REPO/Source/Installer.real/Installer-Info.plist" "$app/Contents/Info.plist" 2>/dev/null ||
    cp "$STUB_REPO/Source/Installer/Installer-Info.plist" "$app/Contents/Info.plist"
echo binary >"$app/Contents/MacOS/McBopomofoInstaller"
src="$STUB_REPO/Source/Installer.real/NotarizedArchives"
[ -d "$src" ] || src="$STUB_REPO/Source/Installer/NotarizedArchives"
for f in "$src"/*; do [ -e "$f" ] && cp "$f" "$app/Contents/Resources/NotarizedArchives/"; done
exit 0
STUB
cat >"$WORK/bin/getconf" <<'STUB'
#!/bin/bash
# Stub: the per-user temp folder is the one the test names; everything else is the real getconf.
[ "${1:-}" = "DARWIN_USER_TEMP_DIR" ] && [ -n "${FAKE_USER_TMP:-}" ] && { echo "$FAKE_USER_TMP/"; exit 0; }
exec /usr/bin/getconf "$@"
STUB
printf '#!/bin/bash\nexit 0\n' >"$WORK/bin/codesign"
printf '#!/bin/bash\necho "x86_64 arm64"\n' >"$WORK/bin/lipo"
# nm stub (v3.3l): ordinary symbols by default; STUB_NM=profiled adds LLVM's profile runtime,
# fail = nm cannot read the file (status 1, nothing on stdout), empty = status 0 and no output.
cat >"$WORK/bin/nm" <<'STUB'
#!/bin/bash
case "${STUB_NM:-}" in
fail) echo "nm: file format not recognized" >&2; exit 1 ;;
empty) exit 0 ;;
esac
echo "0000000000001000 T _main"
[ "${STUB_NM:-}" = profiled ] && echo "0000000000002000 S ___llvm_profile_runtime"
exit 0
STUB
chmod +x "$WORK/bin/"*

# ---- fixtures --------------------------------------------------------------------------------
write_plist() { # write_plist <file> <bundle id or empty>
    /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string 2600" \
        -c "Add :CFBundleShortVersionString string 3.3.0" ${2:+-c "Add :CFBundleIdentifier string $2"} "$1" >/dev/null
}

# A fresh fake repository root with the script under test, the installer plist and an IME stand-in.
new_repo() {
    local repo="$WORK/$1"
    mkdir -p "$repo/scripts" "$repo/Source/Installer/NotarizedArchives" "$repo/ime/McBopomofoLM.app/Contents/MacOS"
    cp "$HERE/build-installer.sh" "$repo/scripts/"
    : >"$repo/Source/Installer/NotarizedArchives/.gitkeep"
    write_plist "$repo/Source/Installer/Installer-Info.plist" ""
    write_plist "$repo/ime/McBopomofoLM.app/Contents/Info.plist" org.openvanilla.inputmethod.McBopomofoLM
    echo ime >"$repo/ime/McBopomofoLM.app/Contents/MacOS/McBopomofoLM"
    mkdir -p "$repo/tmp" "$repo/tmp-env" "$repo/out"
    chmod 700 "$repo/tmp"
    : >"$repo/tmp-env/sentinel"
    echo "$repo"
}

# Somebody else's folder, with sentinels at several depths.
victim() {
    mkdir -p "$WORK/victim-$1/NotarizedArchives/deep" "$WORK/victim-$1/build/stuff"
    : >"$WORK/victim-$1/sentinel"
    : >"$WORK/victim-$1/NotarizedArchives/deep/sentinel"
    : >"$WORK/victim-$1/build/stuff/sentinel"
    echo "$WORK/victim-$1"
}

# run <repo> [env...]: the script with the stand-ins; output in $out, status in $status. The
# per-user temp folder is the layout's tmp/ (so leftovers of the script's mktemp folder can be
# seen); TMPDIR points at tmp-env/, which the script must ignore.
run() {
    local repo="$1"
    shift
    set +e
    out="$(env PATH="$WORK/bin:$PATH" TMPDIR="$repo/tmp-env" FAKE_USER_TMP="$repo/tmp" STUB_REPO="$repo" "$@" \
        "$repo/scripts/build-installer.sh" "$repo/ime/McBopomofoLM.app" "$repo/out" 2>&1)"
    status=$?
    set -e
}

# ---- 0. no recursive delete outside the script's own mktemp folder -------------------------------
script="$HERE/build-installer.sh"
recursive="$(grep -nE '(^|[^[:alnum:]_])rm[[:space:]]+-[a-zA-Z]*[rR]|-delete|-exec|rmdir|find .* -prune' "$script" | grep -v '^[0-9]*:[[:space:]]*#' || true)"
check "the only recursive delete is rm -rf -- \"\$TMP\"" \
    $([ "$(echo "$recursive" | grep -c .)" = 1 ] && echo "$recursive" | grep -q 'rm -rf -- "\$TMP"' && echo 0 || echo 1)
check "the TMP folder comes from mktemp -d in the per-user temp folder and is read-only afterwards" \
    $(yes_if sh -c "grep -q 'TMP=\"\$(mktemp -d \"\$USER_TMP_PARENT/' '$script' && grep -q '^readonly TMP' '$script'")
check "the script's code never reads TMPDIR" \
    $([ -z "$(grep -n 'TMPDIR' "$script" | grep -v '^[0-9]*:[[:space:]]*#' || true)" ] && echo 0 || echo 1)
check "the one other delete is rm -f -- of the added zip" \
    $(yes_if sh -c "grep -nE '(^|[^[:alnum:]_])rm ' '$script' | grep -v '^[0-9]*:[[:space:]]*#' | grep -c 'rm -f -- \"\$ARCHIVE_ZIP\"' | grep -q 1")

# ---- 1. an ordinary run -------------------------------------------------------------------------
repo="$(new_repo r1)"
v="$(victim 1)"
mkdir -p "$repo/build/keep"
: >"$repo/build/keep/sentinel"
run "$repo" DERIVED_DATA="$v/build" BUILD_LOG="$v/log"
check "an ordinary run succeeds" $([ "$status" -eq 0 ] && echo 0 || { echo "$out" >&2; echo 1; })
check "the installer zip is in the output folder" $(yes_if test -f "$repo/out/McBopomofoLM-Installer-3.3.0.zip")
check "NotarizedArchives holds only .gitkeep afterwards" \
    $([ "$(ls -A "$repo/Source/Installer/NotarizedArchives")" = ".gitkeep" ] && echo 0 || echo 1)
check "no temp folder is left behind" $([ -z "$(ls -A "$repo/tmp")" ] && echo 0 || echo 1)
check "the build ran in a folder under the per-user temp folder, not under TMPDIR" \
    $([[ "$(cat "$repo/dd-path")" == "$repo/tmp/mcbpmf-installer."*/derived ]] && echo 0 || echo 1)
check "an environment TMPDIR pointing at a folder is ignored (nothing made, nothing deleted there)" \
    $([ "$(ls -A "$repo/tmp-env")" = "sentinel" ] && echo 0 || echo 1)
check "DERIVED_DATA / BUILD_LOG from the environment are ignored" \
    $([ -e "$v/build/stuff/sentinel" ] && [ ! -e "$v/log" ] && echo 0 || echo 1)
check "the repository's build folder is untouched (and none is made by the script)" \
    $([ -e "$repo/build/keep/sentinel" ] && [ "$(ls -A "$repo/build")" = "keep" ] && echo 0 || echo 1)

# ---- 2. a stray file in NotarizedArchives -------------------------------------------------------
repo="$(new_repo r2)"
: >"$repo/Source/Installer/NotarizedArchives/McBopomofoLM-r2510.zip"
mkdir "$repo/Source/Installer/NotarizedArchives/subfolder"
run "$repo"
check "a stray archive makes the script refuse" $([ "$status" -ne 0 ] && echo "$out" | grep -q "unexpected files" && echo 0 || echo 1)
check "...and nothing was deleted" \
    $([ -e "$repo/Source/Installer/NotarizedArchives/McBopomofoLM-r2510.zip" ] && [ -d "$repo/Source/Installer/NotarizedArchives/subfolder" ] && echo 0 || echo 1)
check "...and no temp folder is left" $([ -z "$(ls -A "$repo/tmp")" ] && echo 0 || echo 1)

# ---- 3. a symlinked NotarizedArchives -----------------------------------------------------------
repo="$(new_repo r3)"
v="$(victim 3)"
rm "$repo/Source/Installer/NotarizedArchives/.gitkeep"
rmdir "$repo/Source/Installer/NotarizedArchives"
ln -s "$v/NotarizedArchives" "$repo/Source/Installer/NotarizedArchives"
run "$repo"
check "a symlinked NotarizedArchives makes the script refuse" $([ "$status" -ne 0 ] && echo "$out" | grep -q "is a symlink" && echo 0 || echo 1)
check "...and the folder it points to is untouched" \
    $([ -e "$v/NotarizedArchives/deep/sentinel" ] && [ -z "$(find "$v/NotarizedArchives" -maxdepth 1 -name '*.zip')" ] && echo 0 || echo 1)

# ---- 4. the output file exists ------------------------------------------------------------------
repo="$(new_repo r4)"
echo precious >"$repo/out/McBopomofoLM-Installer-3.3.0.zip"
: >"$repo/out/other"
run "$repo"
check "an existing output file makes the script refuse" $([ "$status" -ne 0 ] && echo "$out" | grep -q "already exists" && echo 0 || echo 1)
check "...and the output folder is untouched" \
    $([ "$(cat "$repo/out/McBopomofoLM-Installer-3.3.0.zip")" = precious ] && [ -e "$repo/out/other" ] && echo 0 || echo 1)
check "...and the repository folder is as it was" $([ "$(ls -A "$repo/Source/Installer/NotarizedArchives")" = ".gitkeep" ] && echo 0 || echo 1)

# ---- 4b. coverage (v3.3k): the build is told not to, and a profiled binary is refused -----------
check "the installer build runs with CLANG_COVERAGE_MAPPING=NO" \
    $(yes_if grep -q -- "CLANG_COVERAGE_MAPPING=NO" "$WORK/r1/xcodebuild-args")
repo="$(new_repo r4b)"
run "$repo" STUB_NM=profiled
check "an IME with LLVM profile instrumentation is refused" \
    $([ "$status" -ne 0 ] && echo "$out" | grep -q "profile instrumentation" && echo 0 || echo 1)
check "...before anything was added or built" \
    $([ "$(ls -A "$repo/Source/Installer/NotarizedArchives")" = ".gitkeep" ] && [ -z "$(ls -A "$repo/out")" ] && [ ! -e "$repo/dd-path" ] && echo 0 || echo 1)
for mode in fail empty; do
    repo="$(new_repo r4c-$mode)"
    run "$repo" STUB_NM=$mode
    check "an nm that ${mode}s makes the script refuse instead of reading it as 'no profiling'" \
        $([ "$status" -ne 0 ] && echo "$out" | grep -q "cannot be checked for profile instrumentation" && echo 0 || echo 1)
    check "...before anything was added or built ($mode)" \
        $([ "$(ls -A "$repo/Source/Installer/NotarizedArchives")" = ".gitkeep" ] && [ -z "$(ls -A "$repo/out")" ] && [ ! -e "$repo/dd-path" ] && echo 0 || echo 1)
done

# ---- 5. a failing build is cleaned up -----------------------------------------------------------
repo="$(new_repo r5)"
run "$repo" STUB_XCODEBUILD=fail
check "a failing build fails the script" $([ "$status" -ne 0 ] && echo 0 || echo 1)
check "...the zip it added is removed again" $([ "$(ls -A "$repo/Source/Installer/NotarizedArchives")" = ".gitkeep" ] && echo 0 || echo 1)
check "...and the temp folder is gone" $([ -z "$(ls -A "$repo/tmp")" ] && echo 0 || echo 1)
check "...and no output was written" $([ -z "$(ls -A "$repo/out")" ] && echo 0 || echo 1)

# ---- 6. the folder holding NotarizedArchives is swapped for a symlink during the build ------------
repo="$(new_repo r6)"
v="$(victim 6)"
: >"$v/NotarizedArchives/McBopomofoLM-r2600.zip" # the exact name: the one thing that may go
run "$repo" STUB_XCODEBUILD=swap STUB_VICTIM="$v"
check "the swap run finishes" $([ "$status" -eq 0 ] && echo 0 || { echo "$out" >&2; echo 1; })
check "after the swap only the zip with the exact name is gone" \
    $([ ! -e "$v/NotarizedArchives/McBopomofoLM-r2600.zip" ] && echo 0 || echo 1)
check "...and nothing else in that folder (files, sub folders, siblings)" \
    $([ -e "$v/sentinel" ] && [ -e "$v/NotarizedArchives/deep/sentinel" ] && [ -e "$v/build/stuff/sentinel" ] && echo 0 || echo 1)
check "...and no temp folder is left" $([ -z "$(ls -A "$repo/tmp")" ] && echo 0 || echo 1)

# ---- 7. the per-user temp folder is not as required ----------------------------------------------
repo="$(new_repo r7)"
chmod 755 "$repo/tmp"
run "$repo"
check "a per-user temp folder with mode 0755 is refused" $([ "$status" -ne 0 ] && echo "$out" | grep -q "mode 0700" && echo 0 || echo 1)
check "...before anything is built or made" $([ ! -e "$repo/dd-path" ] && [ -z "$(ls -A "$repo/tmp")" ] && [ -z "$(ls -A "$repo/out")" ] && echo 0 || echo 1)

repo="$(new_repo r8)"
v="$(victim 8)"
rm -rf "$repo/tmp"
ln -s "$v" "$repo/tmp"
run "$repo"
check "a per-user temp folder that is a symlink is refused" $([ "$status" -ne 0 ] && echo "$out" | grep -q "mode 0700" && echo 0 || echo 1)
check "...and the folder it points to is untouched" $([ -e "$v/sentinel" ] && [ -z "$(find "$v" -maxdepth 1 -name 'mcbpmf-installer.*')" ] && echo 0 || echo 1)

# ---- 8. the per-user temp folder is swapped for a symlink during the build ------------------------
repo="$(new_repo r9)"
v="$(victim 9)"
run "$repo" STUB_XCODEBUILD=swapparent STUB_VICTIM="$v"
tmpname="$(basename "$(dirname "$(cat "$repo/dd-path")")")"
check "after the swap the cleanup warns and deletes nothing through the link" \
    $([ -e "$v/$tmpname/sentinel" ] && echo "$out" | grep -q "leaving" && echo 0 || echo 1)
check "...the real temp folder is left where it was" $([ -d "$repo/tmp.real/$tmpname" ] && echo 0 || echo 1)
check "...and the added zip is still removed by its exact name" \
    $([ "$(ls -A "$repo/Source/Installer/NotarizedArchives")" = ".gitkeep" ] && echo 0 || echo 1)

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo OK
