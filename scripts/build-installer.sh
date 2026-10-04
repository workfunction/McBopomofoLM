#!/bin/bash
# Builds the McBopomofoLM installer around a finished input method.
#
#   scripts/build-installer.sh <path-to-McBopomofoLM.app> <output-dir>
#
# 1. checks that the IME and the installer carry the same CFBundleVersion and
#    CFBundleShortVersionString, that the IME is the right bundle and that its signature is valid;
# 2. zips the IME with `ditto -c -k --keepParent --norsrc` to
#    Source/Installer/NotarizedArchives/McBopomofoLM-r<CFBundleVersion>.zip, the only file the
#    installer's ArchiveUtil accepts there (it aborts on any other visible entry). The script
#    stops if the folder holds anything besides its tracked .gitkeep (a Finder .DS_Store is
#    tolerated, not touched), adds that ONE file and, on exit, removes that ONE file;
# 3. builds the McBopomofoInstaller scheme (Release, arm64 + x86_64) with that zip inside;
# 4. checks the built installer (both architectures, one archive, no dev-mode app, same versions,
#    valid ad-hoc signature);
# 5. zips the installer app into <output-dir>/McBopomofoLM-Installer-<short version>.zip; the
#    script stops if that file already exists and never cleans the output folder.
#
# Environment (all optional):
#   DEVELOPER_DIR         Xcode to use (default: the selected one, else /Applications/Xcode.app)
#   SOURCE_PACKAGES_DIR   a folder with the already resolved Swift packages; the build then runs
#                         without network access (-disableAutomaticPackageResolution). Without it
#                         the packages are fetched into the build's own temp folder every run.
#   KEEP_ARCHIVE=1        leave the zip in Source/Installer/NotarizedArchives afterwards (the next
#                         run then stops until it is removed)
#
# What gets deleted: the one temp folder this run made with `mktemp -d` (derived data, staging,
# build log; removed on exit by its exact path), and the one zip file the script added to
# NotarizedArchives. Nothing in the repository or in the output folder is ever removed
# recursively, and there is no path override.
#
# Threat model: the temp folder is made in the per-user temp folder (`getconf
# DARWIN_USER_TEMP_DIR`, /var/folders/../T/), never in $TMPDIR, which is ignored. That parent must
# be a real directory (not a symlink) owned by the current user with mode 0700, checked before the
# build and again right before the final `rm -rf`; if it is not, the folder is left in place.
# Only a process running as the same user can swap that parent, and such a process can already
# delete everything this user owns without going through this script, so the remaining race (a
# swap between the last check and the delete) is accepted as out of scope.
#
# The result is ad-hoc signed and not notarized: users approve it once under System Settings >
# Privacy & Security > Open Anyway. This script never installs anything.

set -euo pipefail

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 2
}

fail() {
    echo "build-installer: error: $*" >&2
    exit 1
}

step() {
    echo "==> $*"
}

[ $# -eq 2 ] || usage
case "$1" in -h | --help) usage ;; esac

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[ -d "$1" ] || fail "not a folder: $1"
IME_APP="$(cd "$1" && pwd)"
OUT_ARG="$2" # created only once everything has been checked and built

if [ -z "${DEVELOPER_DIR:-}" ] && ! xcodebuild -version >/dev/null 2>&1; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

INSTALLER_PLIST="$ROOT/Source/Installer/Installer-Info.plist"
ARCHIVE_DIR="$ROOT/Source/Installer/NotarizedArchives"
APP_NAME=McBopomofoLM
IME_BUNDLE_ID=org.openvanilla.inputmethod.McBopomofoLM

# Scratch space: derived data, the staged zips, the build log. A private, unique folder made by
# mktemp in the per-user temp folder (see the threat model in the header), so nobody else's path
# is ever deleted: the exit trap removes exactly this one.
USER_TMP_PARENT="$(getconf DARWIN_USER_TEMP_DIR)"
USER_TMP_PARENT="${USER_TMP_PARENT%/}"

# temp_parent_ok: the parent is a real directory (not a symlink) owned by this user, mode 0700.
temp_parent_ok() {
    [ -n "$USER_TMP_PARENT" ] && [ -d "$USER_TMP_PARENT" ] && [ ! -L "$USER_TMP_PARENT" ] &&
        [ "$(stat -f '%u' "$USER_TMP_PARENT")" = "$(id -u)" ] &&
        [ "$(stat -f '%Lp' "$USER_TMP_PARENT")" = 700 ]
}
temp_parent_ok || fail "the per-user temp folder $USER_TMP_PARENT is not a real directory owned by you with mode 0700; not building"

TMP="$(mktemp -d "$USER_TMP_PARENT/mcbpmf-installer.XXXXXX")"
readonly TMP
ARCHIVE_ZIP="" # set once, when this run added the zip to NotarizedArchives
cleanup() {
    if [ -n "$ARCHIVE_ZIP" ] && [ "${KEEP_ARCHIVE:-0}" != 1 ]; then
        rm -f -- "$ARCHIVE_ZIP"
    fi
    # Checked again right before the delete: if the parent was swapped meanwhile, leave the folder.
    if temp_parent_ok && [ "$(dirname "$TMP")" = "$USER_TMP_PARENT" ] && [ ! -L "$TMP" ]; then
        rm -rf -- "$TMP"
    else
        echo "build-installer: warning: the temp parent changed; leaving $TMP in place" >&2
    fi
}
trap cleanup EXIT
trap 'exit 1' INT TERM HUP

plist() { # plist <file> <key>
    /usr/libexec/PlistBuddy -c "Print :$2" "$1" 2>/dev/null || true
}

# ---- 1. the IME and the installer agree ------------------------------------------------------
step "Checking $IME_APP"
[ "$(basename "$IME_APP")" = "$APP_NAME.app" ] || fail "the app must be named $APP_NAME.app (the installer looks for that name inside the zip)"
IME_PLIST="$IME_APP/Contents/Info.plist"
[ -f "$IME_PLIST" ] || fail "no Contents/Info.plist in $IME_APP"

IME_BUILD="$(plist "$IME_PLIST" CFBundleVersion)"
IME_SHORT="$(plist "$IME_PLIST" CFBundleShortVersionString)"
INSTALLER_BUILD="$(plist "$INSTALLER_PLIST" CFBundleVersion)"
INSTALLER_SHORT="$(plist "$INSTALLER_PLIST" CFBundleShortVersionString)"
[ -n "$IME_BUILD" ] && [ -n "$IME_SHORT" ] || fail "the IME has no CFBundleVersion / CFBundleShortVersionString"
[ -n "$INSTALLER_BUILD" ] && [ -n "$INSTALLER_SHORT" ] || fail "the installer has no CFBundleVersion / CFBundleShortVersionString"
if [ "$IME_BUILD" != "$INSTALLER_BUILD" ] || [ "$IME_SHORT" != "$INSTALLER_SHORT" ]; then
    fail "version mismatch: IME $IME_SHORT (r$IME_BUILD), installer $INSTALLER_SHORT (r$INSTALLER_BUILD). Both Info.plists must carry the same CFBundleVersion and CFBundleShortVersionString."
fi
[ "$(plist "$IME_PLIST" CFBundleIdentifier)" = "$IME_BUNDLE_ID" ] || fail "the IME's bundle identifier is not $IME_BUNDLE_ID"
codesign --verify --deep --strict "$IME_APP" || fail "the IME's code signature is not valid"
IME_ARCHS="$(lipo -archs "$IME_APP/Contents/MacOS/$APP_NAME" 2>/dev/null || true)"
case " $IME_ARCHS " in *" arm64 "*) ;; *) fail "the IME binary has no arm64 slice (found: ${IME_ARCHS:-none})" ;; esac
echo "    IME and installer: $IME_SHORT (r$IME_BUILD), IME architectures: $IME_ARCHS"

ZIP_NAME="$APP_NAME-r$IME_BUILD.zip"
OUT_ZIP_NAME="McBopomofoLM-Installer-$IME_SHORT.zip"

# The output file must not exist yet; checked here so a long build is not wasted, and once more
# when the file is written.
if [ -e "$OUT_ARG/$OUT_ZIP_NAME" ] || [ -L "$OUT_ARG/$OUT_ZIP_NAME" ]; then
    fail "$OUT_ARG/$OUT_ZIP_NAME already exists; the script never overwrites or cleans the output folder"
fi

# ---- 2. the archive inside the installer -----------------------------------------------------
[ ! -L "$ARCHIVE_DIR" ] || fail "$ARCHIVE_DIR is a symlink"
[ -d "$ARCHIVE_DIR" ] || fail "$ARCHIVE_DIR is not a folder (it is tracked through its .gitkeep)"
# Fail closed: anything this script did not make (another archive, a stray file, a folder) stops
# the run, and nothing is deleted. A Finder .DS_Store is tolerated: the installer ignores hidden
# entries and the build's resource copy leaves it out.
UNEXPECTED="$(find "$ARCHIVE_DIR" -mindepth 1 -maxdepth 1 ! -name .gitkeep ! -name .DS_Store)"
if [ -n "$UNEXPECTED" ]; then
    fail "unexpected files in $ARCHIVE_DIR (remove them yourself, this script deletes only the one zip it adds):
$UNEXPECTED"
fi

step "Zipping the IME to Source/Installer/NotarizedArchives/$ZIP_NAME"
STAGED_ZIP="$TMP/$ZIP_NAME"
ditto -c -k --keepParent --norsrc "$IME_APP" "$STAGED_ZIP"
ZIP_LISTING="$(unzip -Z1 "$STAGED_ZIP")" # captured: `grep -q` on a pipe would SIGPIPE unzip under pipefail
printf '%s\n' "$ZIP_LISTING" | grep -qx "$APP_NAME.app/Contents/Info.plist" || fail "the zip has no $APP_NAME.app/Contents/Info.plist"
if printf '%s\n' "$ZIP_LISTING" | grep -q '^__MACOSX'; then fail "the zip carries __MACOSX resource entries"; fi
# Add that one file, never over something that is there; from now on the exit trap may remove it.
[ ! -e "$ARCHIVE_DIR/$ZIP_NAME" ] && [ ! -L "$ARCHIVE_DIR/$ZIP_NAME" ] || fail "$ARCHIVE_DIR/$ZIP_NAME already exists"
cp -n -- "$STAGED_ZIP" "$ARCHIVE_DIR/$ZIP_NAME"
cmp -s "$STAGED_ZIP" "$ARCHIVE_DIR/$ZIP_NAME" || fail "could not add $ZIP_NAME to $ARCHIVE_DIR"
ARCHIVE_ZIP="$ARCHIVE_DIR/$ZIP_NAME"
# Exactly that one file besides the tracked .gitkeep (and a tolerated .DS_Store).
COUNT="$(find "$ARCHIVE_DIR" -mindepth 1 -maxdepth 1 ! -name .gitkeep ! -name .DS_Store | wc -l | tr -d ' ')"
[ "$COUNT" = 1 ] || fail "$ARCHIVE_DIR must hold exactly $ZIP_NAME, found $COUNT entries"

# ---- 3. build the installer ------------------------------------------------------------------
# Derived data lives in the temp folder: every run is a clean build, and nothing under the
# repository is deleted to make it so.
DERIVED="$TMP/derived"
BUILD_LOG="$TMP/xcodebuild.log"
temp_parent_ok || fail "the per-user temp folder $USER_TMP_PARENT changed before the build; not building"
step "Building the McBopomofoInstaller scheme (Release, arm64 + x86_64)"
PRODUCT="$DERIVED/Build/Products/Release/McBopomofoInstaller.app"
PACKAGE_ARGS=()
if [ -n "${SOURCE_PACKAGES_DIR:-}" ]; then
    PACKAGE_ARGS=(-clonedSourcePackagesDirPath "$SOURCE_PACKAGES_DIR" -disableAutomaticPackageResolution)
fi
if ! (cd "$ROOT" && xcodebuild -project McBopomofo.xcodeproj -scheme McBopomofoInstaller \
    -configuration Release -derivedDataPath "$DERIVED" ${PACKAGE_ARGS[@]+"${PACKAGE_ARGS[@]}"} \
    ARCHS="arm64 x86_64" ONLY_ACTIVE_ARCH=NO CODE_SIGN_IDENTITY=- build >"$BUILD_LOG" 2>&1); then
    tail -60 "$BUILD_LOG" >&2
    fail "xcodebuild failed (the log was in the temp folder, shown above)"
fi
[ -d "$PRODUCT" ] || fail "the build left no $PRODUCT"

# ---- 4. check what was built -----------------------------------------------------------------
step "Checking the built installer"
BIN="$PRODUCT/Contents/MacOS/McBopomofoInstaller"
ARCHS_FOUND="$(lipo -archs "$BIN")"
case " $ARCHS_FOUND " in *" arm64 "*) ;; *) fail "no arm64 slice in the installer ($ARCHS_FOUND)" ;; esac
case " $ARCHS_FOUND " in *" x86_64 "*) ;; *) fail "no x86_64 slice in the installer ($ARCHS_FOUND)" ;; esac
BUILT_PLIST="$PRODUCT/Contents/Info.plist"
[ "$(plist "$BUILT_PLIST" CFBundleVersion)" = "$IME_BUILD" ] || fail "the built installer's CFBundleVersion is not $IME_BUILD"
[ "$(plist "$BUILT_PLIST" CFBundleShortVersionString)" = "$IME_SHORT" ] || fail "the built installer's CFBundleShortVersionString is not $IME_SHORT"
EMBEDDED="$PRODUCT/Contents/Resources/NotarizedArchives"
[ -f "$EMBEDDED/$ZIP_NAME" ] || fail "the installer does not carry $ZIP_NAME"
[ "$(find "$EMBEDDED" -mindepth 1 ! -name '.*' | wc -l | tr -d ' ')" = 1 ] || fail "the installer's NotarizedArchives holds more than $ZIP_NAME"
[ ! -e "$PRODUCT/Contents/Resources/$APP_NAME.app" ] || fail "the installer carries a dev-mode $APP_NAME.app next to the archive (ArchiveUtil refuses that)"
cmp -s "$STAGED_ZIP" "$EMBEDDED/$ZIP_NAME" || fail "the embedded zip differs from the one just made"
codesign --verify --deep --strict "$PRODUCT" || fail "the installer's code signature is not valid"
echo "    architectures: $ARCHS_FOUND"
echo "    embedded: NotarizedArchives/$ZIP_NAME ($(du -h "$EMBEDDED/$ZIP_NAME" | cut -f1 | tr -d ' '))"

# ---- 5. the installer zip --------------------------------------------------------------------
mkdir -p "$OUT_ARG"
OUT_DIR="$(cd "$OUT_ARG" && pwd)"
OUT_ZIP="$OUT_DIR/$OUT_ZIP_NAME"
step "Zipping the installer to $OUT_ZIP"
STAGED_OUT="$TMP/$OUT_ZIP_NAME"
ditto -c -k --keepParent --norsrc "$PRODUCT" "$STAGED_OUT"
# Written without overwriting; the output folder is never cleaned.
[ ! -e "$OUT_ZIP" ] && [ ! -L "$OUT_ZIP" ] || fail "$OUT_ZIP already exists"
cp -n -- "$STAGED_OUT" "$OUT_ZIP"
cmp -s "$STAGED_OUT" "$OUT_ZIP" || fail "could not write $OUT_ZIP"
echo "    sha256 $(shasum -a 256 "$OUT_ZIP" | cut -d' ' -f1)  $(basename "$OUT_ZIP")"
echo "    sha256 $(shasum -a 256 "$BIN" | cut -d' ' -f1)  McBopomofoInstaller (binary)"
echo "done: $OUT_ZIP"
