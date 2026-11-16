#!/usr/bin/env bash
# Package one compiled binary into the installer formats of the release build: `.deb`, `.rpm`,
# `.tar.gz`, `.dmg`, or a Windows setup `.exe`. Run it on the platform you are packaging for
# (dpkg, rpmbuild, hdiutil and makensis are not portable), from any working directory.
#
#   scripts/package.sh deb target/release/animalpose animalpose 0.1.0
#   scripts/package.sh rpm target/release/animalpose animalpose 0.1.0
#   scripts/package.sh tar target/release/animalpose animalpose 0.1.0
#   scripts/package.sh dmg target/release/animalpose-gui animalpose-gui 0.1.0   # macOS
#   scripts/package.sh exe target/release/animalpose target/release/animalpose-gui 0.1.0  # Windows
#
# The Linux metadata comes from the environment: SUMMARY and DESCRIPTION (one line, then a
# paragraph) and DEPENDS, the space-separated runtime packages of the binary. `deb` and `rpm`
# therefore need DEPENDS for the GUI build (X11/GL libraries, which the headless one does not
# touch) and nothing for the headless one. OUT is the output directory, `dist` by default; the
# file name always carries the version and the architecture, so artifacts of several targets can
# share one folder.
set -euo pipefail

OUT=${OUT:-dist}
SUMMARY=${SUMMARY:-animalpose: species-agnostic multi-animal pose estimation}
DESCRIPTION=${DESCRIPTION:-Detect, track and classify animals in videos: keypoints, tracks, ethograms and multi-view 3D.}
MAINTAINER=${MAINTAINER:-animalpose contributors}
LICENSE=${LICENSE:-GPL-3.0-only}

die() {
    echo "package.sh: $*" >&2
    exit 1
}

# The version is embedded in the binary by `cargo`, so `--version` is the cheapest cross-check that
# the artifact matches the version the installer claims.
check_binary() {
    local bin=$1 want=$2 have
    [[ -x "$bin" ]] || die "no executable at $bin"
    have=$("$bin" --version 2>/dev/null | tail -1) || have="?"
    if [[ "$have" != "$want" && "$have" != *" $want"* ]]; then
        echo "package.sh: note: $bin reports '$have', packaging it as $want" >&2
    fi
}

# The packager name for the machine architecture, per format.
deb_arch() {
    case "$(uname -m)" in
        x86_64) echo amd64 ;;
        aarch64 | arm64) echo arm64 ;;
        armv7l | armv8l) echo armhf ;;
        i686) echo i386 ;;
        riscv64) echo riscv64 ;;
        *) echo "$(uname -m)" ;;
    esac
}

rpm_arch() {
    case "$(uname -m)" in
        x86_64) echo x86_64 ;;
        aarch64 | arm64) echo aarch64 ;;
        armv7l) echo armv7hl ;;
        i686) echo i686 ;;
        *) echo "$(uname -m)" ;;
    esac
}

# The architecture of the build, not of the machine: a cross-built binary would otherwise be
# labelled with the builder's arch. `cargo` prints the target triple in `--version`-independent
# ways, so the caller passes the target through TARGET when it cross-compiles.
arch_for() {
    case ${TARGET:-$(uname -m)} in
        x86_64* | amd64*) echo x86_64 ;;
        aarch64* | arm64*) echo arm64 ;;
        *) echo "${TARGET:-$(uname -m)}" ;;
    esac
}

# A staging tree laid out as the installed package: /usr/bin/<name> plus the licence and README.
stage_linux() {
    local stage=$1 bin=$2 name=$3
    mkdir -p "$stage/usr/bin" "$stage/usr/share/doc/$name"
    install -m 755 "$bin" "$stage/usr/bin/$name"
    install -m 644 "$LICENSE_FILE" "$stage/usr/share/doc/$name/LICENSE" 2>/dev/null || true
    install -m 644 README.md "$stage/usr/share/doc/$name/README.md" 2>/dev/null || true
}

LICENSE_FILE=$(cd "$(dirname "$0")/.." && pwd)/LICENSE

depends_deb() {
    local d=${DEPENDS// /, }
    [[ -n "$d" ]] && printf 'Depends: %s\n' "$d"
}

do_deb() {
    local bin=$1 name=$2 version=$3 arch stage out
    command -v dpkg-deb >/dev/null || die "dpkg-deb is not installed (apt install dpkg-dev)"
    arch=$(deb_arch)
    check_binary "$bin" "$version"
    stage=$(mktemp -d)
    trap 'rm -rf "$stage"' EXIT
    stage_linux "$stage" "$bin" "$name"
    mkdir -p "$stage/DEBIAN"
    {
        printf 'Package: %s\nVersion: %s\nArchitecture: %s\nMaintainer: %s\n' \
            "$name" "$version" "$arch" "$MAINTAINER"
        printf 'Priority: optional\nSection: science\nHomepage: https://github.com/MarkSong535/animalpose\n'
        printf '%s\nInstalled-Size: %s\n' "$(depends_deb)" \
            "$(( $(du -sk "$stage/usr" | cut -f1) ))"
        printf 'Description: %s\n %s\n' "$SUMMARY" "$DESCRIPTION"
    } > "$stage/DEBIAN/control"
    mkdir -p "$OUT"
    out=$OUT/${name}_${version}_${arch}.deb
    dpkg-deb --root-owner-group --build "$stage" "$out"
    echo "wrote $out"
}

do_rpm() {
    local bin=$1 name=$2 version=$3 arch top
    command -v rpmbuild >/dev/null || die "rpmbuild is not installed (apt install rpm)"
    arch=$(rpm_arch)
    check_binary "$bin" "$version"
    top=$(mktemp -d)
    trap 'rm -rf "$top"' EXIT
    mkdir -p "$top/BUILD" "$top/RPMS" "$top/SOURCES" "$top/SPECS" "$top/SRPMS" "$top/BUILDROOT"
    stage_linux "$top/SRC" "$bin" "$name"
    cp -a "$top/SRC/usr" "$top/SOURCES/usr"
    cat > "$top/SPECS/$name.spec" <<SPEC
Name: $name
Version: $version
Release: 1
Summary: $SUMMARY
License: $LICENSE
URL: https://github.com/MarkSong535/animalpose
$( [[ -n ${DEPENDS:-} ]] && for d in $DEPENDS; do echo "Requires: $d"; done )
%description
$DESCRIPTION
%prep
rm -rf %{buildroot}; mkdir -p %{buildroot}
cp -a %{_sourcedir}/usr %{buildroot}/usr
%files
/usr/bin/$name
/usr/share/doc/$name
SPEC
    mkdir -p "$OUT"
    rpmbuild -bb --noclean --target "$arch" \
        --define "_topdir $top" --define "debug_package %{nil}" \
        --define "_rpmfilename %%{ARCH}/%%{NAME}-%%{VERSION}-%%{RELEASE}.%%{ARCH}.rpm" \
        "$top/SPECS/$name.spec" >/dev/null
    local built=$top/RPMS/$arch/$name-$version-1.$arch.rpm
    [[ -f $built ]] || die "rpmbuild produced no $built"
    mv "$built" "$OUT/$name-$version-1.$arch.rpm"
    echo "wrote $OUT/$name-$version-1.$arch.rpm"
}

do_tar() {
    local bin=$1 name=$2 version=$3 arch stage out
    check_binary "$bin" "$version"
    arch=$(arch_for)
    stage=$(mktemp -d)
    trap 'rm -rf "$stage"' EXIT
    stage_linux "$stage" "$bin" "$name"
    mkdir -p "$OUT"
    out=$OUT/${name}-${version}-linux-${arch}.tar.gz
    tar --sort=name --owner=0 --group=0 --numeric-owner -czf "$out" -C "$stage" usr
    echo "wrote $out"
}

do_dmg() {
    local bin=$1 name=$2 version=$3 arch root out
    command -v hdiutil >/dev/null || die "hdiutil is macOS only"
    arch=$(arch_for)
    check_binary "$bin" "$version"
    # An ad-hoc signature: the copy out of the disk image passes the notarisation-less
    # `spctl` check on the machine that built it, and Gatekeeper only warns once for the rest.
    codesign --force --sign - --timestamp=none "$bin" 2>/dev/null || true
    mkdir -p "$OUT"
    root=$(mktemp -d)
    trap 'rm -rf "$root"' EXIT
    mkdir -p "$root/$name.app/Contents/MacOS"
    install -m 755 "$bin" "$root/$name.app/Contents/MacOS/$name"
    printf '%s\n' "$SUMMARY" > "$root/README.txt"
    printf 'Version %s (%s), %s licence.\n\n%s\n' "$version" "$arch" "$LICENSE" "$DESCRIPTION" >> "$root/README.txt"
    ln -s /Applications "$root/Applications"
    out=$OUT/${name}-${version}-macos-${arch}.dmg
    hdiutil create -quiet -ov -format UDZO -volname "$name" -srcfolder "$root" "$out"
    echo "wrote $out"
}

# A Windows installer with both executables: the headless CLI and the GUI app.
do_exe() {
    local headless=$1 gui=$2 version=$3 nsi out
    command -v makensis >/dev/null || die "makensis is not installed (choco install nsis)"
    [[ -f "$LICENSE" ]] && license=$LICENSE || license=""
    mkdir -p "$OUT"
    out=$OUT/animalpose-${version}-windows-x86_64-setup.exe
    nsi=$(mktemp --suffix=.nsi)
    trap 'rm -f "$nsi"' EXIT
    cat > "$nsi" <<'NSI'
!include MUI2.nsh
!ifndef HEADLESS
  !define HEADLESS "animalpose.exe"
!endif
!ifndef GUI
  !define GUI "animalpose-gui.exe"
!endif
!ifndef VERSION
  !define VERSION "0.0.0"
!endif
Name "animalpose"
OutFile "animalpose-setup.exe"
InstallDir "$PROGRAMFILES64\animalpose"
InstallDirRegKey HKLM "Software\animalpose" ""
RequestExecutionLevel admin
Unicode true
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
Section "animalpose" SecMain
  SectionIn RO
  SetOutPath $INSTDIR
  File "${HEADLESS}"
  File "${GUI}"
  File "README.txt"
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\animalpose" "" $INSTDIR
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\animalpose" \
    "DisplayName" "animalpose"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\animalpose" \
    "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\animalpose" \
    "UninstallString" "$\"$INSTDIR\uninstall.exe$\""
SectionEnd
Section "Uninstall"
  Delete "$INSTDIR\animalpose.exe"
  Delete "$INSTDIR\animalpose-gui.exe"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKLM "Software\animalpose"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\animalpose"
SectionEnd
NSI
    printf 'animalpose %s (Windows x86_64)\n\n%s\n\n%s\n' "$version" "$SUMMARY" \
        "The console build is animalpose.exe; the windowed build is animalpose-gui.exe." > README.txt
    "$(dirname "$0")/../.github/build-nsis.sh" >/dev/null 2>&1 || true
    makensis -DHEADLESS="$headless" -DGUI="$gui" -DVERSION="$version" \
        -DOUT="$out" -OUTFILE="$out" "$nsi" >/dev/null
    rm -f "$nsi" README.txt
    echo "wrote $out"
}

format=${1:-}
shift || true
case "$format" in
    deb) do_deb "$@" ;;
    rpm) do_rpm "$@" ;;
    tar) do_tar "$@" ;;
    dmg) do_dmg "$@" ;;
    exe) do_exe "$@" ;;
    *)
        sed -n '2,16p' "$0"
        exit 2
        ;;
esac
