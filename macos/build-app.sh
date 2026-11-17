#!/bin/sh
# Builds AnimalPose.app: the Rust engine as a static library, the SwiftUI app linked against it,
# and the bundle around them (ad-hoc signed). Usage: ./build-app.sh [output-dir]
set -eu
cd "$(dirname "$0")"
out="${1:-build}"
version="${ANIMALPOSE_VERSION:-$(sed -n 's/^version = "\(.*\)"/\1/p' ../rust/Cargo.toml | head -n 1)}"
# Keep the C code bundled by Rust crates (libjpeg) loadable on the oldest supported macOS.
export MACOSX_DEPLOYMENT_TARGET=14.0

cargo rustc --manifest-path ../rust/Cargo.toml --release --lib --features ffi --crate-type staticlib
# The CLI with the app layer, to check the sample profiles before they are bundled.
cargo build --manifest-path ../rust/Cargo.toml --release --features ffi --bin animalpose
swift build -c release --product AnimalPose
bin="$(swift build -c release --show-bin-path)/AnimalPose"

app="$out/AnimalPose.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$bin" "$app/Contents/MacOS/AnimalPose"
# The icon, from docs/icons/minimal.svg (scripts/make_icons.sh).
cp AppIcon.icns "$app/Contents/Resources/AppIcon.icns"
sed "s/@VERSION@/$version/g" Info.plist > "$app/Contents/Info.plist"
# Sample profiles (../samples, built by scripts/build_samples.py): installed into ~/AnimalPose on
# launch. Checked where they ship: hashes, then every profile installed, loaded and run.
cp -R ../samples "$app/Contents/Resources/samples"
../rust/target/release/animalpose samples check "$app/Contents/Resources/samples"
codesign --force --sign - "$app"
codesign --verify --strict "$app"
echo "$app"
