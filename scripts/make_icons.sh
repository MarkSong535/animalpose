#!/bin/sh
# The app icons of every build, from docs/icons/minimal.svg (run on macOS: AppKit renders the SVG,
# iconutil packs the .icns, ImageMagick the .ico). The outputs are committed, so builds on other
# systems need none of these tools:
#   macos/AppIcon.icns      the macOS app bundle (CFBundleIconFile)
#   rust/assets/icon.png    the egui window and dock icon; the Linux packages' hicolor icon
#   rust/assets/icon.ico    embedded in the Windows executables (rust/build.rs)
set -eu
cd "$(dirname "$0")/.."
svg=docs/icons/minimal.svg
margin=0.06
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

render() { swift scripts/render_svg.swift "$svg" "$2" "$1" "$margin"; }

set_dir="$work/AppIcon.iconset"
mkdir -p "$set_dir"
for size in 16 32 128 256 512; do
    render "$size" "$set_dir/icon_${size}x${size}.png"
    render $((size * 2)) "$set_dir/icon_${size}x${size}@2x.png"
done
iconutil --convert icns --output macos/AppIcon.icns "$set_dir"

render 256 rust/assets/icon.png
for size in 16 24 32 48 64 128; do
    render "$size" "$work/ico_$size.png"
done
magick "$work"/ico_16.png "$work"/ico_24.png "$work"/ico_32.png "$work"/ico_48.png \
    "$work"/ico_64.png "$work"/ico_128.png rust/assets/icon.png rust/assets/icon.ico
ls -l macos/AppIcon.icns rust/assets/icon.png rust/assets/icon.ico
