"""Package the native binaries built on this host, or the shared model bundle."""

import argparse
import platform
import plistlib
import shutil
import struct
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
LICENSE = ROOT / "LICENSE"
BINARIES = ("animalpose", "animalpose-gui")
RUST_PACKAGE = tomllib.loads((ROOT / "rust/Cargo.toml").read_text())["package"]
VERSION = RUST_PACKAGE["version"]
PACKAGE_LICENSE = RUST_PACKAGE["license"]


def run(*args):
    subprocess.run(args, check=True)


def portable(binary, dest, name):
    with tarfile.open(dest / f"{name}.tar.gz", "w:gz") as archive:
        archive.add(binary, arcname=f"{name}/{binary.name}")
        archive.add(LICENSE, arcname=f"{name}/LICENSE")


def linux_packages(binary, dest, arch):
    name = binary.name
    gui = name.endswith("-gui")
    description = "AnimalPose desktop app" if gui else "AnimalPose command-line tools"
    deb_arch = {"x86_64": "amd64", "aarch64": "arm64"}[arch]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        root = work / "package"
        (root / "usr/bin").mkdir(parents=True)
        shutil.copy2(binary, root / "usr/bin")
        license_path = Path(f"usr/share/doc/{name}/copyright")
        (root / license_path.parent).mkdir(parents=True)
        shutil.copy2(LICENSE, root / license_path)

        deb_deps = ["libc6 (>= 2.35)", "libgcc-s1"]
        deb_recommends = ["ffmpeg", "curl"]
        rpm_deps = []
        if gui:
            deb_recommends.append("xdg-desktop-portal")
            desktop = root / "usr/share/applications/animalpose-gui.desktop"
            desktop.parent.mkdir(parents=True)
            desktop.write_text(
                "[Desktop Entry]\nType=Application\nName=AnimalPose\nExec=animalpose-gui\n"
                "Terminal=false\nCategories=Science;Video;\n"
            )
            deb_deps.extend(
                [
                    "libgl1",
                    "libegl1",
                    "libx11-6",
                    "libxcursor1",
                    "libxi6",
                    "libxrandr2",
                    "libxkbcommon0",
                    "libxkbcommon-x11-0",
                    "libwayland-client0",
                    "libwayland-egl1",
                ]
            )
            # OpenGL and window-system libraries can be loaded at runtime by the GUI.
            rpm_deps = [
                f"Requires: {lib}()(64bit)"
                for lib in (
                    "libGL.so.1",
                    "libEGL.so.1",
                    "libX11.so.6",
                    "libXcursor.so.1",
                    "libXi.so.6",
                    "libXrandr.so.2",
                    "libxkbcommon.so.0",
                    "libxkbcommon-x11.so.0",
                    "libwayland-client.so.0",
                    "libwayland-egl.so.1",
                )
            ]
            rpm_deps.append("Recommends: xdg-desktop-portal")

        (root / "DEBIAN").mkdir()
        (root / "DEBIAN/control").write_text(
            f"Package: {name}\nVersion: {VERSION}\nSection: science\nPriority: optional\n"
            f"Architecture: {deb_arch}\nMaintainer: AnimalPose contributors\n"
            f"Depends: {', '.join(deb_deps)}\nRecommends: {', '.join(deb_recommends)}\nDescription: {description}\n"
        )
        deb = dest / f"{name}_{VERSION}_{deb_arch}.deb"
        run("dpkg-deb", "--root-owner-group", "--build", root, deb)
        run("dpkg-deb", "--info", deb)

        spec = work / f"{name}.spec"
        spec.write_text(
            "\n".join(
                [
                    f"Name: {name}",
                    f"Version: {VERSION}",
                    "Release: 1",
                    f"Summary: {description}",
                    f"License: {PACKAGE_LICENSE}",
                    *rpm_deps,
                    "",
                    "%description",
                    description,
                    "",
                    "%install",
                    'mkdir -p "%{buildroot}"',
                    f'cp -a "{root}/usr" "%{{buildroot}}/"',
                    "",
                    "%files",
                    f"%license /{license_path}",
                    f"/usr/bin/{name}",
                    *(["/usr/share/applications/animalpose-gui.desktop"] if gui else []),
                    "",
                ]
            )
        )
        run("rpmbuild", "--define", f"_topdir {work}", "--define", "_build_id_links none", "--target", arch, "-bb", spec)
        rpm = work / f"RPMS/{arch}/{name}-{VERSION}-1.{arch}.rpm"
        run("rpm", "-qpl", rpm)
        shutil.copy2(rpm, dest)


def macos_app(binary, dest, name):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        app = root / "AnimalPose.app"
        contents = app / "Contents"
        (contents / "MacOS").mkdir(parents=True)
        shutil.copy2(binary, contents / "MacOS")
        with (contents / "Info.plist").open("wb") as plist:
            plistlib.dump(
                {
                    "CFBundleIdentifier": "com.marksong535.animalpose",
                    "CFBundleName": "AnimalPose",
                    "CFBundleExecutable": binary.name,
                    "CFBundlePackageType": "APPL",
                    "CFBundleShortVersionString": VERSION,
                    "CFBundleVersion": VERSION,
                    "NSCameraUsageDescription": "AnimalPose uses your camera to track and record animals.",
                },
                plist,
            )
        run("codesign", "--force", "--sign", "-", app)
        run("codesign", "--verify", "--strict", app)
        (root / "Applications").symlink_to("/Applications")
        shutil.copy2(LICENSE, root)
        image = dest / f"{name}.dmg"
        run(
            "hdiutil", "create", "-ov", "-format", "UDZO", "-fs", "HFS+", "-volname", "AnimalPose", "-srcfolder", root, image
        )
        run("hdiutil", "verify", image)


def windows_exe(binary, dest, name, gui):
    # PE has distinct subsystem values for a desktop app (2) and a console app (3).
    with binary.open("rb") as exe:
        exe.seek(0x3C)
        pe_offset = struct.unpack("<I", exe.read(4))[0]
        exe.seek(pe_offset)
        if exe.read(4) != b"PE\0\0":
            raise ValueError(f"{binary} is not a Windows executable")
        exe.seek(pe_offset + 4 + 20 + 68)
        subsystem = struct.unpack("<H", exe.read(2))[0]
    if subsystem != (2 if gui else 3):
        raise ValueError(f"{binary} has the wrong Windows subsystem ({subsystem})")
    shutil.copy2(binary, dest / f"{name}.exe")


def native(binaries, dest):
    system = platform.system().lower()
    arch = platform.machine().lower()
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
    label = {"linux": "linux", "darwin": "macos", "windows": "windows"}[system]
    suffix = ".exe" if system == "windows" else ""
    cli = binaries / f"animalpose{suffix}"
    version = subprocess.check_output([cli, "--version"], text=True).strip()
    if version != f"animalpose {VERSION}":
        raise ValueError(f"{cli} reports {version!r}, expected animalpose {VERSION}")
    for program in BINARIES:
        binary = binaries / f"{program}{suffix}"
        name = f"{program}-v{VERSION}-{label}-{arch}"
        if system == "windows":
            windows_exe(binary, dest, name, gui=program == "animalpose-gui")
        elif system == "darwin" and program == "animalpose-gui":
            macos_app(binary, dest, name)
        else:
            portable(binary, dest, name)
            if system == "linux":
                linux_packages(binary, dest, arch)


def models(dest):
    checkpoints = [ROOT / f"weights/{model}_general.pt" for model in ("detector", "pose", "temporal", "behavior")]
    for path in checkpoints:
        if not zipfile.is_zipfile(path):
            raise ValueError(f"{path} is missing or invalid. Run `git lfs pull`.")
    configs = [
        ROOT / f"configs/{name}.yaml"
        for name in (
            "pipeline",
            "detector",
            "pose",
            "temporal",
            "temporal3d",
            "behavior",
            "behavior3d",
        )
    ]
    with zipfile.ZipFile(dest / f"animalpose-models-v{VERSION}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [*configs, *checkpoints, LICENSE]:
            archive.write(path, path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("version", "native", "models"))
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--binaries", type=Path, default=ROOT / "rust/target/release")
    args = parser.parse_args()
    python_package = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    for field in ("version", "license"):
        if RUST_PACKAGE[field] != python_package[field]:
            parser.error(f"Rust and Python disagree on {field}: {RUST_PACKAGE[field]!r} and {python_package[field]!r}")
    gpl_text = LICENSE.read_text().lstrip().startswith("GNU GENERAL PUBLIC LICENSE")
    if gpl_text != PACKAGE_LICENSE.startswith("GPL-"):
        parser.error(f"LICENSE and the package license {PACKAGE_LICENSE!r} disagree")
    if args.kind == "version":
        print(VERSION)
        return
    dest = args.out.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if args.kind == "native":
        native(args.binaries.resolve(), dest)
    else:
        models(dest)


if __name__ == "__main__":
    main()
