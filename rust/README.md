# animalpose (Rust)

A pure-Rust port of the Python package: same configs, same checkpoints, same outputs. No PyTorch, no
OpenCV, no C toolchain. Video goes through the `ffmpeg` / `ffprobe` executables. Downloads go through
`curl`.

```bash
cargo build --release
./target/release/animalpose predict --video myvideo.mp4 --out outputs/run
cargo test --release
```

## Release downloads

The **Release** workflow in GitHub Actions builds separate `animalpose` and `animalpose-gui`
downloads. Linux x86-64 and ARM64 have `.deb`, `.rpm`, and `.tar.gz` packages. macOS Intel and
Apple Silicon have a CLI `.tar.gz` and an ad-hoc signed desktop `.dmg`. Windows x86-64 has two
standalone `.exe` files. The workflow also builds the Python wheel and source archive.

Extract `animalpose-models-v<version>.zip` into `~/.animalpose` (`%USERPROFILE%\.animalpose` on
Windows), with `configs/` and `weights/` at the top level. Portable executables also look for those
folders beside the binary. Install `ffmpeg` and `ffprobe` to open or record videos.

Run **Actions → Release → Run workflow** to build downloadable artifacts. Push a matching tag
such as `v0.1.0` to publish the packages and checksums to GitHub Releases.

## Commands

| Command | What it does |
|---|---|
| `predict` | detect → pose → track → temporal → behavior. Takes `--video` (a file or frame folder), `--views` + `--cameras` for 3D, or `--synthetic`. Writes `tracks.csv/npz`, `behavior.csv`, `timing.json`, `overlay.mp4` |
| `train`, `finetune` | all four networks: detector, pose, temporal, behavior |
| `label` | pseudo-label unlabeled recordings |
| `behavior` | `cluster`, `name`, `label`, `plot` |
| `data` | `audit`, `calibrate`, `convert`, `prepare`, `download` |
| `eval` | `unlabeled`, `real`, `real-tracking`, `behavior-sessions`, `benchmark` |
| `export` | `safetensors`, `import-safetensors` |

Checkpoints load from `.pt` (read directly, no torch needed) or `.safetensors`.

## Desktop app (`--features gui`)

```bash
cargo run --release --features gui --bin animalpose-gui
```

1. Tick one or more cameras, or type a video file path, then press **Start**. Each view shows live
   skeletons and track IDs.
2. Press **Record** / **Stop recording**. By default, each camera is saved under
   `~/AnimalPose/recordings/rec-<time>/cam<i>.mp4`.
3. Press **Analyze last recording**. This runs the full pipeline (temporal model and behavior
   included) into `<recording>/analysis/`. With several cameras plus a `cameras.json` from
   `animalpose data calibrate`, it runs in 3D.

The cameras are not hardware-synchronized. On macOS the first **Start** asks for camera access.

## GPU (`--features mps`, macOS)

On the first frame, the detector and pose forward passes are recorded on the CPU. They are then
replayed as one Metal Performance Shaders graph, with the weights kept on the GPU. Outputs match the
CPU within 1e-3. `device: cpu` in the config turns it off. The `gui` feature includes it.

| Apple Silicon | CPU | GPU |
|---|---|---|
| detector, 256×320 | 17 ms | 2.7 ms |
| pose (ResNet-18), 1 crop | 20 ms | 3.5 ms |
| pose, 8 crops | 105 ms | 11 ms |
| `predict`, 300 frames, 2 mice | 26 s | 4.4 s |

`cargo run --release --features mps --example gpu_bench` reproduces the network timings.

## Modules

| Module | Contents |
|---|---|
| `tensor` | f32 autodiff; im2col + `matrixmultiply` GEMM, multithreaded with `rayon`; MPSGraph replay (`mpsgraph`) |
| `models` | the four networks and their losses; `.pt` and `.safetensors` loading |
| `algos` | decoding, NMS, Kalman + Hungarian tracking, blob proposals, behavior post-processing |
| `data` | configs, simulator and renderer, datasets, degradation, frames/video, converters |
| `multiview` | cameras, calibration, triangulation, fusion, left/right symmetry |
| `eval` | AP, PCK, MOTA/IDF1, F1, AUROC and the eval harnesses |
| `cli` | the command line |
| `gui` | the desktop app (`--features gui`) |

## Parity with Python

Python is the reference. Tests pin Rust to torch and NumPy dumps under `../tests/fixtures`:
- network outputs within 5e-5;
- pipeline keypoints within 1e-3 px;
- the four general `.pt` checkpoints load bit-exact.

Every intentional difference is marked `// differs from Python:` in the source. Examples: OpenCV
drawing and warping replaced by Rust rasterizers, and a different random number generator.

Speed on Apple Silicon, CPU only: ResNet-18 pose forward pass ≈ 20 ms per crop, about the same as
torch on the CPU.

## Not ported

ONNX / TorchScript export, checkerboard corner detection, `behavior sessions`, and several
dataset-specific `eval` reports (`rat7m-3d`, `mirror-paws`, `mirror-videos`, `blur`,
`crim13-identity`, `real-tracks`, `behavior-loao`). These print `unsupported:` and name the Python
command to use.

## Toolchain

`rust-toolchain.toml` pins 1.88.0 for local builds. CI uses stable Rust on Linux (x86-64, arm64),
Windows and macOS (x86-64, arm64), and enforces `cargo fmt` and `cargo clippy -D warnings`.
Use `git lfs pull` at the repository root to fetch the test fixtures and general checkpoints.

GPL-3.0-only. See [LICENSE](../LICENSE).
