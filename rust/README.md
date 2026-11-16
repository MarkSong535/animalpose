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

1. Select cameras and press **Start preview**. Choose the **Recording folder**, then start a
   manual or timed recording. On macOS, grant camera access when prompted.
2. Under **Protocol timers**, edit the **Accustomization** and **Experiment** durations in seconds.
   Add stages for additional experiments. **Wait for Continue** records the interval after a stage
   in a separate clip until you click **Continue to next stage**.
3. Each stage or wait interval gets a named folder with one MP4 per camera. **Stop and save**
   flushes the encoders. The session files list the protocol, the verified frame counts, and the
   first and last capture times of each clip. The default target is `~/AnimalPose/recordings`.

Recording takes priority over pose analysis. The app queues every frame it receives in order and
uses a separate, sampled raw preview while recording. If encoding falls behind, capture waits for
queue space. Frames lost upstream by the camera or driver cannot be recovered. Cameras are
free-running without hardware synchronization.

In **Video processing**, use **Add videos…** (several files at once), **Add folders…** (searched
recursively; hidden folders and earlier analysis folders are skipped) or type a path, then press
**Process**. One video writes `tracks.csv`, `tracks.npz`, and `overlay.mp4` into its own folder
under `~/AnimalPose/processed`. Several videos go into one `batch-…` folder with a subfolder per
video and a `batch.json` listing each input's output or error. A failed video is reported and the
batch continues with the next. Each video is analyzed as a separate recording: tracking and the
temporal model start fresh at its first frame, so nothing carries over from the previous video.
**Analyze last recording** processes each recorded stage separately. Supply a calibration JSON for
multi-camera 3D analysis of synchronized views.

The progress bar shows which pass is running and counts frames during analysis and overlay
writing. With several videos, a second bar shows which video is in progress. **Pause** and **Stop
processing** take effect after the current operation. **Resume** continues the same job. Stopping
discards unfinished analysis and keeps completed videos, stages and source recordings.

Under **Models and configs**, choose **Import folder** to find YAML configs and `.pt` or
`.safetensors` checkpoints in a folder and its subfolders. Folder imports link to library copies
of models they can locate and skip missing paths. Several configs can use the same checkpoint,
and importing identical model files reuses the stored copy.

**Import config** imports the YAML settings, including inherited settings. Choose the detector
and pose checkpoints in the app. Behavior classification is optional. The app produces
coordinates and an overlay without a behavior model. Clear **Classify behavior if available** to
skip classification even when a model is available.

Configs are stored in `~/AnimalPose/configs`. Model selections, recording location, and the timer
protocol are restored from `~/AnimalPose/settings.json`.
On Linux, file pickers use the desktop's XDG portal.

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
