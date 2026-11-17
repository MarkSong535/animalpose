# AnimalPose for macOS

The native app: SwiftUI over the Rust engine (`../rust/src/app`), linked as a static library
through the C interface in `../rust/src/ffi.rs`. It shares `~/AnimalPose` (models, configs,
settings) with the egui window used on Windows and Linux.

```bash
./build-app.sh          # writes build/AnimalPose.app
open build/AnimalPose.app
```

Needs Rust and Xcode (Swift 6). Liquid Glass needs Xcode 26 or later to build and macOS 26 or
later to run. Older toolchains and systems get translucent materials instead. The app runs on
macOS 14 and later.
