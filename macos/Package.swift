// swift-tools-version: 6.0
// The native macOS app: SwiftUI over the Rust engine (`rust/src/ffi.rs`), linked as a static
// library. Build the library first, or use `./build-app.sh`, which does both and makes the .app:
//
//     cargo rustc --manifest-path ../rust/Cargo.toml --release --lib --features ffi --crate-type staticlib

import Foundation
import PackageDescription

let rustLibrary = Context.environment["ANIMALPOSE_RUST_LIB"]
    ?? URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("../rust/target/release")
    .standardizedFileURL.path

let package = Package(
    name: "AnimalPose",
    platforms: [.macOS(.v14)],
    targets: [
        .systemLibrary(name: "CAnimalPose", path: "Sources/CAnimalPose"),
        .executableTarget(
            name: "AnimalPose",
            dependencies: ["CAnimalPose"],
            path: "Sources/AnimalPose",
            linkerSettings: [
                .unsafeFlags(["-L", rustLibrary]),
                .linkedFramework("AppKit"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
                .linkedFramework("QuartzCore"),
                .linkedLibrary("iconv"),
            ]
        ),
    ]
)
