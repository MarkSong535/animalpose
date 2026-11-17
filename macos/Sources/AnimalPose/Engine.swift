import CAnimalPose
import CoreGraphics
import Foundation
import Observation
import AppKit

/// Settings saved in `~/AnimalPose/settings.json`, shared with the egui app.
struct AppSettings: Codable, Equatable {
    var profile: String?
    var recentResults: [String] = []
    var recordings: String?
    var output: String?
    var config: String?
    var models: [String: String] = [:]
    var stages: [Stage] = []
    var live = true
    var classifyBehavior = true
    var cameraModel: String?
    var views: [String: String] = [:]
    var multiview = false
}

struct Stage: Codable, Equatable, Identifiable {
    var id = UUID()
    /// Optional; an unnamed stage is "Stage 2" and so on.
    var name: String
    var seconds: UInt32
    var waitAfter: Bool

    enum CodingKeys: String, CodingKey {
        case name, seconds, waitAfter
    }

    /// Equal content, whatever the row identity.
    static func == (a: Stage, b: Stage) -> Bool {
        a.name == b.name && a.seconds == b.seconds && a.waitAfter == b.waitAfter
    }

    var json: [String: Any] { ["name": name, "seconds": seconds, "wait_after": waitAfter] }
}

/// A named setup: config, models, behavior, recording stages and the arena's cameras.
struct Profile: Codable, Equatable, Identifiable {
    var name: String
    var config: String?
    var models: [String: String] = [:]
    var classifyBehavior = true
    var stages: [Stage] = []
    var cameraModel: String?
    var views: [String: String] = [:]
    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name, config, models, classifyBehavior, stages, cameraModel, views
    }

    var json: [String: Any] {
        [
            "name": name, "config": config ?? NSNull(), "models": models,
            "classify_behavior": classifyBehavior, "stages": stages.map(\.json),
            "camera_model": cameraModel ?? NSNull(), "views": views,
        ]
    }
}

/// A JSON scalar from the config form.
enum FieldValue: Codable, Equatable {
    case bool(Bool), number(Double), text(String), none

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let b = try? c.decode(Bool.self) {
            self = .bool(b)
        } else if let n = try? c.decode(Double.self) {
            self = .number(n)
        } else if let t = try? c.decode(String.self) {
            self = .text(t)
        } else {
            self = .none
        }
    }

    func encode(to encoder: Encoder) throws {}

    var bool: Bool { if case .bool(let b) = self { b } else { false } }
    var number: Double { if case .number(let n) = self { n } else { 0 } }
    var text: String { if case .text(let t) = self { t } else { "" } }
}

struct ConfigField: Codable, Equatable, Identifiable {
    var key: String
    var label: String
    var type: String
    var min: Double?
    var max: Double?
    var step: Double?
    var options: [String]?
    var value: FieldValue
    var id: String { key }
}

struct ConfigSection: Codable, Equatable, Identifiable {
    var title: String
    var fields: [ConfigField]
    var id: String { title }
}

/// The config open in the editor.
struct ConfigForm: Codable, Equatable {
    var name: String?
    var editable: Bool
    var sections: [ConfigSection]
}

struct CameraSource: Codable, Equatable {
    var name: String
    var selected: Bool
    /// What a profile's `views` stores this camera's label under.
    var key: String
    /// Its label before it is named: `Camera <ID>`.
    var defaultView: String
}

struct WorkerState: Codable, Equatable {
    var name: String
    /// The view this camera records as.
    var view: String
    var ready: Bool
    var ended: Bool
    var captureError: String?
    var previewError: String?
}

struct SessionState: Codable, Equatable {
    var label: String
    var timed: Bool
    var waiting: Bool
    var stageFraction: Double?
    var remaining: Double
    var elapsed: Double
    var dir: String
}

struct ProcessingState: Codable, Equatable {
    var total: Int
    var fraction: Double
    var clipFraction: Double
    var videoLabel: String
    var stageLabel: String
    var label: String
    var current: String
    var mode: String
}

/// Everything the window shows, as `ap_engine_state` describes it.
struct EngineState: Codable, Equatable {
    var status = ""
    var result: String?
    var recentResults: [String] = []
    var logDir = ""
    var profiles: [Profile] = []
    var profile: String?
    var recordingName = ""
    var configForm: ConfigForm?
    var libraryRoot = ""
    var recordingsDir = ""
    var outputDir: String?
    var settings = AppSettings()
    var configs: [String] = []
    var models: [String: [String]] = [:]
    var sources: [CameraSource] = []
    var videos: [String] = []
    var playback = false
    var cameraModels: [String] = []
    /// The cameras of the profile's camera model.
    var cameraNames: [String] = []
    var cameraModelError: String?
    /// With Multiview, the video list by folder.
    var viewGroups: [ViewGroup]?
    var workers: [WorkerState] = []
    var sourcesReady = false
    var session: SessionState?
    var processing: ProcessingState?
    var idle = true
    var busy = false
    var importing = false
    var finishing = false
    var canAnalyzeLast = false
    var ffmpeg = true
}

/// One folder of views for Multiview, and the camera each video is.
struct ViewGroup: Codable, Equatable {
    struct Member: Codable, Equatable {
        var index: Int
        var camera: String?
    }
    var title: String
    var videos: [Member]
    var error: String?
}

/// One preview tile's newest image.
struct PreviewFrame {
    var sequence: UInt64
    var overlay: Bool
    var image: CGImage
    var animals: Int?
    var analysisMs: Double
}

/// The model kinds in pipeline order.
let modelKinds = ["detector", "pose", "temporal", "behavior"]

/// The Rust engine behind the window. Every call stays on the main actor.
@MainActor
@Observable
final class Engine {
    private(set) var state = EngineState()
    private(set) var frames: [Int: PreviewFrame] = [:]
    /// The last command the engine rejected.
    var error: String?

    @ObservationIgnored private let handle: OpaquePointer?
    @ObservationIgnored private var polling: Task<Void, Never>?
    @ObservationIgnored private let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()

    init() {
        ap_camera_init()
        handle = ap_engine_new()
        refresh()
        polling = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.tick()
                let busy = !self.state.workers.isEmpty
                try? await Task.sleep(for: .milliseconds(busy ? 33 : 150))
            }
        }
    }

    /// Stops every job; called when the app quits.
    func shutdown() {
        polling?.cancel()
        if let handle { ap_engine_free(handle) }
    }

    private var frameTick = 0

    private func tick() {
        // Frames at ~30 Hz while previewing; the state at ~7 Hz.
        frameTick += 1
        if !state.workers.isEmpty { pullFrames() }
        if state.workers.isEmpty || frameTick % 4 == 0 { refresh() }
    }

    func refresh() {
        guard let handle, let text = ap_engine_state(handle) else { return }
        apply(text)
    }

    private func apply(_ text: UnsafeMutablePointer<CChar>) {
        let data = Data(bytes: text, count: strlen(text))
        ap_string_free(text)
        if let failure = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let message = failure["error"] as? String
        {
            error = message
            return
        }
        do {
            let next = try decoder.decode(EngineState.self, from: data)
            if next != state { state = next }
            if next.workers.isEmpty, !frames.isEmpty { frames = [:] }
        } catch {
            self.error = "Unexpected engine state: \(error)"
        }
    }

    /// Runs one engine command, `{"cmd": name, ...arguments}`.
    func send(_ name: String, _ arguments: [String: Any] = [:]) {
        guard let handle else { return }
        var command = arguments
        command["cmd"] = name
        guard let data = try? JSONSerialization.data(withJSONObject: command),
            let json = String(data: data, encoding: .utf8)
        else { return }
        guard let reply = json.withCString({ ap_engine_command(handle, $0) }) else { return }
        apply(reply)
    }

    private func pullFrames() {
        guard let handle else { return }
        for index in state.workers.indices {
            let known = frames[index]
            var frame = ApFrame()
            guard
                ap_engine_frame(
                    handle, UInt32(index), known?.sequence ?? 0, known?.overlay ?? false, &frame)
            else { continue }
            guard let rgba = frame.rgba else { continue }
            let data = Data(bytes: rgba, count: frame.len)
            ap_frame_free(rgba, frame.len)
            guard let image = Self.image(data, width: Int(frame.width), height: Int(frame.height))
            else { continue }
            frames[index] = PreviewFrame(
                sequence: frame.sequence, overlay: frame.overlay, image: image,
                animals: frame.animals >= 0 ? Int(frame.animals) : nil,
                analysisMs: frame.analysis_ms)
        }
    }

    private static func image(_ data: Data, width: Int, height: Int) -> CGImage? {
        guard width > 0, height > 0, let provider = CGDataProvider(data: data as CFData) else {
            return nil
        }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    // MARK: Actions

    func addVideos(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        send("add_videos", ["paths": urls.map(\.path)])
    }

    func setOutput(_ url: URL?) {
        send("set_output", ["path": url?.path ?? NSNull()])
    }

    func setRecordings(_ url: URL?) {
        send("set_recordings", ["path": url?.path ?? NSNull()])
    }

    func importCameraModel(_ url: URL, into profile: String) {
        send("import_camera_model", ["path": url.path, "profile": profile])
    }

    func setConfig(_ name: String?) {
        send("set_config", ["name": name ?? NSNull()])
    }

    func setModel(_ kind: String, _ name: String?) {
        send("set_model", ["kind": kind, "name": name ?? NSNull()])
    }

    func setStages(_ stages: [Stage]) {
        let rows = stages.map {
            ["name": $0.name, "seconds": $0.seconds, "wait_after": $0.waitAfter] as [String: Any]
        }
        send("set_stages", ["stages": rows])
    }

    func importItem(_ kind: String, _ url: URL) {
        send("import", ["kind": kind, "path": url.path])
    }

    /// The profile in use.
    var activeProfile: Profile? {
        state.profiles.first { $0.name == state.profile }
    }

    func saveProfile(_ profile: Profile, previous: String?) {
        send("save_profile", ["previous": previous ?? NSNull(), "profile": profile.json])
    }

    func editConfig(_ name: String?) {
        send("edit_config", ["name": name ?? NSNull()])
    }
}

/// Opens a results folder in Finder.
@MainActor
func openFolder(_ path: String) {
    NSWorkspace.shared.open(URL(fileURLWithPath: path))
}
