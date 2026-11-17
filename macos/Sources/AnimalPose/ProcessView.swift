import SwiftUI

/// The video list, where results go, and the batch progress.
struct ProcessView: View {
    @Environment(Engine.self) private var engine
    @State private var selection = Set<Int>()
    @State private var dropping = false

    var body: some View {
        let state = engine.state
        Group {
            if state.videos.isEmpty {
                ContentUnavailableView {
                    Label("No Videos", systemImage: "film.stack")
                } description: {
                    Text("Drop videos or folders here.")
                } actions: {
                    Button("Add…") { engine.addVideos(Files.chooseVideos()) }
                        .controlSize(.large)
                }
            } else {
                List(selection: $selection) {
                    if let groups = state.viewGroups {
                        ForEach(Array(groups.enumerated()), id: \.offset) { _, group in
                            Section {
                                ForEach(group.videos.filter { $0.index < state.videos.count }, id: \.index) {
                                    row($0.index, camera: $0.camera)
                                }
                            } header: {
                                HStack {
                                    Text(group.title)
                                    Spacer()
                                    if group.videos.count > 1, group.error == nil {
                                        Text("\(group.videos.count) Views")
                                    }
                                }
                            } footer: {
                                if let error = group.error {
                                    Label(error, systemImage: "exclamationmark.triangle.fill")
                                        .font(.caption)
                                        .foregroundStyle(.orange)
                                }
                            }
                        }
                    } else {
                        ForEach(state.videos.indices, id: \.self) { row($0, camera: nil) }
                    }
                }
                .onDeleteCommand { remove(selection) }
                .disabled(!state.idle)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .dropDestination(for: URL.self) { urls, _ in
            guard engine.state.idle else { return false }
            engine.addVideos(urls)
            return true
        } isTargeted: {
            dropping = $0
        }
        .overlay {
            if dropping {
                RoundedRectangle(cornerRadius: 18)
                    .strokeBorder(.tint, lineWidth: 3)
                    .padding(8)
                    .allowsHitTesting(false)
            }
        }
        .safeAreaInset(edge: .bottom) {
            ProcessPanel()
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
        }
        .navigationTitle("Process")
        .navigationSubtitle(subtitle)
        .toolbar {
            if state.canAnalyzeLast {
                ToolbarItem {
                    Button("Analyze Recording") { engine.send("analyze_last") }
                        .disabled(!state.idle)
                }
            }
            ToolbarItemGroup {
                Button {
                    engine.addVideos(Files.chooseVideos())
                } label: {
                    Label("Add", systemImage: "plus")
                }
                .help("Add Videos or Folders")
                .disabled(!state.idle)
                Button {
                    engine.send("clear_videos")
                    selection.removeAll()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .help("Clear List")
                .disabled(!state.idle || state.videos.isEmpty)
            }
        }
    }

    private func row(_ index: Int, camera: String?) -> some View {
        let path = engine.state.videos[index]
        return VideoRow(path: path, active: isActive(path), camera: camera)
            .tag(index)
            .contextMenu {
                Button("Show in Finder") { Files.reveal(path) }
                Divider()
                Button("Remove", role: .destructive) { remove([index]) }
            }
    }

    private var subtitle: String {
        switch engine.state.videos.count {
        case 0: ""
        case 1: "1 Video"
        case let n: "\(n) Videos"
        }
    }

    private func isActive(_ path: String) -> Bool {
        guard let current = engine.state.processing?.current, !current.isEmpty else { return false }
        return path.hasSuffix("/" + current) || path == current
    }

    private func remove(_ indices: Set<Int>) {
        for index in indices.sorted(by: >) {
            engine.send("remove_video", ["index": index])
        }
        selection.removeAll()
    }
}

struct VideoRow: View {
    var path: String
    var active: Bool
    /// The camera of the camera model this video is, with Multiview.
    var camera: String? = nil

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: active ? "waveform.circle.fill" : "film")
                .font(.title3)
                .foregroundStyle(active ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                .symbolEffect(.pulse, isActive: active)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(path.fileName)
                Text(abbreviated(path.parentPath))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            if let camera {
                Text(camera)
                    .font(.caption.weight(.medium))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: .capsule)
            }
        }
        .padding(.vertical, 3)
        .help(path)
    }
}

/// The floating glass panel: progress, where results go, and Process.
struct ProcessPanel: View {
    @Environment(Engine.self) private var engine

    var body: some View {
        let state = engine.state
        GlassGroup {
            VStack(alignment: .leading, spacing: 14) {
                if let processing = state.processing {
                    ProgressSection(processing: processing)
                    Divider()
                }
                HStack(spacing: 12) {
                    LabeledContent("Save To") {
                        FolderMenu(
                            path: state.outputDir, isDefault: state.settings.output == nil,
                            defaultTitle: "Next to Video", choose: engine.setOutput)
                    }
                    .disabled(!state.idle)
                    Toggle(
                        "Multiview",
                        isOn: Binding(
                            get: { state.settings.multiview },
                            set: { engine.send("set_multiview", ["on": $0]) })
                    )
                    .toggleStyle(.switch)
                    .disabled(
                        !state.idle || (!state.settings.multiview && state.settings.cameraModel == nil))
                    .help(
                        state.settings.cameraModel == nil
                            ? "Set a Camera Model in the Profile"
                            : "Fuse Each Folder’s Views with the Camera Model")
                    Spacer()
                    if !state.ffmpeg {
                        Label("FFmpeg Not Installed", systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    }
                    Button("Process") { engine.send("process") }
                        .prominentButton()
                        .controlSize(.large)
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(!state.idle || state.videos.isEmpty || !state.ffmpeg)
                }
            }
            .padding(16)
            .glassPanel()
        }
        .animation(.smooth, value: state.processing == nil)
    }
}

/// Two bars for a batch: which video, and how far into it.
struct ProgressSection: View {
    @Environment(Engine.self) private var engine
    var processing: ProcessingState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if processing.total > 1 {
                ProgressView(value: processing.fraction) {
                    Text(processing.videoLabel).font(.headline)
                }
            }
            ProgressView(value: processing.total > 1 ? processing.clipFraction : processing.fraction) {
                HStack {
                    Text(processing.current).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Text(status).monospacedDigit().foregroundStyle(.secondary)
                }
                .font(processing.total > 1 ? .callout : .headline)
            }
            HStack {
                Spacer()
                Button(processing.mode == "running" ? "Pause" : "Resume") {
                    engine.send("pause_or_resume")
                }
                .glassButton()
                Button("Stop", role: .destructive) { engine.send("stop_processing") }
                    .glassButton()
            }
            .disabled(processing.mode == "stopping")
        }
    }

    private var status: String {
        switch processing.mode {
        case "pausing", "paused": "Paused"
        case "stopping": "Stopping…"
        default: processing.stageLabel
        }
    }
}
