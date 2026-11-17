import SwiftUI

/// Camera previews with the live pose overlay, and recording.
struct LiveView: View {
    @Environment(Engine.self) private var engine

    var body: some View {
        let state = engine.state
        Group {
            if state.workers.isEmpty {
                ContentUnavailableView {
                    Label("No Preview", systemImage: "video.slash")
                } description: {
                    Text(state.sources.isEmpty ? "No cameras found." : "Choose a source below.")
                }
            } else {
                ScrollView {
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 360), spacing: 16)], spacing: 16
                    ) {
                        ForEach(Array(state.workers.enumerated()), id: \.offset) { index, worker in
                            PreviewTile(
                                index: index, worker: worker, frame: engine.frames[index],
                                recording: state.session != nil)
                        }
                    }
                    .padding(16)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .safeAreaInset(edge: .bottom) {
            LivePanel()
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
        }
        .navigationTitle("Live")
        .toolbar {
            ToolbarItem {
                Button {
                    engine.send("refresh_cameras")
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .help("Refresh Cameras")
                .disabled(!state.idle)
            }
        }
    }
}

struct PreviewTile: View {
    var index: Int
    var worker: WorkerState
    var frame: PreviewFrame?
    var recording: Bool

    var body: some View {
        ZStack {
            Rectangle().fill(.black.opacity(0.85))
            if let frame {
                Image(decorative: frame.image, scale: 1)
                    .resizable()
                    .interpolation(.medium)
                    .aspectRatio(contentMode: .fit)
            } else if !worker.ended {
                ProgressView().controlSize(.large).tint(.white)
            }
        }
        .aspectRatio(4 / 3, contentMode: .fit)
        .clipShape(.rect(cornerRadius: 18))
        .overlay(alignment: .topLeading) {
            HStack(spacing: 8) {
                if recording {
                    Circle().fill(.red).frame(width: 8, height: 8)
                }
                ViewName(index: index, view: worker.view)
                    .font(.callout.weight(.semibold))
                    .disabled(recording)
                Text(worker.name)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .glassCapsule()
            .padding(10)
        }
        .overlay(alignment: .bottomLeading) {
            VStack(alignment: .leading, spacing: 6) {
                if let frame, let animals = frame.animals {
                    Text("\(animals) animal\(animals == 1 ? "" : "s") · \(Int(frame.analysisMs)) ms")
                        .font(.caption.monospacedDigit())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .glassCapsule()
                }
                if let error = worker.captureError ?? (recording ? nil : worker.previewError) {
                    Label(error, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .lineLimit(2)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .glassCapsule()
                }
            }
            .padding(10)
        }
    }
}

/// Sources, preview, recording and the protocol timer, on one glass bar.
struct LivePanel: View {
    @Environment(Engine.self) private var engine

    var body: some View {
        let state = engine.state
        GlassGroup {
            VStack(alignment: .leading, spacing: 14) {
                if let session = state.session {
                    RecordingSection(session: session)
                } else {
                    SourcePicker()
                    Divider()
                    HStack(spacing: 10) {
                        Toggle(
                            "Pose Overlay",
                            isOn: Binding(
                                get: { state.settings.live },
                                set: { engine.send("set_live", ["on": $0]) })
                        )
                        .toggleStyle(.switch)
                        .disabled(state.busy)
                        Spacer()
                        if !state.workers.isEmpty {
                            RecordingNameField()
                        }
                        if state.workers.isEmpty {
                            Button {
                                engine.send("start_preview")
                            } label: {
                                Label("Preview", systemImage: "play.fill")
                            }
                            .prominentButton()
                            .disabled(!state.idle)
                        } else {
                            Button("Stop") { engine.send("stop_preview") }
                                .glassButton()
                            Button {
                                engine.send("record", ["timed": false])
                            } label: {
                                Label("Record", systemImage: "record.circle")
                            }
                            .glassButton()
                            .help("Record Until Stopped")
                            .disabled(!state.sourcesReady)
                            Button {
                                engine.send("record", ["timed": true])
                            } label: {
                                Label("Timed", systemImage: "timer")
                            }
                            .prominentButton()
                            .tint(.red)
                            .help("Record the Protocol Stages")
                            .disabled(!state.sourcesReady || state.settings.stages.isEmpty)
                        }
                    }
                    .controlSize(.large)
                }
            }
            .padding(18)
            .glassPanel()
        }
    }
}

struct SourcePicker: View {
    @Environment(Engine.self) private var engine

    var body: some View {
        let state = engine.state
        VStack(alignment: .leading, spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(Array(state.sources.enumerated()), id: \.offset) { index, source in
                        Toggle(
                            isOn: Binding(
                                get: { source.selected },
                                set: { engine.send("select_camera", ["index": index, "on": $0]) })
                        ) {
                            Label(source.name, systemImage: "web.camera")
                        }
                        .toggleStyle(.button)
                    }
                    if !state.videos.isEmpty {
                        Toggle(
                            isOn: Binding(
                                get: { state.playback },
                                set: { engine.send("set_playback", ["on": $0]) })
                        ) {
                            Label(state.videos[0].fileName, systemImage: "film")
                        }
                        .toggleStyle(.button)
                        .help("Play as a Camera")
                    }
                }
            }
            .disabled(!state.idle)
        }
    }
}

struct RecordingSection: View {
    @Environment(Engine.self) private var engine
    var session: SessionState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "record.circle.fill")
                    .foregroundStyle(.red)
                    .symbolEffect(.pulse)
                Text(session.label).font(.headline)
                Spacer()
                Text(Duration.seconds(session.elapsed), format: .time(pattern: .minuteSecond))
                    .font(.title3.monospacedDigit())
            }
            if let fraction = session.stageFraction {
                ProgressView(value: fraction) {
                    Text("\(Int(session.remaining.rounded(.up))) s")
                        .font(.callout.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            HStack {
                Spacer()
                if session.waiting {
                    Button("Continue") { engine.send("continue_stage") }
                        .prominentButton()
                }
                Button("Stop") { engine.send("stop_recording") }
                    .glassButton()
            }
            .controlSize(.large)
        }
    }
}

/// An optional name for the next recording's folder.
struct RecordingNameField: View {
    @Environment(Engine.self) private var engine
    @State private var name = ""

    var body: some View {
        TextField("Name", text: $name, prompt: Text("Recording Name"))
            .textFieldStyle(.roundedBorder)
            .frame(width: 180)
            .onAppear { name = engine.state.recordingName }
            .onChange(of: name) { _, new in engine.send("set_recording_name", ["name": new]) }
            .help("Names the recording folder")
    }
}

/// The view a camera records as: one of the camera model's cameras, or a name of its own.
struct ViewName: View {
    @Environment(Engine.self) private var engine
    var index: Int
    var view: String
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        let cameras = engine.state.cameraNames
        if cameras.isEmpty {
            TextField("Name", text: $text, prompt: Text("Camera"))
                .textFieldStyle(.plain)
                .fixedSize()
                .focused($focused)
                .onSubmit(commit)
                .onChange(of: focused) { _, now in if !now { commit() } }
                .onAppear { text = view }
                .onChange(of: view) { _, new in text = new }
                .help("Camera Name")
        } else {
            Menu {
                Picker("View", selection: Binding(get: { view }, set: { set($0) })) {
                    ForEach(cameras, id: \.self) { Text($0).tag($0) }
                    if !cameras.contains(view) {
                        Text(view).tag(view)
                    }
                }
                .pickerStyle(.inline)
                .labelsHidden()
            } label: {
                Text(view)
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .help("Camera in the Camera Model")
        }
    }

    private func commit() {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        if trimmed != view { set(trimmed) }
    }

    private func set(_ name: String) {
        engine.send("set_view", ["index": index, "name": name])
    }
}
