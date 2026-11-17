import SwiftUI

struct ContentView: View {
    @Environment(Engine.self) private var engine
    @Binding var page: Page

    var body: some View {
        NavigationSplitView {
            Sidebar(page: $page)
                .navigationSplitViewColumnWidth(min: 200, ideal: 230, max: 300)
        } detail: {
            Group {
                switch page {
                case .process: ProcessView()
                case .live: LiveView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .overlay(alignment: .top) { StatusBanner().animation(.smooth, value: engine.state.status) }
        }
        .alert(
            "Something went wrong",
            isPresented: Binding(
                get: { engine.error != nil }, set: { if !$0 { engine.error = nil } })
        ) {
            Button("OK") { engine.error = nil }
        } message: {
            Text(engine.error ?? "")
        }
    }
}

struct Sidebar: View {
    @Environment(Engine.self) private var engine
    @Binding var page: Page

    var body: some View {
        List(selection: Binding(get: { page }, set: { if let next = $0 { page = next } })) {
            Section {
                ForEach(Page.allCases) { item in
                    Label(item.title, systemImage: item.symbol)
                        .badge(badge(for: item))
                        .tag(item)
                }
            }
            Section {
                ProfileRow()
            }
            if !engine.state.settings.recentResults.isEmpty {
                Section("Results") {
                    ForEach(engine.state.settings.recentResults.prefix(5), id: \.self) { path in
                        ResultRow(path: path)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom) {
            if let processing = engine.state.processing {
                VStack(alignment: .leading, spacing: 6) {
                    Text(processing.total > 1 ? processing.videoLabel : "Processing…")
                        .font(.caption.weight(.semibold))
                    ProgressView(value: processing.fraction)
                }
                .padding(12)
                .glassPanel(cornerRadius: 14)
                .padding(10)
            }
        }
    }

    private func badge(for item: Page) -> Int {
        switch item {
        case .process: engine.state.videos.count
        case .live: engine.state.workers.count
        }
    }
}

/// One sidebar line: "Profile", and the profile in use as a menu that switches it.
struct ProfileRow: View {
    var body: some View {
        HStack(spacing: 8) {
            Label("Profile", systemImage: "person.crop.rectangle.stack")
                .layoutPriority(1)
            Spacer(minLength: 4)
            ProfileMenu()
        }
    }
}

/// Switches the profile in use: its config, models and stages.
struct ProfileMenu: View {
    @Environment(Engine.self) private var engine

    var body: some View {
        let state = engine.state
        Menu {
            Picker(
                "Profile",
                selection: Binding(
                    get: { state.profile ?? "" },
                    set: { engine.send("select_profile", ["name": $0]) })
            ) {
                ForEach(state.profiles) { Text($0.name).tag($0.name) }
            }
            .pickerStyle(.inline)
            .labelsHidden()
            Divider()
            SettingsLink { Text("Edit Profiles…") }
        } label: {
            Text(state.profile ?? "Default")
                .lineLimit(1)
                .truncationMode(.tail)
        }
        .menuStyle(.borderlessButton)
        .foregroundStyle(.secondary)
        .disabled(!state.idle)
        .help(state.profile ?? "Default")
    }
}

/// A result folder: opens in Finder; its log is a right-click away.
struct ResultRow: View {
    var path: String

    var body: some View {
        Button {
            openFolder(path)
        } label: {
            Label(path.fileName, systemImage: "folder")
        }
        .buttonStyle(.plain)
        .help(path)
        .contextMenu {
            Button("Open") { openFolder(path) }
            Button("Show in Finder") { Files.reveal(path) }
            let log = (path as NSString).appendingPathComponent("animalpose.log")
            if FileManager.default.fileExists(atPath: log) {
                Button("Open Log") { NSWorkspace.shared.open(URL(fileURLWithPath: log)) }
            }
        }
    }
}

/// `pipeline_lab.yaml` → `pipeline_lab`.
func profileName(_ file: String) -> String {
    (file as NSString).deletingPathExtension
}

/// The engine's latest message, dismissable.
struct StatusBanner: View {
    @Environment(Engine.self) private var engine
    @State private var hovering = false

    /// How long a message stays before it fades.
    private static let shown: Duration = .seconds(5)

    var body: some View {
        let status = engine.state.status
        Group { banner(status) }
            .task(id: status) {
                guard !status.isEmpty else { return }
                // Five seconds on screen; the clock waits while the pointer is on the banner.
                try? await Task.sleep(for: Self.shown)
                while hovering {
                    try? await Task.sleep(for: .milliseconds(300))
                }
                guard !Task.isCancelled, engine.state.status == status else { return }
                withAnimation(.easeOut(duration: 0.4)) { engine.send("clear_status") }
            }
    }

    @ViewBuilder
    private func banner(_ status: String) -> some View {
        if !status.isEmpty {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: icon(status))
                    .foregroundStyle(tint(status))
                Text(status)
                    .font(.callout)
                    .textSelection(.enabled)
                    .lineLimit(6)
                    .frame(maxWidth: 560, alignment: .leading)
                if let result = engine.state.result {
                    Button("Open") { openFolder(result) }
                        .help(result)
                }
                Button {
                    engine.send("clear_status")
                } label: {
                    Label("Dismiss", systemImage: "xmark").labelStyle(.iconOnly)
                }
                .buttonStyle(.borderless)
                .help("Dismiss")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .glassPanel(cornerRadius: 18)
            .onHover { hovering = $0 }
            .padding(.top, 12)
            .transition(.move(edge: .top).combined(with: .opacity))
        }
    }

    private func failed(_ status: String) -> Bool {
        ["fail", "cannot", "could not", "incomplete", "install"].contains {
            status.localizedCaseInsensitiveContains($0)
        }
    }

    private func icon(_ status: String) -> String {
        failed(status) ? "exclamationmark.triangle.fill" : "info.circle.fill"
    }

    private func tint(_ status: String) -> Color { failed(status) ? .orange : .accentColor }
}
