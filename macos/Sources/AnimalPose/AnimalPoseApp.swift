import AppKit
import SwiftUI

@main
struct AnimalPoseApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
    @State private var engine = Engine()
    @State private var page: Page = .process

    var body: some Scene {
        Window("AnimalPose", id: "main") {
            ContentView(page: $page)
                .environment(engine)
                .frame(minWidth: 820, minHeight: 540)
                .onAppear { delegate.engine = engine }
        }
        .defaultSize(width: 1180, height: 760)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Add Videos or Folders…") {
                    page = .process
                    engine.addVideos(Files.chooseVideos())
                }
                .keyboardShortcut("o")
                .disabled(!engine.state.idle)
            }
            CommandMenu("Analysis") {
                Button("Process Videos") { engine.send("process") }
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!engine.state.idle || engine.state.videos.isEmpty)
                Button("Analyze Last Recording") { engine.send("analyze_last") }
                    .disabled(!engine.state.idle || !engine.state.canAnalyzeLast)
                Divider()
                Button(engine.state.processing?.mode == "running" ? "Pause" : "Resume") {
                    engine.send("pause_or_resume")
                }
                .keyboardShortcut("p", modifiers: [.command, .shift])
                .disabled(engine.state.processing == nil)
                Button("Stop Processing") { engine.send("stop_processing") }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(engine.state.processing == nil)
            }
            CommandGroup(after: .sidebar) {
                Button("Process") { page = .process }.keyboardShortcut("1")
                Button("Live") { page = .live }.keyboardShortcut("2")
                Divider()
            }
        }

        Settings {
            SettingsView()
                .environment(engine)
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var engine: Engine?

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        engine?.shutdown()
    }
}

enum Page: String, CaseIterable, Identifiable {
    case process, live
    var id: Self { self }

    var title: String {
        switch self {
        case .process: "Process"
        case .live: "Live"
        }
    }

    var symbol: String {
        switch self {
        case .process: "film.stack"
        case .live: "video"
        }
    }
}
