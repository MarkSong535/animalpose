import AppKit
import SwiftUI
import UniformTypeIdentifiers

// Liquid Glass on macOS 26 and later (built with Xcode 26 or newer); translucent materials before.

extension View {
    /// A floating glass panel.
    @ViewBuilder
    func glassPanel(cornerRadius: CGFloat = 22) -> some View {
        #if compiler(>=6.2)
            if #available(macOS 26.0, *) {
                self.glassEffect(.regular, in: .rect(cornerRadius: cornerRadius))
            } else {
                self.background(.regularMaterial, in: .rect(cornerRadius: cornerRadius))
            }
        #else
            self.background(.regularMaterial, in: .rect(cornerRadius: cornerRadius))
        #endif
    }

    /// A small glass capsule, for badges over video.
    @ViewBuilder
    func glassCapsule() -> some View {
        #if compiler(>=6.2)
            if #available(macOS 26.0, *) {
                self.glassEffect(.regular, in: .capsule)
            } else {
                self.background(.ultraThinMaterial, in: .capsule)
            }
        #else
            self.background(.ultraThinMaterial, in: .capsule)
        #endif
    }

    /// The main action of a view.
    @ViewBuilder
    func prominentButton() -> some View {
        #if compiler(>=6.2)
            if #available(macOS 26.0, *) {
                self.buttonStyle(.glassProminent)
            } else {
                self.buttonStyle(.borderedProminent)
            }
        #else
            self.buttonStyle(.borderedProminent)
        #endif
    }

    /// A secondary action on a glass panel.
    @ViewBuilder
    func glassButton() -> some View {
        #if compiler(>=6.2)
            if #available(macOS 26.0, *) {
                self.buttonStyle(.glass)
            } else {
                self.buttonStyle(.bordered)
            }
        #else
            self.buttonStyle(.bordered)
        #endif
    }
}

/// Groups glass shapes so they blend into each other where they meet.
struct GlassGroup<Content: View>: View {
    var spacing: CGFloat = 12
    @ViewBuilder var content: Content

    var body: some View {
        #if compiler(>=6.2)
            if #available(macOS 26.0, *) {
                GlassEffectContainer(spacing: spacing) { content }
            } else {
                content
            }
        #else
            content
        #endif
    }
}

/// Open panels and Finder.
@MainActor
enum Files {
    static let videoTypes: [UTType] = ["mp4", "avi", "mov", "mkv", "mpg", "mpeg", "m4v", "wmv"]
        .compactMap { UTType(filenameExtension: $0) }

    /// Videos and folders of videos, several at once.
    static func chooseVideos() -> [URL] {
        let panel = NSOpenPanel()
        panel.title = "Add Videos or Folders"
        panel.prompt = "Add"
        panel.message = "Choose videos, or folders to search for videos."
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = videoTypes + [.folder]
        return panel.runModal() == .OK ? panel.urls : []
    }

    static func chooseFolder(title: String, prompt: String = "Choose", at start: String? = nil)
        -> URL?
    {
        let panel = NSOpenPanel()
        panel.title = title
        panel.prompt = prompt
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        if let start { panel.directoryURL = URL(fileURLWithPath: start) }
        return panel.runModal() == .OK ? panel.url : nil
    }

    static func chooseFile(title: String, extensions: [String]) -> URL? {
        let panel = NSOpenPanel()
        panel.title = title
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = extensions.compactMap { UTType(filenameExtension: $0) }
        return panel.runModal() == .OK ? panel.url : nil
    }

    /// Shows `path`, or the nearest folder above it that exists, in Finder.
    static func reveal(_ path: String) {
        var url = URL(fileURLWithPath: path)
        while !FileManager.default.fileExists(atPath: url.path), url.pathComponents.count > 1 {
            url.deleteLastPathComponent()
        }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}

/// `~/Movies/day1` for paths in the home folder.
func abbreviated(_ path: String) -> String {
    (path as NSString).abbreviatingWithTildeInPath
}

extension String {
    var fileName: String { (self as NSString).lastPathComponent }
    var parentPath: String { (self as NSString).deletingLastPathComponent }
}
