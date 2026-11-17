import SwiftUI

/// A save-location popup in the style of macOS: the folder in use, a default choice, and Other….
struct FolderMenu: View {
    /// The folder in use; `nil` when the default has no folder yet.
    var path: String?
    /// The default (`true`) or a chosen folder.
    var isDefault: Bool
    /// The default choice, e.g. “Next to Video”.
    var defaultTitle: String
    var choose: (URL?) -> Void

    var body: some View {
        Menu {
            Picker(
                "",
                selection: Binding(
                    get: { isDefault },
                    set: { if $0 { choose(nil) } })
            ) {
                Text(defaultTitle).tag(true)
                if !isDefault, let path {
                    Text(path.fileName).tag(false)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
            Divider()
            Button("Other…") {
                if let url = Files.chooseFolder(title: "Choose a Folder", at: path) {
                    choose(url)
                }
            }
            if let path {
                Divider()
                Button("Show in Finder") { Files.reveal(path) }
            }
        } label: {
            Label(title, systemImage: "folder")
        }
        .fixedSize()
        .help(path ?? defaultTitle)
    }

    private var title: String {
        guard let path else { return defaultTitle }
        return path.fileName
    }
}
