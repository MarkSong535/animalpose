import SwiftUI

/// The Settings window (⌘,).
struct SettingsView: View {
    @Environment(Engine.self) private var engine

    var body: some View {
        TabView {
            GeneralSettings().tabItem { Label("General", systemImage: "gearshape") }
            ProfileEditor().tabItem { Label("Profiles", systemImage: "person.crop.rectangle.stack") }
            ConfigEditor().tabItem { Label("Configs", systemImage: "slider.horizontal.3") }
        }
        .frame(width: 680, height: 520)
        .disabled(!engine.state.idle)
    }
}

struct GeneralSettings: View {
    @Environment(Engine.self) private var engine

    var body: some View {
        let state = engine.state
        Form {
            Section {
                LabeledContent("Processed Videos") {
                    FolderMenu(
                        path: state.settings.output, isDefault: state.settings.output == nil,
                        defaultTitle: "Next to Video", choose: engine.setOutput)
                }
                LabeledContent("Recordings") {
                    FolderMenu(
                        path: state.recordingsDir, isDefault: state.settings.recordings == nil,
                        defaultTitle: "Default", choose: engine.setRecordings)
                }
            }
            Section {
                LabeledContent("Logs") {
                    Button("Show in Finder") { openFolder(state.logDir) }
                }
                LabeledContent("Library") {
                    HStack {
                        if state.importing { ProgressView().controlSize(.small) }
                        ImportMenu()
                        Button("Show in Finder") { openFolder(state.libraryRoot) }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }
}

struct ImportMenu: View {
    @Environment(Engine.self) private var engine

    var body: some View {
        Menu("Import") {
            Button("Model…") {
                if let url = Files.chooseFile(title: "Import Model", extensions: ["pt", "safetensors"]) {
                    engine.importItem("model", url)
                }
            }
            Button("Config…") {
                if let url = Files.chooseFile(title: "Import Config", extensions: ["yaml", "yml"]) {
                    engine.importItem("config", url)
                }
            }
            Button("Folder…") {
                if let url = Files.chooseFolder(title: "Import Folder", prompt: "Import") {
                    engine.importItem("folder", url)
                }
            }
        }
        .fixedSize()
    }
}

// MARK: Profiles

struct ProfileEditor: View {
    @Environment(Engine.self) private var engine
    @State private var selection: String?

    var body: some View {
        let profiles = engine.state.profiles
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                List(profiles, selection: $selection) { profile in
                    HStack {
                        Text(profile.name)
                        Spacer()
                        if profile.name == engine.state.profile {
                            Image(systemName: "checkmark")
                                .foregroundStyle(.tint)
                                .help("In Use")
                        }
                    }
                    .tag(profile.name)
                    .contextMenu {
                        Button("Duplicate") { duplicate(profile.name) }
                        Divider()
                        Button("Delete", role: .destructive) {
                            engine.send("delete_profile", ["name": profile.name])
                            selection = engine.state.profile
                        }
                        .disabled(profiles.count < 2)
                    }
                }
                .listStyle(.bordered(alternatesRowBackgrounds: false))
                HStack(spacing: 0) {
                    Button {
                        engine.send("new_profile", ["name": "New Profile"])
                        selection = engine.state.profile
                    } label: {
                        Label("New Profile", systemImage: "plus").labelStyle(.iconOnly).frame(width: 24, height: 20)
                    }
                    .help("New Profile")
                    Divider().frame(height: 16)
                    Button {
                        if let selection { engine.send("delete_profile", ["name": selection]) }
                        selection = engine.state.profile
                    } label: {
                        Label("Delete Profile", systemImage: "minus").labelStyle(.iconOnly).frame(width: 24, height: 20)
                    }
                    .help("Delete Profile")
                    .disabled(selection == nil || profiles.count < 2)
                    Divider().frame(height: 16)
                    Menu {
                        Button("Duplicate") {
                            if let selection { duplicate(selection) }
                        }
                        .disabled(selection == nil)
                    } label: {
                        Label("More", systemImage: "ellipsis.circle").labelStyle(.iconOnly).frame(width: 24, height: 20)
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .fixedSize()
                    .help("More")
                    Spacer()
                }
                .buttonStyle(.borderless)
                .padding(4)
            }
            .frame(width: 190)
            .padding([.leading, .vertical], 20)

            if let profile = profiles.first(where: { $0.name == selection }) {
                ProfileForm(profile: profile, renamed: { selection = $0 })
                    .id(profile.name)
            } else {
                ContentUnavailableView("No Profile Selected", systemImage: "person.crop.rectangle")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear { selection = selection ?? engine.state.profile }
    }

    /// Copies a profile and selects the copy; the profile in use stays.
    private func duplicate(_ name: String) {
        let before = Set(engine.state.profiles.map(\.name))
        engine.send("duplicate_profile", ["name": name])
        selection = engine.state.profiles.map(\.name).first { !before.contains($0) } ?? selection
    }
}

/// One profile's name, pipeline and stages; every change is saved as it is made.
struct ProfileForm: View {
    @Environment(Engine.self) private var engine
    let profile: Profile
    var renamed: (String) -> Void
    @State private var name = ""
    @State private var stages: [Stage] = []

    var body: some View {
        let state = engine.state
        Form {
            Section {
                TextField("Name", text: $name)
                    .onSubmit(rename)
                if profile.name != state.profile {
                    LabeledContent("") {
                        Button("Use") { engine.send("select_profile", ["name": profile.name]) }
                    }
                }
            }
            Section("Pipeline") {
                Picker("Config", selection: binding(\.config)) {
                    Text("Default").tag(String?.none)
                    if !state.configs.isEmpty { Divider() }
                    ForEach(state.configs, id: \.self) { Text(profileName($0)).tag(Optional($0)) }
                }
                ForEach(modelKinds, id: \.self) { kind in
                    Picker(kind.capitalized, selection: modelBinding(kind)) {
                        Text("From Config").tag(String?.none)
                        let names = state.models[kind] ?? []
                        if !names.isEmpty { Divider() }
                        ForEach(names, id: \.self) { Text($0).tag(Optional($0)) }
                    }
                    .disabled(kind == "behavior" && !profile.classifyBehavior)
                }
                Toggle("Classify Behavior", isOn: binding(\.classifyBehavior))
            }
            Section("Arena") {
                Picker("Arena Detector", selection: modelBinding("arena")) {
                    Text("From Config").tag(String?.none)
                    let names = state.models["arena"] ?? []
                    if !names.isEmpty { Divider() }
                    ForEach(names, id: \.self) { Text($0).tag(Optional($0)) }
                }
                .help("Finds the arena's floor and zones in each video")
                LabeledContent("Camera Model") {
                    CameraModelMenu(profile: profile, choose: { name in
                        var next = profile
                        next.cameraModel = name
                        engine.saveProfile(next, previous: profile.name)
                    })
                }
                if profile.name == state.profile, profile.cameraModel != nil {
                    if let error = state.cameraModelError {
                        Label(error, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    } else {
                        LabeledContent("Views", value: state.cameraNames.joined(separator: ", "))
                    }
                }
            }
            if !state.sources.isEmpty {
                Section("Camera Names") {
                    ForEach(state.sources, id: \.key) { source in
                        CameraNameField(
                            source: source, name: profile.views[source.key] ?? "",
                            commit: { name in
                                var next = profile
                                next.views[source.key] = name.isEmpty ? nil : name
                                engine.saveProfile(next, previous: profile.name)
                            })
                    }
                }
            }
            Section("Stages") {
                ForEach(Array($stages.enumerated()), id: \.element.id) { index, $stage in
                    HStack(spacing: 10) {
                        TextField("Name", text: $stage.name, prompt: Text("Stage \(index + 1)"))
                            .labelsHidden()
                        TextField("Seconds", value: $stage.seconds, format: .number)
                            .labelsHidden()
                            .multilineTextAlignment(.trailing)
                            .frame(width: 64)
                        Text("s").foregroundStyle(.secondary)
                        if index < stages.count - 1 {
                            Toggle("Wait", isOn: $stage.waitAfter)
                                .help("Wait for Continue")
                        }
                        Button {
                            stages.removeAll { $0.id == stage.id }
                        } label: {
                            Label("Remove Stage", systemImage: "minus.circle.fill").labelStyle(.iconOnly)
                        }
                        .buttonStyle(.borderless)
                        .foregroundStyle(.secondary)
                        .help("Remove Stage")
                    }
                }
                Button {
                    stages.append(Stage(name: "", seconds: 300, waitAfter: false))
                } label: {
                    Label("Add Stage", systemImage: "plus").labelStyle(.iconOnly)
                }
                .buttonStyle(.borderless)
                .help("Add Stage")
            }
        }
        .formStyle(.grouped)
        .onAppear {
            name = profile.name
            stages = profile.stages
        }
        .onChange(of: stages) { _, new in
            guard new != profile.stages else { return }
            var next = profile
            next.stages = new.map { stage in
                var stage = stage
                stage.seconds = max(1, stage.seconds)
                return stage
            }
            engine.saveProfile(next, previous: profile.name)
        }
    }

    private func rename() {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, trimmed != profile.name else {
            name = profile.name
            return
        }
        var next = profile
        next.name = trimmed
        engine.saveProfile(next, previous: profile.name)
        if engine.state.profiles.contains(where: { $0.name == trimmed }) {
            renamed(trimmed)
        } else {
            name = profile.name
        }
    }

    private func binding<T>(_ key: WritableKeyPath<Profile, T>) -> Binding<T> {
        Binding(
            get: { profile[keyPath: key] },
            set: { value in
                var next = profile
                next[keyPath: key] = value
                engine.saveProfile(next, previous: profile.name)
            })
    }

    private func modelBinding(_ kind: String) -> Binding<String?> {
        Binding(
            get: { profile.models[kind] },
            set: { value in
                var next = profile
                next.models[kind] = value
                engine.saveProfile(next, previous: profile.name)
            })
    }
}

// MARK: Configs

struct ConfigEditor: View {
    @Environment(Engine.self) private var engine
    @State private var duplicating = false
    @State private var newName = ""

    var body: some View {
        let state = engine.state
        let form = state.configForm
        VStack(spacing: 0) {
            HStack {
                Picker(
                    "Config",
                    selection: Binding(get: { form?.name }, set: { engine.editConfig($0) })
                ) {
                    Text("Default").tag(String?.none)
                    if !state.configs.isEmpty { Divider() }
                    ForEach(state.configs, id: \.self) { Text(profileName($0)).tag(Optional($0)) }
                }
                .fixedSize()
                Spacer()
                Menu {
                    Button("Duplicate…") { startDuplicate(form?.name) }
                    if let name = form?.name {
                        Divider()
                        Button("Delete", role: .destructive) {
                            engine.send("delete_config", ["name": name])
                        }
                    }
                } label: {
                    Label("Config Actions", systemImage: "ellipsis.circle").labelStyle(.iconOnly)
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .help("Config Actions")
            }
            .padding(.horizontal, 20)
            .padding(.top, 16)

            if let form {
                Form {
                    if !form.editable {
                        Section {
                            HStack {
                                Text("The default config can’t be changed.")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Button("Duplicate…") { startDuplicate(nil) }
                            }
                        }
                    }
                    ForEach(form.sections) { section in
                        Section(section.title) {
                            ForEach(section.fields) { field in
                                FieldRow(field: field)
                                    .disabled(!form.editable)
                            }
                        }
                    }
                }
                .formStyle(.grouped)
                .id(form.name ?? "")
            } else {
                Spacer()
            }
        }
        .onAppear {
            if engine.state.configForm == nil {
                engine.editConfig(engine.activeProfile?.config)
            }
        }
        .alert("Duplicate Config", isPresented: $duplicating) {
            TextField("Name", text: $newName)
            Button("Cancel", role: .cancel) {}
            Button("Duplicate") {
                engine.send(
                    "duplicate_config",
                    ["source": engine.state.configForm?.name ?? NSNull(), "name": newName])
            }
        }
    }

    private func startDuplicate(_ source: String?) {
        newName = source.map { "\(profileName($0)) Copy" } ?? "Pipeline"
        duplicating = true
    }
}

/// One setting: a switch, a slider for fractions, a number with a stepper, or a menu.
struct FieldRow: View {
    @Environment(Engine.self) private var engine
    let field: ConfigField
    @State private var number = 0.0

    var body: some View {
        Group {
            switch field.type {
            case "bool":
                Toggle(field.label, isOn: Binding(get: { field.value.bool }, set: { commit($0) }))
            case "choice":
                Picker(
                    field.label,
                    selection: Binding(get: { field.value.text }, set: { commit($0) })
                ) {
                    ForEach(field.options ?? [], id: \.self) {
                        Text($0.replacingOccurrences(of: "_", with: " ").capitalized).tag($0)
                    }
                }
            case "float" where (field.max ?? 0) <= 1:
                LabeledContent(field.label) {
                    HStack {
                        Slider(value: $number, in: (field.min ?? 0)...(field.max ?? 1)) { editing in
                            if !editing {
                                let step = field.step ?? 0.01
                                number = (number / step).rounded() * step
                                commit(number)
                            }
                        }
                        .frame(width: 180)
                        Text(number, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 40, alignment: .trailing)
                    }
                }
            default:
                LabeledContent(field.label) {
                    HStack(spacing: 6) {
                        TextField(
                            field.label, value: $number,
                            format: .number.precision(
                                .fractionLength(field.type == "float" ? 0...2 : 0...0))
                        )
                        .labelsHidden()
                        .multilineTextAlignment(.trailing)
                        .frame(width: 70)
                        .onSubmit { commit(number) }
                        Stepper(
                            "", value: $number, in: (field.min ?? 0)...(field.max ?? 1_000_000),
                            step: field.step ?? 1
                        ) { editing in
                            if !editing { commit(number) }
                        }
                        .labelsHidden()
                    }
                    .help(field.type == "limit" ? "0 for No Limit" : "")
                }
            }
        }
        .onAppear { number = field.value.number }
        .onChange(of: field.value) { _, new in number = new.number }
    }

    private func commit(_ value: Any) {
        engine.send("set_config_value", ["key": field.key, "value": value])
    }
}

/// The arena's camera model: a calibration of the rig, for Multiview.
struct CameraModelMenu: View {
    @Environment(Engine.self) private var engine
    let profile: Profile
    var choose: (String?) -> Void

    var body: some View {
        Menu {
            Picker("Camera Model", selection: Binding(get: { profile.cameraModel }, set: { choose($0) })) {
                Text("None").tag(String?.none)
                let models = engine.state.cameraModels
                if !models.isEmpty { Divider() }
                ForEach(models, id: \.self) { Text(profileName($0)).tag(Optional($0)) }
            }
            .pickerStyle(.inline)
            .labelsHidden()
            Divider()
            Button("Import…") {
                if let url = Files.chooseFile(title: "Import Camera Model", extensions: ["json"]) {
                    engine.importCameraModel(url, into: profile.name)
                }
            }
        } label: {
            Text(profile.cameraModel.map(profileName) ?? "None")
        }
        .fixedSize()
    }
}

/// One camera's label in recordings; blank keeps its default (`Camera 1`, …).
struct CameraNameField: View {
    let source: CameraSource
    let name: String
    var commit: (String) -> Void
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        TextField(source.name, text: $text, prompt: Text(source.defaultView))
            .focused($focused)
            .onSubmit(save)
            .onChange(of: focused) { _, now in if !now { save() } }
            .onAppear { text = name }
            .onChange(of: name) { _, new in text = new }
    }

    private func save() {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        if trimmed != name { commit(trimmed) }
    }
}
