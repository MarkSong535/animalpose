//! The egui window over [`crate::app::Engine`]: the desktop application on Windows and Linux (macOS
//! has a native app in `macos/`). Two pages, Live (cameras, preview, recording) and Process (the
//! video list, output folder and batch progress), and a Settings window on ⌘, / Ctrl+,.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Poll, Wake, Waker};
use std::time::Duration;

use anyhow::Result;
use eframe::egui;
use egui::{Color32, RichText};

use crate::app::library::{CONFIG_EXTENSIONS, MODEL_EXTENSIONS, MODEL_KINDS};
use crate::app::processing::Mode as ProcessingMode;
use crate::app::recording::{self, Phase};
pub use crate::app::{analyze, list_cameras, Source};
use crate::app::{display_name, Engine, ImportKind};

/// The system accent blue.
const ACCENT: Color32 = Color32::from_rgb(10, 132, 255);
const SETTINGS: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Comma);

#[derive(Clone, Copy)]
enum Selection {
    RecordingFolder,
    OutputFolder,
    Videos,
    VideoFolders,
    Calibration,
    Model,
    Config,
    ImportFolder,
}

type Picked = Pin<Box<dyn Future<Output = Vec<PathBuf>>>>;

struct Picker {
    selection: Selection,
    result: Picked,
}

fn picked_one(pick: impl Future<Output = Option<rfd::FileHandle>> + 'static) -> Picked {
    Box::pin(async move {
        pick.await
            .map(|file| vec![file.path().to_path_buf()])
            .unwrap_or_default()
    })
}

fn picked_many(pick: impl Future<Output = Option<Vec<rfd::FileHandle>>> + 'static) -> Picked {
    Box::pin(async move {
        pick.await
            .unwrap_or_default()
            .iter()
            .map(|file| file.path().to_path_buf())
            .collect()
    })
}

struct Repaint(egui::Context);

impl Wake for Repaint {
    fn wake(self: Arc<Self>) {
        self.0.request_repaint();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Live,
    Process,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    General,
    Pipeline,
    Protocol,
}

pub struct App {
    engine: Engine,
    textures: Vec<Option<((u64, bool), egui::TextureHandle)>>,
    picker: Option<Picker>,
    page: Page,
    settings_open: bool,
    settings_tab: SettingsTab,
    /// A typed video or folder path, added to the list with Add.
    video_entry: String,
}

impl App {
    fn new(ctx: &egui::Context) -> Self {
        style(ctx);
        let repaint = ctx.clone();
        Self {
            engine: Engine::new(move || repaint.request_repaint()),
            textures: Vec::new(),
            picker: None,
            page: Page::Process,
            settings_open: false,
            settings_tab: SettingsTab::General,
            video_entry: String::new(),
        }
    }

    fn busy(&self) -> bool {
        self.engine.busy() || self.picker.is_some()
    }

    fn choose(&mut self, selection: Selection, frame: &eframe::Frame) {
        if self.picker.is_some() || self.engine.importing() {
            return;
        }
        // A parent window lets rfd open a sheet on macOS without a nested event loop.
        let dialog = rfd::AsyncFileDialog::new().set_parent(frame);
        let result = match selection {
            Selection::RecordingFolder => picked_one(
                dialog
                    .set_title("Recording folder")
                    .set_directory(self.engine.recording_dir())
                    .pick_folder(),
            ),
            Selection::OutputFolder => {
                let dialog = dialog.set_title("Save results in");
                let dialog = match self.engine.output_dir() {
                    Some(dir) if dir.is_dir() => dialog.set_directory(dir),
                    _ => dialog,
                };
                picked_one(dialog.pick_folder())
            }
            Selection::Videos => {
                let exts = crate::data::video::VIDEO_EXT.map(|e| &e[1..]);
                picked_many(
                    dialog
                        .set_title("Add videos")
                        .add_filter("Video", &exts)
                        .pick_files(),
                )
            }
            Selection::VideoFolders => {
                picked_many(dialog.set_title("Add folders of videos").pick_folders())
            }
            Selection::Calibration => picked_one(
                dialog
                    .add_filter("Camera calibration", &["json"])
                    .pick_file(),
            ),
            Selection::Model => picked_one(
                dialog
                    .add_filter("Checkpoint", &MODEL_EXTENSIONS)
                    .pick_file(),
            ),
            Selection::Config => picked_one(
                dialog
                    .add_filter("Pipeline config", &CONFIG_EXTENSIONS)
                    .pick_file(),
            ),
            Selection::ImportFolder => picked_one(dialog.set_title("Import folder").pick_folder()),
        };
        self.picker = Some(Picker { selection, result });
    }

    fn poll_picker(&mut self, ctx: &egui::Context) {
        let Some(picker) = self.picker.as_mut() else {
            return;
        };
        let waker = Waker::from(Arc::new(Repaint(ctx.clone())));
        let Poll::Ready(paths) = picker
            .result
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker))
        else {
            return;
        };
        let selection = picker.selection;
        self.picker = None;
        ctx.request_repaint();
        if let Selection::Videos | Selection::VideoFolders = selection {
            if !paths.is_empty() {
                self.engine.add_videos(paths);
            }
            return;
        }
        let Some(path) = paths.into_iter().next() else {
            return;
        };
        match selection {
            Selection::RecordingFolder => {
                self.engine.settings.recordings = Some(path);
                self.engine.save();
            }
            Selection::OutputFolder => {
                self.engine.settings.output = Some(path);
                self.engine.save();
            }
            Selection::Calibration => self.engine.cameras_json = Some(path),
            Selection::Model => self.engine.import(ImportKind::Model, path),
            Selection::Config => self.engine.import(ImportKind::Config, path),
            Selection::ImportFolder => self.engine.import(ImportKind::Folder, path),
            Selection::Videos | Selection::VideoFolders => unreachable!(),
        }
    }

    fn toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("toolbar")
            .frame(
                egui::Frame::side_top_panel(&ctx.style())
                    .inner_margin(egui::Margin::symmetric(16.0, 10.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("AnimalPose").size(17.0).strong());
                    ui.add_space(16.0);
                    for (page, label) in [(Page::Live, "Live"), (Page::Process, "Process")] {
                        let selected = self.page == page;
                        let text = RichText::new(label).size(14.0);
                        if ui.add(egui::SelectableLabel::new(selected, text)).clicked() {
                            self.page = page;
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let hint = format!("Settings ({})", ctx.format_shortcut(&SETTINGS));
                        if ui.button("⚙ Settings").on_hover_text(hint).clicked() {
                            self.settings_open = true;
                        }
                        if let Some(job) = &self.engine.processing {
                            let snapshot = job.control.snapshot();
                            ui.add(
                                egui::ProgressBar::new(snapshot.fraction())
                                    .desired_width(140.0)
                                    .show_percentage(),
                            );
                        }
                    });
                });
            });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        if self.engine.status.is_empty() {
            return;
        }
        egui::TopBottomPanel::bottom("status")
            .frame(
                egui::Frame::side_top_panel(&ctx.style())
                    .inner_margin(egui::Margin::symmetric(16.0, 8.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.small_button("🗙").on_hover_text("Dismiss").clicked() {
                        self.engine.status.clear();
                    }
                    ui.label(&self.engine.status);
                });
            });
    }

    fn live_page(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let session = self.engine.session();
        let idle = self.engine.idle() && self.picker.is_none();
        egui::SidePanel::left("live")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(8.0);
                    card(ui, "Sources", |ui| {
                        ui.add_enabled_ui(idle, |ui| {
                            for (source, on) in
                                self.engine.sources.iter().zip(&mut self.engine.selected)
                            {
                                ui.checkbox(on, source.name());
                            }
                            if self.engine.sources.is_empty() {
                                ui.label(RichText::new("No cameras found").weak());
                            }
                            ui.horizontal(|ui| {
                                if ui.button("Refresh cameras").clicked() {
                                    self.engine.refresh_cameras();
                                }
                            });
                            ui.add_enabled(
                                !self.engine.videos.is_empty(),
                                egui::Checkbox::new(
                                    &mut self.engine.playback,
                                    "Play the first listed video as a source",
                                ),
                            );
                        });
                        ui.add_enabled_ui(session.is_none() && !self.busy(), |ui| {
                            let mut live = self.engine.settings.live;
                            if ui.checkbox(&mut live, "Live pose preview").changed() {
                                self.engine.set_live(live);
                            }
                        });
                    });
                    card(ui, "Recording", |ui| self.recording_controls(ui, frame));
                });
            });
        egui::CentralPanel::default().show(ctx, |ui| self.preview(ui, session.is_some()));
    }

    fn recording_controls(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        if let Some(session) = self.engine.session() {
            let (phase, elapsed, remaining) = session.position();
            ui.label(
                RichText::new(format!("● {}", session.label(phase)))
                    .color(Color32::from_rgb(255, 69, 58))
                    .strong(),
            );
            if let Phase::Timed(i) = phase {
                let seconds = f64::from(session.stages[i].seconds);
                ui.add(
                    egui::ProgressBar::new((1.0 - remaining.as_secs_f64() / seconds) as f32)
                        .text(format!("{:.1} s remaining", remaining.as_secs_f64())),
                );
            }
            ui.label(format!("{:.1} s recorded", elapsed.as_secs_f64()));
            ui.label(RichText::new("Raw preview during recording").weak());
            ui.horizontal_wrapped(|ui| {
                if matches!(phase, Phase::Waiting(_))
                    && ui.button("Continue to next stage").clicked()
                {
                    self.engine.continue_stage();
                }
                if ui.button("Stop and save").clicked() {
                    self.engine.stop_recording();
                }
            });
            return;
        }
        let busy = self.busy();
        let running = !self.engine.workers.is_empty();
        let ready = self.engine.sources_ready();
        ui.add_enabled_ui(!busy, |ui| {
            path_row(
                ui,
                "Saved in",
                &self.engine.recording_dir(),
                self.engine.settings.recordings.is_none(),
            );
            if ui.button("Change…").clicked() {
                self.choose(Selection::RecordingFolder, frame);
            }
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                if running {
                    if ui.button("Stop preview").clicked() {
                        self.engine.stop_sources(false);
                    }
                } else if ui.add(primary("Start preview")).clicked() {
                    self.engine.start_preview();
                }
                if ui
                    .add_enabled(ready, egui::Button::new("● Record"))
                    .clicked()
                {
                    self.engine.start_recording(false);
                }
                if ui
                    .add_enabled(
                        ready && !self.engine.settings.stages.is_empty(),
                        egui::Button::new("Timed recording"),
                    )
                    .on_hover_text("Runs the protocol stages from Settings")
                    .clicked()
                {
                    self.engine.start_recording(true);
                }
            });
        });
    }

    fn preview(&mut self, ui: &mut egui::Ui, recording: bool) {
        let count = self.engine.workers.len();
        self.textures.resize_with(count, || None);
        if count == 0 {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("Select a camera and press Start preview")
                        .size(16.0)
                        .weak(),
                );
            });
            return;
        }
        let columns = (count as f32).sqrt().ceil() as usize;
        let spacing = ui.spacing().item_spacing.x;
        let width = (ui.available_width() - spacing * (columns - 1) as f32) / columns as f32;
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("previews")
                .num_columns(columns)
                .spacing([spacing, spacing])
                .show(ui, |ui| {
                    for index in 0..count {
                        ui.vertical(|ui| {
                            ui.set_width(width);
                            self.tile(ui, index, width, recording);
                        });
                        if (index + 1) % columns == 0 {
                            ui.end_row();
                        }
                    }
                });
        });
    }

    fn tile(&mut self, ui: &mut egui::Ui, index: usize, width: f32, recording: bool) {
        let worker = &self.engine.workers[index];
        ui.horizontal(|ui| {
            ui.label(RichText::new(worker.source.name()).strong());
            if recording {
                ui.label(RichText::new("● REC").color(Color32::from_rgb(255, 69, 58)));
            }
        });
        if let Some(error) = worker.capture_error.lock().unwrap().as_deref() {
            ui.colored_label(Color32::LIGHT_RED, error);
        }
        if !recording && self.engine.settings.live {
            if let Some(error) = worker.preview_error.lock().unwrap().as_deref() {
                ui.label(RichText::new(error).weak());
            }
        }
        let ended = worker.ended.load(std::sync::atomic::Ordering::Acquire);
        let known = self.textures[index].as_ref().map(|(key, _)| *key);
        let preview = self.engine.preview(index, known);
        if let Some(preview) = &preview {
            if let Some(rgba) = &preview.rgba {
                let image = egui::ColorImage::from_rgba_unmultiplied(preview.size, rgba);
                match &mut self.textures[index] {
                    Some((key, texture)) => {
                        texture.set(image, egui::TextureOptions::LINEAR);
                        *key = preview.key;
                    }
                    slot => {
                        let texture = ui.ctx().load_texture(
                            format!("source-{index}"),
                            image,
                            egui::TextureOptions::LINEAR,
                        );
                        *slot = Some((preview.key, texture));
                    }
                }
            }
        }
        if let Some((_, texture)) = &self.textures[index] {
            let size = texture.size_vec2();
            ui.add(egui::Image::new((texture.id(), size * (width / size.x))).rounding(10.0));
            if let Some((animals, ms)) = preview.and_then(|p| p.analysis) {
                ui.label(RichText::new(format!("{animals} animals · {ms:.0} ms")).weak());
            }
        } else if !ended {
            ui.label(RichText::new("Waiting for frames…").weak());
        }
    }

    fn process_page(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.set_max_width(760.0);
                ui.add_space(8.0);
                if self.engine.processing.is_some() {
                    card(ui, "Progress", |ui| self.processing_controls(ui));
                }
                let idle = self.engine.idle() && self.picker.is_none();
                ui.add_enabled_ui(idle, |ui| {
                    card(ui, "Videos", |ui| self.video_list(ui, frame));
                    card(ui, "Output", |ui| self.output_controls(ui, frame));
                    card(ui, "Pipeline", |ui| self.pipeline_summary(ui));
                    ui.horizontal(|ui| {
                        let process = match self.engine.videos.len() {
                            0 | 1 => "Process video".to_owned(),
                            n => format!("Process {n} videos"),
                        };
                        if ui
                            .add_enabled(!self.engine.videos.is_empty(), primary(&process))
                            .clicked()
                        {
                            self.engine.process_videos();
                            ui.ctx().request_repaint();
                        }
                        if ui
                            .add_enabled(
                                !self.engine.last_recording.is_empty(),
                                egui::Button::new("Analyze last recording"),
                            )
                            .on_hover_text("Saves an analysis folder in each recorded stage")
                            .clicked()
                        {
                            self.engine.analyze_last();
                            ui.ctx().request_repaint();
                        }
                    });
                });
            });
        });
    }

    fn processing_controls(&self, ui: &mut egui::Ui) {
        let Some(job) = &self.engine.processing else {
            return;
        };
        let snapshot = job.control.snapshot();
        if snapshot.total() > 1 {
            // One bar for the videos, one for the frames of the video in progress.
            ui.add(egui::ProgressBar::new(snapshot.fraction()).text(snapshot.video_label()));
            let current = job.control.current();
            if !current.is_empty() {
                ui.label(RichText::new(current).strong());
            }
            ui.add(egui::ProgressBar::new(snapshot.clip_fraction()).text(snapshot.stage_label()));
        } else {
            ui.add(egui::ProgressBar::new(snapshot.fraction()).show_percentage());
            ui.label(snapshot.label());
        }
        match snapshot.mode {
            ProcessingMode::Running => {}
            ProcessingMode::Pausing => {
                ui.label("Pausing after the current operation…");
            }
            ProcessingMode::Paused => {
                ui.label("Paused");
            }
            ProcessingMode::Stopping => {
                ui.label("Stopping after the current operation…");
            }
        }
        ui.add_enabled_ui(snapshot.mode != ProcessingMode::Stopping, |ui| {
            ui.horizontal(|ui| {
                let pause = if snapshot.mode == ProcessingMode::Running {
                    "Pause"
                } else {
                    "Resume"
                };
                if ui.button(pause).clicked() {
                    self.engine.pause_or_resume();
                }
                if ui.button("Stop").clicked() {
                    self.engine.stop_processing();
                }
            });
        });
        if snapshot.mode != ProcessingMode::Paused {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
    }

    /// The add buttons, a typed path field and the listed videos, each removable.
    fn video_list(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Add videos…").clicked() {
                self.choose(Selection::Videos, frame);
            }
            if ui.button("Add folders…").clicked() {
                self.choose(Selection::VideoFolders, frame);
            }
            if ui
                .add_enabled(!self.engine.videos.is_empty(), egui::Button::new("Clear"))
                .clicked()
            {
                self.engine.videos.clear();
            }
        });
        let mut submit = false;
        ui.horizontal(|ui| {
            let field = ui.add(
                egui::TextEdit::singleline(&mut self.video_entry)
                    .hint_text("Or type a video or folder path")
                    .desired_width((ui.available_width() - 48.0).max(80.0)),
            );
            submit |= field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            submit |= ui
                .add_enabled(
                    !self.video_entry.trim().is_empty(),
                    egui::Button::new("Add"),
                )
                .clicked();
        });
        if submit && !self.video_entry.trim().is_empty() {
            let path = PathBuf::from(self.video_entry.trim());
            self.video_entry.clear();
            self.engine.add_videos([path]);
        }
        if self.engine.videos.is_empty() {
            ui.label(RichText::new("No videos yet. Folders are searched recursively.").weak());
            return;
        }
        let mut remove = None;
        egui::ScrollArea::vertical()
            .id_salt("video-list")
            .max_height(220.0)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                for (index, video) in self.engine.videos.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("🗙")
                            .on_hover_text("Remove from the list")
                            .clicked()
                        {
                            remove = Some(index);
                        }
                        ui.label(display_name(video))
                            .on_hover_text(video.display().to_string());
                    });
                }
            });
        if let Some(index) = remove {
            self.engine.remove_video(index);
        }
    }

    fn output_controls(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        match self.engine.output_dir() {
            Some(dir) => path_row(
                ui,
                "Results go to",
                &dir,
                self.engine.settings.output.is_none(),
            ),
            None => {
                ui.label(RichText::new("Next to the first listed video").weak());
            }
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("Choose folder…").clicked() {
                self.choose(Selection::OutputFolder, frame);
            }
            if ui
                .add_enabled(
                    self.engine.settings.output.is_some(),
                    egui::Button::new("Next to first video"),
                )
                .clicked()
            {
                self.engine.settings.output = None;
                self.engine.save();
            }
            if let Some(dir) = self.engine.output_dir() {
                if ui.button("Show").clicked() {
                    reveal(&dir);
                }
            }
        });
        ui.label(
            RichText::new(
                "One video: a <name>-analysis folder. Several: one animalpose-batch folder.",
            )
            .weak()
            .small(),
        );
    }

    fn pipeline_summary(&mut self, ui: &mut egui::Ui) {
        let settings = &self.engine.settings;
        egui::Grid::new("pipeline-summary")
            .num_columns(2)
            .spacing([16.0, 4.0])
            .show(ui, |ui| {
                ui.label(RichText::new("Config").weak());
                ui.label(settings.config.as_deref().unwrap_or("Default pipeline"));
                ui.end_row();
                for kind in MODEL_KINDS {
                    let key = kind.as_str();
                    ui.label(RichText::new(capitalized(key)).weak());
                    let off =
                        kind == crate::models::ModelKind::Behavior && !settings.classify_behavior;
                    ui.label(if off {
                        "Off"
                    } else {
                        settings
                            .models
                            .get(key)
                            .map_or("From config", String::as_str)
                    });
                    ui.end_row();
                }
            });
        if ui.link("Change in Settings…").clicked() {
            self.settings_open = true;
            self.settings_tab = SettingsTab::Pipeline;
        }
    }

    fn settings_window(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let mut open = self.settings_open;
        egui::Window::new("Settings")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(520.0)
            .anchor(egui::Align2::CENTER_TOP, [0.0, 72.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    for (tab, label) in [
                        (SettingsTab::General, "General"),
                        (SettingsTab::Pipeline, "Models & config"),
                        (SettingsTab::Protocol, "Protocol"),
                    ] {
                        ui.selectable_value(&mut self.settings_tab, tab, label);
                    }
                });
                ui.separator();
                let enabled = self.engine.idle() && self.picker.is_none();
                ui.add_enabled_ui(enabled, |ui| match self.settings_tab {
                    SettingsTab::General => self.general_settings(ui, frame),
                    SettingsTab::Pipeline => self.library_controls(ui, frame),
                    SettingsTab::Protocol => self.timer_editor(ui),
                });
                if !enabled {
                    ui.label(RichText::new("Settings are locked while a job runs.").weak());
                }
            });
        self.settings_open = open;
    }

    fn general_settings(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.label(RichText::new("Processed results").strong());
        let custom = self.engine.settings.output.is_some();
        if ui.radio(!custom, "Next to the first video").clicked() && custom {
            self.engine.settings.output = None;
            self.engine.save();
        }
        ui.horizontal(|ui| {
            if ui.radio(custom, "In a folder:").clicked() || ui.button("Choose…").clicked() {
                self.choose(Selection::OutputFolder, frame);
            }
        });
        if let Some(dir) = &self.engine.settings.output {
            ui.label(RichText::new(dir.display().to_string()).monospace().small());
        }
        ui.add_space(8.0);
        ui.label(RichText::new("Recordings").strong());
        ui.label(
            RichText::new(self.engine.recording_dir().display().to_string())
                .monospace()
                .small(),
        );
        ui.horizontal(|ui| {
            if ui.button("Choose…").clicked() {
                self.choose(Selection::RecordingFolder, frame);
            }
            if ui
                .add_enabled(
                    self.engine.settings.recordings.is_some(),
                    egui::Button::new("Use default"),
                )
                .clicked()
            {
                self.engine.settings.recordings = None;
                self.engine.save();
            }
        });
        ui.add_space(8.0);
        ui.label(RichText::new("Library").strong());
        ui.label(
            RichText::new(self.engine.library.root.display().to_string())
                .monospace()
                .small(),
        );
        if ui.button("Show in file browser").clicked() {
            reveal(&self.engine.library.root);
        }
    }

    fn library_controls(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Import model…").clicked() {
                self.choose(Selection::Model, frame);
            }
            if ui.button("Import config…").clicked() {
                self.choose(Selection::Config, frame);
            }
            if ui.button("Import folder…").clicked() {
                self.choose(Selection::ImportFolder, frame);
            }
        });
        ui.add_space(6.0);
        let engine = &mut self.engine;
        let before = engine.settings.config.clone();
        let configs = engine.library.configs();
        let mut changed = false;
        egui::Grid::new("models")
            .num_columns(2)
            .spacing([16.0, 8.0])
            .show(ui, |ui| {
                ui.label("Pipeline config");
                egui::ComboBox::from_id_salt("config")
                    .width(260.0)
                    .selected_text(before.as_deref().unwrap_or("Default pipeline"))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut engine.settings.config, None, "Default pipeline");
                        for name in configs {
                            ui.selectable_value(
                                &mut engine.settings.config,
                                Some(name.clone()),
                                name,
                            );
                        }
                    });
                ui.end_row();
                changed |= before != engine.settings.config;
                for kind in MODEL_KINDS {
                    let key = kind.as_str();
                    let prior = engine.settings.models.get(key).cloned();
                    let mut chosen = prior.clone();
                    let enabled = kind != crate::models::ModelKind::Behavior
                        || engine.settings.classify_behavior;
                    ui.label(capitalized(key));
                    ui.add_enabled_ui(enabled, |ui| {
                        egui::ComboBox::from_id_salt(key)
                            .width(260.0)
                            .selected_text(prior.as_deref().unwrap_or("From config"))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut chosen, None, "From config");
                                for name in engine.library.models(kind) {
                                    ui.selectable_value(&mut chosen, Some(name.clone()), name);
                                }
                            });
                    });
                    ui.end_row();
                    if chosen != prior {
                        changed = true;
                        if let Some(name) = chosen {
                            engine.settings.models.insert(key.to_owned(), name);
                        } else {
                            engine.settings.models.remove(key);
                        }
                    }
                }
            });
        changed |= ui
            .checkbox(
                &mut engine.settings.classify_behavior,
                "Classify behavior if available",
            )
            .changed();
        if changed {
            engine.save();
        }
        let mut live = engine.settings.live;
        if ui.checkbox(&mut live, "Live pose preview").changed() {
            engine.set_live(live);
        }
        ui.add_space(8.0);
        ui.label(RichText::new("3D calibration").strong());
        match &self.engine.cameras_json {
            Some(path) => ui.label(RichText::new(path.display().to_string()).small()),
            None => ui.label(RichText::new("None: views are analyzed separately").weak()),
        };
        ui.horizontal(|ui| {
            if ui.button("Choose calibration…").clicked() {
                self.choose(Selection::Calibration, frame);
            }
            if ui
                .add_enabled(
                    self.engine.cameras_json.is_some(),
                    egui::Button::new("Clear"),
                )
                .clicked()
            {
                self.engine.cameras_json = None;
            }
        });
    }

    fn timer_editor(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        let mut remove = None;
        let settings = &mut self.engine.settings;
        let count = settings.stages.len();
        for (index, stage) in settings.stages.iter_mut().enumerate() {
            ui.push_id(index, |ui| {
                ui.horizontal(|ui| {
                    changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut stage.name)
                                .desired_width(160.0)
                                .hint_text("Stage name"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut stage.seconds)
                                .range(1..=u32::MAX)
                                .suffix(" s"),
                        )
                        .changed();
                    if index + 1 < count {
                        changed |= ui
                            .checkbox(&mut stage.wait_after, "Wait for Continue")
                            .changed();
                    }
                    if ui.small_button("🗙").on_hover_text("Remove stage").clicked() {
                        remove = Some(index);
                    }
                });
            });
        }
        if let Some(index) = remove {
            settings.stages.remove(index);
            changed = true;
        }
        if ui.button("Add stage").clicked() {
            settings.stages.push(recording::Stage {
                name: "Experiment".into(),
                seconds: 300,
                wait_after: false,
            });
            changed = true;
        }
        if changed {
            self.engine.save();
        }
    }
}

/// A rounded, filled section with a heading.
fn card(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::none()
        .fill(ui.visuals().faint_bg_color)
        .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
        .rounding(12.0)
        .inner_margin(egui::Margin::same(14.0))
        .outer_margin(egui::Margin::symmetric(4.0, 6.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new(title).size(15.0).strong());
            ui.add_space(4.0);
            add(ui);
        });
}

/// A prominent, accent-filled button.
fn primary(text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text).color(Color32::WHITE).strong())
        .fill(ACCENT)
        .min_size(egui::vec2(0.0, 28.0))
}

fn path_row(ui: &mut egui::Ui, label: &str, path: &Path, default: bool) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(label).weak());
        ui.label(RichText::new(path.display().to_string()).monospace());
        if default {
            ui.label(RichText::new("(default)").weak().small());
        }
    });
}

fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

/// Opens `path` (or the nearest folder that exists) in the system file browser.
fn reveal(path: &Path) {
    let Some(existing) = path.ancestors().find(|p| p.exists()) else {
        return;
    };
    let program = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(program).arg(existing).spawn();
}

/// Follows the system light or dark appearance, with rounder widgets, roomier spacing and the
/// system accent color.
fn style(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::System);
    ctx.all_styles_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(12.0, 5.0);
        style.spacing.interact_size.y = 24.0;
        let visuals = &mut style.visuals;
        visuals.selection.bg_fill = ACCENT;
        visuals.selection.stroke.color = Color32::WHITE;
        visuals.hyperlink_color = ACCENT;
        visuals.window_rounding = egui::Rounding::same(14.0);
        visuals.menu_rounding = egui::Rounding::same(10.0);
        for widgets in [
            &mut visuals.widgets.noninteractive,
            &mut visuals.widgets.inactive,
            &mut visuals.widgets.hovered,
            &mut visuals.widgets.active,
            &mut visuals.widgets.open,
        ] {
            widgets.rounding = egui::Rounding::same(7.0);
        }
    });
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.engine.poll();
        if ctx.input_mut(|i| i.consume_shortcut(&SETTINGS)) {
            self.settings_open = !self.settings_open;
        }
        self.toolbar(ctx);
        self.status_bar(ctx);
        match self.page {
            Page::Live => self.live_page(ctx, frame),
            Page::Process => self.process_page(ctx, frame),
        }
        if self.settings_open {
            self.settings_window(ctx, frame);
        }
        if self.engine.finishing() || !self.engine.workers.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(33));
        } else if self.engine.importing() || self.engine.processing.is_some() {
            // Also check for a worker that could not send its repaint signal.
            ctx.request_repaint_after(Duration::from_millis(200));
        }
        self.poll_picker(ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.engine.shutdown();
    }
}

/// Opens the window.
pub fn run() -> Result<()> {
    // macOS asks the user for camera access once; the answer arrives asynchronously.
    #[cfg(target_os = "macos")]
    nokhwa::nokhwa_initialize(|_| {});
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AnimalPose")
            .with_inner_size([1200.0, 780.0])
            .with_min_inner_size([760.0, 520.0]),
        ..Default::default()
    };
    eframe::run_native(
        "animalpose",
        options,
        Box::new(|cc| Ok(Box::new(App::new(&cc.egui_ctx)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}
