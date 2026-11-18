//! The egui window over [`crate::app::Engine`]: the desktop application on Windows and Linux. It
//! follows the native macOS app (`macos/`) control for control: a sidebar with Process, Live, the
//! Profile menu and recent Results; a toolbar; content with a floating glass panel at the bottom; a
//! status banner; and a Settings window (Ctrl+, / ⌘,) with General, Profiles and Configs.
//!
//! egui cannot blur what lies behind a panel, so "glass" is a translucent fill with a light rim and
//! a soft shadow. Switches and hover states animate as on macOS.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Poll, Wake, Waker};
use std::time::Duration;

use anyhow::Result;
use eframe::egui;
use egui::{
    vec2, Align, Align2, Color32, FontId, Id, Layout, Rect, Response, RichText, Rounding, Sense,
    Stroke, Ui,
};
use serde_json::{json, Value};

use crate::app::library::{Profile, CONFIG_EXTENSIONS, MODEL_EXTENSIONS, MODEL_KINDS};
use crate::app::processing::Mode as ProcessingMode;
use crate::app::recording::{Phase, Stage};
pub use crate::app::{analyze, list_cameras, Source};
use crate::app::{display_name, Engine, ImportKind};

const SETTINGS: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Comma);
/// The macOS app's menu commands, with the same keys (⌘ is Ctrl on Windows and Linux).
const fn command(key: egui::Key) -> egui::KeyboardShortcut {
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, key)
}
const ADD_VIDEOS: egui::KeyboardShortcut = command(egui::Key::O);
const PROCESS: egui::KeyboardShortcut = command(egui::Key::Enter);
const PAUSE: egui::KeyboardShortcut = egui::KeyboardShortcut::new(
    egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
    egui::Key::P,
);
const STOP: egui::KeyboardShortcut = command(egui::Key::Period);
const PAGE_PROCESS: egui::KeyboardShortcut = command(egui::Key::Num1);
const PAGE_LIVE: egui::KeyboardShortcut = command(egui::Key::Num2);
const SIDEBAR: f32 = 230.0;
/// How long a banner message stays, and how long it takes to fade (s).
const BANNER_SECONDS: f64 = 5.0;
const BANNER_FADE: f64 = 0.4;
const TOOLBAR: f32 = 52.0;
const ACCENT: Color32 = Color32::from_rgb(10, 132, 255);
const RED: Color32 = Color32::from_rgb(255, 69, 58);
const ORANGE: Color32 = Color32::from_rgb(255, 159, 10);

/// Colors of the macOS appearance, light or dark.
#[derive(Clone, Copy)]
struct Palette {
    dark: bool,
    content: Color32,
    sidebar: Color32,
    group: Color32,
    separator: Color32,
    secondary: Color32,
    tertiary: Color32,
    control: Color32,
    control_hover: Color32,
    selection: Color32,
    glass: Color32,
    glass_rim: Color32,
    shadow: Color32,
}

impl Palette {
    fn of(ui: &Ui) -> Self {
        Self::new(ui.visuals().dark_mode)
    }

    fn new(dark: bool) -> Self {
        if dark {
            Self {
                dark,
                content: Color32::from_rgb(30, 30, 30),
                sidebar: Color32::from_rgb(40, 40, 42),
                group: Color32::from_rgb(44, 44, 46),
                separator: Color32::from_white_alpha(22),
                secondary: Color32::from_rgb(152, 152, 157),
                tertiary: Color32::from_rgb(99, 99, 102),
                control: Color32::from_white_alpha(26),
                control_hover: Color32::from_white_alpha(40),
                selection: Color32::from_white_alpha(28),
                glass: Color32::from_rgba_unmultiplied(50, 50, 54, 205),
                glass_rim: Color32::from_white_alpha(34),
                shadow: Color32::from_black_alpha(110),
            }
        } else {
            Self {
                dark,
                content: Color32::WHITE,
                sidebar: Color32::from_rgb(236, 236, 238),
                group: Color32::from_rgb(246, 246, 248),
                separator: Color32::from_black_alpha(22),
                secondary: Color32::from_rgb(110, 110, 115),
                tertiary: Color32::from_rgb(174, 174, 178),
                control: Color32::from_black_alpha(14),
                control_hover: Color32::from_black_alpha(26),
                selection: Color32::from_black_alpha(20),
                glass: Color32::from_rgba_unmultiplied(252, 252, 254, 215),
                glass_rim: Color32::from_white_alpha(230),
                shadow: Color32::from_black_alpha(40),
            }
        }
    }

    fn text(&self) -> Color32 {
        if self.dark {
            Color32::from_rgb(235, 235, 240)
        } else {
            Color32::from_rgb(28, 28, 30)
        }
    }

    /// A translucent floating panel with a light rim and a soft shadow.
    fn glass(&self, radius: f32) -> egui::Frame {
        egui::Frame::none()
            .fill(self.glass)
            .stroke(Stroke::new(1.0_f32, self.glass_rim))
            .rounding(radius)
            .shadow(egui::epaint::Shadow {
                offset: vec2(0.0, 8.0),
                blur: 28.0,
                spread: 0.0,
                color: self.shadow,
            })
    }
}

#[derive(Clone, Copy)]
enum Selection {
    RecordingFolder,
    OutputFolder,
    Videos,
    VideoFolders,
    /// A camera model for the profile in `App::camera_model_for`.
    CameraModel,
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
    Process,
    Live,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    General,
    Profiles,
    Configs,
}

/// The profile open in the profile editor: its saved name, the name being typed, its stages.
struct Editing {
    profile: String,
    name: String,
    stages: Vec<Stage>,
    /// Camera labels being typed, by camera key.
    views: std::collections::BTreeMap<String, String>,
}

impl Editing {
    fn of(profile: &Profile) -> Self {
        Self {
            profile: profile.name.clone(),
            name: profile.name.clone(),
            stages: profile.stages.clone(),
            views: profile.views.clone(),
        }
    }
}

pub struct App {
    engine: Engine,
    textures: Vec<Option<((u64, bool), egui::TextureHandle)>>,
    picker: Option<Picker>,
    page: Page,
    settings_open: bool,
    tab: Tab,
    editing: Option<Editing>,
    /// The message the banner shows and when it appeared (seconds, egui time).
    banner_since: Option<(String, f64)>,
    banner_hovered: bool,
    /// The name typed for a config copy, while that dialog is open.
    duplicate: Option<String>,
    /// The height of the floating panel last frame, kept clear below the list.
    panel_height: f32,
    /// The profile a camera model is being imported for.
    camera_model_for: Option<String>,
    /// A file dialog asked for where no window frame is at hand; opened next.
    wanted_pick: Option<Selection>,
    /// The view name being typed on a preview tile.
    view_draft: String,
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
            tab: Tab::General,
            editing: None,
            banner_since: None,
            banner_hovered: false,
            duplicate: None,
            panel_height: 120.0,
            camera_model_for: None,
            wanted_pick: None,
            view_draft: String::new(),
        }
    }

    fn idle(&self) -> bool {
        self.engine.idle() && self.picker.is_none()
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
                    .set_title("Choose a Folder")
                    .set_directory(self.engine.recording_dir())
                    .pick_folder(),
            ),
            Selection::OutputFolder => {
                let dialog = dialog.set_title("Choose a Folder");
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
                        .set_title("Add Videos")
                        .add_filter("Video", &exts)
                        .pick_files(),
                )
            }
            Selection::VideoFolders => picked_many(dialog.set_title("Add Folders").pick_folders()),
            Selection::CameraModel => picked_one(
                dialog
                    .set_title("Import Camera Model")
                    .add_filter("Camera model", &["json"])
                    .pick_file(),
            ),
            Selection::Model => picked_one(
                dialog
                    .set_title("Import Model")
                    .add_filter("Checkpoint", &MODEL_EXTENSIONS)
                    .pick_file(),
            ),
            Selection::Config => picked_one(
                dialog
                    .set_title("Import Config")
                    .add_filter("Pipeline config", &CONFIG_EXTENSIONS)
                    .pick_file(),
            ),
            Selection::ImportFolder => picked_one(dialog.set_title("Import Folder").pick_folder()),
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
        if let Selection::CameraModel = selection {
            if let Some(profile) = self.camera_model_for.take() {
                self.engine.import_camera_model(&path, &profile);
            }
            return;
        }
        let engine = &mut self.engine;
        match selection {
            Selection::RecordingFolder => {
                engine.settings.recordings = Some(path);
                engine.save();
            }
            Selection::OutputFolder => {
                engine.settings.output = Some(path);
                engine.save();
            }
            Selection::Model => engine.import(ImportKind::Model, path),
            Selection::Config => engine.import(ImportKind::Config, path),
            Selection::ImportFolder => engine.import(ImportKind::Folder, path),
            Selection::Videos | Selection::VideoFolders | Selection::CameraModel => unreachable!(),
        }
    }

    fn report(&mut self, result: Result<()>) {
        if let Err(err) = result {
            self.engine.status = format!("{err:#}");
        }
    }

    // MARK: Sidebar

    fn sidebar(&mut self, ui: &mut Ui) {
        let pal = Palette::of(ui);
        ui.add_space(TOOLBAR - 8.0);
        for (page, icon, title, badge) in [
            (Page::Process, "🎞", "Process", self.engine.videos.len()),
            (Page::Live, "📹", "Live", self.engine.workers.len()),
        ] {
            if sidebar_row(ui, &pal, icon, title, self.page == page, badge).clicked() {
                self.page = page;
            }
        }
        ui.add_space(10.0);
        self.profile_menu(ui, &pal);
        let recent = self.engine.settings.recent_results.clone();
        if !recent.is_empty() {
            section_header(ui, &pal, "Results");
            for path in recent.iter().take(5) {
                let row = sidebar_row(ui, &pal, "🗀", &file_name(path), false, 0)
                    .on_hover_text(path.display().to_string());
                if row.clicked() {
                    open(path);
                }
                row.context_menu(|ui| {
                    if ui.button("Open").clicked() {
                        open(path);
                        ui.close_menu();
                    }
                    let log = path.join("animalpose.log");
                    if log.is_file() && ui.button("Open Log").clicked() {
                        open(&log);
                        ui.close_menu();
                    }
                });
            }
        }
        if let Some(job) = &self.engine.processing {
            let snapshot = job.control.snapshot();
            ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
                ui.add_space(10.0);
                pal.glass(14.0)
                    .inner_margin(egui::Margin::same(12.0))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        let title = if snapshot.total() > 1 {
                            snapshot.video_label()
                        } else {
                            "Processing…".into()
                        };
                        ui.label(RichText::new(title).small().strong());
                        progress(ui, snapshot.fraction(), ACCENT);
                    });
            });
        }
    }

    fn profile_menu(&mut self, ui: &mut Ui, pal: &Palette) {
        let active = self.engine.settings.profile.clone().unwrap_or_default();
        let enabled = self.idle();
        // One line: "Profile" on the left, the profile in use (a menu) on the right.
        let response = ui
            .add_enabled_ui(enabled, |ui| {
                let row = sidebar_row(ui, pal, "👤", "Profile", false, 0);
                let label_end = row.rect.left() + 34.0 + 56.0;
                let mut job = egui::text::LayoutJob::simple_singleline(
                    format!("{active} ⏷"),
                    FontId::proportional(13.0),
                    pal.secondary,
                );
                job.wrap = egui::text::TextWrapping::truncate_at_width(
                    (row.rect.right() - 10.0 - label_end).max(20.0),
                );
                let galley = ui.painter().layout_job(job);
                ui.painter().galley(
                    egui::pos2(
                        row.rect.right() - 10.0 - galley.size().x,
                        row.rect.center().y - galley.size().y / 2.0,
                    ),
                    galley,
                    pal.secondary,
                );
                row
            })
            .inner
            .on_hover_text(active.as_str());
        let id = ui.make_persistent_id("profile-menu");
        if response.clicked() {
            ui.memory_mut(|m| m.toggle_popup(id));
        }
        let mut chosen = None;
        let mut edit = false;
        egui::popup_below_widget(
            ui,
            id,
            &response,
            egui::PopupCloseBehavior::CloseOnClick,
            |ui| {
                ui.set_min_width(180.0);
                for profile in &self.engine.profiles {
                    if menu_check(ui, &profile.name, profile.name == active) {
                        chosen = Some(profile.name.clone());
                    }
                }
                ui.separator();
                if ui.button("Edit Profiles…").clicked() {
                    edit = true;
                }
            },
        );
        if let Some(name) = chosen {
            self.engine.select_profile(&name);
        }
        if edit {
            self.settings_open = true;
            self.tab = Tab::Profiles;
        }
    }

    // MARK: Toolbar and banner

    fn toolbar(&mut self, ui: &mut Ui, frame: &eframe::Frame) {
        let pal = Palette::of(ui);
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), TOOLBAR), Sense::hover());
        ui.painter().hline(
            rect.x_range(),
            rect.bottom() - 0.5,
            Stroke::new(1.0_f32, pal.separator),
        );
        let mut bar = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(rect.shrink2(vec2(18.0, 8.0)))
                .layout(Layout::left_to_right(Align::Center)),
        );
        let (title, subtitle) = match self.page {
            Page::Process => (
                "Process",
                match self.engine.videos.len() {
                    0 => String::new(),
                    1 => "1 Video".into(),
                    n => format!("{n} Videos"),
                },
            ),
            Page::Live => ("Live", String::new()),
        };
        bar.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            ui.label(RichText::new(title).size(15.0).strong());
            if !subtitle.is_empty() {
                ui.label(RichText::new(subtitle).size(11.0).color(pal.secondary));
            }
        });
        bar.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let idle = self.idle();
            match self.page {
                Page::Process => {
                    if icon_button(ui, &pal, "🗑", idle && !self.engine.videos.is_empty())
                        .on_hover_text("Clear List")
                        .clicked()
                    {
                        self.engine.videos.clear();
                    }
                    let add =
                        icon_button(ui, &pal, "➕", idle).on_hover_text("Add Videos or Folders");
                    let id = ui.make_persistent_id("add-menu");
                    if add.clicked() {
                        ui.memory_mut(|m| m.toggle_popup(id));
                    }
                    let mut pick = None;
                    egui::popup_below_widget(
                        ui,
                        id,
                        &add,
                        egui::PopupCloseBehavior::CloseOnClick,
                        |ui| {
                            ui.set_min_width(150.0);
                            if ui.button("Videos…").clicked() {
                                pick = Some(Selection::Videos);
                            }
                            if ui.button("Folders…").clicked() {
                                pick = Some(Selection::VideoFolders);
                            }
                        },
                    );
                    if let Some(selection) = pick {
                        self.choose(selection, frame);
                    }
                    if !self.engine.last_recording.is_empty()
                        && glass_button(ui, &pal, "Analyze Recording", idle).clicked()
                    {
                        self.engine.analyze_last();
                    }
                }
                Page::Live => {
                    if icon_button(ui, &pal, "🔄", idle)
                        .on_hover_text("Refresh Cameras")
                        .clicked()
                    {
                        self.engine.refresh_cameras();
                    }
                }
            }
        });
    }

    fn banner(&mut self, ctx: &egui::Context, area: Rect) {
        if self.engine.status.is_empty() {
            self.banner_since = None;
            return;
        }
        // Five seconds on screen, then a 0.4 s fade; the clock waits while the pointer is on it.
        let now = ctx.input(|i| i.time);
        match &mut self.banner_since {
            Some((text, since)) if *text == self.engine.status => {
                if self.banner_hovered {
                    *since = now;
                }
            }
            slot => *slot = Some((self.engine.status.clone(), now)),
        }
        let elapsed = now - self.banner_since.as_ref().map_or(now, |(_, t)| *t);
        let opacity = (1.0 - (elapsed - BANNER_SECONDS) / BANNER_FADE).clamp(0.0, 1.0) as f32;
        if opacity <= 0.0 {
            self.engine.status.clear();
            self.engine.result = None;
            self.banner_since = None;
            return;
        }
        if elapsed >= BANNER_SECONDS {
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(Duration::from_secs_f64(BANNER_SECONDS - elapsed));
        }
        let pal = Palette::new(ctx.style().visuals.dark_mode);
        let status = self.engine.status.to_lowercase();
        let failed = ["fail", "cannot", "could not", "incomplete", "install"]
            .iter()
            .any(|w| status.contains(w));
        let shown = egui::Area::new(Id::new("status-banner"))
            .order(egui::Order::Foreground)
            .pivot(Align2::CENTER_TOP)
            .fixed_pos(egui::pos2(area.center().x, area.top() + TOOLBAR + 12.0))
            .show(ctx, |ui| {
                ui.set_opacity(opacity);
                pal.glass(16.0)
                    .inner_margin(egui::Margin::symmetric(16.0, 10.0))
                    .show(ui, |ui| {
                        ui.set_max_width(area.width().min(640.0) - 40.0);
                        ui.horizontal(|ui| {
                            let (icon, color) = if failed {
                                ("⚠", ORANGE)
                            } else {
                                ("ℹ", ACCENT)
                            };
                            ui.label(RichText::new(icon).color(color).size(15.0));
                            ui.add(egui::Label::new(&self.engine.status).wrap());
                            if let Some(result) = self.engine.result.clone() {
                                if glass_button(ui, &pal, "Open", true)
                                    .on_hover_text(result.display().to_string())
                                    .clicked()
                                {
                                    open(&result);
                                }
                            }
                            let close =
                                egui::Button::new(RichText::new("✖").size(11.0)).frame(false);
                            if ui.add(close).on_hover_text("Dismiss").clicked() {
                                self.engine.status.clear();
                                self.engine.result = None;
                            }
                        });
                    });
            });
        self.banner_hovered = shown.response.contains_pointer();
    }

    // MARK: Process

    fn process_body(&mut self, ui: &mut Ui, frame: &eframe::Frame) {
        let pal = Palette::of(ui);
        if self.engine.videos.is_empty() {
            let note = "Drop videos or folders here.";
            if empty_state(
                ui,
                &pal,
                "🎞",
                "No Videos",
                note,
                Some("Add…"),
                self.panel_height,
            ) {
                self.choose(Selection::Videos, frame);
            }
            return;
        }
        let idle = self.idle();
        let mut remove = None;
        let active = self
            .engine
            .processing
            .as_ref()
            .map(|job| job.control.current())
            .unwrap_or_default();
        let groups = self.engine.view_groups();
        let videos = &self.engine.videos;
        let mut show = |ui: &mut Ui, index: usize, camera: Option<&str>| {
            let Some(video) = videos.get(index) else {
                return;
            };
            let is_active = !active.is_empty() && display_name(video) == active;
            let (row, removed) = video_row(ui, &pal, video, is_active, idle, camera);
            if removed {
                remove = Some(index);
            }
            row.context_menu(|ui| {
                if ui.button("Show in Folder").clicked() {
                    open(video.parent().unwrap_or(video));
                    ui.close_menu();
                }
                ui.separator();
                if ui.add_enabled(idle, egui::Button::new("Remove")).clicked() {
                    remove = Some(index);
                    ui.close_menu();
                }
            });
        };
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(6.0);
            match &groups {
                Some(groups) => {
                    for group in groups {
                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.add_space(14.0);
                            ui.label(
                                RichText::new(&group.title)
                                    .size(11.5)
                                    .strong()
                                    .color(pal.secondary),
                            );
                            if group.videos.len() > 1 && group.error.is_none() {
                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    ui.add_space(14.0);
                                    let views = format!("{} Views", group.videos.len());
                                    ui.label(RichText::new(views).size(11.5).color(pal.secondary));
                                });
                            }
                        });
                        for (index, camera) in &group.videos {
                            show(ui, *index, camera.as_deref());
                        }
                        if let Some(error) = &group.error {
                            ui.horizontal(|ui| {
                                ui.add_space(14.0);
                                ui.label(
                                    RichText::new(format!("⚠ {error}")).size(11.5).color(ORANGE),
                                );
                            });
                        }
                    }
                }
                None => {
                    for index in 0..videos.len() {
                        show(ui, index, None);
                    }
                }
            }
            ui.add_space(self.panel_height + 24.0);
        });
        if let Some(index) = remove {
            self.engine.remove_video(index);
        }
    }

    fn process_panel(&mut self, ui: &mut Ui, frame: &eframe::Frame) {
        let pal = Palette::of(ui);
        if let Some(job) = &self.engine.processing {
            let snapshot = job.control.snapshot();
            let current = job.control.current();
            let batch = snapshot.total() > 1;
            if batch {
                ui.label(RichText::new(snapshot.video_label()).strong());
                progress(ui, snapshot.fraction(), ACCENT);
                ui.add_space(2.0);
            }
            ui.horizontal(|ui| {
                let size = if batch { 12.5 } else { 13.5 };
                ui.label(RichText::new(&current).size(size));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let status = match snapshot.mode {
                        ProcessingMode::Pausing | ProcessingMode::Paused => "Paused".to_owned(),
                        ProcessingMode::Stopping => "Stopping…".to_owned(),
                        ProcessingMode::Running => snapshot.stage_label(),
                    };
                    ui.label(RichText::new(status).color(pal.secondary));
                });
            });
            if batch {
                progress(ui, snapshot.clip_fraction(), pal.secondary);
            } else {
                progress(ui, snapshot.fraction(), ACCENT);
            }
            ui.horizontal(|ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let enabled = snapshot.mode != ProcessingMode::Stopping;
                    if glass_button(ui, &pal, "Stop", enabled).clicked() {
                        self.engine.stop_processing();
                    }
                    let pause = if snapshot.mode == ProcessingMode::Running {
                        "Pause"
                    } else {
                        "Resume"
                    };
                    if glass_button(ui, &pal, pause, enabled).clicked() {
                        self.engine.pause_or_resume();
                    }
                });
            });
            separator(ui, &pal);
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        let idle = self.idle();
        ui.horizontal(|ui| {
            ui.label("Save To");
            ui.add_enabled_ui(idle, |ui| {
                let path = self.engine.output_dir();
                let custom = self.engine.settings.output.is_some();
                match folder_menu(
                    ui,
                    &pal,
                    "save-to",
                    path.as_deref(),
                    custom,
                    "Next to Video",
                ) {
                    Some(FolderChoice::Default) => {
                        self.engine.settings.output = None;
                        self.engine.save();
                    }
                    Some(FolderChoice::Other) => self.choose(Selection::OutputFolder, frame),
                    None => {}
                }
            });
            ui.add_space(14.0);
            ui.label("Multiview");
            let has_model = self.engine.settings.camera_model.is_some();
            let mut multiview = self.engine.settings.multiview;
            let toggled = ui
                .add_enabled_ui(idle && (has_model || multiview), |ui| {
                    switch(ui, &mut multiview)
                })
                .inner
                .on_hover_text("Fuse Each Folder’s Views with the Camera Model")
                .on_disabled_hover_text("Set a Camera Model in the Profile");
            if toggled.changed() {
                self.engine.set_multiview(multiview);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let ffmpeg = crate::data::video::ffmpeg_available();
                let enabled = idle && !self.engine.videos.is_empty() && ffmpeg;
                if prominent_button(ui, "Process", enabled, ACCENT).clicked() {
                    self.engine.process_videos();
                }
                if !ffmpeg {
                    ui.label(RichText::new("⚠ FFmpeg Not Installed").color(ORANGE));
                }
            });
        });
    }

    // MARK: Live

    fn live_body(&mut self, ui: &mut Ui) {
        let pal = Palette::of(ui);
        let count = self.engine.workers.len();
        self.textures.resize_with(count, || None);
        if count == 0 {
            let note = if self.engine.sources.is_empty() {
                "No cameras found."
            } else {
                "Choose a source below."
            };
            empty_state(ui, &pal, "📹", "No Preview", note, None, self.panel_height);
            return;
        }
        let recording = self.engine.session().is_some();
        let spacing = 16.0;
        let available = ui.available_width() - 32.0;
        let columns = ((available / 360.0).floor().max(1.0) as usize).min(count);
        let width = (available - spacing * (columns - 1) as f32) / columns as f32;
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(16.0);
            let indices: Vec<usize> = (0..count).collect();
            for row in indices.chunks(columns) {
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.spacing_mut().item_spacing.x = spacing;
                    for &index in row {
                        self.tile(ui, &pal, index, width, recording);
                    }
                });
                ui.add_space(spacing);
            }
            ui.add_space(self.panel_height + 24.0);
        });
    }

    fn tile(&mut self, ui: &mut Ui, pal: &Palette, index: usize, width: f32, recording: bool) {
        let height = width * 3.0 / 4.0;
        let (rect, _) = ui.allocate_exact_size(vec2(width, height), Sense::hover());
        ui.painter()
            .rect_filled(rect, 18.0, Color32::from_black_alpha(220));
        let known = self.textures[index].as_ref().map(|(key, _)| *key);
        let preview = self.engine.preview(index, known);
        if let Some(p) = &preview {
            if let Some(rgba) = &p.rgba {
                let image = egui::ColorImage::from_rgba_unmultiplied(p.size, rgba);
                match &mut self.textures[index] {
                    Some((key, texture)) => {
                        texture.set(image, egui::TextureOptions::LINEAR);
                        *key = p.key;
                    }
                    slot => {
                        let texture = ui.ctx().load_texture(
                            format!("source-{index}"),
                            image,
                            egui::TextureOptions::LINEAR,
                        );
                        *slot = Some((p.key, texture));
                    }
                }
            }
        }
        let worker = &self.engine.workers[index];
        if let Some((_, texture)) = &self.textures[index] {
            let size = texture.size_vec2();
            let scale = (width / size.x).min(height / size.y);
            let fit = Rect::from_center_size(rect.center(), size * scale);
            let corner = if (fit.height() - height).abs() < 1.0 && (fit.width() - width).abs() < 1.0
            {
                18.0
            } else {
                0.0
            };
            egui::Image::new((texture.id(), fit.size()))
                .rounding(corner)
                .paint_at(ui, fit);
        } else if !worker.ended.load(std::sync::atomic::Ordering::Acquire) {
            ui.put(
                Rect::from_center_size(rect.center(), vec2(24.0, 24.0)),
                egui::Spinner::new().color(Color32::WHITE),
            );
        }
        // Badges: the view and source on top, the analysis and errors at the bottom.
        let dot = recording.then_some(RED);
        let name = worker.source.name();
        let mut bottom = rect.left_bottom() + vec2(10.0, -10.0);
        let error = worker.capture_error.lock().unwrap().clone().or_else(|| {
            if recording {
                None
            } else {
                worker.preview_error.lock().unwrap().clone()
            }
        });
        if let Some(error) = error {
            let text = format!("⚠ {error}");
            let badge = glass_capsule(ui, pal, bottom, Align2::LEFT_BOTTOM, &text, None, 11.5);
            bottom.y = badge.top() - 6.0;
        }
        if let Some((animals, ms)) = preview.and_then(|p| p.analysis) {
            let s = if animals == 1 { "" } else { "s" };
            let text = format!("{animals} animal{s} · {ms:.0} ms");
            glass_capsule(ui, pal, bottom, Align2::LEFT_BOTTOM, &text, None, 11.5);
        }
        self.view_badge(ui, pal, index, rect, &name, dot, recording);
    }

    /// The view a camera records as, on its tile; click to name it, or with a camera model to
    /// choose which of its cameras this is.
    #[allow(clippy::too_many_arguments)]
    fn view_badge(
        &mut self,
        ui: &mut Ui,
        pal: &Palette,
        index: usize,
        tile: Rect,
        source: &str,
        dot: Option<Color32>,
        recording: bool,
    ) {
        let view = self.engine.views().get(index).cloned().unwrap_or_default();
        let text = if recording {
            format!("{view}   {source}")
        } else {
            format!("{view} ⏷   {source}")
        };
        let anchor = tile.left_top() + vec2(10.0, 10.0);
        let capsule = glass_capsule(ui, pal, anchor, Align2::LEFT_TOP, &text, dot, 13.0);
        if recording {
            return;
        }
        let id = ui.make_persistent_id(("view", index));
        let response = ui
            .interact(capsule, id.with("badge"), Sense::click())
            .on_hover_text("View");
        if response.clicked() {
            self.view_draft = view.clone();
            ui.memory_mut(|m| m.toggle_popup(id));
        }
        let names = self.engine.camera_names();
        let mut chosen = None;
        egui::popup_below_widget(
            ui,
            id,
            &response,
            egui::PopupCloseBehavior::CloseOnClickOutside,
            |ui| {
                ui.set_min_width(170.0);
                if names.is_empty() {
                    let field = ui.add(
                        egui::TextEdit::singleline(&mut self.view_draft)
                            .hint_text("Camera")
                            .desired_width(160.0),
                    );
                    if !field.has_focus() && !field.lost_focus() && response.clicked() {
                        field.request_focus();
                    }
                    if field.lost_focus() {
                        chosen = Some(self.view_draft.clone());
                    }
                } else {
                    for camera in &names {
                        if menu_check(ui, camera, *camera == view) {
                            chosen = Some(camera.clone());
                        }
                    }
                }
            },
        );
        if let Some(chosen) = chosen {
            ui.memory_mut(|m| m.close_popup());
            if chosen.trim() != view {
                self.engine.set_view(index, &chosen);
            }
        }
    }

    fn live_panel(&mut self, ui: &mut Ui) {
        let pal = Palette::of(ui);
        if let Some(session) = self.engine.session() {
            let (phase, elapsed, remaining) = session.position();
            ui.horizontal(|ui| {
                let pulse = (ui.input(|i| i.time) * 2.0).sin() as f32 * 0.25 + 0.75;
                ui.label(
                    RichText::new("⏺")
                        .color(RED.gamma_multiply(pulse))
                        .size(15.0),
                );
                ui.label(RichText::new(session.label(phase)).strong());
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let secs = elapsed.as_secs();
                    let clock = format!("{}:{:02}", secs / 60, secs % 60);
                    ui.label(RichText::new(clock).size(16.0).monospace());
                });
            });
            if let Phase::Timed(i) = phase {
                let seconds = f64::from(session.stages[i].seconds);
                progress(ui, (1.0 - remaining.as_secs_f64() / seconds) as f32, ACCENT);
                let left = format!("{} s", remaining.as_secs_f64().ceil());
                ui.label(RichText::new(left).color(pal.secondary));
            }
            ui.horizontal(|ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if glass_button(ui, &pal, "Stop", true).clicked() {
                        self.engine.stop_recording();
                    }
                    if matches!(phase, Phase::Waiting(_))
                        && prominent_button(ui, "Continue", true, ACCENT).clicked()
                    {
                        self.engine.continue_stage();
                    }
                });
            });
            ui.ctx().request_repaint_after(Duration::from_millis(50));
            return;
        }
        let idle = self.idle();
        ui.add_enabled_ui(idle, |ui| {
            ui.horizontal_wrapped(|ui| {
                let engine = &mut self.engine;
                for (source, on) in engine.sources.iter().zip(&mut engine.selected) {
                    chip(ui, &pal, &format!("📷 {}", source.name()), on);
                }
                if let Some(first) = engine.videos.first() {
                    let name = format!("🎞 {}", file_name(first));
                    chip(ui, &pal, &name, &mut engine.playback).on_hover_text("Play as a Camera");
                }
            });
        });
        separator(ui, &pal);
        ui.horizontal(|ui| {
            ui.label("Pose Overlay");
            let mut live = self.engine.settings.live;
            let busy = self.engine.busy();
            if ui
                .add_enabled_ui(!busy, |ui| switch(ui, &mut live))
                .inner
                .changed()
            {
                self.engine.set_live(live);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if self.engine.workers.is_empty() {
                    if prominent_button(ui, "▶  Preview", idle, ACCENT).clicked() {
                        self.engine.start_preview();
                    }
                    return;
                }
                let ready = self.engine.sources_ready();
                let timed = ready && !self.engine.settings.stages.is_empty();
                if prominent_button(ui, "⏱  Timed", timed, RED)
                    .on_hover_text("Record the Protocol Stages")
                    .clicked()
                {
                    self.engine.start_recording(true);
                }
                if glass_button(ui, &pal, "⏺  Record", ready)
                    .on_hover_text("Record Until Stopped")
                    .clicked()
                {
                    self.engine.start_recording(false);
                }
                if glass_button(ui, &pal, "Stop", true).clicked() {
                    self.engine.stop_sources(false);
                }
                ui.add(
                    egui::TextEdit::singleline(&mut self.engine.recording_name)
                        .hint_text("Recording Name")
                        .desired_width(170.0),
                )
                .on_hover_text("Names the recording folder");
            });
        });
    }

    // MARK: Settings

    fn settings(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let id = egui::ViewportId::from_hash_of("settings");
        let builder = egui::ViewportBuilder::default()
            .with_title("Settings")
            .with_inner_size([680.0, 560.0])
            .with_min_inner_size([560.0, 420.0]);
        ctx.show_viewport_immediate(id, builder, |ctx, class| {
            if ctx.input_mut(|i| i.viewport().close_requested() || i.consume_shortcut(&SETTINGS)) {
                self.settings_open = false;
            }
            let pal = Palette::new(ctx.style().visuals.dark_mode);
            if class == egui::ViewportClass::Embedded {
                let mut open = self.settings_open;
                egui::Window::new("Settings")
                    .open(&mut open)
                    .default_size([640.0, 520.0])
                    .show(ctx, |ui| self.settings_body(ui, &pal, frame));
                self.settings_open &= open;
            } else {
                egui::CentralPanel::default()
                    .frame(egui::Frame::none().fill(pal.content))
                    .show(ctx, |ui| self.settings_body(ui, &pal, frame));
            }
        });
    }

    fn settings_body(&mut self, ui: &mut Ui, pal: &Palette, frame: &eframe::Frame) {
        self.settings_tabs(ui, pal);
        let enabled = self.idle();
        ui.add_enabled_ui(enabled, |ui| match self.tab {
            Tab::General => self.general(ui, pal, frame),
            Tab::Profiles => self.profiles(ui, pal),
            Tab::Configs => self.configs(ui, pal),
        });
        if let Some(selection) = self.wanted_pick.take() {
            self.choose(selection, frame);
        }
    }

    fn settings_tabs(&mut self, ui: &mut Ui, pal: &Palette) {
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 62.0), Sense::hover());
        ui.painter().rect_filled(rect, 0.0, pal.sidebar);
        ui.painter().hline(
            rect.x_range(),
            rect.bottom(),
            Stroke::new(1.0_f32, pal.separator),
        );
        let tabs = [
            (Tab::General, "⚙", "General"),
            (Tab::Profiles, "👤", "Profiles"),
            (Tab::Configs, "🔧", "Configs"),
        ];
        let width = 76.0;
        let left = rect.center().x - width * tabs.len() as f32 / 2.0;
        for (i, (tab, icon, title)) in tabs.into_iter().enumerate() {
            let cell = Rect::from_min_size(
                egui::pos2(left + width * i as f32, rect.top() + 6.0),
                vec2(width - 6.0, 50.0),
            );
            let response = ui.interact(cell, Id::new(("settings-tab", i)), Sense::click());
            let selected = self.tab == tab;
            let fill = if selected {
                pal.selection
            } else if response.hovered() {
                pal.control
            } else {
                Color32::TRANSPARENT
            };
            ui.painter().rect_filled(cell, 8.0, fill);
            let color = if selected { ACCENT } else { pal.secondary };
            ui.painter().text(
                cell.center_top() + vec2(0.0, 17.0),
                Align2::CENTER_CENTER,
                icon,
                FontId::proportional(20.0),
                color,
            );
            ui.painter().text(
                cell.center_bottom() - vec2(0.0, 10.0),
                Align2::CENTER_CENTER,
                title,
                FontId::proportional(11.5),
                if selected { pal.text() } else { pal.secondary },
            );
            if response.clicked() {
                self.tab = tab;
            }
        }
    }

    fn general(&mut self, ui: &mut Ui, pal: &Palette, frame: &eframe::Frame) {
        let mut pick = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            form(ui, |ui| {
                group(ui, pal, None, |ui, rows| {
                    row(ui, pal, rows, "Processed Videos", |ui| {
                        let custom = self.engine.settings.output.clone();
                        let path = custom.as_deref();
                        let custom = path.is_some();
                        match folder_menu(ui, pal, "processed", path, custom, "Next to Video") {
                            Some(FolderChoice::Default) => {
                                self.engine.settings.output = None;
                                self.engine.save();
                            }
                            Some(FolderChoice::Other) => pick = Some(Selection::OutputFolder),
                            None => {}
                        }
                    });
                    row(ui, pal, rows, "Recordings", |ui| {
                        let path = self.engine.recording_dir();
                        let custom = self.engine.settings.recordings.is_some();
                        match folder_menu(ui, pal, "recordings", Some(&path), custom, "Default") {
                            Some(FolderChoice::Default) => {
                                self.engine.settings.recordings = None;
                                self.engine.save();
                            }
                            Some(FolderChoice::Other) => pick = Some(Selection::RecordingFolder),
                            None => {}
                        }
                    });
                });
                group(ui, pal, None, |ui, rows| {
                    row(ui, pal, rows, "Logs", |ui| {
                        if glass_button(ui, pal, "Show in Folder", true).clicked() {
                            open(self.engine.log.dir());
                        }
                    });
                    row(ui, pal, rows, "Library", |ui| {
                        if glass_button(ui, pal, "Show in Folder", true).clicked() {
                            open(&self.engine.library.root);
                        }
                        let import = glass_button(ui, pal, "Import  ⏷", true);
                        let id = ui.make_persistent_id("import-menu");
                        if import.clicked() {
                            ui.memory_mut(|m| m.toggle_popup(id));
                        }
                        egui::popup_below_widget(
                            ui,
                            id,
                            &import,
                            egui::PopupCloseBehavior::CloseOnClick,
                            |ui| {
                                ui.set_min_width(120.0);
                                if ui.button("Model…").clicked() {
                                    pick = Some(Selection::Model);
                                }
                                if ui.button("Config…").clicked() {
                                    pick = Some(Selection::Config);
                                }
                                if ui.button("Folder…").clicked() {
                                    pick = Some(Selection::ImportFolder);
                                }
                            },
                        );
                        if self.engine.importing() {
                            ui.add(egui::Spinner::new());
                        }
                    });
                });
            });
        });
        if let Some(selection) = pick {
            self.choose(selection, frame);
        }
    }

    fn profiles(&mut self, ui: &mut Ui, pal: &Palette) {
        let active = self.engine.settings.profile.clone();
        let wanted = self
            .editing
            .as_ref()
            .map(|e| e.profile.clone())
            .or_else(|| active.clone());
        let Some(profile) = wanted
            .and_then(|name| self.engine.profiles.iter().find(|p| p.name == name))
            .or_else(|| self.engine.profiles.first())
            .cloned()
        else {
            return;
        };
        if self.editing.as_ref().map(|e| &e.profile) != Some(&profile.name) {
            self.editing = Some(Editing::of(&profile));
        }
        let mut copy: Option<String> = None;
        let mut delete: Option<String> = None;
        ui.add_space(16.0);
        ui.horizontal_top(|ui| {
            ui.add_space(20.0);
            // The list, with + and − below it.
            ui.vertical(|ui| {
                ui.set_width(190.0);
                let height = (ui.available_height() - 48.0).max(120.0);
                egui::Frame::none()
                    .fill(pal.group)
                    .stroke(Stroke::new(1.0_f32, pal.separator))
                    .rounding(8.0)
                    .inner_margin(egui::Margin::same(4.0))
                    .show(ui, |ui| {
                        ui.set_min_height(height);
                        ui.set_width(182.0);
                        let count = self.engine.profiles.len();
                        for p in self.engine.profiles.clone() {
                            let check = if Some(&p.name) == active.as_ref() {
                                "   ✔"
                            } else {
                                ""
                            };
                            let title = format!("{}{check}", p.name);
                            let row = sidebar_row(ui, pal, "", &title, p.name == profile.name, 0);
                            if row.clicked() {
                                self.editing = Some(Editing::of(&p));
                            }
                            row.context_menu(|ui| {
                                if ui.button("Duplicate").clicked() {
                                    copy = Some(p.name.clone());
                                    ui.close_menu();
                                }
                                ui.separator();
                                if ui
                                    .add_enabled(count > 1, egui::Button::new("Delete"))
                                    .clicked()
                                {
                                    delete = Some(p.name.clone());
                                    ui.close_menu();
                                }
                            });
                        }
                    });
                ui.horizontal(|ui| {
                    if icon_button(ui, pal, "➕", true)
                        .on_hover_text("New Profile")
                        .clicked()
                    {
                        let mut copy = profile.clone();
                        copy.name = self.engine.new_profile_name("New Profile");
                        let result = self.engine.save_profile(None, copy);
                        self.report(result);
                        self.editing = None;
                    }
                    let can_delete = self.engine.profiles.len() > 1;
                    if icon_button(ui, pal, "➖", can_delete)
                        .on_hover_text("Delete Profile")
                        .clicked()
                    {
                        delete = Some(profile.name.clone());
                    }
                    if glass_button(ui, pal, "Duplicate", true)
                        .on_hover_text("Duplicate Profile")
                        .clicked()
                    {
                        copy = Some(profile.name.clone());
                    }
                });
            });
            ui.add_space(8.0);
            ui.vertical(|ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.set_width(ui.available_width() - 20.0);
                    self.profile_form(ui, pal, &profile, active.as_deref());
                });
            });
        });
        if let Some(name) = delete {
            let result = self.engine.delete_profile(&name);
            self.report(result);
            self.editing = None;
        }
        // The copy is selected; the profile in use stays.
        if let Some(name) = copy {
            match self.engine.duplicate_profile(&name) {
                Ok(new) => {
                    let saved = self.engine.profiles.iter().find(|p| p.name == new);
                    self.editing = saved.map(Editing::of);
                }
                Err(err) => self.engine.status = format!("{err:#}"),
            }
        }
    }

    fn profile_form(
        &mut self,
        ui: &mut Ui,
        pal: &Palette,
        profile: &Profile,
        active: Option<&str>,
    ) {
        let mut next = profile.clone();
        let mut rename = None;
        let mut use_it = false;
        let mut stages_changed = false;
        let configs = self.engine.library.configs();
        let models: Vec<Vec<String>> = MODEL_KINDS
            .iter()
            .map(|kind| self.engine.library.models(*kind))
            .collect();
        let camera_models = self.engine.library.camera_models();
        let in_use = active == Some(profile.name.as_str());
        let cameras: Option<Result<String, String>> = match &self.engine.cameras {
            Some(Ok(cams)) if in_use => Some(Ok(cams
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>()
                .join(", "))),
            Some(Err(err)) if in_use => Some(Err(err.clone())),
            _ => None,
        };
        let mut import_model = false;
        let sources: Vec<(String, String, String)> = self
            .engine
            .sources
            .iter()
            .map(|s| {
                (
                    s.name(),
                    self.engine.source_key(s),
                    self.engine.default_view(s),
                )
            })
            .collect();
        let editing = self.editing.as_mut().expect("a profile is being edited");
        group(ui, pal, None, |ui, rows| {
            row(ui, pal, rows, "Name", |ui| {
                let field =
                    ui.add(egui::TextEdit::singleline(&mut editing.name).desired_width(220.0));
                let name = editing.name.trim();
                if field.lost_focus() && name != profile.name && !name.is_empty() {
                    rename = Some(name.to_owned());
                }
            });
            if active != Some(profile.name.as_str()) {
                row(ui, pal, rows, "", |ui| {
                    use_it = glass_button(ui, pal, "Use", true).clicked();
                });
            }
        });
        group(ui, pal, Some("Pipeline"), |ui, rows| {
            row(ui, pal, rows, "Config", |ui| {
                let label = next.config.as_deref().map_or("Default".into(), stem);
                egui::ComboBox::from_id_salt("profile-config")
                    .selected_text(label)
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut next.config, None, "Default");
                        for c in &configs {
                            ui.selectable_value(&mut next.config, Some(c.clone()), stem(c));
                        }
                    });
            });
            for (kind, names) in MODEL_KINDS.iter().zip(&models) {
                if *kind == crate::models::ModelKind::Arena {
                    continue; // under Arena, with the camera model
                }
                let enabled = *kind != crate::models::ModelKind::Behavior || next.classify_behavior;
                row(ui, pal, rows, &capitalized(kind.as_str()), |ui| {
                    ui.add_enabled_ui(enabled, |ui| {
                        model_picker(ui, &mut next.models, kind.as_str(), names)
                    });
                });
            }
            row(ui, pal, rows, "Classify Behavior", |ui| {
                switch(ui, &mut next.classify_behavior);
            });
        });
        group(ui, pal, Some("Arena"), |ui, rows| {
            let arena = crate::models::ModelKind::Arena;
            if let Some(names) = MODEL_KINDS
                .iter()
                .position(|k| *k == arena)
                .map(|i| &models[i])
            {
                row(ui, pal, rows, "Arena Detector", |ui| {
                    model_picker(ui, &mut next.models, arena.as_str(), names);
                });
            }
            row(ui, pal, rows, "Camera Model", |ui| {
                let label = next.camera_model.as_deref().map_or("None".into(), stem);
                egui::ComboBox::from_id_salt("profile-camera-model")
                    .selected_text(label)
                    .width(200.0)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut next.camera_model, None, "None");
                        for m in &camera_models {
                            ui.selectable_value(&mut next.camera_model, Some(m.clone()), stem(m));
                        }
                        ui.separator();
                        import_model = ui.button("Import…").clicked();
                    });
            });
            match &cameras {
                Some(Ok(names)) => row(ui, pal, rows, "Views", |ui| {
                    ui.label(RichText::new(names).color(pal.secondary));
                }),
                Some(Err(err)) => row(ui, pal, rows, "", |ui| {
                    ui.label(RichText::new(format!("⚠ {err}")).color(ORANGE));
                }),
                None => {}
            }
        });
        let cameras: Vec<(String, String, String)> = sources
            .iter()
            .map(|(name, key, default)| (name.clone(), key.clone(), default.clone()))
            .collect();
        if !cameras.is_empty() {
            group(ui, pal, Some("Camera Names"), |ui, rows| {
                for (name, key, default) in &cameras {
                    row(ui, pal, rows, name, |ui| {
                        let text = editing.views.entry(key.clone()).or_default();
                        let field = ui.add(
                            egui::TextEdit::singleline(text)
                                .hint_text(default.as_str())
                                .desired_width(200.0),
                        );
                        if field.lost_focus() {
                            match text.trim() {
                                "" => next.views.remove(key),
                                label => next.views.insert(key.clone(), label.to_owned()),
                            };
                        }
                    });
                }
            });
        }
        group(ui, pal, Some("Stages"), |ui, rows| {
            let count = editing.stages.len();
            let mut remove = None;
            for (i, stage) in editing.stages.iter_mut().enumerate() {
                row(ui, pal, rows, "", |ui| {
                    let minus = RichText::new("✖").size(11.0).color(pal.secondary);
                    if ui
                        .add(egui::Button::new(minus).frame(false))
                        .on_hover_text("Remove")
                        .clicked()
                    {
                        remove = Some(i);
                    }
                    if i + 1 < count {
                        stages_changed |= switch(ui, &mut stage.wait_after)
                            .on_hover_text("Wait for Continue")
                            .changed();
                        ui.label("Wait");
                    }
                    ui.label(RichText::new("s").color(pal.secondary));
                    stages_changed |= ui
                        .add(egui::DragValue::new(&mut stage.seconds).range(1..=86_400))
                        .changed();
                    let field = ui.add(
                        egui::TextEdit::singleline(&mut stage.name)
                            .hint_text(format!("Stage {}", i + 1))
                            .desired_width(ui.available_width() - 8.0),
                    );
                    stages_changed |= field.lost_focus();
                });
            }
            if let Some(i) = remove {
                editing.stages.remove(i);
                stages_changed = true;
            }
            row(ui, pal, rows, "", |ui| {
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    let plus = RichText::new("➕").color(pal.secondary);
                    if ui
                        .add(egui::Button::new(plus).frame(false))
                        .on_hover_text("Add Stage")
                        .clicked()
                    {
                        editing.stages.push(Stage {
                            name: String::new(),
                            seconds: 300,
                            wait_after: false,
                        });
                        stages_changed = true;
                    }
                });
            });
        });
        if stages_changed {
            next.stages = editing.stages.clone();
        }
        if import_model {
            self.camera_model_for = Some(profile.name.clone());
            self.wanted_pick = Some(Selection::CameraModel);
        }
        if use_it {
            self.engine.select_profile(&profile.name);
        }
        if let Some(new) = rename {
            next.name = new.clone();
            let result = self.engine.save_profile(Some(&profile.name), next);
            match &result {
                Ok(()) => {
                    let saved = self.engine.profiles.iter().find(|p| p.name == new);
                    self.editing = saved.map(Editing::of);
                }
                Err(_) => {
                    if let Some(editing) = self.editing.as_mut() {
                        editing.name = profile.name.clone();
                    }
                }
            }
            self.report(result);
        } else if next != *profile {
            let result = self.engine.save_profile(Some(&profile.name), next);
            self.report(result);
        }
    }

    fn configs(&mut self, ui: &mut Ui, pal: &Palette) {
        if self.engine.config_form.is_none() {
            let config = self.engine.settings.config.clone();
            let result = self.engine.edit_config(config);
            self.report(result);
        }
        let Some((name, form_json)) = self.engine.config_form.clone() else {
            return;
        };
        let configs = self.engine.library.configs();
        let mut chosen = name.clone();
        let mut delete = false;
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            ui.add_space(20.0);
            ui.label("Config");
            egui::ComboBox::from_id_salt("config-editor")
                .selected_text(chosen.as_deref().map_or("Default".into(), stem))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut chosen, None, "Default");
                    for c in &configs {
                        ui.selectable_value(&mut chosen, Some(c.clone()), stem(c));
                    }
                });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.add_space(20.0);
                let more = icon_button(ui, pal, "…", true);
                let id = ui.make_persistent_id("config-actions");
                if more.clicked() {
                    ui.memory_mut(|m| m.toggle_popup(id));
                }
                egui::popup_below_widget(
                    ui,
                    id,
                    &more,
                    egui::PopupCloseBehavior::CloseOnClick,
                    |ui| {
                        ui.set_min_width(130.0);
                        if ui.button("Duplicate…").clicked() {
                            self.duplicate = Some(copy_name(name.as_deref()));
                        }
                        if name.is_some() {
                            ui.separator();
                            delete = ui.button(RichText::new("Delete").color(RED)).clicked();
                        }
                    },
                );
            });
        });
        if chosen != name {
            let result = self.engine.edit_config(chosen);
            self.report(result);
            return;
        }
        if delete {
            if let Some(n) = &name {
                let result = self.engine.delete_config(n);
                self.report(result);
                return;
            }
        }
        let editable = form_json["editable"].as_bool().unwrap_or(false);
        let mut commits = Vec::new();
        egui::ScrollArea::vertical().show(ui, |ui| {
            form(ui, |ui| {
                if !editable {
                    group(ui, pal, None, |ui, rows| {
                        row(
                            ui,
                            pal,
                            rows,
                            "The default config can’t be changed.",
                            |ui| {
                                if glass_button(ui, pal, "Duplicate…", true).clicked() {
                                    self.duplicate = Some(copy_name(None));
                                }
                            },
                        );
                    });
                }
                for section in form_json["sections"].as_array().into_iter().flatten() {
                    let title = section["title"].as_str().unwrap_or_default();
                    group(ui, pal, Some(title), |ui, rows| {
                        for field in section["fields"].as_array().into_iter().flatten() {
                            let label = field["label"].as_str().unwrap_or_default();
                            row(ui, pal, rows, label, |ui| {
                                ui.add_enabled_ui(editable, |ui| {
                                    if let Some(value) = field_control(ui, field) {
                                        let key = field["key"].as_str().unwrap_or_default();
                                        commits.push((key.to_owned(), value));
                                    }
                                });
                            });
                        }
                    });
                }
            });
        });
        for (key, value) in commits {
            let result = self.engine.set_config_value(&key, &value);
            self.report(result);
        }
        self.duplicate_dialog(ui.ctx(), pal, name.as_deref());
    }

    /// Names a copy of config `source` in a small glass dialog.
    fn duplicate_dialog(&mut self, ctx: &egui::Context, pal: &Palette, source: Option<&str>) {
        let Some(new_name) = self.duplicate.as_mut() else {
            return;
        };
        let (mut submit, mut cancel) = (false, false);
        egui::Window::new("Duplicate Config")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .frame(pal.glass(14.0).inner_margin(egui::Margin::same(16.0)))
            .show(ctx, |ui| {
                let field = ui.add(egui::TextEdit::singleline(new_name).desired_width(240.0));
                submit |= field.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let ok = !new_name.trim().is_empty();
                        submit |= prominent_button(ui, "Duplicate", ok, ACCENT).clicked();
                        cancel |= glass_button(ui, pal, "Cancel", true).clicked();
                    });
                });
            });
        if submit && !new_name.trim().is_empty() {
            let new_name = new_name.clone();
            let result = self.engine.duplicate_config(source, &new_name);
            self.report(result);
            cancel = true;
        }
        if cancel {
            self.duplicate = None;
        }
    }
}

fn copy_name(source: Option<&str>) -> String {
    source.map_or("Pipeline".into(), |s| format!("{} Copy", stem(s)))
}

/// The control for one config setting; returns the value to save when it changed.
fn field_control(ui: &mut Ui, field: &Value) -> Option<Value> {
    let kind = field["type"].as_str().unwrap_or_default();
    let min = field["min"].as_f64().unwrap_or(0.0);
    let max = field["max"].as_f64().unwrap_or(1.0);
    let id = Id::new(("config-field", field["key"].as_str().unwrap_or_default()));
    match kind {
        "bool" => {
            let mut on = field["value"].as_bool().unwrap_or(false);
            switch(ui, &mut on).changed().then(|| json!(on))
        }
        "choice" => {
            let current = field["value"].as_str().unwrap_or_default().to_owned();
            let mut chosen = current.clone();
            egui::ComboBox::from_id_salt(id)
                .selected_text(capitalized(&chosen))
                .show_ui(ui, |ui| {
                    for option in field["options"].as_array().into_iter().flatten() {
                        let option = option.as_str().unwrap_or_default();
                        ui.selectable_value(&mut chosen, option.to_owned(), capitalized(option));
                    }
                });
            (chosen != current).then(|| json!(chosen))
        }
        "float" if max <= 1.0 => {
            // The slider keeps its value while dragged and saves it when released.
            let saved = field["value"].as_f64().unwrap_or(0.0);
            let mut value = ui.data(|d| d.get_temp::<f64>(id)).unwrap_or(saved);
            ui.label(RichText::new(format!("{value:.2}")).monospace());
            let response = ui.add(
                egui::Slider::new(&mut value, min..=max)
                    .show_value(false)
                    .trailing_fill(true),
            );
            if response.dragged() {
                ui.data_mut(|d| d.insert_temp(id, value));
            } else {
                ui.data_mut(|d| d.remove::<f64>(id));
            }
            let step = field["step"].as_f64().unwrap_or(0.01);
            (response.drag_stopped() || (response.changed() && !response.dragged()))
                .then(|| json!((value / step).round() * step))
        }
        _ => {
            let mut value = field["value"].as_f64().unwrap_or(0.0);
            let step = field["step"].as_f64().unwrap_or(1.0);
            let max = field["max"].as_f64().unwrap_or(1_000_000.0);
            let decimals = if kind == "float" { 2 } else { 0 };
            let mut response = ui.add(
                egui::DragValue::new(&mut value)
                    .range(min..=max)
                    .speed(step)
                    .max_decimals(decimals),
            );
            if kind == "limit" {
                response = response.on_hover_text("0 for No Limit");
            }
            let done = response.drag_stopped()
                || response.lost_focus()
                || (response.changed() && !response.dragged() && !response.has_focus());
            done.then(|| json!(value))
        }
    }
}

// MARK: Widgets

fn sidebar_row(
    ui: &mut Ui,
    pal: &Palette,
    icon: &str,
    title: &str,
    selected: bool,
    badge: usize,
) -> Response {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 28.0), Sense::click());
    let hover = ui
        .ctx()
        .animate_bool(response.id.with("hover"), response.hovered() && !selected);
    let fill = if selected {
        pal.selection
    } else {
        pal.control.gamma_multiply(hover)
    };
    ui.painter().rect_filled(rect, 6.0, fill);
    let mut x = rect.left() + 10.0;
    if !icon.is_empty() {
        ui.painter().text(
            egui::pos2(x + 8.0, rect.center().y),
            Align2::CENTER_CENTER,
            icon,
            FontId::proportional(14.0),
            if selected { ACCENT } else { pal.secondary },
        );
        x += 24.0;
    }
    let color = if ui.is_enabled() {
        pal.text()
    } else {
        pal.tertiary
    };
    ui.painter().text(
        egui::pos2(x, rect.center().y),
        Align2::LEFT_CENTER,
        title,
        FontId::proportional(13.5),
        color,
    );
    if badge > 0 {
        let galley = ui.painter().layout_no_wrap(
            badge.to_string(),
            FontId::proportional(11.0),
            pal.secondary,
        );
        let pill = Rect::from_center_size(
            egui::pos2(rect.right() - 18.0, rect.center().y),
            vec2(galley.size().x + 12.0, 18.0),
        );
        ui.painter().rect_filled(pill, 9.0, pal.control);
        ui.painter()
            .galley(pill.center() - galley.size() / 2.0, galley, pal.secondary);
    }
    response
}

fn section_header(ui: &mut Ui, pal: &Palette, title: &str) {
    ui.add_space(14.0);
    ui.horizontal(|ui| {
        ui.add_space(10.0);
        ui.label(
            RichText::new(title)
                .size(11.0)
                .strong()
                .color(pal.secondary),
        );
    });
    ui.add_space(2.0);
}

fn separator(ui: &mut Ui, pal: &Palette) {
    ui.add_space(4.0);
    let y = ui.cursor().top();
    ui.painter().hline(
        ui.max_rect().x_range(),
        y,
        Stroke::new(1.0_f32, pal.separator),
    );
    ui.add_space(10.0);
}

/// A capsule button filled with `color`: the main action.
fn prominent_button(ui: &mut Ui, text: &str, enabled: bool, color: Color32) -> Response {
    let fill = if enabled {
        color
    } else {
        color.gamma_multiply(0.35)
    };
    let button = egui::Button::new(RichText::new(text).color(Color32::WHITE).strong())
        .fill(fill)
        .rounding(Rounding::same(16.0))
        .min_size(vec2(88.0, 32.0));
    ui.add_enabled(enabled, button)
}

/// A translucent capsule button: a secondary action on glass.
fn glass_button(ui: &mut Ui, pal: &Palette, text: &str, enabled: bool) -> Response {
    let button = egui::Button::new(RichText::new(text).size(13.0))
        .fill(pal.control)
        .stroke(Stroke::new(1.0_f32, pal.glass_rim.gamma_multiply(0.6)))
        .rounding(Rounding::same(14.0))
        .min_size(vec2(0.0, 28.0));
    ui.add_enabled(enabled, button)
}

fn icon_button(ui: &mut Ui, pal: &Palette, icon: &str, enabled: bool) -> Response {
    let button = egui::Button::new(RichText::new(icon).size(15.0).color(pal.secondary))
        .frame(false)
        .min_size(vec2(30.0, 28.0));
    ui.add_enabled(enabled, button)
}

/// A toggle chip for a source, tinted when on.
fn chip(ui: &mut Ui, pal: &Palette, text: &str, on: &mut bool) -> Response {
    let fill = if *on {
        ACCENT.gamma_multiply(0.85)
    } else {
        pal.control
    };
    let color = if *on { Color32::WHITE } else { pal.text() };
    let response = ui.add(
        egui::Button::new(RichText::new(text).color(color))
            .fill(fill)
            .rounding(Rounding::same(8.0))
            .min_size(vec2(0.0, 28.0)),
    );
    if response.clicked() {
        *on = !*on;
    }
    response
}

/// The macOS switch: a capsule whose knob slides across.
fn switch(ui: &mut Ui, on: &mut bool) -> Response {
    let pal = Palette::of(ui);
    let (rect, mut response) = ui.allocate_exact_size(vec2(38.0, 22.0), Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    let t = ui.ctx().animate_bool_responsive(response.id, *on);
    let off = if pal.dark {
        Color32::from_white_alpha(40)
    } else {
        Color32::from_black_alpha(30)
    };
    let fill = lerp_color(off, ACCENT, t);
    let fill = if ui.is_enabled() {
        fill
    } else {
        fill.gamma_multiply(0.5)
    };
    ui.painter().rect_filled(rect, 11.0, fill);
    let x = egui::lerp(rect.left() + 11.0..=rect.right() - 11.0, t);
    let knob = egui::pos2(x, rect.center().y);
    ui.painter()
        .circle_filled(knob + vec2(0.0, 0.8), 9.4, Color32::from_black_alpha(40));
    ui.painter().circle_filled(knob, 9.0, Color32::WHITE);
    response
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgba_premultiplied(
        mix(a.r(), b.r()),
        mix(a.g(), b.g()),
        mix(a.b(), b.b()),
        mix(a.a(), b.a()),
    )
}

/// A thin capsule progress bar.
fn progress(ui: &mut Ui, fraction: f32, color: Color32) {
    let pal = Palette::of(ui);
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 6.0), Sense::hover());
    ui.painter().rect_filled(rect, 3.0, pal.control_hover);
    let mut fill = rect;
    fill.set_width(rect.width() * fraction.clamp(0.0, 1.0));
    if fill.width() > 0.0 {
        ui.painter().rect_filled(fill, 3.0, color);
    }
}

/// A text badge on glass, drawn at `anchor`; returns its rectangle.
fn glass_capsule(
    ui: &Ui,
    pal: &Palette,
    anchor: egui::Pos2,
    align: Align2,
    text: &str,
    dot: Option<Color32>,
    size: f32,
) -> Rect {
    let galley =
        ui.painter()
            .layout_no_wrap(text.to_owned(), FontId::proportional(size), pal.text());
    let dot_width = if dot.is_some() { 14.0 } else { 0.0 };
    let size = vec2(galley.size().x + 22.0 + dot_width, galley.size().y + 10.0);
    let rect = align.anchor_size(anchor, size);
    let radius = rect.height() / 2.0;
    let shadow = egui::epaint::Shadow {
        offset: vec2(0.0, 3.0),
        blur: 10.0,
        spread: 0.0,
        color: Color32::from_black_alpha(60),
    };
    ui.painter().add(shadow.as_shape(rect, radius));
    ui.painter().rect_filled(rect, radius, pal.glass);
    ui.painter()
        .rect_stroke(rect, radius, Stroke::new(1.0_f32, pal.glass_rim));
    let mut x = rect.left() + 11.0;
    if let Some(color) = dot {
        ui.painter()
            .circle_filled(egui::pos2(x + 4.0, rect.center().y), 4.0, color);
        x += dot_width;
    }
    let top = rect.center().y - galley.size().y / 2.0;
    ui.painter().galley(egui::pos2(x, top), galley, pal.text());
    rect
}

/// The empty view of a page; returns true when its button was pressed.
fn empty_state(
    ui: &mut Ui,
    pal: &Palette,
    icon: &str,
    title: &str,
    note: &str,
    button: Option<&str>,
    bottom: f32,
) -> bool {
    let area = ui.available_rect_before_wrap();
    let center = egui::pos2(area.center().x, area.top() + (area.height() - bottom) / 2.0);
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(Rect::from_center_size(
                center,
                vec2(area.width().min(420.0), 200.0),
            ))
            .layout(Layout::top_down(Align::Center)),
    );
    child.label(RichText::new(icon).size(44.0).color(pal.tertiary));
    child.add_space(8.0);
    child.label(
        RichText::new(title)
            .size(17.0)
            .strong()
            .color(pal.secondary),
    );
    child.label(RichText::new(note).color(pal.secondary));
    match button {
        Some(text) => {
            child.add_space(8.0);
            glass_button(&mut child, pal, text, true).clicked()
        }
        None => false,
    }
}

/// One listed video; returns its row and whether its remove button was pressed.
fn video_row(
    ui: &mut Ui,
    pal: &Palette,
    video: &Path,
    active: bool,
    idle: bool,
    camera: Option<&str>,
) -> (Response, bool) {
    let (rect, response) = ui.allocate_exact_size(vec2(ui.available_width(), 46.0), Sense::click());
    let rect = rect.shrink2(vec2(10.0, 1.0));
    if response.hovered() {
        ui.painter().rect_filled(rect, 8.0, pal.control);
    }
    let (icon, icon_color) = if active {
        let pulse = (ui.input(|i| i.time) * 3.0).sin() as f32 * 0.3 + 0.7;
        ("◉", ACCENT.gamma_multiply(pulse))
    } else {
        ("🎞", pal.secondary)
    };
    ui.painter().text(
        egui::pos2(rect.left() + 18.0, rect.center().y),
        Align2::CENTER_CENTER,
        icon,
        FontId::proportional(17.0),
        icon_color,
    );
    let folder = video
        .parent()
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    ui.painter().text(
        egui::pos2(rect.left() + 40.0, rect.center().y - 8.0),
        Align2::LEFT_CENTER,
        file_name(video),
        FontId::proportional(13.5),
        pal.text(),
    );
    ui.painter().text(
        egui::pos2(rect.left() + 40.0, rect.center().y + 9.0),
        Align2::LEFT_CENTER,
        shorten(&folder, ((rect.width() - 90.0) / 6.2).max(0.0) as usize),
        FontId::proportional(11.0),
        pal.secondary,
    );
    ui.painter().hline(
        (rect.left() + 40.0)..=rect.right(),
        rect.bottom() + 0.5,
        Stroke::new(1.0_f32, pal.separator),
    );
    if let Some(camera) = camera {
        let galley =
            ui.painter()
                .layout_no_wrap(camera.to_owned(), FontId::proportional(11.0), pal.text());
        let pill = Align2::RIGHT_CENTER.anchor_size(
            egui::pos2(rect.right() - 40.0, rect.center().y),
            galley.size() + vec2(16.0, 6.0),
        );
        ui.painter()
            .rect_filled(pill, pill.height() / 2.0, pal.control);
        ui.painter()
            .galley(pill.center() - galley.size() / 2.0, galley, pal.text());
    }
    let mut remove = false;
    if response.hovered() && idle {
        let spot = Rect::from_center_size(
            egui::pos2(rect.right() - 18.0, rect.center().y),
            vec2(24.0, 24.0),
        );
        let close = RichText::new("✖").size(11.0).color(pal.secondary);
        remove = ui
            .put(spot, egui::Button::new(close).frame(false))
            .on_hover_text("Remove")
            .clicked();
    }
    (response.on_hover_text(video.display().to_string()), remove)
}

enum FolderChoice {
    Default,
    Other,
}

/// The save-location popup: the folder in use, the default choice, Other… and Show in Folder.
fn folder_menu(
    ui: &mut Ui,
    pal: &Palette,
    id: &str,
    path: Option<&Path>,
    custom: bool,
    default_title: &str,
) -> Option<FolderChoice> {
    let title = path.map_or(default_title.to_owned(), file_name);
    let hint = path.map_or(default_title.to_owned(), |p| p.display().to_string());
    let button = ui
        .add(
            egui::Button::new(RichText::new(format!("🗀  {title}  ⏷")).size(13.0))
                .fill(pal.control)
                .rounding(Rounding::same(7.0))
                .min_size(vec2(0.0, 26.0)),
        )
        .on_hover_text(hint);
    let popup = ui.make_persistent_id(("folder-menu", id));
    if button.clicked() {
        ui.memory_mut(|m| m.toggle_popup(popup));
    }
    let mut choice = None;
    egui::popup_below_widget(
        ui,
        popup,
        &button,
        egui::PopupCloseBehavior::CloseOnClick,
        |ui| {
            ui.set_min_width(200.0);
            if menu_check(ui, default_title, !custom) {
                choice = Some(FolderChoice::Default);
            }
            if let Some(path) = path.filter(|_| custom) {
                menu_check(ui, &file_name(path), true);
            }
            ui.separator();
            if ui.button("Other…").clicked() {
                choice = Some(FolderChoice::Other);
            }
            if let Some(path) = path {
                ui.separator();
                if ui.button("Show in Folder").clicked() {
                    open(path);
                }
            }
        },
    );
    choice
}

/// A menu item with a check mark when `checked`; returns whether it was clicked.
fn menu_check(ui: &mut Ui, text: &str, checked: bool) -> bool {
    let mark = if checked { "✔" } else { "    " };
    ui.button(format!("{mark}  {text}")).clicked()
}

/// A settings page: grouped rows with a margin, like a macOS grouped form.
fn form(ui: &mut Ui, add: impl FnOnce(&mut Ui)) {
    egui::Frame::none()
        .inner_margin(egui::Margin::symmetric(20.0, 14.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

/// A rounded group of rows, with an optional heading above it.
fn group(ui: &mut Ui, pal: &Palette, title: Option<&str>, add: impl FnOnce(&mut Ui, &mut usize)) {
    if let Some(title) = title {
        ui.add_space(6.0);
        ui.label(RichText::new(title).strong());
    }
    egui::Frame::none()
        .fill(pal.group)
        .stroke(Stroke::new(1.0_f32, pal.separator))
        .rounding(10.0)
        .inner_margin(egui::Margin::symmetric(12.0, 2.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            let mut rows = 0;
            add(ui, &mut rows);
        });
    ui.add_space(12.0);
}

/// One row of a group: the label on the left, its control on the right.
fn row(ui: &mut Ui, pal: &Palette, rows: &mut usize, label: &str, control: impl FnOnce(&mut Ui)) {
    if *rows > 0 {
        let y = ui.cursor().top();
        ui.painter().hline(
            ui.max_rect().x_range(),
            y,
            Stroke::new(1.0_f32, pal.separator),
        );
    }
    *rows += 1;
    ui.allocate_ui_with_layout(
        vec2(ui.available_width(), 36.0),
        Layout::left_to_right(Align::Center),
        |ui| {
            ui.set_min_height(36.0);
            if !label.is_empty() {
                ui.label(label);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), control);
        },
    );
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// `pipeline_lab.yaml` → `pipeline_lab`.
fn stem(file: &str) -> String {
    Path::new(file)
        .file_stem()
        .map_or_else(|| file.to_owned(), |s| s.to_string_lossy().into_owned())
}

/// `open_field` → `Open Field`.
/// A profile's network of one kind: "From Config", or one of the library's `names`.
fn model_picker(
    ui: &mut Ui,
    models: &mut std::collections::BTreeMap<String, String>,
    key: &str,
    names: &[String],
) {
    let mut chosen = models.get(key).cloned();
    egui::ComboBox::from_id_salt(("profile-model", key))
        .selected_text(chosen.as_deref().unwrap_or("From Config"))
        .width(200.0)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut chosen, None, "From Config");
            for m in names {
                ui.selectable_value(&mut chosen, Some(m.clone()), m);
            }
        });
    match chosen {
        Some(m) => models.insert(key.to_owned(), m),
        None => models.remove(key),
    };
}

fn capitalized(text: &str) -> String {
    text.split('_')
        .map(|word| {
            let mut chars = word.chars();
            chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect::<String>())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// `text`, with its middle replaced by "…" when longer than `max` characters.
fn shorten(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max || max < 8 {
        return text.to_owned();
    }
    let keep = (max - 1) / 2;
    let head: String = text.chars().take(keep).collect();
    let tail: String = text.chars().skip(count - keep).collect();
    format!("{head}…{tail}")
}

/// Opens a folder or file with the system (the file browser for folders).
fn open(path: &Path) {
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

/// The platform's interface font (Segoe UI, or Inter / Noto Sans / Cantarell on Linux), the macOS
/// type scale, and widgets styled like macOS controls, in light and dark.
fn style(ctx: &egui::Context) {
    let candidates: &[&str] = if cfg!(target_os = "windows") {
        &[r"C:\Windows\Fonts\segoeui.ttf"]
    } else {
        &[
            "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/cantarell/Cantarell-VF.otf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    };
    let system = cfg!(not(target_os = "macos"))
        .then(|| candidates.iter().find_map(|path| std::fs::read(path).ok()))
        .flatten();
    if let Some(bytes) = system {
        let mut fonts = egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("system".into(), egui::FontData::from_owned(bytes));
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "system".into());
        ctx.set_fonts(fonts);
    }
    ctx.set_theme(egui::ThemePreference::System);
    for theme in [egui::Theme::Light, egui::Theme::Dark] {
        let pal = Palette::new(theme == egui::Theme::Dark);
        ctx.style_mut_of(theme, |style| {
            use egui::TextStyle;
            style.text_styles = [
                (TextStyle::Small, FontId::proportional(11.0)),
                (TextStyle::Body, FontId::proportional(13.5)),
                (TextStyle::Button, FontId::proportional(13.5)),
                (TextStyle::Heading, FontId::proportional(17.0)),
                (TextStyle::Monospace, FontId::monospace(12.5)),
            ]
            .into();
            style.spacing.item_spacing = vec2(8.0, 6.0);
            style.spacing.button_padding = vec2(12.0, 4.0);
            style.spacing.interact_size.y = 24.0;
            style.spacing.slider_width = 180.0;
            style.spacing.menu_margin = egui::Margin::same(6.0);
            let v = &mut style.visuals;
            v.panel_fill = pal.content;
            v.window_fill = pal.content;
            v.window_stroke = Stroke::new(1.0_f32, pal.separator);
            v.window_rounding = Rounding::same(12.0);
            v.menu_rounding = Rounding::same(8.0);
            v.window_shadow = egui::epaint::Shadow {
                offset: vec2(0.0, 10.0),
                blur: 30.0,
                spread: 0.0,
                color: pal.shadow,
            };
            v.popup_shadow = egui::epaint::Shadow {
                offset: vec2(0.0, 6.0),
                blur: 18.0,
                spread: 0.0,
                color: pal.shadow,
            };
            v.extreme_bg_color = if pal.dark {
                Color32::from_rgb(28, 28, 30)
            } else {
                Color32::WHITE
            };
            v.faint_bg_color = pal.group;
            v.selection.bg_fill = ACCENT.gamma_multiply(0.45);
            v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
            v.hyperlink_color = ACCENT;
            v.slider_trailing_fill = true;
            v.override_text_color = Some(pal.text());
            for (w, fill) in [
                (&mut v.widgets.inactive, pal.control),
                (&mut v.widgets.hovered, pal.control_hover),
                (&mut v.widgets.active, pal.control_hover),
                (&mut v.widgets.open, pal.control_hover),
            ] {
                w.weak_bg_fill = fill;
                w.bg_fill = fill;
                w.bg_stroke = Stroke::NONE;
                w.rounding = Rounding::same(7.0);
                w.expansion = 0.0;
            }
            v.widgets.noninteractive.rounding = Rounding::same(7.0);
            v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, pal.separator);
            v.widgets.active.bg_fill = ACCENT;
        });
    }
}

impl App {
    /// The menu commands of the macOS app ([`ADD_VIDEOS`], [`PROCESS`], [`PAUSE`], [`STOP`] and the
    /// two pages), enabled when the macOS menu items are.
    fn shortcuts(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let pressed = |s: &egui::KeyboardShortcut| ctx.input_mut(|i| i.consume_shortcut(s));
        let idle = self.idle();
        if pressed(&PAGE_PROCESS) {
            self.page = Page::Process;
        }
        if pressed(&PAGE_LIVE) {
            self.page = Page::Live;
        }
        if pressed(&ADD_VIDEOS) && idle {
            self.page = Page::Process;
            self.choose(Selection::Videos, frame);
        }
        if pressed(&PROCESS)
            && idle
            && !self.engine.videos.is_empty()
            && crate::data::video::ffmpeg_available()
        {
            self.engine.process_videos();
        }
        if self.engine.processing.is_some() {
            if pressed(&PAUSE) {
                self.engine.pause_or_resume();
            }
            if pressed(&STOP) {
                self.engine.stop_processing();
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.engine.poll();
        if ctx.input_mut(|i| i.consume_shortcut(&SETTINGS)) {
            self.settings_open = !self.settings_open;
        }
        self.shortcuts(ctx, frame);
        // Drop videos or folders anywhere on the window.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if !dropped.is_empty() && self.idle() {
            self.page = Page::Process;
            self.engine.add_videos(dropped);
        }
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let pal = Palette::new(ctx.style().visuals.dark_mode);

        egui::SidePanel::left("sidebar")
            .exact_width(SIDEBAR)
            .resizable(false)
            .show_separator_line(true)
            .frame(
                egui::Frame::none()
                    .fill(pal.sidebar)
                    .inner_margin(egui::Margin::symmetric(10.0, 0.0)),
            )
            .show(ctx, |ui| self.sidebar(ui));
        let central = egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(pal.content))
            .show(ctx, |ui| {
                self.toolbar(ui, frame);
                match self.page {
                    Page::Process => self.process_body(ui, frame),
                    Page::Live => self.live_body(ui),
                }
            });
        let area = central.response.rect;
        // The floating glass panel at the bottom of the content.
        let panel = egui::Area::new(Id::new("panel"))
            .order(egui::Order::Middle)
            .pivot(Align2::LEFT_BOTTOM)
            .fixed_pos(area.left_bottom() + vec2(16.0, -16.0))
            .show(ctx, |ui| {
                pal.glass(22.0)
                    .inner_margin(egui::Margin::same(16.0))
                    .show(ui, |ui| {
                        ui.set_width(area.width() - 64.0);
                        match self.page {
                            Page::Process => self.process_panel(ui, frame),
                            Page::Live => self.live_panel(ui),
                        }
                    });
            });
        self.panel_height = panel.response.rect.height();
        self.banner(ctx, area);
        if hovering {
            ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, Id::new("drop")))
                .rect_stroke(area.shrink(8.0), 18.0, Stroke::new(3.0_f32, ACCENT));
        }
        if self.settings_open {
            self.settings(ctx, frame);
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

/// The window and dock icon: `docs/icons/minimal.svg` as rendered by `scripts/make_icons.sh`.
fn icon() -> egui::IconData {
    let rgba = image::load_from_memory(include_bytes!("../assets/icon.png"))
        .expect("the bundled icon is a PNG")
        .to_rgba8();
    egui::IconData {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
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
            // Linux docks match the window to `animalpose-gui.desktop` (and its icon) by this id.
            .with_app_id("animalpose-gui")
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([820.0, 540.0])
            .with_drag_and_drop(true)
            .with_icon(icon()),
        ..Default::default()
    };
    eframe::run_native(
        "animalpose",
        options,
        Box::new(|cc| Ok(Box::new(App::new(&cc.egui_ctx)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_icon_is_a_square_picture() {
        let icon = super::icon();
        assert_eq!((icon.width, icon.height), (256, 256));
        assert_eq!(icon.rgba.len(), 256 * 256 * 4);
        // Transparent corners, the artwork in the middle.
        assert_eq!(icon.rgba[3], 0);
        assert!(icon.rgba.chunks(4).filter(|p| p[3] == 255).count() > 256 * 256 / 10);
    }
}
