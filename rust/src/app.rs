//! The desktop application without its window: camera capture and recording, the model library,
//! and batch video processing. [`crate::gui`] (egui) and [`crate::ffi`] (the native macOS app)
//! are two faces of the same [`Engine`].

mod capture;
mod engine;
pub(crate) mod library;
pub(crate) mod processing;
pub(crate) mod recording;

pub use capture::{list_cameras, Source};
pub use engine::{analyze, Engine};
pub(crate) use engine::{display_name, ImportKind};

#[cfg(test)]
mod tests;
