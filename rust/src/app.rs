//! The desktop application without its window: camera capture and recording, the model library,
//! and batch video processing. [`crate::gui`] (egui) and [`crate::ffi`] (the native macOS app)
//! are two faces of the same [`Engine`].

mod capture;
pub(crate) mod configs;
mod engine;
pub(crate) mod library;
mod log;
mod models;
pub(crate) mod processing;
pub(crate) mod recording;
pub(crate) mod samples;
pub(crate) mod views;

pub use capture::{list_cameras, Source};
#[cfg(feature = "gui")]
pub(crate) use engine::display_name;
pub(crate) use engine::ImportKind;
pub use engine::{analyze, Engine};

/// Checks a sample bundle (`dir`, or the one shipped with this binary) end to end: hashes, then
/// each profile installed into a scratch library, its networks loaded and run, its camera model
/// and assay section read. Prints one line per profile; an error names the first problem.
pub fn check_samples(dir: Option<&std::path::Path>) -> anyhow::Result<()> {
    use anyhow::Context;
    let dir = match dir {
        Some(dir) => dir.to_path_buf(),
        None => samples::locate().context("no sample bundle found next to this binary")?,
    };
    let scratch = tempfile::tempdir()?;
    for line in samples::check(&dir, scratch.path())? {
        println!("{line}");
    }
    println!("sample bundle {} is complete", dir.display());
    Ok(())
}

#[cfg(test)]
mod tests;
