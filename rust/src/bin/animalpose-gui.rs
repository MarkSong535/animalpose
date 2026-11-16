//! The desktop app: live tracking from USB cameras, recording and analysis.
//!
//! ```text
//! cargo run --release --features gui --bin animalpose-gui
//! ```

#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

fn main() -> anyhow::Result<()> {
    animalpose::gui::run()
}
