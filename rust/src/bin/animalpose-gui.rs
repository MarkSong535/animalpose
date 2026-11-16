//! The desktop app: live tracking from USB cameras, recording and analysis.
//!
//! ```text
//! cargo run --release --features gui --bin animalpose-gui
//! ```

fn main() -> anyhow::Result<()> {
    animalpose::gui::run()
}
