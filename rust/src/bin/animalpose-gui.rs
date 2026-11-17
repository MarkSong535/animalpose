//! The desktop app: live tracking from USB cameras, recording and analysis.
//!
//! ```text
//! cargo run --release --features gui --bin animalpose-gui
//! ```

#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

fn main() -> anyhow::Result<()> {
    // `animalpose-gui --check-samples [DIR]`: verify the bundled sample profiles and exit (the
    // release packaging runs it before shipping the bundle).
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--check-samples") {
        return animalpose::app::check_samples(args.get(1).map(std::path::Path::new));
    }
    animalpose::gui::run()
}
