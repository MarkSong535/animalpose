//! Embeds the app icon (`assets/icon.ico`, from `docs/icons/minimal.svg` via
//! `scripts/make_icons.sh`) in the Windows executables. Other targets need nothing here: the
//! macOS bundle carries its own `.icns` and the window icon is set at run time.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/icon.ico");
        resource
            .compile()
            .expect("cannot embed the icon in the Windows executable");
    }
}
