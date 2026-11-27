//! Sets `XMALAB_BUILD_DATE`, the build date the About dialog shows (`dialogs::about::build_date`).
//!
//! - `XMALAB_BUILD_DATE` set in the environment: passed through unchanged (a release build
//!   can name its date).
//! - Else the UTC date, `YYYY-MM-DD`, of `SOURCE_DATE_EPOCH` when that is set (reproducible
//!   builds), and of the system clock otherwise.
//!
//! Cargo reruns this script only when `build.rs`, a file under `src/` or one of the two
//! variables changes, so the date is the day of the last build in which the program's own
//! sources changed. Standard library only: no dependency, no network, no `git`.
//!
//! When building for Windows with the Microsoft linker it also gives `rustxmalab.exe` its
//! icon (`build/windows_icon.rs`), written as a resource file from
//! `assets/icons/RustXMALab.ico` and passed to the linker; the icon is data, not code.

#[path = "src/civil_date.rs"]
mod civil_date;

#[path = "build/windows_icon.rs"]
mod windows_icon;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-env-changed=XMALAB_BUILD_DATE");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    windows_icon_resource();
    let date = match std::env::var("XMALAB_BUILD_DATE") {
        Ok(d) => {
            // One line: anything after a line break would be read by cargo as a directive.
            assert!(
                !d.contains(['\n', '\r']),
                "XMALAB_BUILD_DATE must be one line"
            );
            d
        }
        Err(_) => civil_date::utc_date(epoch_seconds()),
    };
    println!("cargo:rustc-env=XMALAB_BUILD_DATE={date}");
}

/// On a Windows target with the Microsoft linker, writes the icon's resource file and links
/// it into `rustxmalab.exe`. Other targets (and the GNU linker, which cannot read `.res`
/// files) get no icon resource.
fn windows_icon_resource() {
    const ICO: &str = "assets/icons/RustXMALab.ico";
    println!("cargo:rerun-if-changed={ICO}");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os != "windows" || target_env != "msvc" {
        return;
    }
    let ico = std::fs::read(ICO).unwrap_or_else(|e| panic!("cannot read {ICO}: {e}"));
    let res = windows_icon::res_from_ico(&ico).unwrap_or_else(|e| panic!("{ICO}: {e}"));
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("rustxmalab-icon.res");
    std::fs::write(&out, res).unwrap_or_else(|e| panic!("cannot write {}: {e}", out.display()));
    println!("cargo:rustc-link-arg-bin=rustxmalab={}", out.display());
}

/// `SOURCE_DATE_EPOCH` when it is a number of seconds, else the system clock.
fn epoch_seconds() -> u64 {
    if let Ok(s) = std::env::var("SOURCE_DATE_EPOCH") {
        match s.trim().parse::<u64>() {
            Ok(n) => return n,
            Err(_) => println!(
                "cargo:warning=SOURCE_DATE_EPOCH is not a number of seconds ({s:?}); using the clock"
            ),
        }
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
