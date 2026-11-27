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

#[path = "src/civil_date.rs"]
mod civil_date;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-env-changed=XMALAB_BUILD_DATE");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
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
