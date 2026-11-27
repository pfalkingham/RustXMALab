//! `cargo xtask dist`: the Linux release archive.
//!
//! Builds `rustxmalab` and `xma` in release mode and writes
//! `target/dist/rustxmalab-<version>-<arch>-linux.tar.gz`, holding one folder
//! `rustxmalab-<version>-<arch>-linux/` with the two programs, the Linux menu entry
//! `rustxmalab.desktop` and its picture `rustxmalab.png` (from `packaging/linux/`), `README.md`, `LICENSE`,
//! `NOTICE.md`, `INSTALL.txt` (from `xtask/dist/INSTALL-linux.txt`), `THIRD-PARTY-LICENSES.txt`
//! (written by `licenses.rs`) and the folder `licenses/` (the licence texts and notices of the
//! works whose code was ported, copied from the workspace's `licenses/`). The archive is made
//! with the system's `tar` (no archive crate, so no new dependency); the programs are not
//! stripped here (`[profile.release]` decides).

use std::path::{Path, PathBuf};

use crate::{cargo, licenses};

/// The programs in the archive, with the package that builds each.
const PROGRAMS: &[(&str, &str)] = &[("xma-app", "rustxmalab"), ("xma-cli", "xma")];

/// The folder of the Linux menu entry and its picture, under the workspace root.
const PACKAGING: &str = "packaging/linux";

/// The menu entry and its picture, copied from [`PACKAGING`] beside the programs.
const MENU_FILES: &[&str] = &["rustxmalab.desktop", "rustxmalab.png"];

/// The names placed at the top of the archive's folder, in order: the programs, the menu
/// entry and its picture, the documents, the install text, the crates' licences and the
/// folder of the ported works' licences.
#[must_use]
pub fn staged_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = PROGRAMS.iter().map(|(_, b)| *b).collect();
    v.extend(MENU_FILES);
    v.extend(DOCS);
    v.extend(["INSTALL.txt", "THIRD-PARTY-LICENSES.txt", LICENSES_DIR]);
    v
}

/// Files copied from the workspace root.
const DOCS: &[&str] = &["README.md", "LICENSE", "NOTICE.md"];

/// The folder of licence texts of the ported works, copied from the workspace root.
const LICENSES_DIR: &str = "licenses";

/// The archive's file name without `.tar.gz`.
#[must_use]
pub fn archive_stem(version: &str, arch: &str) -> String {
    format!("rustxmalab-{version}-{arch}-linux")
}

pub fn dist(root: &Path) -> Result<(), String> {
    if std::env::consts::OS != "linux" {
        return Err("`dist` makes the Linux archive; run it on Linux".into());
    }
    let version = env!("CARGO_PKG_VERSION");
    let arch = std::env::consts::ARCH;

    let mut build = cargo();
    build.args(["build", "--release", "--locked"]);
    for (pkg, bin) in PROGRAMS {
        build.args(["-p", pkg, "--bin", bin]);
    }
    eprintln!("==> cargo build --release (rustxmalab, xma)");
    let status = build
        .current_dir(root)
        .status()
        .map_err(|e| format!("could not start cargo: {e}"))?;
    if !status.success() {
        return Err(format!("release build failed ({status})"));
    }

    let target = target_dir(root)?;
    let dist = target.join("dist");
    let stem = archive_stem(version, arch);
    let stage = dist.join(&stem);
    if stage.exists() {
        std::fs::remove_dir_all(&stage)
            .map_err(|e| format!("cannot remove {}: {e}", stage.display()))?;
    }
    std::fs::create_dir_all(&stage)
        .map_err(|e| format!("cannot create {}: {e}", stage.display()))?;

    for (_, bin) in PROGRAMS {
        copy(&target.join("release").join(bin), &stage.join(bin))?;
    }
    for f in MENU_FILES {
        copy(&root.join(PACKAGING).join(f), &stage.join(f))?;
    }
    for doc in DOCS {
        copy(&root.join(doc), &stage.join(doc))?;
    }
    copy_dir(&root.join(LICENSES_DIR), &stage.join(LICENSES_DIR))?;
    eprintln!("==> THIRD-PARTY-LICENSES.txt");
    licenses::write(root, &stage.join("THIRD-PARTY-LICENSES.txt"))?;
    let mut install = std::fs::read_to_string(root.join("xtask/dist/INSTALL-linux.txt"))
        .map_err(|e| format!("cannot read xtask/dist/INSTALL-linux.txt: {e}"))?;
    for (_, bin) in PROGRAMS {
        let bytes =
            std::fs::read(stage.join(bin)).map_err(|e| format!("cannot read {bin}: {e}"))?;
        let v =
            newest_glibc(&bytes).map_or_else(|| "(any)".to_string(), |(a, b)| format!("{a}.{b}"));
        eprintln!("==> {bin} needs glibc {v}");
        install = install.replace(&format!("@GLIBC_{bin}@"), &v);
    }
    std::fs::write(stage.join("INSTALL.txt"), install)
        .map_err(|e| format!("cannot write INSTALL.txt: {e}"))?;
    for name in staged_names() {
        if !stage.join(name).exists() {
            return Err(format!("{name} is missing from {}", stage.display()));
        }
    }

    let archive = dist.join(format!("{stem}.tar.gz"));
    let _ = std::fs::remove_file(&archive);
    eprintln!("==> tar -czf {}", archive.display());
    let status = std::process::Command::new("tar")
        .arg("--owner=0")
        .arg("--group=0")
        .arg("--numeric-owner")
        .arg("-czf")
        .arg(&archive)
        .arg("-C")
        .arg(&dist)
        .arg(&stem)
        .status()
        .map_err(|e| format!("could not run tar: {e}"))?;
    if !status.success() {
        return Err(format!("tar failed ({status})"));
    }
    let size = std::fs::metadata(&archive).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "==> wrote {} ({:.1} MB)",
        archive.display(),
        size as f64 / 1e6
    );
    Ok(())
}

/// The newest `GLIBC_x.y` symbol version named in a program (an upper bound: a weak
/// symbol's version counts too).
#[must_use]
pub fn newest_glibc(bytes: &[u8]) -> Option<(u32, u32)> {
    const TAG: &[u8] = b"GLIBC_";
    let mut best = None;
    let mut i = 0;
    while i + TAG.len() < bytes.len() {
        if &bytes[i..i + TAG.len()] == TAG {
            let rest = &bytes[i + TAG.len()..];
            let end = rest
                .iter()
                .position(|c| !(c.is_ascii_digit() || *c == b'.'))
                .unwrap_or(rest.len());
            let text = std::str::from_utf8(&rest[..end]).unwrap_or("");
            let mut parts = text.split('.');
            if let (Some(Ok(a)), Some(Ok(b))) = (
                parts.next().map(str::parse::<u32>),
                parts.next().map(str::parse::<u32>),
            ) {
                best = best.max(Some((a, b)));
            }
            i += TAG.len() + end;
        } else {
            i += 1;
        }
    }
    best
}

/// Copies a file, keeping its permissions (the programs stay executable).
fn copy(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::copy(from, to)
        .map(|_| ())
        .map_err(|e| format!("cannot copy {} to {}: {e}", from.display(), to.display()))
}

/// Copies a folder of plain files (no subfolders are expected; any found are copied too).
fn copy_dir(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to).map_err(|e| format!("cannot create {}: {e}", to.display()))?;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(from)
        .map_err(|e| format!("cannot read {}: {e}", from.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for p in entries {
        let dest = to.join(p.file_name().unwrap_or_default());
        if p.is_dir() {
            copy_dir(&p, &dest)?;
        } else {
            copy(&p, &dest)?;
        }
    }
    Ok(())
}

/// Cargo's target directory (`CARGO_TARGET_DIR`, a configuration, or `<root>/target`), as
/// `cargo metadata` reports it.
pub(crate) fn target_dir(root: &Path) -> Result<PathBuf, String> {
    let out = cargo()
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("cargo metadata output: {e}"))?;
    v["target_directory"]
        .as_str()
        .map(PathBuf::from)
        .ok_or_else(|| "cargo metadata gave no target_directory".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_name() {
        assert_eq!(
            archive_stem("0.1.0", "x86_64"),
            "rustxmalab-0.1.0-x86_64-linux"
        );
    }

    #[test]
    fn glibc_versions_are_compared_as_numbers() {
        let bin = b"..GLIBC_2.2.5\0GLIBC_2.39\0xGLIBC_2.4\0GLIBC_PRIVATE\0";
        assert_eq!(newest_glibc(bin), Some((2, 39)));
        assert_eq!(newest_glibc(b"nothing"), None);
    }

    #[test]
    fn install_text_is_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(root.join("dist/INSTALL-linux.txt")).unwrap();
        assert!(text.contains("ffmpeg"));
        for (_, bin) in PROGRAMS {
            assert!(text.contains(&format!("@GLIBC_{bin}@")), "{bin}");
        }
        for doc in DOCS {
            assert!(root.parent().unwrap().join(doc).is_file(), "{doc}");
        }
        assert!(text.contains("THIRD-PARTY-LICENSES.txt"));
        assert!(text.contains("licenses/"));
        for f in MENU_FILES {
            assert!(
                root.parent().unwrap().join(PACKAGING).join(f).is_file(),
                "{f}"
            );
        }
        let dir = root.parent().unwrap().join(LICENSES_DIR);
        assert!(dir.join("README.txt").is_file());
        // Every file the folder's README lists is there, and every file there is listed.
        let readme = std::fs::read_to_string(dir.join("README.txt")).unwrap();
        for e in std::fs::read_dir(&dir).unwrap() {
            let name = e.unwrap().file_name().into_string().unwrap();
            assert!(
                name == "README.txt" || readme.contains(&name),
                "{name} not in README.txt"
            );
        }
    }

    /// The archive's folder holds the two programs, the menu entry and its picture, and the
    /// install text names each of them and says how to add the menu entry by hand.
    #[test]
    fn archive_holds_the_programs_and_the_menu_entry() {
        let names = staged_names();
        for want in [
            "rustxmalab",
            "xma",
            "rustxmalab.desktop",
            "rustxmalab.png",
            "INSTALL.txt",
        ] {
            assert!(names.contains(&want), "{want} not staged: {names:?}");
        }
        assert!(!names.contains(&"xmalab"));
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(root.join("dist/INSTALL-linux.txt")).unwrap();
        for f in ["rustxmalab", "xma", "rustxmalab.desktop", "rustxmalab.png"] {
            assert!(
                text.lines().any(|l| l.split_whitespace().next() == Some(f)),
                "INSTALL text does not list {f}"
            );
        }
        assert!(text.contains("~/.local/share/applications/"));
        assert!(text.contains("~/.local/share/icons/hicolor/256x256/apps/"));
        assert!(text.contains("Exec="));
    }

    /// The menu entry has the keys the brief names, with these values.
    #[test]
    fn desktop_file_has_its_keys() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let text =
            std::fs::read_to_string(root.join(PACKAGING).join("rustxmalab.desktop")).unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("[Desktop Entry]"));
        let keys: std::collections::BTreeMap<&str, &str> =
            lines.filter_map(|l| l.split_once('=')).collect();
        for (k, v) in [
            ("Type", "Application"),
            ("Name", "RustXMALab"),
            ("GenericName", "X-ray motion analysis"),
            (
                "Comment",
                "Marker tracking and 3D reconstruction from X-ray video (XROMM)",
            ),
            ("Exec", "rustxmalab %f"),
            ("Icon", "rustxmalab"),
            ("Terminal", "false"),
            ("StartupWMClass", "rustxmalab"),
            ("Categories", "Science;Biology;"),
            ("Keywords", "XMALab;XROMM;X-ray;marker;tracking;"),
        ] {
            assert_eq!(keys.get(k), Some(&v), "{k}");
        }
        // The program file and the application id are the names the entry uses.
        assert!(PROGRAMS.contains(&("xma-app", "rustxmalab")));
    }
}
