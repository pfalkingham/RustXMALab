//! Dependency policy check over the JSON from `cargo metadata --format-version 1`.
//!
//! Policy (PLAN.md section 3, principle 4): no package in the resolved graph,
//! other than allowlisted names, may
//! (a) have a `links` key, (b) be named `*-sys`, `*_sys` or `*-sys-*` (hyphen
//! and underscore are the same to crates.io), or (c) have a build script that
//! depends on a native build tool crate: directly (normal or build dependency),
//! or anywhere in the libraries its build dependencies pull in. Workspace members
//! are checked too, and the allowlist never excuses a member.
//!
//! The Rust sources of the workspace's own members are scanned too
//! ([`link_attributes`]): a `#[link(...)]` attribute (or one inside
//! `#[cfg_attr(..., link(...))]`) outside comments and string literals, in
//! `src/` or `build.rs` next to a member's `Cargo.toml` (from `cargo metadata`'s
//! `manifest_path`), links a native library without a build script or a `links`
//! key, and is a violation the allowlist never excuses. The scan reads the text;
//! it does not catch an attribute a macro generates (`macro_rules!` with the
//! attribute in its body is caught, as text; one assembled from tokens or by a
//! procedural macro is not), sources outside `src/` (`tests/`, `examples/`,
//! `benches/`, a build script at a path other than `build.rs`, a file pulled in
//! with `include!` from elsewhere), a build script that prints
//! `cargo:rustc-link-lib`, or link flags given through `RUSTFLAGS` or
//! `.cargo/config.toml`.
//!
//! Architecture rules (PLAN.md section 4), not subject to the allowlist:
//! `xma-vision` must not depend on `xma-model`, `xma-oracle` depends on no
//! other workspace crate, and only `xma-app` may depend on the GUI toolkit
//! (`GUI_CRATES`).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Crates whose presence as a dependency of a build script means the package
/// compiles native code or locates a system library.
const NATIVE_BUILD_CRATES: &[&str] = &[
    "cc",
    "cmake",
    "bindgen",
    "pkg-config",
    "nasm-rs",
    "vcpkg",
    "cxx-build",
    "meson",
    // Wrappers over the above: `system-deps` over pkg-config, `autotools` over cc.
    "system-deps",
    "autotools",
];

/// GUI crates only `xma-app` may depend on.
const GUI_CRATES: &[&str] = &["eframe", "egui", "wgpu", "winit", "epaint", "egui-wgpu"];
const GUI_CRATE_OWNER: &str = "xma-app";

/// One entry of `deps-allowlist.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowEntry {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    Links {
        package: String,
        links: String,
    },
    SysCrate {
        package: String,
    },
    NativeBuildScript {
        package: String,
        tools: Vec<String>,
    },
    Architecture {
        package: String,
        forbidden: String,
        rule: String,
    },
    /// A `#[link(...)]` attribute in a member's own source.
    LinkAttribute {
        package: String,
        /// The file, relative to the member's folder.
        file: String,
        /// 1-based.
        line: usize,
    },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Violation::Links { package, links } => write!(
                f,
                "{package}: has `links = \"{links}\"`, so it links a native library"
            ),
            Violation::SysCrate { package } => write!(
                f,
                "{package}: name ends in `-sys` (or `_sys`, or contains `-sys-`), the convention \
                 for native library bindings"
            ),
            Violation::NativeBuildScript { package, tools } => write!(
                f,
                "{package}: has a build script that depends on native build tooling ({})",
                tools.join(", ")
            ),
            Violation::Architecture {
                package,
                forbidden,
                rule,
            } => write!(f, "{package}: depends on {forbidden}; {rule}"),
            Violation::LinkAttribute {
                package,
                file,
                line,
            } => write!(
                f,
                "{package}: {file} line {line}: a `#[link(...)]` attribute links a native library"
            ),
        }
    }
}

#[derive(Debug, Default)]
pub struct Report {
    pub packages_checked: usize,
    pub violations: Vec<Violation>,
}

struct Package<'a> {
    name: &'a str,
    links: Option<&'a str>,
    has_build_script: bool,
    manifest_path: Option<&'a str>,
}

/// An edge in the resolved graph.
struct Edge<'a> {
    to: &'a str,
    /// True if the edge is used by normal or build compilation.
    normal_or_build: bool,
    /// True if the edge is a build dependency only (used by the build script).
    build: bool,
    normal: bool,
}

fn str_field<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("cargo metadata: missing string field `{key}`"))
}

fn array_field<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("cargo metadata: missing array field `{key}`"))
}

/// Run all checks against `cargo metadata` JSON text.
pub fn check(metadata_json: &str, allowlist: &[AllowEntry]) -> Result<Report, String> {
    let meta: Value = serde_json::from_str(metadata_json)
        .map_err(|e| format!("cargo metadata: invalid JSON: {e}"))?;

    let mut packages: HashMap<&str, Package> = HashMap::new();
    for p in array_field(&meta, "packages")? {
        let has_build_script = array_field(p, "targets")?.iter().any(|t| {
            t.get("kind")
                .and_then(Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("custom-build")))
        });
        packages.insert(
            str_field(p, "id")?,
            Package {
                name: str_field(p, "name")?,
                links: p.get("links").and_then(Value::as_str),
                has_build_script,
                manifest_path: p.get("manifest_path").and_then(Value::as_str),
            },
        );
    }

    let members: HashSet<&str> = array_field(&meta, "workspace_members")?
        .iter()
        .filter_map(Value::as_str)
        .collect();

    let resolve = meta
        .get("resolve")
        .filter(|r| !r.is_null())
        .ok_or("cargo metadata: no `resolve` graph (was --no-deps used?)")?;

    let mut graph: HashMap<&str, Vec<Edge>> = HashMap::new();
    for node in array_field(resolve, "nodes")? {
        let id = str_field(node, "id")?;
        let mut edges = Vec::new();
        for dep in array_field(node, "deps")? {
            let kinds: Vec<Option<&str>> = dep
                .get("dep_kinds")
                .and_then(Value::as_array)
                .map(|ks| {
                    ks.iter()
                        .map(|k| k.get("kind").and_then(Value::as_str))
                        .collect()
                })
                .unwrap_or_else(|| vec![None]);
            let normal = kinds.contains(&None);
            let build = kinds.contains(&Some("build"));
            edges.push(Edge {
                to: str_field(dep, "pkg")?,
                normal_or_build: normal || build,
                build,
                normal,
            });
        }
        graph.insert(id, edges);
    }

    let name_of = |id: &str| -> Result<&str, String> {
        packages
            .get(id)
            .map(|p| p.name)
            .ok_or_else(|| format!("cargo metadata: unknown package id {id}"))
    };
    let allowed: HashSet<&str> = allowlist.iter().map(|e| e.name.as_str()).collect();

    let mut report = Report::default();
    let mut node_ids: Vec<&str> = graph.keys().copied().collect();
    node_ids.sort_unstable();

    // Native-code policy, on every package including the workspace's own.
    for &id in &node_ids {
        report.packages_checked += 1;
        let pkg = packages
            .get(id)
            .ok_or_else(|| format!("cargo metadata: unknown package id {id}"))?;
        if !members.contains(id) && allowed.contains(pkg.name) {
            continue;
        }
        if let Some(links) = pkg.links {
            report.violations.push(Violation::Links {
                package: pkg.name.to_string(),
                links: links.to_string(),
            });
        }
        if is_sys_name(pkg.name) {
            report.violations.push(Violation::SysCrate {
                package: pkg.name.to_string(),
            });
        }
        if pkg.has_build_script {
            let mut tools = BTreeSet::new();
            for e in &graph[id] {
                if e.normal || e.build {
                    let dep = name_of(e.to)?;
                    if is_native_build_crate(dep) {
                        tools.insert(dep.to_string());
                    }
                }
            }
            // The build script also links every library its build dependencies
            // pull in, so a wrapper over a tool counts as the tool.
            for dep in build_closure_names(id, &graph, &packages) {
                if is_native_build_crate(dep) {
                    tools.insert(dep.to_string());
                }
            }
            if !tools.is_empty() {
                report.violations.push(Violation::NativeBuildScript {
                    package: pkg.name.to_string(),
                    tools: tools.into_iter().collect(),
                });
            }
        }
    }

    // Architecture rules, checked on every workspace member.
    let mut member_ids: Vec<&str> = members.iter().copied().collect();
    member_ids.sort_unstable();

    // `#[link(...)]` in the members' own sources; no allowlist. A member without a
    // `manifest_path` (only in hand-written test graphs) has no sources to read.
    for &id in &member_ids {
        let pkg = packages
            .get(id)
            .ok_or_else(|| format!("cargo metadata: unknown package id {id}"))?;
        if let Some(manifest) = pkg.manifest_path {
            report
                .violations
                .extend(member_link_attributes(pkg.name, Path::new(manifest))?);
        }
    }
    for &id in &member_ids {
        let name = name_of(id)?;
        let reachable = reachable_names(id, &graph, &packages);
        if name == "xma-vision" && reachable.contains("xma-model") {
            report.violations.push(Violation::Architecture {
                package: name.to_string(),
                forbidden: "xma-model".to_string(),
                rule: "xma-vision must not depend on xma-model".to_string(),
            });
        }
        if name == "xma-oracle" {
            let mut others: Vec<&str> = member_ids
                .iter()
                .filter(|&&m| m != id)
                .filter_map(|m| packages.get(m).map(|p| p.name))
                .filter(|m| reachable.contains(m))
                .collect();
            others.sort_unstable();
            for other in others {
                report.violations.push(Violation::Architecture {
                    package: name.to_string(),
                    forbidden: other.to_string(),
                    rule: "xma-oracle depends on no other workspace crate".to_string(),
                });
            }
        }
        if name != GUI_CRATE_OWNER {
            for gui in GUI_CRATES {
                if reachable.iter().any(|r| same_name(r, gui)) {
                    report.violations.push(Violation::Architecture {
                        package: name.to_string(),
                        forbidden: (*gui).to_string(),
                        rule: format!("only {GUI_CRATE_OWNER} may depend on the GUI toolkit"),
                    });
                }
            }
        }
    }

    Ok(report)
}

/// Crate names are the same to crates.io whether written with `-` or `_`.
fn same_name(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .all(|(x, y)| x == y || (matches!(x, b'-' | b'_') && matches!(y, b'-' | b'_')))
}

/// True for the names of native-library binding crates: `*-sys`, `*_sys`, and
/// the infix forms `*-sys-*`, `*_sys_*` (in any mix of `-` and `_`).
fn is_sys_name(name: &str) -> bool {
    let n = name.replace('_', "-");
    n.ends_with("-sys") || n.contains("-sys-")
}

fn is_native_build_crate(name: &str) -> bool {
    NATIVE_BUILD_CRATES.iter().any(|t| same_name(name, t))
}

/// Names of the packages a build script of `start` links: its build
/// dependencies, then their normal dependencies transitively. (A build
/// dependency's own build dependencies serve that dependency's build script,
/// which is checked as a package of its own.)
fn build_closure_names<'a>(
    start: &'a str,
    graph: &HashMap<&'a str, Vec<Edge<'a>>>,
    packages: &HashMap<&'a str, Package<'a>>,
) -> Vec<&'a str> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut stack: Vec<&str> = graph
        .get(start)
        .map(|es| es.iter().filter(|e| e.build).map(|e| e.to).collect())
        .unwrap_or_default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Some(edges) = graph.get(id) {
            stack.extend(edges.iter().filter(|e| e.normal).map(|e| e.to));
        }
    }
    let mut names: Vec<&str> = seen
        .iter()
        .filter_map(|id| packages.get(id).map(|p| p.name))
        .collect();
    names.sort_unstable();
    names
}

/// Names of all packages reachable from `start`: its direct dependencies of any
/// kind (including dev), then normal and build dependencies transitively.
fn reachable_names<'a>(
    start: &'a str,
    graph: &HashMap<&'a str, Vec<Edge<'a>>>,
    packages: &HashMap<&'a str, Package<'a>>,
) -> HashSet<&'a str> {
    let mut seen: HashSet<&str> = HashSet::new();
    let mut stack: Vec<&str> = graph
        .get(start)
        .map(|es| es.iter().map(|e| e.to).collect())
        .unwrap_or_default();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Some(edges) = graph.get(id) {
            stack.extend(edges.iter().filter(|e| e.normal_or_build).map(|e| e.to));
        }
    }
    seen.iter()
        .filter_map(|id| packages.get(id).map(|p| p.name))
        .collect()
}

/// The `#[link(...)]` attributes in the sources of the member whose manifest is
/// `manifest`: every `.rs` file under `src/` next to it, and `build.rs`.
fn member_link_attributes(package: &str, manifest: &Path) -> Result<Vec<Violation>, String> {
    let dir = manifest.parent().unwrap_or(Path::new("."));
    let mut files = Vec::new();
    rust_files(&dir.join("src"), &mut files)?;
    let build = dir.join("build.rs");
    if build.is_file() {
        files.push(build);
    }
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f)
            .map_err(|e| format!("depcheck: cannot read {}: {e}", f.display()))?;
        let rel = f.strip_prefix(dir).unwrap_or(&f).display().to_string();
        for line in link_attributes(&text) {
            out.push(Violation::LinkAttribute {
                package: package.to_string(),
                file: rel.clone(),
                line,
            });
        }
    }
    Ok(out)
}

/// The `.rs` files under `dir`, recursively (nothing when `dir` does not exist).
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if !dir.is_dir() {
        return Ok(());
    }
    let entries = std::fs::read_dir(dir)
        .map_err(|e| format!("depcheck: cannot list {}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|e| format!("depcheck: cannot list {}: {e}", dir.display()))?
            .path();
        if path.is_dir() {
            rust_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The 1-based lines of the `#[link(...)]` attributes in Rust source `src`
/// (also `#![link(...)]`, and `link(...)` inside `#[cfg_attr(...)]`), outside
/// comments, string literals and character literals. `link_name`,
/// `link_section` and the like are not `link`.
pub fn link_attributes(src: &str) -> Vec<usize> {
    let code: Vec<char> = strip_comments_and_literals(src).chars().collect();
    let ident = |c: char| c.is_alphanumeric() || c == '_';
    let skip_ws = |mut i: usize| {
        while i < code.len() && code[i].is_whitespace() {
            i += 1;
        }
        i
    };
    let line_of = |i: usize| code[..i].iter().filter(|&&c| c == '\n').count() + 1;
    let mut lines = Vec::new();
    let mut i = 0;
    while i < code.len() {
        if code[i] != '#' {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = skip_ws(i + 1);
        if code.get(j) == Some(&'!') {
            j = skip_ws(j + 1);
        }
        if code.get(j) != Some(&'[') {
            i += 1;
            continue;
        }
        let open = j;
        j = skip_ws(j + 1);
        let name_start = j;
        while j < code.len() && ident(code[j]) {
            j += 1;
        }
        let name: String = code[name_start..j].iter().collect();
        let after = skip_ws(j);
        if name == "link" && code.get(after) == Some(&'(') {
            lines.push(line_of(start));
        } else if name == "cfg_attr" && code.get(after) == Some(&'(') {
            // The attribute's text up to its closing `]`.
            let mut depth = 0usize;
            let mut end = open;
            while end < code.len() {
                match code[end] {
                    '[' | '(' => depth += 1,
                    ']' | ')' => {
                        depth = depth.saturating_sub(1);
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                end += 1;
            }
            let mut k = after;
            while k + 4 <= end {
                if code[k..k + 4] == ['l', 'i', 'n', 'k']
                    && !ident(code[k - 1])
                    && code.get(k + 4).is_none_or(|&c| !ident(c))
                    && code.get(skip_ws(k + 4)) == Some(&'(')
                {
                    lines.push(line_of(start));
                    break;
                }
                k += 1;
            }
        }
        i = j.max(i + 1);
    }
    lines
}

/// `src` with comments (line, block and nested block comments, doc comments
/// included), string literals (plain, byte, C and raw) and character literals
/// replaced by spaces, line breaks kept.
fn strip_comments_and_literals(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let blank = |ch: char| if ch == '\n' { '\n' } else { ' ' };
    let ident = |ch: char| ch.is_alphanumeric() || ch == '_';
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        let next = c.get(i + 1).copied();
        if ch == '/' && next == Some('/') {
            while i < c.len() && c[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else if ch == '/' && next == Some('*') {
            let mut depth = 0usize;
            while i < c.len() {
                if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(c[i]));
                    i += 1;
                }
            }
        } else if let Some((open_len, hashes)) = raw_string_start(&c, i, ident) {
            // r"...", r#"..."#, br"...", cr#"..."#: ends at `"` and as many `#`.
            for &x in &c[i..i + open_len] {
                out.push(blank(x));
            }
            i += open_len;
            while i < c.len() {
                if c[i] == '"' && (1..=hashes).all(|h| c.get(i + h) == Some(&'#')) {
                    for _ in 0..=hashes {
                        out.push(' ');
                    }
                    i += hashes + 1;
                    break;
                }
                out.push(blank(c[i]));
                i += 1;
            }
        } else if ch == '"' {
            out.push(' ');
            i += 1;
            while i < c.len() {
                if c[i] == '\\' {
                    out.push(' ');
                    i += 1;
                    if i < c.len() {
                        out.push(blank(c[i]));
                        i += 1;
                    }
                } else if c[i] == '"' {
                    out.push(' ');
                    i += 1;
                    break;
                } else {
                    out.push(blank(c[i]));
                    i += 1;
                }
            }
        } else if ch == '\'' && (next == Some('\\') || c.get(i + 2) == Some(&'\'')) {
            // A character literal ('x', '\n', '\'', '\u{..}'); otherwise a lifetime.
            out.push(' ');
            i += 1;
            if c.get(i) == Some(&'\\') {
                out.push_str("  ");
                i += 2;
            }
            while i < c.len() && c[i] != '\'' {
                out.push(blank(c[i]));
                i += 1;
            }
            if i < c.len() {
                out.push(' ');
                i += 1;
            }
        } else {
            out.push(ch);
            i += 1;
        }
    }
    out
}

/// At `i`, the start of a raw string literal (`r`, `br` or `cr`, then `#`s,
/// then `"`) that is not the end of an identifier: its opening length and the
/// number of `#`s.
fn raw_string_start(c: &[char], i: usize, ident: impl Fn(char) -> bool) -> Option<(usize, usize)> {
    if i > 0 && ident(c[i - 1]) {
        return None;
    }
    let mut j = i;
    if matches!(c.get(j), Some('b' | 'c')) {
        j += 1;
    }
    if c.get(j) != Some(&'r') {
        return None;
    }
    j += 1;
    let mut hashes = 0;
    while c.get(j) == Some(&'#') {
        hashes += 1;
        j += 1;
    }
    (c.get(j) == Some(&'"')).then_some((j + 1 - i, hashes))
}

/// Parse `deps-allowlist.toml`: a sequence of `[[allow]]` tables, each with
/// `name = "..."` and a non-empty `reason = "..."`. Only this subset of TOML is
/// accepted, which keeps xtask free of a TOML dependency.
pub fn parse_allowlist(text: &str) -> Result<Vec<AllowEntry>, String> {
    let mut entries: Vec<(Option<String>, Option<String>)> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        let lineno = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[allow]]" {
            entries.push((None, None));
            continue;
        }
        let err = |what: &str| format!("deps-allowlist.toml line {lineno}: {what}: {raw}");
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| err("expected key = \"value\""))?;
        let value = value.trim();
        let value = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .ok_or_else(|| err("value must be a double-quoted string"))?
            .to_string();
        let entry = entries
            .last_mut()
            .ok_or_else(|| err("key outside an [[allow]] table"))?;
        match key.trim() {
            "name" => entry.0 = Some(value),
            "reason" => entry.1 = Some(value),
            other => return Err(err(&format!("unknown key `{other}`"))),
        }
    }
    entries
        .into_iter()
        .enumerate()
        .map(|(i, e)| match e {
            (Some(name), Some(reason)) if !name.is_empty() && !reason.trim().is_empty() => {
                Ok(AllowEntry { name, reason })
            }
            _ => Err(format!(
                "deps-allowlist.toml: entry {} needs a non-empty `name` and `reason`",
                i + 1
            )),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLEAN: &str = include_str!("../fixtures/clean.json");
    const SYS: &str = include_str!("../fixtures/sys-crate.json");
    const CC: &str = include_str!("../fixtures/build-script-cc.json");
    const ARCH: &str = include_str!("../fixtures/architecture.json");

    #[test]
    fn clean_graph_passes() {
        let r = check(CLEAN, &[]).unwrap();
        assert_eq!(r.violations, vec![]);
        // The member `app` is checked too (review 2026-11-20, finding 2a).
        assert_eq!(r.packages_checked, 3);
    }

    #[test]
    fn sys_crate_with_links_fails() {
        let r = check(SYS, &[]).unwrap();
        assert_eq!(
            r.violations,
            vec![
                Violation::Links {
                    package: "libfoo-sys".into(),
                    links: "foo".into()
                },
                Violation::SysCrate {
                    package: "libfoo-sys".into()
                },
            ]
        );
    }

    #[test]
    fn build_script_using_cc_fails() {
        let r = check(CC, &[]).unwrap();
        assert_eq!(
            r.violations,
            vec![Violation::NativeBuildScript {
                package: "fastzip".into(),
                tools: vec!["cc".into()]
            }]
        );
        // `autocfg` in a build script is fine, and `cc` itself has no build script.
        assert!(
            r.violations
                .iter()
                .all(|v| !v.to_string().contains("numcrate"))
        );
    }

    #[test]
    fn allowlisted_package_is_exempt() {
        let allow = parse_allowlist(
            "# approved by the user\n[[allow]]\nname = \"libfoo-sys\"\nreason = \"test\"\n",
        )
        .unwrap();
        assert_eq!(allow.len(), 1);
        let r = check(SYS, &allow).unwrap();
        assert_eq!(r.violations, vec![]);
    }

    #[test]
    fn architecture_rules() {
        let r = check(ARCH, &[]).unwrap();
        let msgs: Vec<String> = r.violations.iter().map(ToString::to_string).collect();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        assert!(msgs[0].starts_with("xma-engine: depends on egui"));
        assert!(msgs[1].starts_with("xma-vision: depends on xma-model"));
    }

    const MEMBER_NATIVE: &str = include_str!("../fixtures/member-native.json");
    const SYS_UNDERSCORE: &str = include_str!("../fixtures/sys-underscore.json");
    const BUILD_CLOSURE: &str = include_str!("../fixtures/build-closure.json");
    const ARCH_MORE: &str = include_str!("../fixtures/architecture-more.json");

    /// Review 2026-11-20 (parity-xtask finding 2a): the native-code rules
    /// apply to workspace members, and the allowlist never excuses a member.
    #[test]
    fn review12_member_with_native_code_fails() {
        let expected = vec![
            Violation::Links {
                package: "xma-video".into(),
                links: "avcodec".into(),
            },
            Violation::NativeBuildScript {
                package: "xma-vision".into(),
                tools: vec!["cc".into()],
            },
        ];
        let r = check(MEMBER_NATIVE, &[]).unwrap();
        assert_eq!(r.violations, expected);
        let allow = parse_allowlist(
            "[[allow]]\nname = \"xma-vision\"\nreason = \"t\"\n\
             [[allow]]\nname = \"xma-video\"\nreason = \"t\"\n",
        )
        .unwrap();
        assert_eq!(check(MEMBER_NATIVE, &allow).unwrap().violations, expected);
    }

    /// Finding 2b: `_sys` and the `-sys-` / `_sys_` infix forms are binding names.
    #[test]
    fn review12_sys_name_with_underscore_or_infix_fails() {
        let r = check(SYS_UNDERSCORE, &[]).unwrap();
        let flagged: Vec<&str> = r
            .violations
            .iter()
            .map(|v| match v {
                Violation::SysCrate { package } => package.as_str(),
                other => panic!("unexpected {other}"),
            })
            .collect();
        // Sorted by package id; `sysinfo`, `sys-locale` and `mysys` are not binding names.
        assert_eq!(
            flagged,
            vec!["bar_sys_util", "foo-sys-util", "glutin_egl_sys"]
        );
    }

    /// Finding 2c: a native build tool reached through a wrapper in the build
    /// dependencies counts; a build dependency's own build tool does not count
    /// against the package that uses it (it is flagged on its own).
    #[test]
    fn review12_build_tool_behind_wrapper_fails() {
        let r = check(BUILD_CLOSURE, &[]).unwrap();
        assert_eq!(
            r.violations,
            vec![
                Violation::NativeBuildScript {
                    package: "autoish".into(),
                    tools: vec!["cc".into()],
                },
                Violation::NativeBuildScript {
                    package: "gtkish".into(),
                    tools: vec!["pkg-config".into(), "system-deps".into()],
                },
                Violation::NativeBuildScript {
                    package: "helper".into(),
                    tools: vec!["cc".into()],
                },
            ]
        );
    }

    /// Finding 2d: `xma-oracle` depends on no other workspace crate, and the
    /// GUI toolkit includes `winit`, `epaint` and `egui-wgpu`.
    #[test]
    fn review12_more_architecture_rules() {
        let r = check(ARCH_MORE, &[]).unwrap();
        let mut msgs: Vec<String> = r.violations.iter().map(ToString::to_string).collect();
        msgs.sort();
        let starts: Vec<&str> = msgs.iter().map(|m| m.split(';').next().unwrap()).collect();
        assert_eq!(
            starts,
            vec![
                "xma-engine: depends on egui-wgpu",
                "xma-engine: depends on epaint",
                "xma-io: depends on winit",
                "xma-oracle: depends on xma-model",
            ],
            "{msgs:?}"
        );
    }

    /// Review 2026-11-20, third round: a `#[link(...)]` attribute in a member's
    /// own source is a violation the allowlist does not excuse.
    #[test]
    fn review12c_link_attribute_in_member_source_fails() {
        let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");
        let meta =
            include_str!("../fixtures/member-link-attr.json").replace("@FIXTURES@", fixtures);
        let expected = vec![
            Violation::LinkAttribute {
                package: "xma-video".into(),
                file: "build.rs".into(),
                line: 3,
            },
            Violation::LinkAttribute {
                package: "xma-video".into(),
                file: "src/ffi/mod.rs".into(),
                line: 1,
            },
            Violation::LinkAttribute {
                package: "xma-video".into(),
                file: "src/lib.rs".into(),
                line: 9,
            },
        ];
        assert_eq!(check(&meta, &[]).unwrap().violations, expected);
        let allow = parse_allowlist("[[allow]]\nname = \"xma-video\"\nreason = \"t\"\n").unwrap();
        assert_eq!(check(&meta, &allow).unwrap().violations, expected);
    }

    #[test]
    fn review12c_link_attribute_scanner() {
        let src = r##"
// #[link(name = "a")]
/// #[link(name = "doc")]
/* outer /* #[link(name = "nested")] */ still a comment #[link(name = "b")] */
const S: &str = "#[link(name = \"c\")]";
const R: &str = r#"#[link(name = "d")]"#;
const C: char = '"';
fn f<'a>(x: &'a str) -> &'a str { x }
#[link_name = "e"]
#[link_section = ".f"]
#[cfg_attr(unix, link_name = "g")]
#[ link ( name = "m" ) ]
#![link(name = "inner")]
#[cfg_attr(all(unix, not(target_os = "macos")), link(name = "z"))]
#[cfg_attr(feature = "link", inline)]
"##;
        assert_eq!(link_attributes(src), vec![12, 13, 14]);
        assert_eq!(
            link_attributes("#[link(name = \"x\")] extern \"C\" {}"),
            vec![1]
        );
        assert!(link_attributes("").is_empty());
        assert!(link_attributes("#[").is_empty());
        assert!(link_attributes("#[cfg_attr(x, link").is_empty());
        assert!(link_attributes("\"unterminated #[link(").is_empty());
    }

    #[test]
    fn allowlist_rejects_missing_reason() {
        assert!(parse_allowlist("[[allow]]\nname = \"x\"\n").is_err());
        assert!(parse_allowlist("name = \"x\"\n").is_err());
        assert_eq!(parse_allowlist("# empty\n").unwrap(), vec![]);
    }
}
