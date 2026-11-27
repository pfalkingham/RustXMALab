//! `cargo xtask licenses`: `THIRD-PARTY-LICENSES.txt`, the licences of the Rust crates the
//! two programs are built from.
//!
//! The crates are those `cargo metadata` resolves for `xma-app` (`rustxmalab`) and `xma-cli`
//! (`xma`) on the host platform, following normal and build dependencies (not
//! dev-dependencies). Each is marked as linked into the programs, or used only while
//! building them (build scripts, procedural macros and what they depend on).
//!
//! For every crate the file gives its name, version, the licence expression of its
//! `Cargo.toml`, the licence used where the expression offers a choice (one compatible with
//! distributing the programs under GPL-3.0-only), its authors, the copyright lines its licence
//! files give, and the full text of every licence file it ships (read from the crate's
//! source in Cargo's registry). Identical texts are stored once and referred to by number.
//! A licence that none of the crate's top-level files carries gets the standard text from
//! `xtask/dist/licenses/` (MIT with a copyright line made from the `Cargo.toml` authors).
//!
//! Generation fails, listing the crates, when an expression names a licence this module
//! does not know, offers nothing compatible with GPL-3.0-only, or uses a font licence in a
//! crate not reviewed for it ([`FONT_CRATES`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use serde_json::Value;

/// The packages whose dependency graphs are covered, with the program each builds.
pub const PROGRAMS: &[(&str, &str)] = &[("xma-app", "rustxmalab"), ("xma-cli", "xma")];

/// Crates allowed to carry a font licence: their fonts are embedded as data (NOTICE.md,
/// "Fonts"). Any other crate with a font licence makes generation fail until reviewed.
const FONT_CRATES: &[&str] = &["epaint_default_fonts"];

/// How a licence relates to distributing the programs under GPL-3.0-only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// May be combined with GPL-3.0-only code.
    Compatible,
    /// A font licence: the fonts are data distributed with the program under their own
    /// licence (only in [`FONT_CRATES`]).
    Font,
    /// Cannot be combined with GPL-3.0-only code.
    Incompatible,
}

/// The licences known, in order of preference when an expression offers a choice, with a
/// phrase that identifies the licence's full text (matched case-insensitively after collapsing
/// white space). The phrase must be one only the full text has, not a mention of the licence
/// such as a NOTICE's "Licensed under the Apache License"; [`DISTINCT`] adds the phrases that
/// tell apart licences whose texts share one.
const KNOWN: &[(&str, Class, &str)] = &[
    (
        "MIT",
        Class::Compatible,
        "permission is hereby granted, free of charge",
    ),
    (
        "Apache-2.0",
        Class::Compatible,
        "terms and conditions for use, reproduction, and distribution",
    ),
    (
        "Zlib",
        Class::Compatible,
        "altered source versions must be plainly marked",
    ),
    (
        "BSD-3-Clause",
        Class::Compatible,
        "redistribution and use in source and binary forms",
    ),
    (
        "BSD-2-Clause",
        Class::Compatible,
        "redistribution and use in source and binary forms",
    ),
    (
        "ISC",
        Class::Compatible,
        "with or without fee is hereby granted",
    ),
    (
        "0BSD",
        Class::Compatible,
        "with or without fee is hereby granted",
    ),
    (
        "Unlicense",
        Class::Compatible,
        "this is free and unencumbered software released into the public domain",
    ),
    (
        "Apache-2.0 WITH LLVM-exception",
        Class::Compatible,
        "llvm exceptions to the apache 2.0 license",
    ),
    ("Unicode-3.0", Class::Compatible, "unicode license v3"),
    ("BSL-1.0", Class::Compatible, "boost software license"),
    ("CC0-1.0", Class::Compatible, "cc0 1.0 universal"),
    (
        "MPL-2.0",
        Class::Compatible,
        "mozilla public license version 2.0",
    ),
    (
        "LGPL-2.1-or-later",
        Class::Compatible,
        "gnu lesser general public license",
    ),
    ("OFL-1.1", Class::Font, "sil open font license"),
    ("Ubuntu-font-1.0", Class::Font, "ubuntu font licence"),
    ("GPL-2.0-only", Class::Incompatible, ""),
    ("GPL-2.0", Class::Incompatible, ""),
    ("LGPL-2.0-only", Class::Incompatible, ""),
];

/// For licences whose texts share their [`KNOWN`] phrase: phrases the text must also have,
/// and phrases it must not have. BSD-3-Clause has the third (no endorsement) clause that
/// BSD-2-Clause lacks; ISC has the condition of keeping the notice that 0BSD drops.
const DISTINCT: &[(&str, &[&str], &[&str])] = &[
    (
        "BSD-3-Clause",
        &["endorse or promote products derived from this software"],
        &[],
    ),
    (
        "BSD-2-Clause",
        &[],
        &["endorse or promote products derived from this software"],
    ),
    (
        "ISC",
        &[
            "provided that the above copyright notice and this permission notice appear in all copies",
        ],
        &[],
    ),
    (
        "0BSD",
        &[],
        &[
            "provided that the above copyright notice and this permission notice appear in all copies",
        ],
    ),
];

/// True if `text` (one file, after [`squash`]) is the full text of licence `id`.
fn carries_text(id: &str, text: &str) -> bool {
    let Some(marker) = known(id).map(|k| k.2).filter(|m| !m.is_empty()) else {
        return false;
    };
    if !text.contains(marker) {
        return false;
    }
    DISTINCT
        .iter()
        .filter(|d| d.0 == id)
        .all(|(_, need, deny)| {
            need.iter().all(|p| text.contains(p)) && !deny.iter().any(|p| text.contains(p))
        })
}

fn known(id: &str) -> Option<(usize, Class, &'static str)> {
    KNOWN
        .iter()
        .position(|(k, _, _)| *k == id)
        .map(|i| (i, KNOWN[i].1, KNOWN[i].2))
}

// ---------------------------------------------------------------------------------------
// Licence expressions
// ---------------------------------------------------------------------------------------

/// A licence expression (SPDX, with the older `/` for OR).
#[derive(Debug, PartialEq)]
enum Expr {
    /// A licence, or `licence WITH exception`.
    Id(String),
    And(Vec<Expr>),
    Or(Vec<Expr>),
}

fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_whitespace() || c == '(' || c == ')' || c == '/' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            match c {
                '(' | ')' => out.push(c.to_string()),
                '/' => out.push("OR".into()),
                _ => {}
            }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn parse_expr(s: &str) -> Result<Expr, String> {
    let toks = tokenize(s);
    let mut pos = 0;
    let e = parse_or(&toks, &mut pos)?;
    if pos != toks.len() {
        return Err(format!("cannot parse licence expression '{s}'"));
    }
    Ok(e)
}

fn parse_or(t: &[String], pos: &mut usize) -> Result<Expr, String> {
    let mut alts = vec![parse_and(t, pos)?];
    while t.get(*pos).is_some_and(|x| x.eq_ignore_ascii_case("OR")) {
        *pos += 1;
        alts.push(parse_and(t, pos)?);
    }
    Ok(if alts.len() == 1 {
        alts.pop().unwrap()
    } else {
        Expr::Or(alts)
    })
}

fn parse_and(t: &[String], pos: &mut usize) -> Result<Expr, String> {
    let mut parts = vec![parse_atom(t, pos)?];
    while t.get(*pos).is_some_and(|x| x.eq_ignore_ascii_case("AND")) {
        *pos += 1;
        parts.push(parse_atom(t, pos)?);
    }
    Ok(if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        Expr::And(parts)
    })
}

fn parse_atom(t: &[String], pos: &mut usize) -> Result<Expr, String> {
    let tok = t.get(*pos).ok_or("licence expression ends early")?;
    *pos += 1;
    if tok == "(" {
        let e = parse_or(t, pos)?;
        if t.get(*pos).map(String::as_str) != Some(")") {
            return Err("unbalanced parenthesis in licence expression".into());
        }
        *pos += 1;
        return Ok(e);
    }
    if tok == ")"
        || ["AND", "OR", "WITH"]
            .iter()
            .any(|k| tok.eq_ignore_ascii_case(k))
    {
        return Err(format!("unexpected '{tok}' in licence expression"));
    }
    let mut id = tok.clone();
    if t.get(*pos).is_some_and(|x| x.eq_ignore_ascii_case("WITH")) {
        let exc = t.get(*pos + 1).ok_or("WITH without an exception")?;
        id = format!("{id} WITH {exc}");
        *pos += 2;
    }
    Ok(Expr::Id(id))
}

/// The licences to comply with: for OR, the most preferred alternative that is usable; for
/// AND, all parts. The error lists what is unknown or incompatible.
fn choose(e: &Expr) -> Result<Vec<String>, String> {
    match e {
        Expr::Id(id) => match known(id) {
            Some((_, Class::Compatible | Class::Font, _)) => Ok(vec![id.clone()]),
            Some((_, Class::Incompatible, _)) => {
                Err(format!("{id} is not compatible with GPL-3.0-only"))
            }
            None => Err(format!("{id} is not a licence xtask knows")),
        },
        Expr::And(parts) => {
            let mut all = Vec::new();
            for p in parts {
                for id in choose(p)? {
                    if !all.contains(&id) {
                        all.push(id);
                    }
                }
            }
            Ok(all)
        }
        Expr::Or(alts) => {
            let mut best: Option<(usize, Vec<String>)> = None;
            let mut errors = Vec::new();
            for a in alts {
                match choose(a) {
                    Ok(ids) => {
                        let rank = ids
                            .iter()
                            .map(|id| known(id).map_or(usize::MAX, |k| k.0))
                            .max()
                            .unwrap_or(usize::MAX);
                        if best.as_ref().is_none_or(|(r, _)| rank < *r) {
                            best = Some((rank, ids));
                        }
                    }
                    Err(e) => errors.push(e),
                }
            }
            best.map(|(_, ids)| ids).ok_or_else(|| errors.join("; "))
        }
    }
}

// ---------------------------------------------------------------------------------------
// The dependency graph
// ---------------------------------------------------------------------------------------

/// One third-party crate of the graph.
#[derive(Debug, Clone)]
pub struct Crate {
    pub name: String,
    pub version: String,
    license: Option<String>,
    authors: Vec<String>,
    dir: PathBuf,
    /// The programs it is linked into (empty: used only while building).
    pub linked_into: Vec<&'static str>,
}

/// The third-party crates of the two programs' graphs, sorted by name and version.
pub fn crates(metadata: &str) -> Result<Vec<Crate>, String> {
    let v: Value =
        serde_json::from_str(metadata).map_err(|e| format!("cargo metadata output: {e}"))?;
    let packages: HashMap<&str, &Value> = v["packages"]
        .as_array()
        .ok_or("cargo metadata: no packages")?
        .iter()
        .filter_map(|p| Some((p["id"].as_str()?, p)))
        .collect();
    let nodes: HashMap<&str, &Value> = v["resolve"]["nodes"]
        .as_array()
        .ok_or("cargo metadata: no resolve graph")?
        .iter()
        .filter_map(|n| Some((n["id"].as_str()?, n)))
        .collect();
    let members: BTreeSet<&str> = v["workspace_members"]
        .as_array()
        .ok_or("cargo metadata: no workspace_members")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let is_proc_macro = |id: &str| {
        packages.get(id).is_some_and(|p| {
            p["targets"].as_array().is_some_and(|ts| {
                ts.iter().any(|t| {
                    t["kind"]
                        .as_array()
                        .is_some_and(|k| k.iter().any(|k| k == "proc-macro"))
                })
            })
        })
    };
    let deps = |id: &str, build_too: bool| -> Vec<String> {
        let mut out = Vec::new();
        if let Some(deps) = nodes.get(id).and_then(|n| n["deps"].as_array()) {
            for d in deps {
                let usable = d["dep_kinds"].as_array().is_some_and(|ks| {
                    ks.iter().any(|k| {
                        k["kind"].is_null() || (build_too && k["kind"].as_str() == Some("build"))
                    })
                });
                if let (true, Some(p)) = (usable, d["pkg"].as_str()) {
                    out.push(p.to_string());
                }
            }
        }
        out
    };

    let mut all: BTreeSet<String> = BTreeSet::new();
    let mut linked: BTreeMap<String, Vec<&'static str>> = BTreeMap::new();
    for (pkg, program) in PROGRAMS {
        let root = members
            .iter()
            .find(|id| packages.get(*id).and_then(|p| p["name"].as_str()) == Some(*pkg))
            .ok_or_else(|| format!("package {pkg} is not in the workspace"))?
            .to_string();
        // Everything needed to build the program.
        let mut stack = vec![root.clone()];
        while let Some(id) = stack.pop() {
            if all.insert(id.clone()) {
                stack.extend(deps(&id, true));
            }
        }
        // What ends up in it: normal dependencies, not through procedural macros.
        let mut seen = BTreeSet::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if is_proc_macro(&id) || !seen.insert(id.clone()) {
                continue;
            }
            linked.entry(id.clone()).or_default().push(program);
            stack.extend(deps(&id, false));
        }
    }

    let mut out = Vec::new();
    for id in &all {
        if members.contains(id.as_str()) {
            continue;
        }
        let p = packages
            .get(id.as_str())
            .ok_or_else(|| format!("cargo metadata: unknown package {id}"))?;
        let manifest = p["manifest_path"]
            .as_str()
            .ok_or_else(|| format!("{id}: no manifest_path"))?;
        out.push(Crate {
            name: p["name"].as_str().unwrap_or_default().to_string(),
            version: p["version"].as_str().unwrap_or_default().to_string(),
            license: p["license"].as_str().map(str::to_string),
            authors: p["authors"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default(),
            dir: Path::new(manifest)
                .parent()
                .ok_or_else(|| format!("{id}: bad manifest_path"))?
                .to_path_buf(),
            linked_into: linked.get(id).cloned().unwrap_or_default(),
        });
    }
    out.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then_with(|| version_key(&a.version).cmp(&version_key(&b.version)))
            .then_with(|| a.version.cmp(&b.version))
    });
    Ok(out)
}

fn version_key(v: &str) -> Vec<u64> {
    v.split(['.', '-', '+'])
        .map(|p| p.parse().unwrap_or(0))
        .collect()
}

// ---------------------------------------------------------------------------------------
// Licence files
// ---------------------------------------------------------------------------------------

/// Folders not searched for licence files: they hold tests, examples and build output,
/// none of which is part of the programs.
const SKIP_DIRS: &[&str] = &[
    "target",
    ".git",
    ".github",
    "tests",
    "test",
    "testdata",
    "autotests",
    "benches",
    "examples",
    "fuzz",
];

/// File extensions that are code or data, never a licence text.
const CODE_EXTS: &[&str] = &[
    "rs", "toml", "json", "lock", "xml", "html", "yml", "yaml", "sh", "py", "c", "h", "cpp", "in",
    "cmake",
];

/// One licence file of a crate, its path relative to the crate's folder.
struct LicenceFile {
    rel: String,
    text: String,
    top_level: bool,
}

fn licence_files(dir: &Path) -> Result<Vec<LicenceFile>, String> {
    let mut out = Vec::new();
    walk(dir, dir, &mut out)?;
    out.sort_by(|a, b| {
        b.top_level
            .cmp(&a.top_level)
            .then_with(|| a.rel.cmp(&b.rel))
    });
    Ok(out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<LicenceFile>) -> Result<(), String> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("cannot read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    let has_fonts = entries.iter().any(|p| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("ttf") || e.eq_ignore_ascii_case("otf"))
    });
    let dir_name = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    for p in entries {
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let lower = name.to_ascii_lowercase();
        if p.is_dir() {
            if !SKIP_DIRS.contains(&lower.as_str()) {
                walk(root, &p, out)?;
            }
            continue;
        }
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let by_name = [
            "license",
            "licence",
            "copying",
            "notice",
            "unlicense",
            "copyright",
            "authors",
        ]
        .iter()
        .any(|s| lower.starts_with(s))
            && !CODE_EXTS.contains(&ext.as_str());
        let in_licenses_dir = dir_name == "licenses" || dir_name == "licences";
        let font_text = has_fonts && ext == "txt";
        if !(by_name || in_licenses_dir || font_text) {
            continue;
        }
        let bytes = std::fs::read(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
        let text = normalise(&String::from_utf8_lossy(&bytes));
        if text.trim().is_empty() {
            continue;
        }
        let rel_path = p.strip_prefix(root).unwrap_or(&p);
        let rel = rel_path
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        out.push(LicenceFile {
            top_level: rel_path.components().count() == 1,
            rel,
            text,
        });
    }
    Ok(())
}

/// Unix line ends, no byte-order mark, no trailing blank lines, one final newline.
fn normalise(s: &str) -> String {
    let s = s.strip_prefix('\u{feff}').unwrap_or(s);
    let mut t = s.replace("\r\n", "\n").replace('\r', "\n");
    let trimmed = t.trim_end().len();
    t.truncate(trimmed);
    t.push('\n');
    t
}

/// Lower case with white space collapsed, for finding a licence's identifying phrase.
fn squash(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The copyright lines of a licence or notice file (not the template lines of licence
/// texts, such as Apache-2.0's `Copyright [yyyy] [name of copyright owner]`).
fn copyright_lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line
            .trim()
            .trim_start_matches(['/', '#', '*', '-', ';'])
            .trim();
        let lower = t.to_lowercase();
        let starts =
            lower.starts_with("copyright") || t.starts_with('©') || lower.starts_with("(c)");
        if !starts {
            continue;
        }
        let rest = lower
            .trim_start_matches("copyright")
            .trim_start_matches(['s', ':'])
            .trim_start();
        // A line opening with "(c)" is a notice only with a year after it (not a list
        // item such as Apache-2.0's "(c) You must retain ...").
        let paren_c_item = lower.starts_with("(c)")
            && !lower[3..]
                .trim_start()
                .starts_with(|c: char| c.is_ascii_digit());
        let looks_like_notice = !paren_c_item
            && (t.chars().any(|c| c.is_ascii_digit())
                || rest.starts_with("(c)")
                || rest.starts_with('©')
                || rest.starts_with("the "));
        let template = [
            "[yyyy]",
            "{yyyy}",
            "<year>",
            "[year]",
            "copyright notice",
            "copyright owner",
            "copyright holder",
            "copyright license",
            "copyright statement",
            "free software foundation",
            "copyright and permission notice",
            "copyright and related rights",
        ]
        .iter()
        .any(|p| lower.contains(p));
        if looks_like_notice && !template && !out.iter().any(|o: &String| o == t) {
            out.push(t.to_string());
        }
    }
    out
}

/// The `Cargo.toml` authors without e-mail addresses.
fn author_names(c: &Crate) -> Vec<String> {
    c.authors
        .iter()
        .map(|a| match a.find('<') {
            Some(i) => a[..i].trim().to_string(),
            None => a.trim().to_string(),
        })
        .filter(|a| !a.is_empty())
        .collect()
}

/// A standard licence text from `xtask/dist/licenses/<id>.txt`, without its leading
/// `#` comment lines.
fn standard_text(root: &Path, id: &str) -> Result<Option<String>, String> {
    let path = root.join("xtask/dist/licenses").join(format!("{id}.txt"));
    if !path.is_file() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let body: Vec<&str> = raw.lines().skip_while(|l| l.starts_with('#')).collect();
    Ok(Some(normalise(&body.join("\n"))))
}

// ---------------------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------------------

/// What [`generate`] found, for the report printed by the command and for tests.
pub struct Report {
    pub text: String,
    /// Per crate: name, version, licence used, whether a choice was made, the numbers of
    /// its texts, and the standard texts it was given.
    pub entries: Vec<Entry>,
    /// The stored texts, numbered from 1.
    pub texts: Vec<String>,
}

pub struct Entry {
    pub name: String,
    pub version: String,
    pub expression: String,
    pub used: String,
    pub chosen: bool,
    #[cfg_attr(not(test), allow(dead_code))]
    pub text_ids: Vec<usize>,
    pub standard: Vec<String>,
    pub linked: bool,
}

/// Builds `THIRD-PARTY-LICENSES.txt` from `cargo metadata` output.
pub fn generate(root: &Path, metadata: &str, target: &str) -> Result<Report, String> {
    let crates = crates(metadata)?;
    let mut problems = Vec::new();
    let mut texts: Vec<String> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut text_users: Vec<Vec<String>> = Vec::new();
    let mut store = |text: String, user: String| -> usize {
        let id = *index.entry(text.clone()).or_insert_with(|| {
            texts.push(text);
            text_users.push(Vec::new());
            texts.len()
        });
        text_users[id - 1].push(user);
        id
    };

    let mut entries = Vec::new();
    let mut blocks = Vec::new();
    for c in &crates {
        let label = format!("{} {}", c.name, c.version);
        let Some(expression) = c.license.clone() else {
            problems.push(format!(
                "{label}: Cargo.toml declares no licence expression"
            ));
            continue;
        };
        let used = match parse_expr(&expression).and_then(|e| choose(&e)) {
            Ok(u) => u,
            Err(e) => {
                problems.push(format!("{label}: '{expression}': {e}"));
                continue;
            }
        };
        if used
            .iter()
            .any(|id| known(id).is_some_and(|k| k.1 == Class::Font))
            && !FONT_CRATES.contains(&c.name.as_str())
        {
            problems.push(format!(
                "{label}: '{expression}' names a font licence; review the fonts and add the \
                 crate to FONT_CRATES in xtask/src/licenses.rs"
            ));
            continue;
        }
        let files = licence_files(&c.dir)?;
        // Each file on its own, so that one file's text cannot pass for another licence's
        // (BSD-2-Clause is recognised by a phrase BSD-3-Clause also has, less one it lacks).
        let top: Vec<String> = files
            .iter()
            .filter(|f| f.top_level)
            .map(|f| squash(&f.text))
            .collect();
        let below: Vec<String> = files
            .iter()
            .filter(|f| !f.top_level)
            .map(|f| squash(&f.text))
            .collect();
        let names = author_names(c);
        let holder = if names.is_empty() {
            format!("The {} authors", c.name)
        } else {
            names.join(", ")
        };
        let mut lines_out: Vec<(String, usize)> = Vec::new();
        for f in &files {
            let id = store(f.text.clone(), format!("{label}, {}", f.rel));
            lines_out.push((f.rel.clone(), id));
        }
        let mut standard = Vec::new();
        for lic in &used {
            if top.iter().any(|t| carries_text(lic, t)) {
                continue;
            }
            match standard_text(root, lic)? {
                Some(body) => {
                    let text = if lic == "MIT" {
                        format!("Copyright (c) {holder}\n\n{body}")
                    } else {
                        body
                    };
                    let id = store(text, format!("{label}, standard {lic} text"));
                    lines_out.push((
                        format!(
                            "{lic}, standard text (the crate ships no {lic} licence file; \
                             xtask/dist/licenses/{lic}.txt)"
                        ),
                        id,
                    ));
                    standard.push(lic.clone());
                }
                None if below.iter().any(|t| carries_text(lic, t)) => {}
                None => problems.push(format!(
                    "{label}: no text of {lic} in the crate and no standard text in \
                     xtask/dist/licenses/"
                )),
            }
        }
        if lines_out.is_empty() {
            problems.push(format!("{label}: no licence text"));
            continue;
        }
        let mut copyrights = Vec::new();
        for f in &files {
            for l in copyright_lines(&f.text) {
                if !copyrights.contains(&l) {
                    copyrights.push(l);
                }
            }
        }
        let used_text = used.join(" AND ");
        let chosen = used_text != expression.trim();
        let role = if c.linked_into.is_empty() {
            "used only while building (build script or procedural macro); not part of the \
             programs"
                .to_string()
        } else {
            format!("linked into {}", c.linked_into.join(" and "))
        };
        let mut b = String::new();
        b.push_str(&format!("{label}\n"));
        b.push_str(&format!("{}\n", "-".repeat(label.chars().count())));
        b.push_str(&format!("Licence (Cargo.toml): {expression}\n"));
        if chosen {
            b.push_str(&format!(
                "Licence used:         {used_text} (chosen from the options offered)\n"
            ));
        } else {
            b.push_str(&format!("Licence used:         {used_text}\n"));
        }
        let fonts: Vec<&str> = used
            .iter()
            .filter(|id| known(id).is_some_and(|k| k.1 == Class::Font))
            .map(String::as_str)
            .collect();
        if !fonts.is_empty() {
            b.push_str(&format!(
                "Note:                 {} cover font files, which are distributed as data \
                 under their own licences, not combined with the program's code; see the \
                 section Fonts of NOTICE.md for which fonts are embedded.\n",
                fonts.join(" and ")
            ));
        }
        b.push_str(&format!("Use:                  {role}\n"));
        b.push_str(&format!(
            "Authors (Cargo.toml): {}\n",
            if names.is_empty() {
                "(none given)".to_string()
            } else {
                names.join("; ")
            }
        ));
        if copyrights.is_empty() {
            b.push_str("Copyright lines:      (none in its licence files)\n");
        } else {
            b.push_str("Copyright lines in its licence files:\n");
            for l in &copyrights {
                b.push_str(&format!("    {l}\n"));
            }
        }
        b.push_str("Licence texts:\n");
        for (rel, id) in &lines_out {
            b.push_str(&format!("    {rel}: Text {id}\n"));
        }
        blocks.push(b);
        entries.push(Entry {
            name: c.name.clone(),
            version: c.version.clone(),
            expression: expression.clone(),
            used: used_text,
            chosen,
            text_ids: lines_out.iter().map(|(_, id)| *id).collect(),
            standard,
            linked: !c.linked_into.is_empty(),
        });
    }
    if !problems.is_empty() {
        return Err(format!(
            "licences FAILED for {} crate(s):\n  - {}",
            problems.len(),
            problems.join("\n  - ")
        ));
    }

    let n_linked = entries.iter().filter(|e| e.linked).count();
    let mut s = String::new();
    s.push_str("THIRD-PARTY LICENCES\n====================\n\n");
    s.push_str(&format!(
        "RustXMALab {} (the programs rustxmalab and xma), built for {target}.\n\n",
        env!("CARGO_PKG_VERSION")
    ));
    s.push_str(&wrap(&format!(
        "RustXMALab is distributed under GPL-3.0-only (see LICENSE). It is written in Rust and \
         is built from the {} Rust crates listed here: {} are compiled into the programs and \
         linked statically; the other {} are build scripts and procedural macros, and what \
         they depend on, which run only while the programs are being built and are not part \
         of them (listed for completeness). The licence texts of the works whose code was \
         ported into RustXMALab are in the folder licenses/ and are listed in NOTICE.md.",
        entries.len(),
        n_linked,
        entries.len() - n_linked
    )));
    s.push('\n');
    s.push_str(&wrap(
        "For each crate: its licence expression as its Cargo.toml declares it; the licence \
         this distribution uses, where the expression offers a choice (an option compatible \
         with GPL-3.0-only: MIT is preferred, then Apache-2.0, Zlib, the BSD licences, ISC, \
         0BSD and the Unlicense); its authors as its Cargo.toml names them; the copyright \
         lines its licence files give; and every licence or notice file it ships, read from \
         the crate's published source. Each distinct text is printed once, at the end of this \
         file, as \"Text N\". A crate that ships no file for a licence it uses is given the \
         standard text of that licence (for MIT with a copyright line made from its authors).",
    ));
    s.push_str("\nSummary\n=======\n\n");
    let w = entries
        .iter()
        .map(|e| e.name.len() + e.version.len() + 1)
        .max()
        .unwrap_or(0);
    for e in &entries {
        let label = format!("{} {}", e.name, e.version);
        s.push_str(&format!(
            "{label:<w$}  {}{}\n",
            e.used,
            if e.linked { "" } else { "  [build only]" }
        ));
    }
    s.push_str("\nCrates\n======\n\n");
    for b in &blocks {
        s.push_str(b);
        s.push('\n');
    }
    s.push_str("Texts\n=====\n");
    for (i, t) in texts.iter().enumerate() {
        let users = &text_users[i];
        s.push_str(&format!(
            "\n======================================== Text {} ========================================\n",
            i + 1
        ));
        s.push_str(&format!("From: {}\n\n", users.join("; ")));
        s.push_str(t);
    }
    Ok(Report {
        text: s,
        entries,
        texts,
    })
}

/// Wraps a paragraph at 90 columns.
fn wrap(p: &str) -> String {
    let mut out = String::new();
    let mut line = String::new();
    for word in p.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > 90 {
            out.push_str(&line);
            out.push('\n');
            line.clear();
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push_str(&line);
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------------------
// The command
// ---------------------------------------------------------------------------------------

/// The host's target triple (`rustc -vV`).
pub fn host_target() -> Result<String, String> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out = std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .map_err(|e| format!("could not run rustc: {e}"))?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("host: "))
        .map(|h| h.trim().to_string())
        .ok_or_else(|| "rustc -vV gave no host".to_string())
}

/// `cargo metadata` for one platform.
pub fn metadata(root: &Path, target: &str) -> Result<String, String> {
    let out = crate::cargo()
        .args([
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--filter-platform",
            target,
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("could not run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("cargo metadata output: {e}"))
}

/// Writes `THIRD-PARTY-LICENSES.txt` to `out` and prints a summary.
/// The crates are resolved for the host platform, or for the target triple in
/// `XTASK_LICENSES_TARGET` when that is set (to check another platform's list from this one).
pub fn write(root: &Path, out: &Path) -> Result<(), String> {
    let target = match std::env::var("XTASK_LICENSES_TARGET") {
        Ok(t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => host_target()?,
    };
    let report = generate(root, &metadata(root, &target)?, &target)?;
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    std::fs::write(out, &report.text)
        .map_err(|e| format!("cannot write {}: {e}", out.display()))?;
    let mut per: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &report.entries {
        *per.entry(e.used.as_str()).or_default() += 1;
    }
    eprintln!(
        "licenses: {} crates ({} linked), {} distinct texts, {} bytes -> {}",
        report.entries.len(),
        report.entries.iter().filter(|e| e.linked).count(),
        report.texts.len(),
        report.text.len(),
        out.display()
    );
    for (lic, n) in per {
        eprintln!("  {n:4}  {lic}");
    }
    let chosen: Vec<&Entry> = report.entries.iter().filter(|e| e.chosen).collect();
    eprintln!("  licence chosen from a choice for {} crates", chosen.len());
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for e in &chosen {
        *kinds
            .entry(format!("{} -> {}", e.expression, e.used))
            .or_default() += 1;
    }
    for (k, n) in kinds {
        eprintln!("    {n:4}  {k}");
    }
    for e in report.entries.iter().filter(|e| !e.standard.is_empty()) {
        eprintln!(
            "  standard text for {} {}: {}",
            e.name,
            e.version,
            e.standard.join(", ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(s: &str) -> Result<Vec<String>, String> {
        parse_expr(s).and_then(|e| choose(&e))
    }

    #[test]
    fn expressions_choose_a_gpl3_compatible_licence() {
        assert_eq!(ids("MIT OR Apache-2.0").unwrap(), ["MIT"]);
        assert_eq!(ids("Apache-2.0/MIT").unwrap(), ["MIT"]);
        assert_eq!(ids("Unlicense OR MIT").unwrap(), ["MIT"]);
        assert_eq!(ids("Apache-2.0 OR GPL-2.0-only").unwrap(), ["Apache-2.0"]);
        assert_eq!(
            ids("Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT").unwrap(),
            ["MIT"]
        );
        assert_eq!(
            ids("(MIT OR Apache-2.0) AND Unicode-3.0").unwrap(),
            ["MIT", "Unicode-3.0"]
        );
        assert_eq!(
            ids("(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0").unwrap(),
            ["MIT", "OFL-1.1", "Ubuntu-font-1.0"]
        );
        assert_eq!(ids("Zlib").unwrap(), ["Zlib"]);
    }

    #[test]
    fn unknown_or_incompatible_expressions_fail() {
        assert!(ids("GPL-2.0-only").unwrap_err().contains("not compatible"));
        assert!(ids("SSPL-1.0 OR GPL-2.0-only").is_err());
        assert!(ids("WTFPL").unwrap_err().contains("not a licence"));
        assert!(ids("MIT AND GPL-2.0-only").is_err());
        assert!(ids("(MIT").is_err());
    }

    /// Review 2026-11-20 (parity-xtask finding 13): a file that only mentions a licence
    /// does not carry its text, and BSD-2/BSD-3 and ISC/0BSD are told apart.
    #[test]
    fn review12_licence_text_is_recognised_by_its_full_text() {
        let notice = squash(
            "Copyright 2020 The Foo Authors.\nLicensed under the Apache License, Version 2.0\n\
             <LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0>.",
        );
        assert!(!carries_text("Apache-2.0", &notice));
        let apache = squash(
            "Apache License\nVersion 2.0, January 2004\n\nTERMS AND CONDITIONS FOR USE,\n\
             REPRODUCTION, AND DISTRIBUTION\n",
        );
        assert!(carries_text("Apache-2.0", &apache));

        let bsd2 = squash(
            "Redistribution and use in source and binary forms, with or without modification, \
             are permitted provided that the following conditions are met:\n1. ...\n2. ...",
        );
        let bsd3 = format!(
            "{bsd2} 3. neither the name of the copyright holder nor the names of its \
             contributors may be used to endorse or promote products derived from this \
             software without specific prior written permission."
        );
        assert!(carries_text("BSD-2-Clause", &bsd2));
        assert!(!carries_text("BSD-3-Clause", &bsd2));
        assert!(carries_text("BSD-3-Clause", &bsd3));
        assert!(!carries_text("BSD-2-Clause", &bsd3));

        let zero_bsd = squash(
            "Permission to use, copy, modify, and/or distribute this software for any purpose \
             with or without fee is hereby granted.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\"",
        );
        let isc = squash(
            "Permission to use, copy, modify, and/or distribute this software for any purpose \
             with or without fee is hereby granted, provided that the above copyright notice \
             and this permission notice appear in all copies.\n\nTHE SOFTWARE IS PROVIDED",
        );
        assert!(carries_text("0BSD", &zero_bsd));
        assert!(!carries_text("ISC", &zero_bsd));
        assert!(carries_text("ISC", &isc));
        assert!(!carries_text("0BSD", &isc));
    }

    #[test]
    fn copyright_lines_skip_templates() {
        let text = "Copyright (c) 2014 Jane Doe\n   Copyright [yyyy] [name of copyright owner]\n\
                    2. Grant of Copyright License.\nThe above copyright notice\n\
                    // Copyright 2018 Source Foundry Authors\n\
                    (c) You must retain, in the Source form\n(C) 2024 Trifecta Tech Foundation\n";
        assert_eq!(
            copyright_lines(text),
            [
                "Copyright (c) 2014 Jane Doe",
                "Copyright 2018 Source Foundry Authors",
                "(C) 2024 Trifecta Tech Foundation"
            ]
        );
    }

    /// Every crate of the graph is in the file with a licence text of real length, and two
    /// runs give the same bytes.
    #[test]
    fn every_crate_is_covered_and_output_is_deterministic() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let target = host_target().unwrap();
        let meta = metadata(root, &target).unwrap();
        let a = generate(root, &meta, &target).unwrap();
        let b = generate(root, &metadata(root, &target).unwrap(), &target).unwrap();
        assert_eq!(a.text.as_bytes(), b.text.as_bytes());

        let graph = crates(&meta).unwrap();
        assert!(graph.len() > 100, "{} crates", graph.len());
        assert_eq!(graph.len(), a.entries.len());
        for c in &graph {
            let heading = format!(
                "\n{} {}\n{}\n",
                c.name,
                c.version,
                "-".repeat(c.name.len() + c.version.len() + 1)
            );
            assert!(
                a.text.contains(&heading),
                "{} {} missing",
                c.name,
                c.version
            );
            let e = a
                .entries
                .iter()
                .find(|e| e.name == c.name && e.version == c.version)
                .unwrap();
            let longest = e
                .text_ids
                .iter()
                .map(|&i| a.texts[i - 1].len())
                .max()
                .unwrap_or(0);
            assert!(
                longest >= 500,
                "{} {}: longest text {longest} bytes",
                c.name,
                c.version
            );
        }
        for p in PROGRAMS {
            assert!(
                graph.iter().any(|c| c.linked_into.contains(&p.1)),
                "{}",
                p.1
            );
        }
    }
}
