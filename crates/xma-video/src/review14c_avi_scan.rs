//! Review 14c (ledger V37): the scan of `movi` without recursion, and
//! the frames after the last indexed one, compared with the reader as it was before.
//!
//! `old_scan_list` and `old_build_index` are copies of the code before this change (the scan
//! recursed into every `LIST` and `RIFF`, clipping each at its declared end; the index alone,
//! when one was usable). They are the reference:
//!
//! - every AVI fixture and every reference AVI (and the pig videos, when they are AVI) gives
//!   the same frames in the same order through the new code (the index is built the same way
//!   in both modes; the modes differ only in the count taken from it);
//! - on damaged variants of them (cut at many offsets, with and without the `RIFF` and `movi`
//!   sizes 0 or 0xFFFFFFFF; every chunk size set to 0, 0xFFFFFFFF, one or eight more or eight
//!   less, or past its parent's end) the new scan finds the same frames as the old one unless a
//!   list's children do not end exactly at the list's declared end (`lists_disagree`), which
//!   is asserted; where they differ, the frames each finds are counted as real (the data offset
//!   of a frame chunk of the undamaged file) or not (a chunk header read from elsewhere, such as
//!   from inside frame data), and printed.
//!
//! Run with `--nocapture` to see the counts.

use std::path::{Path, PathBuf};

use super::*;

// ---- the code before review 14c, kept as the reference ----

/// The scan of `movi` before review 14c (recursive).
fn old_scan_list(
    file: &mut File,
    start: u64,
    end: u64,
    is_frame: &dyn Fn(&[u8]) -> bool,
    out: &mut Vec<Entry>,
) -> std::io::Result<()> {
    let mut p = start;
    while p + 8 <= end {
        let ch = read_at(file, p, 8)?;
        let size = u32_at(&ch, 4);
        if &ch[0..4] == b"LIST" || &ch[0..4] == b"RIFF" {
            old_scan_list(
                file,
                p + 12,
                (p + 8 + u64::from(size)).min(end),
                is_frame,
                out,
            )?;
        } else if is_frame(&ch[0..4]) {
            out.push(Entry {
                offset: p + 8,
                len: size,
            });
        }
        p = p + 8 + u64::from(size) + u64::from(size & 1);
    }
    Ok(())
}

/// `build_index` before review 14c.
fn old_build_index(
    file: &mut File,
    file_len: u64,
    layout: &Layout,
    stream: usize,
    indx: Option<&[u8]>,
) -> std::io::Result<Vec<Entry>> {
    let tag = format!("{stream:02}");
    let is_frame = |id: &[u8]| {
        id.len() == 4 && id[0..2] == *tag.as_bytes() && (&id[2..4] == b"dc" || &id[2..4] == b"db")
    };
    if let Some(indx) = indx
        && let Some(entries) = odml_index(file, file_len, indx)?
    {
        return Ok(entries);
    }
    if let (Some((pos, size)), Some(&(movi, _))) = (layout.idx1, layout.movi.first()) {
        let data = read_at(file, pos, size as usize)?;
        let raw: Vec<(u32, u32)> = data
            .as_chunks::<16>()
            .0
            .iter()
            .filter(|e| is_frame(&e[0..4]))
            .map(|e| (u32_at(e, 8), u32_at(e, 12)))
            .collect();
        if !raw.is_empty() {
            let mut fits = |base: u64| -> std::io::Result<bool> {
                let at = base + u64::from(raw[0].0);
                if at + 8 > file_len {
                    return Ok(false);
                }
                Ok(is_frame(&read_at(file, at, 4)?))
            };
            let base = if fits(movi)? {
                Some(movi)
            } else if fits(0)? {
                Some(0)
            } else {
                None
            };
            if let Some(base) = base {
                return Ok(raw
                    .into_iter()
                    .map(|(off, len)| Entry {
                        offset: base + u64::from(off) + 8,
                        len,
                    })
                    .collect());
            }
        }
    }
    let mut entries = Vec::new();
    for &(movi, end) in &layout.movi {
        old_scan_list(file, movi + 4, end, &is_frame, &mut entries)?;
    }
    Ok(entries)
}

// ---- helpers ----

/// The parts `AviSource::open_with` reads before the index: `None` when the file is not a
/// RIFF AVI with a video stream.
struct Parts {
    file: File,
    len: u64,
    layout: Layout,
    stream: usize,
    indx: Option<Vec<u8>>,
}

fn parts(path: &Path) -> Option<Parts> {
    let mut file = File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    if len < 12 {
        return None;
    }
    let sig = read_at(&mut file, 0, 12).ok()?;
    if &sig[0..4] != b"RIFF" || &sig[8..12] != b"AVI " {
        return None;
    }
    let layout = read_layout(&mut file, len).ok()?;
    let (stream, indx) = layout
        .streams
        .iter()
        .find(|(_, strh, _, _)| strh.len() >= 36 && &strh[0..4] == b"vids")
        .map(|(n, _, _, indx)| (*n, indx.clone()))?;
    Some(Parts {
        file,
        len,
        layout,
        stream,
        indx,
    })
}

fn frame_test(stream: usize) -> impl Fn(&[u8]) -> bool {
    let tag = format!("{stream:02}").into_bytes();
    move |id: &[u8]| {
        id.len() == 4 && id[0..2] == tag[..] && (&id[2..4] == b"dc" || &id[2..4] == b"db")
    }
}

/// The old and the new scan over every `movi` range of the file.
fn both_scans(p: &mut Parts) -> (Vec<Entry>, Vec<Entry>) {
    let is_frame = frame_test(p.stream);
    let (mut old, mut new) = (Vec::new(), Vec::new());
    for &(movi, end) in &p.layout.movi.clone() {
        old_scan_list(&mut p.file, movi + 4, end, &is_frame, &mut old).unwrap();
        scan_list(&mut p.file, movi + 4, end, &is_frame, &mut new).unwrap();
    }
    (old, new)
}

/// True when, somewhere in a `movi` range, the old scan's walk through a list's children ends
/// at another position than the list's declared end (padded), with at least one of the two
/// positions leaving room for a chunk header before the end of the range: there the old scan
/// goes on at the declared end and the new one at the end of the children. When this is false
/// the two scans visit the same chunks in the same order.
fn lists_disagree(p: &mut Parts) -> bool {
    fn walk(file: &mut File, start: u64, end: u64, outer: u64, flag: &mut bool) -> u64 {
        let mut p = start;
        while p + 8 <= end {
            let ch = read_at(file, p, 8).unwrap();
            let size = u64::from(u32_at(&ch, 4));
            let resume = p + 8 + size + (size & 1);
            if &ch[0..4] == b"LIST" || &ch[0..4] == b"RIFF" {
                let q = walk(file, p + 12, (p + 8 + size).min(end), outer, flag);
                if q != resume && (q + 8 <= outer || resume + 8 <= outer) {
                    *flag = true;
                }
            }
            p = resume;
        }
        p
    }
    let mut flag = false;
    for &(movi, end) in &p.layout.movi.clone() {
        walk(&mut p.file, movi + 4, end, end, &mut flag);
    }
    flag
}

fn fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "avi"))
        .collect();
    v.sort();
    v
}

/// Every AVI of the reference video datasets (the folders next to `video_avi_raw`'s) and the
/// pig videos (empty when absent).
fn references() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(ds) = xma_oracle::dataset("video_avi_raw") {
        let video = ds.project.parent().unwrap().parent().unwrap().to_path_buf();
        for d in std::fs::read_dir(video).unwrap() {
            let d = d.unwrap().path();
            if !d.is_dir() {
                continue;
            }
            for e in std::fs::read_dir(d).unwrap() {
                let p = e.unwrap().path();
                if p.extension().is_some_and(|e| e == "avi") {
                    v.push(p);
                }
            }
        }
    }
    let pig = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ExampleDatasets/1 - Pig dataset");
    for name in ["Camera1.avi", "Camera2.avi"] {
        if pig.join(name).exists() {
            v.push(pig.join(name));
        }
    }
    v.sort();
    v
}

/// `d` (one `RIFF` list, one `movi`) with its `movi` chunks grouped in pairs into `LIST rec `
/// lists, all of them inside one more `LIST rec `, and without what follows `movi` (`idx1`,
/// whose offsets would no longer fit): a file whose frames are found by the scan, through
/// two levels of nested lists.
fn wrap_rec(d: &[u8]) -> Option<Vec<u8>> {
    let hs = headers(d);
    if hs.iter().filter(|h| &d[h.0..h.0 + 4] == b"RIFF").count() != 1 {
        return None;
    }
    let &(m, msize, _) = hs
        .iter()
        .find(|h| &d[h.0..h.0 + 4] == b"LIST" && &d[h.0 + 8..h.0 + 12] == b"movi")?;
    let end = m + 8 + msize as usize;
    let mut chunks = Vec::new();
    let mut p = m + 12;
    while p + 8 <= end {
        let size = u32le(d, p + 4) as usize;
        let next = (p + 8 + size + (size & 1)).min(end);
        chunks.push(&d[p..next]);
        p = next;
    }
    let list = |body: &[u8]| {
        let mut v = b"LIST".to_vec();
        v.extend_from_slice(&((4 + body.len()) as u32).to_le_bytes());
        v.extend_from_slice(b"rec ");
        v.extend_from_slice(body);
        v
    };
    let pairs: Vec<u8> = chunks.chunks(2).flat_map(|c| list(&c.concat())).collect();
    let mut out = d[..m + 12].to_vec();
    out.extend(list(&pairs));
    let movi_size = (out.len() - m - 8) as u32;
    out[m + 4..m + 8].copy_from_slice(&movi_size.to_le_bytes());
    let riff_size = (out.len() - 8) as u32;
    out[4..8].copy_from_slice(&riff_size.to_le_bytes());
    Some(out)
}

fn temp_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("xma_review14c_scan_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn u32le(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(d[o..o + 4].try_into().unwrap())
}

/// (position, size, end of its parent) of every chunk header, lists included, to depth 4.
fn headers(d: &[u8]) -> Vec<(usize, u32, usize)> {
    fn walk(d: &[u8], start: usize, end: usize, depth: usize, out: &mut Vec<(usize, u32, usize)>) {
        let mut p = start;
        while p + 8 <= end {
            let size = u32le(d, p + 4);
            out.push((p, size, end));
            let id = &d[p..p + 4];
            if (id == b"LIST" || id == b"RIFF") && depth < 4 {
                let cend = (p as u64 + 8 + u64::from(size)).min(end as u64) as usize;
                walk(d, p + 12, cend, depth + 1, out);
            }
            p = (p as u64 + 8 + u64::from(size) + u64::from(size & 1)).min(usize::MAX as u64)
                as usize;
        }
    }
    let mut out = Vec::new();
    walk(d, 0, d.len(), 0, &mut out);
    out
}

/// The damaged variants of `d` (named).
fn variants(d: &[u8], evenly: usize) -> Vec<(String, Vec<u8>)> {
    let hs = headers(d);
    let movi = (0..d.len().saturating_sub(4))
        .find(|&p| &d[p..p + 4] == b"movi")
        .map(|p| p - 8);
    let mut out = Vec::new();
    let mut cuts: Vec<usize> = hs
        .iter()
        .flat_map(|&(p, size, _)| [p, p + 4, p + 8, p + 12, p + 8 + size as usize / 2])
        .chain((1..evenly).map(|k| d.len() * k / evenly))
        .filter(|&c| c > 12 && c < d.len())
        .collect();
    cuts.sort();
    cuts.dedup();
    for c in cuts {
        let cut = d[..c].to_vec();
        out.push((format!("cut{c}"), cut.clone()));
        for v in [0u32, u32::MAX] {
            let mut e = cut.clone();
            e[4..8].copy_from_slice(&v.to_le_bytes());
            if let Some(m) = movi.filter(|&m| m + 8 <= c) {
                e[m + 4..m + 8].copy_from_slice(&v.to_le_bytes());
            }
            out.push((format!("cut{c}_sizes{v:x}"), e));
        }
    }
    for &(p, size, parent_end) in &hs {
        let past_parent = (parent_end - p - 8) as u32;
        for v in [
            0,
            u32::MAX,
            size.wrapping_add(1),
            size.wrapping_add(8),
            size.saturating_sub(8),
            past_parent.wrapping_add(4),
            past_parent.wrapping_add(12),
        ] {
            if v == size {
                continue;
            }
            let mut e = d.to_vec();
            e[p + 4..p + 8].copy_from_slice(&v.to_le_bytes());
            out.push((format!("size@{p}={v:#x}"), e));
        }
    }
    out
}

/// The positions at which the old (`recursive`) or the new scan reads a chunk header, over every
/// `movi` range: the same walks as `old_scan_list` and `scan_list`, recording where they read.
fn visits(p: &mut Parts, recursive: bool) -> Vec<u64> {
    fn old(file: &mut File, start: u64, end: u64, out: &mut Vec<u64>) {
        let mut p = start;
        while p + 8 <= end {
            out.push(p);
            let ch = read_at(file, p, 8).unwrap();
            let size = u64::from(u32_at(&ch, 4));
            if &ch[0..4] == b"LIST" || &ch[0..4] == b"RIFF" {
                old(file, p + 12, (p + 8 + size).min(end), out);
            }
            p = p + 8 + size + (size & 1);
        }
    }
    let mut out = Vec::new();
    for &(movi, end) in &p.layout.movi.clone() {
        if recursive {
            old(&mut p.file, movi + 4, end, &mut out);
        } else {
            let mut q = movi + 4;
            while q + 8 <= end {
                out.push(q);
                let ch = read_at(&mut p.file, q, 8).unwrap();
                let size = u64::from(u32_at(&ch, 4));
                q = if &ch[0..4] == b"LIST" || &ch[0..4] == b"RIFF" {
                    q + 12
                } else {
                    q + 8 + size + (size & 1)
                };
            }
        }
    }
    out
}

#[derive(Default, Debug)]
struct Tally {
    variants: usize,
    scans_differ: usize,
    predicted: usize,
    old_real: usize,
    old_spurious: usize,
    new_real: usize,
    new_spurious: usize,
    /// Where the scans differ: headers read at a position where the undamaged file has no
    /// chunk header (from inside a chunk's data), and the variants with any.
    old_off_grid: usize,
    new_off_grid: usize,
    old_variants_off_grid: usize,
    new_variants_off_grid: usize,
    /// Where the scans differ: variants in which one finds more real frames than the other.
    old_more_real: usize,
    new_more_real: usize,
    index_differs: usize,
    tail_added: usize,
    tail_spurious: usize,
}

/// Compares old and new on one file; `real` holds the data offsets of the undamaged file's
/// frames.
fn compare(
    path: &Path,
    real: &std::collections::HashSet<u64>,
    grid: &std::collections::HashSet<u64>,
    t: &mut Tally,
    examples: &mut Vec<String>,
    differ: &mut Vec<String>,
) {
    let Some(mut p) = parts(path) else { return };
    t.variants += 1;
    let (old, new) = both_scans(&mut p);
    let disagree = lists_disagree(&mut p);
    if disagree {
        t.predicted += 1;
    }
    if old != new {
        assert!(
            disagree,
            "{}: the scans differ where no list disagrees with its children",
            path.display()
        );
        t.scans_differ += 1;
        let real_of = |v: &[Entry]| v.iter().filter(|e| real.contains(&e.offset)).count();
        let (or, nr) = (real_of(&old), real_of(&new));
        t.old_real += or;
        t.old_spurious += old.len() - or;
        t.new_real += nr;
        t.new_spurious += new.len() - nr;
        let off = |v: Vec<u64>| v.iter().filter(|q| !grid.contains(q)).count();
        let (oo, no) = (off(visits(&mut p, true)), off(visits(&mut p, false)));
        t.old_off_grid += oo;
        t.new_off_grid += no;
        t.old_variants_off_grid += usize::from(oo > 0);
        t.new_variants_off_grid += usize::from(no > 0);
        t.old_more_real += usize::from(or > nr);
        t.new_more_real += usize::from(nr > or);
        differ.push(format!(
            "{}: old {} frames ({} real, {} headers read off the grid), new {} ({} real, {} off)",
            path.file_name().unwrap().to_string_lossy(),
            old.len(),
            or,
            oo,
            new.len(),
            nr,
            no
        ));
    }
    let indx = p.indx.clone();
    let (len, stream) = (p.len, p.stream);
    let old_index = old_build_index(&mut p.file, len, &p.layout, stream, indx.as_deref()).unwrap();
    let now = build_index(&mut p.file, len, &p.layout, stream, indx.as_deref()).unwrap();
    let mut seen = std::collections::HashSet::new();
    for e in &now {
        assert!(
            seen.insert(e.offset),
            "{}: frame at {} twice",
            path.display(),
            e.offset
        );
    }
    if old_index != now {
        t.index_differs += 1;
    }
    if old != new {
        return;
    }
    // Where the scans agree: the old entries, then frames after them that no index held.
    assert_eq!(
        &now[..old_index.len().min(now.len())],
        &old_index[..],
        "{}",
        path.display()
    );
    let tail = &now[old_index.len()..];
    let max = old_index.iter().map(|e| e.offset).max().unwrap_or(0);
    for e in tail {
        assert!(e.offset > max, "{}", path.display());
    }
    t.tail_added += tail.len();
    t.tail_spurious += tail.iter().filter(|e| !real.contains(&e.offset)).count();
    if !tail.is_empty() && examples.len() < 60 {
        examples.push(format!(
            "{}: {} indexed, {} found after them ({} real)",
            path.file_name().unwrap().to_string_lossy(),
            old_index.len(),
            tail.len(),
            tail.iter().filter(|e| real.contains(&e.offset)).count()
        ));
    }
}

// ---- tests ----

/// Undamaged files: the same frames in the same order as before, in both modes; and what the
/// search for frames after the last indexed one costs on each (chunk headers read).
#[test]
fn review14c_intact_files_read_as_before() {
    let mut n = 0;
    let files: Vec<PathBuf> = fixtures().into_iter().chain(references()).collect();
    for path in &files {
        let Some(mut p) = parts(path) else {
            println!("{}: not a RIFF AVI with a video stream", path.display());
            continue;
        };
        n += 1;
        let (old, new) = both_scans(&mut p);
        assert_eq!(old, new, "{}: scan", path.display());
        let indx = p.indx.clone();
        let (len, stream) = (p.len, p.stream);
        let before = old_build_index(&mut p.file, len, &p.layout, stream, indx.as_deref()).unwrap();
        let now = build_index(&mut p.file, len, &p.layout, stream, indx.as_deref()).unwrap();
        assert_eq!(before, now, "{}", path.display());
        // The headers the search after the last indexed frame reads (its walk, counted).
        let last = before.iter().max_by_key(|e| e.offset).copied();
        let mut headers_read = 0;
        if let Some(last) = last {
            let l = u64::from(last.len);
            let start = last.offset + l + (l & 1);
            for &(movi, end) in &p.layout.movi {
                let mut q = start.max(movi + 4);
                while q + 8 <= end {
                    let ch = read_at(&mut p.file, q, 8).unwrap();
                    headers_read += 1;
                    let size = u64::from(u32_at(&ch, 4));
                    q = if &ch[0..4] == b"LIST" || &ch[0..4] == b"RIFF" {
                        q + 12
                    } else {
                        q + 8 + size + (size & 1)
                    };
                }
            }
        }
        println!(
            "R14C intact {}: {} frames, {} movi lists, {} chunk headers read after the last indexed frame (+1 to check that frame's header)",
            path.file_name().unwrap().to_string_lossy(),
            before.len(),
            p.layout.movi.len(),
            headers_read
        );
    }
    assert!(n >= 10, "only {n} AVI files read");
}

fn damaged_run(sources: &[(PathBuf, usize, usize)]) {
    let dir = temp_dir();
    let mut t = Tally::default();
    let mut examples = Vec::new();
    let mut differ = Vec::new();
    for (src, prefix, evenly) in sources {
        let whole = std::fs::read(src).unwrap();
        let d = &whole[..whole.len().min(*prefix)];
        let mut real = std::collections::HashSet::new();
        {
            let mut p = parts(src).unwrap();
            let (old, _) = both_scans(&mut p);
            let indx = p.indx.clone();
            let (len, stream) = (p.len, p.stream);
            let idx =
                old_build_index(&mut p.file, len, &p.layout, stream, indx.as_deref()).unwrap();
            real.extend(old.iter().chain(&idx).map(|e| e.offset));
        }
        let grid: std::collections::HashSet<u64> =
            headers(&whole).iter().map(|h| h.0 as u64).collect();
        let stem = src.file_stem().unwrap().to_string_lossy().into_owned();
        for (name, bytes) in variants(d, *evenly) {
            let path = dir.join(format!("{stem}.{name}.avi"));
            std::fs::write(&path, &bytes).unwrap();
            compare(&path, &real, &grid, &mut t, &mut examples, &mut differ);
            let _ = std::fs::remove_file(&path);
        }
    }
    println!("R14C damaged: {t:?}");
    for e in examples {
        println!("R14C   {e}");
    }
    for e in differ {
        println!("R14C differ {e}");
    }
}

/// Damaged variants of every AVI fixture.
#[test]
fn review14c_damaged_fixtures_scan_as_before_or_as_explained() {
    let mut sources: Vec<(PathBuf, usize, usize)> = fixtures()
        .into_iter()
        .filter(|p| parts(p).is_some())
        .map(|p| (p, usize::MAX, 64))
        .collect();
    // The same with their frames in nested `rec ` lists.
    let dir = temp_dir();
    for (src, _, _) in sources.clone() {
        if let Some(w) = wrap_rec(&std::fs::read(&src).unwrap()) {
            let path = dir.join(format!(
                "{}.rec.avi",
                src.file_stem().unwrap().to_string_lossy()
            ));
            std::fs::write(&path, w).unwrap();
            if parts(&path).is_some() {
                sources.push((path, usize::MAX, 64));
            }
        }
    }
    damaged_run(&sources);
}

/// Damaged variants of the reference AVIs (of their first 4 MB for the large ones, which holds
/// the headers and the first frames).
#[test]
fn review14c_damaged_references_scan_as_before_or_as_explained() {
    let sources: Vec<(PathBuf, usize, usize)> = references()
        .into_iter()
        .filter(|p| parts(p).is_some())
        .map(|p| (p, 4 << 20, 24))
        .collect();
    damaged_run(&sources);
}
