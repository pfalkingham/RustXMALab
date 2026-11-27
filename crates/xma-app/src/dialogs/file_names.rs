//! The file lists of the New dataset and New trial dialogs: how chosen files are sorted, how
//! a folder's images are listed, and what the line edits show.
//!
//! - [`common_prefix`]: ported from ui/CameraBox.cpp and ui/CameraBoxTrial.cpp,
//!   `commonPrefix`: the longest start of the first name that every name **contains**
//!   (`QString::contains`, not "starts with"), plus one more character. Shown as is in the
//!   line edits ("/data/cam1/img001" for `/data/cam1/img001.tif` .. `img0NN.tif`).
//! - [`compare_names`]: ported from core/HelperFunctions.cpp, `littleHelper::compareNames`,
//!   the natural order of a trial's files ("img2" before "img10"), with two defects fixed
//!   (parity ledger rows proposed by the appnew package): a number of more than nine digits
//!   overflowed `QString::toInt`, which then gives 0 and puts that file first; names the C++
//!   comparison finds equal keep the order they had (a stable sort; `std::sort` is not).
//! - [`sort_plain`]: `QStringList::sort` (the New dataset dialog's calibration images).
//! - [`folder_images`]: `QDir::entryList({"*.png", "*.tif", "*.bmp", "*.jpeg", "*.jpg"},
//!   QDir::Files | QDir::NoSymLinks)` (the name filters ignore case, as `QDir`'s do), as
//!   absolute paths in natural order.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

/// The calibration images and videos the New dataset dialog offers (C49).
pub const CALIBRATION_FILES: (&str, &[&str]) = (
    "Image Files",
    &[
        "png", "jpg", "jpeg", "bmp", "tif", "avi", "cine", "mp4", "mov",
    ],
);
/// The undistortion grid images (C50).
pub const GRID_FILES: (&str, &[&str]) = ("Image Files", &["png", "jpg", "jpeg", "bmp", "tif"]);
/// The frame specification of the calibration cube (C51).
pub const FRAMESPEC_FILES: (&str, &[&str]) = ("Framespec File", &["csv"]);
/// The reference points of the calibration cube (C51).
pub const REFERENCE_FILES: (&str, &[&str]) = ("Reference File", &["ref"]);
/// The videos and images of a trial camera (C58).
pub const TRIAL_FILES: (&str, &[&str]) = (
    "Image and Video Files",
    &[
        "cine", "avi", "mp4", "mov", "png", "jpg", "jpeg", "bmp", "tif",
    ],
);
/// The image types a trial folder is searched for (C57).
pub const FOLDER_IMAGE_TYPES: [&str; 5] = ["png", "tif", "bmp", "jpeg", "jpg"];
/// The project files for "Open Dataset with different calibration" (C62).
pub const DATASET_FILES: (&str, &[&str]) = ("Dataset", &["xma", "zip"]);

/// The text a path has in the C++ file lists.
#[must_use]
pub fn path_text(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// The line edit's text for a list of chosen files (`commonPrefix`); empty for no files.
///
/// Ported from ui/CameraBox.cpp, `commonPrefix`.
#[must_use]
pub fn common_prefix(names: &[String]) -> String {
    let Some(first) = names.first() else {
        return String::new();
    };
    let chars: Vec<char> = first.chars().collect();
    let mut count = 0;
    while count < chars.len() {
        let prefix: String = chars[..=count].iter().collect();
        if names.iter().any(|n| !n.contains(&prefix)) {
            break;
        }
        count += 1;
    }
    // `left(count + 1)`: the whole name when `count + 1` passes its end.
    chars[..(count + 1).min(chars.len())].iter().collect()
}

/// The same for paths.
#[must_use]
pub fn common_prefix_of(paths: &[PathBuf]) -> String {
    let names: Vec<String> = paths.iter().map(|p| path_text(p)).collect();
    common_prefix(&names)
}

/// `QChar::isNumber` (the Unicode number categories).
fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// `(n + n1).toInt() < (n + n2).toInt()`, without `toInt`'s overflow to 0: the digit
/// strings compared as numbers of any length (ASCII digits; other number characters, which
/// `toInt` reads as 0, compare as 0 as there). Each number is given as its two parts, `n`
/// and `n1` (or `n2`), so that nothing is copied.
fn number_less(n: &[char], n1: &[char], n2: &[char]) -> bool {
    // The digits without their leading zeros, and how many there are; none when a
    // character is not an ASCII digit.
    fn value<'a>(n: &'a [char], m: &'a [char]) -> (usize, impl Iterator<Item = &'a char>) {
        let digits = n.iter().chain(m).all(char::is_ascii_digit);
        let zeros = if digits {
            n.iter().chain(m).take_while(|c| **c == '0').count()
        } else {
            0
        };
        let len = if digits { n.len() + m.len() - zeros } else { 0 };
        (len, n.iter().chain(m).skip(zeros).take(len))
    }
    let ((la, a), (lb, b)) = (value(n, n1), value(n, n2));
    match la.cmp(&lb) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a.lt(b),
    }
}

/// `littleHelper::compareNames(s1, s2)`: true when `s1` comes before `s2`.
///
/// Ported from core/HelperFunctions.cpp, `littleHelper::compareNames`: skip the common
/// start (ignoring case); at the first difference, if both characters are digits compare
/// the numbers that contain that position (with the digits before it), else a digit after
/// the same start comes last, else the names compare as strings (case-sensitive); when one
/// name ends there, the shorter comes first. Numbers are compared at any length (the C++
/// `toInt` gives 0 past 2147483647).
#[must_use]
pub fn compare_names(s1: &str, s2: &str) -> bool {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    compare_chars(s1, &a, s2, &b)
}

/// [`compare_names`] of `s1` and `s2`, given with their characters `a` and `b`.
fn compare_chars(s1: &str, a: &[char], s2: &str, b: &[char]) -> bool {
    let lower = |c: char| c.to_lowercase().next().unwrap_or(c);
    let mut i = 0;
    while i < a.len() && i < b.len() && lower(a[i]) == lower(b[i]) {
        i += 1;
    }
    i += 1;
    if i < a.len() && i < b.len() {
        let k = i - 1;
        if !is_number(a[k]) || !is_number(b[k]) {
            if is_number(a[k]) {
                return false;
            }
            if is_number(b[k]) {
                return true;
            }
            return utf16_less(s1, s2);
        }
        // The digits before the difference, shared by both (as `s1` has them).
        let mut j = k;
        while j > 0 && is_number(a[j - 1]) {
            j -= 1;
        }
        let run = |v: &[char]| -> usize { v[k..].iter().take_while(|c| is_number(**c)).count() };
        let (l1, l2) = (run(a), run(b));
        number_less(&a[j..k], &a[k..k + l1], &b[k..k + l2])
    } else {
        // The shortest string wins.
        a.len() < b.len()
    }
}

/// `QString::compare(s1, s2, Qt::CaseSensitive) < 0`: UTF-16 code unit order.
fn utf16_less(s1: &str, s2: &str) -> bool {
    s1.encode_utf16().lt(s2.encode_utf16())
}

/// Sorts with a "comes before" test, keeping the order of the items it does not separate
/// (a stable merge sort; it never fails on a test that is not a strict order, as the
/// C++ comparison is not for every pair of names).
pub fn stable_sort_by_less<T: Clone>(v: &mut Vec<T>, less: &impl Fn(&T, &T) -> bool) {
    if v.len() < 2 {
        return;
    }
    let right = v.split_off(v.len() / 2);
    let mut left = std::mem::take(v);
    let mut right = right;
    stable_sort_by_less(&mut left, less);
    stable_sort_by_less(&mut right, less);
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        // Take from the right only when it comes strictly before the left.
        if less(&right[j], &left[i]) {
            out.push(right[j].clone());
            j += 1;
        } else {
            out.push(left[i].clone());
            i += 1;
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    *v = out;
}

/// Sorts paths with [`compare_names`] on their full text (`std::sort(..., compareNames)`),
/// after a plain sort so that names the comparison does not separate come in a fixed order.
pub fn sort_natural(paths: &mut Vec<PathBuf>) {
    paths.sort_by_cached_key(|p| path_text(p));
    sort_by_compare_names(paths);
}

/// [`stable_sort_by_less`] with [`compare_names`] on the paths' full text: the same merges
/// and comparisons, on the positions of the paths, each path's text and characters made
/// once; the paths are put in the order found at the end.
fn sort_by_compare_names(paths: &mut Vec<PathBuf>) {
    let texts: Vec<String> = paths.iter().map(|p| path_text(p)).collect();
    let chars: Vec<Vec<char>> = texts.iter().map(|t| t.chars().collect()).collect();
    let mut order: Vec<usize> = (0..paths.len()).collect();
    stable_sort_by_less(&mut order, &|&i: &usize, &j: &usize| {
        compare_chars(&texts[i], &chars[i], &texts[j], &chars[j])
    });
    let mut slots: Vec<Option<PathBuf>> = std::mem::take(paths).into_iter().map(Some).collect();
    *paths = order
        .into_iter()
        .map(|i| slots[i].take().expect("each position once"))
        .collect();
}

/// `QStringList::sort()`: plain, case-sensitive order of the full text.
pub fn sort_plain(paths: &mut [PathBuf]) {
    paths.sort_by(|a, b| {
        let (a, b) = (path_text(a), path_text(b));
        a.encode_utf16().cmp(b.encode_utf16())
    });
}

/// The images of a folder for a trial camera (C57): files (not symbolic links) whose
/// extension is one of [`FOLDER_IMAGE_TYPES`], ignoring case, as absolute paths in natural
/// order. A folder that cannot be read gives none.
///
/// Ported from ui/CameraBoxTrial.cpp, `CameraBoxTrial::on_toolButtonImage_clicked`.
#[must_use]
pub fn folder_images(folder: &Path) -> Vec<PathBuf> {
    let dir = std::path::absolute(folder).unwrap_or_else(|_| folder.to_path_buf());
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_lowercase();
            FOLDER_IMAGE_TYPES
                .iter()
                .any(|x| name.ends_with(&format!(".{x}")))
        })
        .map(|e| dir.join(e.file_name()))
        .collect();
    // QDir's default order (by name, ignoring case), then the natural order.
    files.sort_by_cached_key(|p| path_text(p).to_lowercase());
    sort_by_compare_names(&mut files);
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn common_prefix_is_one_character_longer_as_in_xmalab() {
        assert_eq!(
            common_prefix(&strings(&[
                "/d/img001.tif",
                "/d/img002.tif",
                "/d/img010.tif"
            ])),
            "/d/img00"
        );
        assert_eq!(
            common_prefix(&strings(&["/d/img001.tif", "/d/img002.tif"])),
            "/d/img001"
        );
        // One file: the whole name.
        assert_eq!(common_prefix(&strings(&["/d/a.avi"])), "/d/a.avi");
        // `contains`, not "starts with": "b" is in both, so the prefix grows past it.
        assert_eq!(common_prefix(&strings(&["ab", "xab"])), "ab");
        assert_eq!(common_prefix(&[]), "");
    }

    #[test]
    fn natural_order_like_compare_names() {
        assert!(compare_names("img2.png", "img10.png"));
        assert!(!compare_names("img10.png", "img2.png"));
        assert!(compare_names("1_a", "12_a"), "a number that ends first");
        assert!(!compare_names("12_a", "1_a"));
        assert!(compare_names("a9", "a10"), "the shorter at the end");
        assert!(
            compare_names("a_x", "a1x"),
            "a digit after the same start comes last"
        );
        assert!(
            compare_names("IMG_b.png", "img_c.png"),
            "the start ignores case"
        );
        // Digits before the difference belong to the number.
        assert!(compare_names("f19.tif", "f110.tif"));
        assert!(!compare_names("f110.tif", "f19.tif"));
    }

    #[test]
    fn long_numbers_are_ordered_by_value() {
        // XMALab 3: `toInt` overflows to 0 and the larger number comes first.
        assert!(compare_names(
            "cam_20231105123456.tif",
            "cam_20231105123457.tif"
        ));
        assert!(compare_names("cam_9999999999.tif", "cam_10000000000.tif"));
        assert!(!compare_names("cam_10000000000.tif", "cam_9999999999.tif"));
    }

    #[test]
    fn sorting_is_stable_and_natural() {
        let mut v: Vec<PathBuf> = ["/d/f10.png", "/d/f9.png", "/d/f1.png", "/d/F2.png"]
            .iter()
            .map(PathBuf::from)
            .collect();
        sort_natural(&mut v);
        let names: Vec<String> = v.iter().map(|p| path_text(p)).collect();
        assert_eq!(names, ["/d/f1.png", "/d/F2.png", "/d/f9.png", "/d/f10.png"]);
        let mut p: Vec<PathBuf> = ["b", "B", "a10", "a9"].iter().map(PathBuf::from).collect();
        sort_plain(&mut p);
        let names: Vec<String> = p.iter().map(|p| path_text(p)).collect();
        assert_eq!(names, ["B", "a10", "a9", "b"]);
    }

    #[test]
    fn folder_listing_filters_and_orders() {
        let d = std::env::temp_dir().join(format!("xma_app_folder_images_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("sub.png")).unwrap();
        for f in ["x10.TIF", "x9.tif", "x1.jpg", "notes.txt", "x2.JPEG"] {
            std::fs::write(d.join(f), b"").unwrap();
        }
        let got: Vec<String> = folder_images(&d)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(got, ["x1.jpg", "x2.JPEG", "x9.tif", "x10.TIF"]);
        assert!(folder_images(&d).iter().all(|p| p.is_absolute()));
        let _ = std::fs::remove_dir_all(&d);
    }
}
