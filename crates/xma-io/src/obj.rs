//! Wavefront OBJ meshes, as a rigid body's bone model is read.
//!
//! Ported from `gl/GLMLoader.cpp`, `GLMLoader::load` (itself based on Nate Robins' GLM
//! loader), which `core/RigidBodyObj.cpp`, `RigidBodyObj::RigidBodyObj` calls. What the
//! C++ reads, and what is kept:
//!
//! - `v x y z` vertices, as `float`; anything after the third number (a `w`, colours) is
//!   ignored. `vn x y z` normals. Every other statement (`vt`, `vp`, `o`, `g`, `s`,
//!   `usemtl`, `mtllib`, `l`, `p`, comments) is ignored.
//! - `f` faces with corners written `v`, `v/t`, `v/t/n` or `v//n` (texture indices are
//!   read and dropped). A face of more than three corners is cut into a fan
//!   (`v0 v1 v2`, `v0 v2 v3`, ...): each new triangle takes the first corner, the last
//!   corner of the previous triangle and the next corner.
//! - Normals: when the file has no `vn` lines, each triangle gets its facet normal
//!   (`(v1 - v0) × (v2 - v0)`, normalised) at its three corners. The C++ averages facet
//!   normals around each vertex only where they make an angle under its threshold
//!   `ANGLE_THRES`, which is 0, so every corner keeps its own facet's normal (flat
//!   shading); this is that result. When the file has normals, a corner takes the normal
//!   its face names.
//! - The mesh scale is not applied here: `RigidBodyObj::render` scales the vertices
//!   (`glScaled`) after the body's pose; see `xma_engine::view::world`.
//!
//! Faults of the C++ loader that are not copied (the C++ reads past its arrays or
//! misreads the file; parity ledger, Interface section, proposed with this reader):
//!
//! - A face index of 0, past the end of the vertex or normal list, or negative: the C++
//!   reads outside its arrays. Here a negative index counts back from the last vertex
//!   (or normal) read so far, as the OBJ format defines, and 0 or an index past the end
//!   is an error naming the line.
//! - A face with fewer than three corners, or a vertex with fewer than three numbers: the
//!   C++ goes on reading the next line's numbers. Here it is an error.
//! - A face whose corners do not all name a normal, in a file that has normals: the C++
//!   reads an uninitialised normal index. Here that triangle gets its facet normal.
//! - A degenerate triangle (zero area): the C++ divides by zero and the normal is NaN.
//!   Here the normal is zero (the triangle is lit by the ambient light only).
//! - Lines longer than 127 characters: the C++ skips the rest of a line with
//!   `fgets(buf, 128)`, so the remainder of a long comment is read as statements. Here
//!   the file is read line by line, whatever the length of a line.
//!
//! Lines end at `\n`, `\r\n` or a lone `\r` (old Mac files; the C++ reads them, as its
//! `fscanf` token reader takes any white space as a separator). A comment runs from `#` to
//! the end of its line wherever it starts, also after a statement (`f 1 2 3 # text` is the
//! face 1 2 3, as the C++ reads it).
//!
//! A file that cannot be opened is an [`ObjError::Io`]; an empty file, or one with no
//! faces, is a mesh with no triangles, as in the C++.

use std::fmt;
use std::path::{Path, PathBuf};

/// A triangle mesh as the C++ loader leaves it, but indexed: the vertices as read, the
/// triangles after the fan cut, and a normal per triangle corner.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObjMesh {
    /// Vertex positions in file order (`float`, as `fscanf("%f")` reads them).
    pub positions: Vec<[f32; 3]>,
    /// Triangles as 0-based indices into [`ObjMesh::positions`].
    pub triangles: Vec<[u32; 3]>,
    /// The normal of each triangle corner, in the order of [`ObjMesh::triangles`].
    pub normals: Vec<[[f32; 3]; 3]>,
    /// True when the normals came from the file's `vn` lines (for every triangle that names
    /// them), false when every triangle has its facet normal.
    pub file_normals: bool,
}

impl ObjMesh {
    /// Number of vertices read (`v` lines).
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.positions.len()
    }

    /// Number of triangles after cutting faces into fans.
    #[must_use]
    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// The smallest and largest coordinates of the vertices, `None` without vertices.
    #[must_use]
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut it = self.positions.iter();
        let first = *it.next()?;
        let (mut lo, mut hi) = (first, first);
        for p in it {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Some((lo, hi))
    }
}

/// Why an OBJ file could not be read.
#[derive(Debug)]
pub enum ObjError {
    /// The file could not be opened or read.
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    /// A statement the reader cannot use (1-based line number).
    Parse { line: usize, message: String },
}

impl fmt::Display for ObjError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjError::Io { path, source } => {
                write!(f, "could not read {}: {source}", path.display())
            }
            ObjError::Parse { line, message } => write!(f, "line {line}: {message}"),
        }
    }
}

impl std::error::Error for ObjError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ObjError::Io { source, .. } => Some(source),
            ObjError::Parse { .. } => None,
        }
    }
}

/// Reads an OBJ file.
///
/// Ported from gl/GLMLoader.cpp, GLMLoader::load (see the module documentation).
///
/// # Errors
///
/// [`ObjError::Io`] when the file cannot be read, [`ObjError::Parse`] for a statement the
/// C++ would misread.
pub fn read_obj(path: &Path) -> Result<ObjMesh, ObjError> {
    let bytes = std::fs::read(path).map_err(|source| ObjError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_obj(&String::from_utf8_lossy(&bytes))
}

/// A face corner: vertex index and optional normal index, both resolved to 0-based.
#[derive(Clone, Copy)]
struct Corner {
    v: u32,
    n: Option<u32>,
}

/// Parses the text of an OBJ file (see [`read_obj`]).
///
/// # Errors
///
/// [`ObjError::Parse`] for a statement the C++ would misread.
pub fn parse_obj(text: &str) -> Result<ObjMesh, ObjError> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut file_normals: Vec<[f32; 3]> = Vec::new();
    // Triangles with their corners' normal indices (`None` where a corner names none).
    let mut tris: Vec<([u32; 3], [Option<u32>; 3])> = Vec::new();

    let text = text.replace("\r\n", "\n");
    for (i, line) in text.split(['\n', '\r']).enumerate() {
        let lineno = i + 1;
        let err = |message: String| ObjError::Parse {
            line: lineno,
            message,
        };
        // A comment runs from `#` to the line end, wherever it starts.
        let line = line.split('#').next().unwrap_or("");
        let mut tokens = line.split_whitespace();
        let Some(key) = tokens.next() else { continue };
        match key {
            "v" | "vn" => {
                let mut xyz = [0f32; 3];
                for (k, c) in xyz.iter_mut().enumerate() {
                    let tok = tokens
                        .next()
                        .ok_or_else(|| err(format!("`{key}` needs three numbers, found {k}")))?;
                    *c = tok
                        .parse::<f32>()
                        .map_err(|_| err(format!("`{key}`: `{tok}` is not a number")))?;
                }
                if key == "v" {
                    positions.push(xyz);
                } else {
                    file_normals.push(xyz);
                }
            }
            "f" => {
                let corners = tokens
                    .map(|t| corner(t, positions.len(), file_normals.len()).map_err(&err))
                    .collect::<Result<Vec<_>, _>>()?;
                if corners.len() < 3 {
                    return Err(err(format!(
                        "a face needs at least three corners, found {}",
                        corners.len()
                    )));
                }
                // The C++ fan: (first, previous last, next).
                let mut prev = [corners[0], corners[1], corners[2]];
                tris.push((
                    [prev[0].v, prev[1].v, prev[2].v],
                    [prev[0].n, prev[1].n, prev[2].n],
                ));
                for c in &corners[3..] {
                    prev = [prev[0], prev[2], *c];
                    tris.push((
                        [prev[0].v, prev[1].v, prev[2].v],
                        [prev[0].n, prev[1].n, prev[2].n],
                    ));
                }
            }
            _ => {}
        }
    }

    let has_file_normals = !file_normals.is_empty();
    let mut triangles = Vec::with_capacity(tris.len());
    let mut normals = Vec::with_capacity(tris.len());
    for (v, n) in tris {
        let named = has_file_normals && n.iter().all(Option::is_some);
        let corner_normals = if named {
            [
                file_normals[n[0].unwrap_or(0) as usize],
                file_normals[n[1].unwrap_or(0) as usize],
                file_normals[n[2].unwrap_or(0) as usize],
            ]
        } else {
            let f = facet_normal(
                positions[v[0] as usize],
                positions[v[1] as usize],
                positions[v[2] as usize],
            );
            [f, f, f]
        };
        triangles.push(v);
        normals.push(corner_normals);
    }
    Ok(ObjMesh {
        positions,
        triangles,
        normals,
        file_normals: has_file_normals,
    })
}

/// One face corner (`v`, `v/t`, `v/t/n`, `v//n`), with the vertex and normal counts read
/// so far (for negative indices and the range check).
fn corner(tok: &str, n_vertices: usize, n_normals: usize) -> Result<Corner, String> {
    let mut parts = tok.split('/');
    let v = parts.next().unwrap_or("");
    let _t = parts.next();
    let n = parts.next().filter(|s| !s.is_empty());
    if parts.next().is_some() {
        return Err(format!("`{tok}` is not a face corner"));
    }
    let v = resolve(v, n_vertices, "vertex", tok)?;
    let n = match n {
        Some(n) => Some(resolve(n, n_normals, "normal", tok)?),
        None => None,
    };
    Ok(Corner { v, n })
}

/// A 1-based (or negative, relative) index into a list of `count` items, 0-based.
fn resolve(s: &str, count: usize, what: &str, tok: &str) -> Result<u32, String> {
    let i: i64 = s
        .parse()
        .map_err(|_| format!("`{tok}`: `{s}` is not a {what} index"))?;
    let idx = if i > 0 {
        i - 1
    } else if i < 0 {
        count as i64 + i
    } else {
        return Err(format!("`{tok}`: {what} index 0 (indices start at 1)"));
    };
    if idx < 0 || idx >= count as i64 {
        return Err(format!(
            "`{tok}`: {what} index {i} is outside the {count} {what}s read so far"
        ));
    }
    u32::try_from(idx).map_err(|_| format!("`{tok}`: {what} index {i} is too large"))
}

/// The unit normal of a triangle (`glmCross`, `glmNormalize`), zero for a degenerate one.
fn facet_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 0.0 && l.is_finite() {
        [n[0] / l, n[1] / l, n[2] / l]
    } else {
        [0.0; 3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_is_cut_into_a_fan() {
        let m = parse_obj("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n").unwrap();
        assert_eq!(m.triangles, vec![[0, 1, 2], [0, 2, 3]]);
        assert_eq!(m.normals[1], [[0.0, 0.0, 1.0]; 3]);
        assert!(!m.file_normals);
    }

    /// Lines end at `\n`, `\r\n` or a lone `\r` (old Mac files); line numbers count each.
    #[test]
    fn carriage_return_line_ends() {
        let unix = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
        let want = parse_obj(unix).unwrap();
        for text in [unix.replace('\n', "\r"), unix.replace('\n', "\r\n")] {
            let m = parse_obj(&text).unwrap();
            assert_eq!(m, want, "{text:?}");
            assert_eq!((m.n_vertices(), m.n_triangles()), (4, 2));
        }
        // Mixed line ends: the error names the fourth line.
        match parse_obj("v 0 0 0\rv 1 0 0\r\nv 1 1 0\nf 1 2 9\r") {
            Err(ObjError::Parse { line, .. }) => assert_eq!(line, 4),
            other => panic!("{other:?}"),
        }
    }

    /// A comment runs to the line end wherever it starts, and only to the line end.
    #[test]
    fn comment_after_a_statement() {
        let m = parse_obj("v 0 0 0 # a\rv 1 0 0#b\nv 0 1 0\nf 1 2 3 # face\r# f 9 9 9\n").unwrap();
        assert_eq!(m.triangles, vec![[0, 1, 2]]);
    }
}
