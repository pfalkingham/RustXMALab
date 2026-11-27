//! A single-channel image held in memory, the type every image routine in this crate
//! takes and returns.
//!
//! Conventions: pixel (0, 0) is the top-left pixel, pixel centres are at
//! integers, rows are stored top to bottom. `x` is the column and `y` the row.
//!
//! Three element types are used: `u8` (what XMALab 3 processes), `u16` (sources deeper than
//! 8 bits, parity ledger V1) and `f32` (intermediate results, as OpenCV's `CV_32F`).

/// A single-channel image: `width * height` values, row by row from the top.
#[derive(Clone, Debug, PartialEq)]
pub struct Image<T> {
    width: usize,
    height: usize,
    data: Vec<T>,
}

/// 8-bit grey image (OpenCV `CV_8UC1`).
pub type ImageU8 = Image<u8>;
/// 16-bit grey image (OpenCV `CV_16UC1`).
pub type ImageU16 = Image<u16>;
/// 32-bit float image (OpenCV `CV_32FC1`).
pub type ImageF32 = Image<f32>;

impl<T: Copy + Default> Image<T> {
    /// An image of `width` by `height` filled with `T::default()` (zero).
    pub fn new(width: usize, height: usize) -> Self {
        Image {
            width,
            height,
            data: vec![T::default(); width * height],
        }
    }

    /// An image filled with `value`.
    pub fn filled(width: usize, height: usize, value: T) -> Self {
        Image {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    /// Wraps row-ordered data.
    ///
    /// # Panics
    /// If `data.len() != width * height`.
    pub fn from_vec(width: usize, height: usize, data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "image data length does not match {width} x {height}"
        );
        Image {
            width,
            height,
            data,
        }
    }

    /// Copies row-ordered data.
    ///
    /// # Panics
    /// If `data.len() != width * height`.
    pub fn from_slice(width: usize, height: usize, data: &[T]) -> Self {
        Self::from_vec(width, height, data.to_vec())
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// True for an image with no pixels.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// All values, row by row from the top.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// All values, row by row from the top, for writing.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// The values, giving up the image.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Row `y`.
    pub fn row(&self, y: usize) -> &[T] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    /// Row `y`, for writing.
    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        &mut self.data[y * self.width..(y + 1) * self.width]
    }

    /// The value at column `x`, row `y`.
    ///
    /// # Panics
    /// If the position is outside the image.
    pub fn at(&self, x: usize, y: usize) -> T {
        assert!(x < self.width, "column {x} outside width {}", self.width);
        self.data[y * self.width + x]
    }

    /// Sets the value at column `x`, row `y`.
    ///
    /// # Panics
    /// If the position is outside the image.
    pub fn set(&mut self, x: usize, y: usize, value: T) {
        assert!(x < self.width, "column {x} outside width {}", self.width);
        self.data[y * self.width + x] = value;
    }

    /// A new image of the same size with `f` applied to every value.
    pub fn map<U: Copy + Default>(&self, f: impl Fn(T) -> U) -> Image<U> {
        Image {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }
}

/// An image read as `f32` grey values, as OpenCV's float routines read a `CV_32F` image:
/// [`ImageU8`] and [`ImageF32`] give their values, [`ScaledU16`] a 16-bit image on the
/// 8-bit scale. The routines that take one compute the same as on an [`ImageF32`] of
/// those values.
pub trait GreySource {
    /// Number of columns.
    fn width(&self) -> usize;
    /// Number of rows.
    fn height(&self) -> usize;
    /// Value `i` in row order, as `f32`.
    fn value(&self, i: usize) -> f32;
}

impl GreySource for ImageU8 {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn value(&self, i: usize) -> f32 {
        f32::from(self.data[i])
    }
}

impl GreySource for ImageF32 {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn value(&self, i: usize) -> f32 {
        self.data[i]
    }
}

/// A 16-bit image whose values are read saturated to `max` and then multiplied by `scale` (a
/// power of two, so exactly): `min(value, 2^bit_depth - 1) / 2^(bit_depth - 8)` puts a
/// `bit_depth`-bit image on the 8-bit scale (`crate::depth`, which makes it with
/// `crate::depth::on_8bit_scale`). Every value the deep path reads goes through
/// [`GreySource::value`] here, so this is the one place where values above the depth are
/// saturated.
#[derive(Clone, Copy, Debug)]
pub struct ScaledU16<'a> {
    /// The image.
    pub image: &'a ImageU16,
    /// The factor, `2^-(bit_depth - 8)`.
    pub scale: f32,
    /// The largest value read, `2^bit_depth - 1`; larger values read as this one.
    pub max: u16,
}

impl GreySource for ScaledU16<'_> {
    fn width(&self) -> usize {
        self.image.width
    }
    fn height(&self) -> usize {
        self.image.height
    }
    fn value(&self, i: usize) -> f32 {
        f32::from(self.image.data[i].min(self.max)) * self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_and_positions() {
        let mut im = ImageU8::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(im.width(), 3);
        assert_eq!(im.height(), 2);
        assert_eq!(im.row(1), &[4, 5, 6]);
        assert_eq!(im.at(2, 0), 3);
        im.set(0, 1, 9);
        assert_eq!(im.data(), &[1, 2, 3, 9, 5, 6]);
        let f = im.map(f32::from);
        assert_eq!(f.at(0, 1), 9.0);
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn wrong_length_panics() {
        let _ = ImageU8::from_vec(2, 2, vec![0; 3]);
    }
}
