//! OpenCV's old Levenberg-Marquardt engine, `CvLevMarq`, as `cv::calibrateCamera`
//! (`updateAlt`, the caller accumulates `J^T J` and `J^T e`) and
//! `cv::findExtrinsicCameraParams2` (`update`, the caller fills `J` and `e`) drive it.
//!
//! Ported from OpenCV 4.13 `modules/calib3d/src/compat_ptsetreg.cpp`. The normal equations
//! are solved with `cv::solve(.., DECOMP_SVD)`, [`crate::linalg::solve_svd`]: OpenCV's own
//! Jacobi SVD. The reference build hands an SVD with 25 or more free parameters to LAPACK
//! instead (parity ledger A37).

use nalgebra::DMatrix;

use super::cvcore;

/// `CvLevMarq::state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum State {
    Done,
    Started,
    CalcJ,
    CheckErr,
}

/// `CvLevMarq` with `CV_64F` matrices held row-major.
pub(crate) struct LevMarq {
    pub nparams: usize,
    pub mask: Vec<bool>,
    pub prev_param: Vec<f64>,
    pub param: Vec<f64>,
    /// `J`, `nerrs x nparams` (only with `nerrs > 0`).
    pub j: Vec<f64>,
    /// `err`, `nerrs` (only with `nerrs > 0`).
    pub err: Vec<f64>,
    pub nerrs: usize,
    pub jtj: Vec<f64>,
    pub jt_err: Vec<f64>,
    pub prev_err_norm: f64,
    pub err_norm: f64,
    pub lambda_lg10: i32,
    pub max_iter: i32,
    pub epsilon: f64,
    pub state: State,
    pub iters: i32,
    pub complete_symm_flag: bool,
}

/// What `update` / `updateAlt` hands back to the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Step {
    /// The return value (`proceed`).
    pub proceed: bool,
    /// `matJ` / `_JtJ` was set: the caller computes derivatives.
    pub want_j: bool,
    /// `_err` / `_errNorm` was set: the caller computes errors.
    pub want_err: bool,
}

impl LevMarq {
    /// `CvLevMarq::init(nparams, nerrs, criteria, completeSymmFlag)` with a criteria of type
    /// `COUNT + EPS` (`max_iter` clamped to 1..1000, `epsilon` to at least 0).
    ///
    /// Ported from OpenCV 4.13 `compat_ptsetreg.cpp`, `CvLevMarq::init`.
    pub fn new(
        nparams: usize,
        nerrs: usize,
        max_iter: i32,
        epsilon: f64,
        complete_symm_flag: bool,
    ) -> Self {
        LevMarq {
            nparams,
            mask: vec![true; nparams],
            prev_param: vec![0.0; nparams],
            param: vec![0.0; nparams],
            j: vec![0.0; nerrs * nparams],
            err: vec![0.0; nerrs],
            nerrs,
            jtj: vec![0.0; nparams * nparams],
            jt_err: vec![0.0; nparams],
            prev_err_norm: f64::MAX,
            err_norm: f64::MAX,
            lambda_lg10: -3,
            max_iter: max_iter.clamp(1, 1000),
            epsilon: epsilon.max(0.0),
            state: State::Started,
            iters: 0,
            complete_symm_flag,
        }
    }

    /// `CvLevMarq::update`: the caller fills `j` (when `want_j`) and `err` (when
    /// `want_err`) at `param`.
    ///
    /// Ported from OpenCV 4.13 `compat_ptsetreg.cpp`, `CvLevMarq::update`.
    pub fn update(&mut self) -> Step {
        let none = Step {
            proceed: false,
            want_j: false,
            want_err: false,
        };
        match self.state {
            State::Done => none,
            State::Started => {
                self.j.fill(0.0);
                self.err.fill(0.0);
                self.state = State::CalcJ;
                Step {
                    proceed: true,
                    want_j: true,
                    want_err: true,
                }
            }
            State::CalcJ => {
                // cvMulTransposed(J, JtJ, 1); cvGEMM(J, err, 1, 0, 0, JtErr, CV_GEMM_A_T).
                self.jtj = cvcore::mul_transposed_ata(&self.j, self.nerrs, self.nparams);
                self.jt_err =
                    cvcore::gemm_at_v(&self.j, self.nparams, self.nparams, &self.err, self.nerrs);
                self.prev_param.copy_from_slice(&self.param);
                self.step();
                if self.iters == 0 {
                    self.prev_err_norm = cvcore::norm_l2(&self.err);
                }
                self.err.fill(0.0);
                self.state = State::CheckErr;
                Step {
                    proceed: true,
                    want_j: false,
                    want_err: true,
                }
            }
            State::CheckErr => {
                self.err_norm = cvcore::norm_l2(&self.err);
                if self.err_norm > self.prev_err_norm {
                    self.lambda_lg10 += 1;
                    if self.lambda_lg10 <= 16 {
                        self.step();
                        self.err.fill(0.0);
                        self.state = State::CheckErr;
                        return Step {
                            proceed: true,
                            want_j: false,
                            want_err: true,
                        };
                    }
                }
                self.lambda_lg10 = (self.lambda_lg10 - 1).max(-16);
                self.iters += 1;
                if self.iters >= self.max_iter
                    || cvcore::norm_relative_l2(&self.param, &self.prev_param) < self.epsilon
                {
                    self.state = State::Done;
                    return Step {
                        proceed: true,
                        want_j: false,
                        want_err: false,
                    };
                }
                self.prev_err_norm = self.err_norm;
                self.j.fill(0.0);
                self.state = State::CalcJ;
                Step {
                    proceed: true,
                    want_j: true,
                    want_err: true,
                }
            }
        }
    }

    /// `CvLevMarq::updateAlt`: the caller accumulates `jtj`, `jt_err` (when `want_j`) and
    /// sets `err_norm` (when `want_err`) at `param`.
    ///
    /// Ported from OpenCV 4.13 `compat_ptsetreg.cpp`, `CvLevMarq::updateAlt`.
    pub fn update_alt(&mut self) -> Step {
        match self.state {
            State::Done => Step {
                proceed: false,
                want_j: false,
                want_err: false,
            },
            State::Started => {
                self.jtj.fill(0.0);
                self.jt_err.fill(0.0);
                self.err_norm = 0.0;
                self.state = State::CalcJ;
                Step {
                    proceed: true,
                    want_j: true,
                    want_err: true,
                }
            }
            State::CalcJ => {
                self.prev_param.copy_from_slice(&self.param);
                self.step();
                self.prev_err_norm = self.err_norm;
                self.err_norm = 0.0;
                self.state = State::CheckErr;
                Step {
                    proceed: true,
                    want_j: false,
                    want_err: true,
                }
            }
            State::CheckErr => {
                if self.err_norm > self.prev_err_norm {
                    self.lambda_lg10 += 1;
                    if self.lambda_lg10 <= 16 {
                        self.step();
                        self.err_norm = 0.0;
                        self.state = State::CheckErr;
                        return Step {
                            proceed: true,
                            want_j: false,
                            want_err: true,
                        };
                    }
                }
                self.lambda_lg10 = (self.lambda_lg10 - 1).max(-16);
                self.iters += 1;
                if self.iters >= self.max_iter
                    || cvcore::norm_relative_l2(&self.param, &self.prev_param) < self.epsilon
                {
                    self.state = State::Done;
                    return Step {
                        proceed: false,
                        want_j: true,
                        want_err: false,
                    };
                }
                self.prev_err_norm = self.err_norm;
                self.jtj.fill(0.0);
                self.jt_err.fill(0.0);
                self.state = State::CalcJ;
                Step {
                    proceed: true,
                    want_j: true,
                    want_err: false,
                }
            }
        }
    }

    /// `CvLevMarq::step`: solves `(JtJ + lambda diag) dp = JtErr` over the free parameters
    /// and sets `param = prevParam - dp`.
    ///
    /// Ported from OpenCV 4.13 `compat_ptsetreg.cpp`, `CvLevMarq::step` (with its
    /// `subMatrix`, `completeSymm` when there is no `err`, `diag() *= 1 + lambda`, and
    /// `cv::solve(.., DECOMP_SVD)`).
    pub fn step(&mut self) {
        let log10 = 10.0f64.ln();
        let lambda = (f64::from(self.lambda_lg10) * log10).exp();
        let n = self.nparams;
        let idx: Vec<usize> = (0..n).filter(|&i| self.mask[i]).collect();
        let nz = idx.len();
        let mut jtjn = vec![0.0f64; nz * nz];
        for (r, &i) in idx.iter().enumerate() {
            for (c, &j) in idx.iter().enumerate() {
                jtjn[r * nz + c] = self.jtj[i * n + j];
            }
        }
        let jterr: Vec<f64> = idx.iter().map(|&i| self.jt_err[i]).collect();
        if self.nerrs == 0 {
            if self.complete_symm_flag {
                // completeSymm(m, true): upper triangle from the lower one.
                for i in 0..nz {
                    for j in (i + 1)..nz {
                        jtjn[i * nz + j] = jtjn[j * nz + i];
                    }
                }
            } else {
                cvcore::complete_symm_upper(&mut jtjn, nz);
            }
        }
        let f = 1.0 + lambda;
        for i in 0..nz {
            jtjn[i * nz + i] = cvcore::scale(jtjn[i * nz + i], f);
        }
        let a = DMatrix::from_row_slice(nz, nz, &jtjn);
        let b = DMatrix::from_row_slice(nz, 1, &jterr);
        let x = crate::linalg::solve_svd(&a, &b);
        let mut j = 0usize;
        for i in 0..n {
            self.param[i] = self.prev_param[i]
                - if self.mask[i] {
                    let v = x[(j, 0)];
                    j += 1;
                    v
                } else {
                    0.0
                };
        }
    }
}
