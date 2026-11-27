# Zero-phase filtering and cubic spline: specification

This document specifies, in mathematical terms, two numerical procedures of `crates/xma-vision/src/filter.rs` whose existing code has an unclear licence (see `NOTICE.md`, sections "liir ... and filtfilt" and "Cubic spline interpolation (CodeCogs)"):

1. zero-phase (forward and backward) application of a recursive filter with given numerator and denominator coefficients, including the edge extension and the initial state;
2. interpolation by a cubic spline with not-a-knot end conditions.

It is written so that the procedures can be implemented afresh from it alone, and so that such an implementation gives the same numbers, bit for bit, as the existing code (which in turn reproduces XMALab 3). The Butterworth coefficient design (the numerator integers, the scale factor and the denominator coefficients) is **not** covered: it stays as it is and its results are inputs here. The splitting of a series into runs of valid frames, and the rule that short runs are copied, are XMALab's own code and are described only as far as the callers' contract needs (section 5).

Contents:

1. Conventions
2. Zero-phase filtering
3. Cubic spline
4. Summary of the details on which bit-identity depends
5. Callers' contract
6. Test vectors
7. Published sources
- Appendix A. Public signatures to keep
- Appendix B. The matrix inverse used for the initial state

---

## 1. Conventions

- **Arithmetic.** Every real quantity is an IEEE 754 binary64 ("double") value, rounded to nearest with ties to even. There is no single-precision arithmetic anywhere. The only integers are lengths, indices and the numerator integers of the filter design (converted exactly to doubles).
- **Notation for rounded operations.** `⊕ ⊖ ⊗ ⊘` denote the correctly rounded double sum, difference, product and quotient. Every such operation rounds once. Parentheses give the order of evaluation exactly; nothing is reassociated or reordered beyond what is written.
- **Fused multiply-add.** `fma(p, q, r)` is the exact value `p·q + r` rounded once. A fused multiply-add is used in exactly one place (the right-hand side of the initial-state system, section 2.4, in one of its two variants). Everywhere else a product followed by a sum is two roundings, and an implementation must make sure no compiler contracts them (Rust does not contract `a * b + c` on its own; do not call `mul_add` except where this document says so).
- **Negation** of a double is exact (sign flip). Multiplication by 2 is exact except on overflow.
- **Signed zero matters** in a few places and is stated where it does: adding `+0` to `−0` gives `+0`.
- **Comparisons** are IEEE ordered comparisons: any comparison with a NaN is false, and `−0 = +0`.
- **Indexing** is 0-based throughout. "Length" means number of elements.

---

## 2. Zero-phase filtering

### 2.1 Inputs

- A numerator coefficient vector `b = (b_0, …, b_{p−1})` and a denominator coefficient vector `a = (a_0, …, a_{q−1})`, doubles. From the Butterworth design, `b_k = c_k ⊗ g`, where `c_k` are the design's numerator integers (the binomial coefficients of the filter order, converted exactly to double) and `g` is the design's scale factor; `a` is the design's denominator vector, whose first entry is exactly 1. Both have length `order + 1`.
- A signal `x = (x_0, …, x_{L−1})`, doubles, `L ≥ 0`.

### 2.2 Sizes

- `N = max(p, q)`. If one vector is shorter it is extended with `+0` entries to length `N`. All further steps use the extended vectors.
- The filter order is `m = N − 1`. The procedure requires `m ≥ 1` (see 2.8).
- The edge-extension length is `E = 3m` samples at each end (12 for order 4, 6 for order 2).
- **Acceptance:** if `L ≤ E` there is no result (this includes the empty signal). Otherwise the result has length `L`. The shortest accepted signal therefore has `3m + 1` samples (13 for order 4).

### 2.3 Odd (point-reflected) extension

The extended signal `u` has length `L + 2E` and is the signal reflected through its end points:

- left part, for `j = 0, …, E−1`: `u_j = (2 ⊗ x_0) ⊖ x_{E−j}`
  (so the left part is `2x_0 − x_E, 2x_0 − x_{E−1}, …, 2x_0 − x_1`);
- middle, for `i = 0, …, L−1`: `u_{E+i} = x_i`;
- right part, for `j = 0, …, E−1`: `u_{E+L+j} = (2 ⊗ x_{L−1}) ⊖ x_{L−2−j}`
  (so the right part is `2x_{L−1} − x_{L−2}, …, 2x_{L−1} − x_{L−1−E}`).

The end samples themselves (`x_0`, `x_{L−1}`) are not repeated in the extension.

### 2.4 Steady-state initial state for a unit step

The initial state is the state vector `ζ = (ζ_0, …, ζ_{m−1})` in which the filter would sit after an infinitely long unit-step input, i.e. the solution of

`M ζ = ρ`,

computed from the **padded but not normalised** coefficient vectors `b` and `a` of 2.2 (even when `a_0 ≠ 1`; see 2.8).

**The matrix** `M` is `m × m`:

- `M_{0,0} = 1 ⊕ a_1`;
- `M_{i,0} = a_{i+1}` for `i = 1, …, m−1`;
- `M_{i,i} = 1` for `i = 1, …, m−1`;
- `M_{i,i+1} = −1` for `i = 0, …, m−2`;
- every other entry is `+0`.

That is, `M = I − C`, where `C` is the companion-type matrix whose first column is `−(a_1, …, a_m)` and whose superdiagonal is 1. For `m = 1`, `M` is the single entry `1 ⊕ a_1`. For order 4:

```
[ 1+a1  -1    0    0 ]
[ a2     1   -1    0 ]
[ a3     0    1   -1 ]
[ a4     0    0    1 ]
```

**The right-hand side** `ρ` has `m` entries, `ρ_{k−1}` for `k = 1, …, m`, mathematically `b_k − a_k b_0`. It is evaluated in one of two ways:

- **fused** (the default): `ρ_{k−1} = fma(a_k, −b_0, b_k)`;
- **unfused**: `ρ_{k−1} = (a_k ⊗ (−b_0)) ⊕ b_k`.

The fused form is what XMALab computed on the reference machine (its matrix library evaluated this expression with a fused multiply-add there). It is used by the filtering itself; the unfused form exists only for tests (section 5 says which entry point uses which). The two differ in the last bits for some coefficient sets (test vectors F4B in section 6).

**The solution method** is: form the inverse `W = M⁻¹` by the procedure of Appendix B, then multiply. It is not a direct solve, and the inverse is not formed by any other method: a different method rounds differently. For each row `r = 0, …, m−1`:

`t_r = ( … ((W_{r,0} ⊗ ρ_0) ⊕ (W_{r,1} ⊗ ρ_1)) ⊕ (W_{r,2} ⊗ ρ_2) … ) ⊕ (W_{r,m−1} ⊗ ρ_{m−1})`

(products rounded individually, summed left to right in increasing column order, no fusion), and then

`ζ_r = (t_r ⊗ 1) ⊕ (+0)`,

which equals `t_r` except that a result of `−0` becomes `+0`.

If Appendix B declares `M` singular, `W` is the zero matrix and the procedure continues with it (`ζ` is then `+0` wherever `ρ` is finite).

### 2.5 One pass of the recursive filter with a given initial state

Input: coefficient vectors `b`, `a` of length `N` (as in 2.2), a sequence `v = (v_0, …, v_{K−1})`, and a state `σ = (σ_0, …, σ_{m−1})`. Output: `w = (w_0, …, w_{K−1})`.

**Normalisation.** If `a_0 ≠ 1` (exact comparison), every coefficient is divided by the original `a_0`: `β_k = b_k ⊘ a_0`, `α_k = a_k ⊘ a_0` for all `k` (so `α_0` becomes `a_0 ⊘ a_0`). If `a_0 = 1`, `β = b` and `α = a` unchanged. (`α_0` is not used below.)

**Recursion.** This is the difference equation

`w_i = β_0 v_i + Σ_{k=1}^{min(i,m)} (β_k v_{i−k} − α_k w_{i−k}) + [i < m] σ_i`,

which is mathematically the same as the transposed direct form II realisation started from state `σ` (the state entry `σ_i` contributes to output `i` only, for `i < m`). Its exact floating-point evaluation is, for each `i = 0, …, K−1` in increasing order:

1. Let `J = min(i, m)`. Start an accumulator `s` at `σ_i` if `i < m`, and at `+0` if `i ≥ m`.
2. For `k = J, J−1, …, 1` (decreasing): `s ← ((β_k ⊗ v_{i−k}) ⊖ (α_k ⊗ w_{i−k})) ⊕ s`.
3. `w_i = (β_0 ⊗ v_i) ⊕ s`.

Points to note:

- The terms are accumulated from the highest lag down to lag 1, each term being "numerator product minus denominator product" formed first and then added to the accumulator.
- For `i ≥ m` the innermost addition is of `+0` (step 1), which turns a `−0` term into `+0`.
- For `i < m` the state entry is the innermost (first-added) quantity, not the last.

### 2.6 The whole zero-phase procedure

Given `b`, `a`, `x` with `L > E`:

1. Pad `b`, `a` to length `N` (2.2).
2. Build the extended signal `u` of length `L + 2E` (2.3).
3. Compute `ζ` (2.4), fused or unfused as the entry point requires (section 5).
4. **Forward pass:** state `σ_k = ζ_k ⊗ u_0` for `k = 0, …, m−1`; filter `u` (2.5) to get `f` (length `L + 2E`).
5. **Reversal:** `g_i = f_{L+2E−1−i}`.
6. **Backward pass:** state `σ_k = ζ_k ⊗ g_0` (note `g_0` is the last output of the forward pass); filter `g` (2.5) to get `h`. The same `ζ` is used for both passes.
7. **Reversal and trimming:** the result is `y_i = h_{L+E−1−i}` for `i = 0, …, L−1`, that is, drop `E` samples at each end of `h` and reverse what is left.

No other scaling, normalisation or rounding takes place.

### 2.7 Building `b` for the entry points

Every entry point that takes a designed filter forms `b_k = c_k ⊗ g` (`c_k` the design's integers converted exactly to double, `g` the design's scale factor) and uses the design's denominator vector unchanged as `a`. The lengths of the design's integer vector and denominator vector need not be checked to be equal; 2.2 pads the shorter one.

### 2.8 Degenerate inputs

- `L ≤ E`, including `L = 0`: no result.
- `a_0 ≠ 1`: the passes normalise (2.5) but the initial state is computed from the unnormalised coefficients (2.4). This mismatch is how the existing code behaves (test vector FA in section 6). It cannot happen with a designed filter, whose `a_0` is exactly 1. `a_0 = 0` gives infinities or NaNs, with no error.
- `m = 0` (both vectors of length 1, or one of length 1 and the other empty): the existing code aborts (Rust panic) for any `L ≥ 1` while building the initial state; for `L = 0` it returns no result. A designed filter always has `m ≥ 1`. Both coefficient vectors empty: also an abort in the existing code. An implementation keeps returning no result for `L ≤ E`; `m = 0` with `L ≥ 1` is the existing code's abort.
- Non-finite samples or coefficients are processed by the same arithmetic; there is no check.

---

## 3. Cubic spline

### 3.1 Inputs and output

- Knots `ξ = (ξ_0, …, ξ_{n−1})` (meant to be strictly increasing) and values `η = (η_0, …, η_{n−1})`, doubles, the same length `n`.
- Evaluation points `τ = (τ_0, …, τ_{P−1})`, doubles, `P ≥ 0`, meant to be non-decreasing.
- Output: either no result, or values `(z_0, …, z_{P−1})`, one per evaluation point, in the same order.

The interpolant is the piecewise cubic in Hermite form: on each interval `[ξ_j, ξ_{j+1}]` it is the cubic that takes the values `η_j`, `η_{j+1}` and the slopes `s_j`, `s_{j+1}` at the ends. The slopes `s` solve the tridiagonal system of 3.3, which makes the second derivative continuous at interior knots and imposes the **not-a-knot** condition at both ends (the third derivative is continuous at `ξ_1` and at `ξ_{n−2}`).

### 3.2 Order of the checks

1. If `ξ` and `η` differ in length the existing code aborts (Rust panic); this is a misuse by the caller (section 5).
2. If `n < 3` or `P = 0`: no result.
3. Solve for the slopes (3.3, 3.4). If the tridiagonal solve fails: no result.
4. If `τ_0 < ξ_0` or `τ_{P−1} > ξ_{n−1}`: no result. Only the first and the last evaluation points are tested here.
5. Evaluate (3.5); the evaluation can still end with no result.

(Steps 3 and 4 both give "no result", so their order does not affect the output.)

### 3.3 The slope equations

Let, for `i = 0, …, n−2`,

- `h_i = ξ_{i+1} ⊖ ξ_i` (interval width),
- `δ_i = (η_{i+1} ⊖ η_i) ⊘ h_i` (divided difference, a true division).

The system has `n` unknowns `s_0, …, s_{n−1}`. Row `i` has a sub-diagonal entry `l_i` (for `i ≥ 1`), a diagonal entry `d_i`, a super-diagonal entry `e_i` (for `i ≤ n−2`) and a right-hand side `q_i`:

- **Row 0** (not-a-knot at the left end), with `σ₀₁ = h_0 ⊕ h_1`:
  - `d_0 = h_1 ⊗ σ₀₁`
  - `e_0 = σ₀₁ ⊗ σ₀₁`
  - `q_0 = (δ_0 ⊗ (((3 ⊗ h_0) ⊗ h_1) ⊕ ((2 ⊗ h_1) ⊗ h_1))) ⊕ ((δ_1 ⊗ h_0) ⊗ h_0)`

  This is the not-a-knot equation `h_1 s_0 + (h_0+h_1) s_1 = ((3h_0h_1 + 2h_1²) δ_0 + h_0² δ_1) / (h_0+h_1)` multiplied through by `h_0 + h_1`.

- **Interior rows** `i = 1, …, n−2` (continuity of the second derivative at `ξ_i`):
  - `l_i = h_i`
  - `d_i = 2 ⊗ (h_{i−1} ⊕ h_i)`
  - `e_i = h_{i−1}`
  - `q_i = 3 ⊗ ((δ_{i−1} ⊗ h_i) ⊕ (δ_i ⊗ h_{i−1}))`

- **Row n−1** (not-a-knot at the right end), with `P = h_{n−3}`, `Q = h_{n−2}`, `σ = P ⊕ Q`:
  - `l_{n−1} = σ ⊗ σ`
  - `d_{n−1} = P ⊗ σ`
  - `q_{n−1} = ((δ_{n−3} ⊗ Q) ⊗ Q) ⊕ (δ_{n−2} ⊗ (((3 ⊗ P) ⊗ Q) ⊕ ((2 ⊗ P) ⊗ P)))`

  This is `(h_{n−3}+h_{n−2})² s_{n−2} + h_{n−3}(h_{n−3}+h_{n−2}) s_{n−1} = h_{n−2}² δ_{n−3} + (3h_{n−3}h_{n−2} + 2h_{n−3}²) δ_{n−2}`.

Every `h_i` and `δ_i` is the same double wherever it appears. With exactly three knots the first and last rows describe the same condition and the system is singular in exact arithmetic (3.6).

### 3.4 Tridiagonal solve (Thomas algorithm, no pivoting)

Gaussian elimination down the diagonal, no row exchanges, then back-substitution:

1. `π_0 = d_0`; `q'_0 = q_0 ⊘ π_0`. (`π_0` is **not** tested for zero.)
2. For `i = 1, …, n−1` in increasing order:
   - `γ_i = e_{i−1} ⊘ π_{i−1}`;
   - `π_i = d_i ⊖ (l_i ⊗ γ_i)`;
   - if `π_i = 0` exactly (either sign), the solve **fails**: no result;
   - `q'_i = (q_i ⊖ (q'_{i−1} ⊗ l_i)) ⊘ π_i`.
3. `s_{n−1} = q'_{n−1}`; for `i = n−2, …, 0` in decreasing order: `s_i = q'_i ⊖ (s_{i+1} ⊗ γ_{i+1})`.

A NaN pivot is not zero and does not fail. There is no tolerance: only an exact zero fails.

### 3.5 Evaluation

The evaluation walks through the points in the given order, keeping a **current interval** `[λ, R]` with its four cubic coefficients `c_0, c_1, c_2, c_3`, and a **knot index** `j` that only moves as described.

**Start:** `j = 0`; `λ = +0`; `c_0 = c_1 = c_2 = c_3 = +0`; `R = τ_0 ⊖ 1`.

For each point `t = τ_p`, `p = 0, 1, …`:

1. **Choose a new interval only if `t > R`.** Otherwise keep the current interval and its coefficients, whatever `t` is (so a point that does not exceed the right end of the current interval is evaluated on the current cubic, by extrapolation if it lies left of `λ`). When a new interval is chosen:
   1. Increase `j` (from its current value; it is never reset) while `j < n` and `t > ξ_j`.
   2. If `j = n`: stop, **no result** for the whole call.
   3. If `t < ξ_j`: if `j = 0`, stop with no result; otherwise decrease `j` by one. (If `t = ξ_j`, `j` is kept: a point exactly on a knot opens the interval that starts at that knot.)
   4. If `j = n−1`: stop, **no result**. (This happens when `t` equals the last knot at the moment a new interval is chosen.)
   5. Set `λ = ξ_j`, `R = ξ_{j+1}`, and the coefficients
      - `ω = 1 ⊘ (ξ_{j+1} ⊖ ξ_j)` (the reciprocal of the width),
      - `μ = (η_{j+1} ⊖ η_j) ⊗ ω` (the slope of the chord, by multiplication with the reciprocal, **not** the `δ_j` of 3.3),
      - `c_0 = η_j`,
      - `c_1 = s_j`,
      - `c_2 = ω ⊗ (((3 ⊗ μ) ⊖ (2 ⊗ s_j)) ⊖ s_{j+1})`,
      - `c_3 = (ω ⊗ ω) ⊗ ((((−2) ⊗ μ) ⊕ s_j) ⊕ s_{j+1})`.
2. **Evaluate** by Horner's rule in the local variable `θ = t ⊖ λ`:
   `z_p = (((((c_3 ⊗ θ) ⊕ c_2) ⊗ θ) ⊕ c_1) ⊗ θ) ⊕ c_0`, that is `(((c_3θ + c_2)θ + c_1)θ + c_0` with each operation rounded.

These coefficients are the power form of the cubic Hermite interpolant on `[ξ_j, ξ_{j+1}]`: `c_2 = (3μ − 2s_j − s_{j+1}) / h`, `c_3 = (s_j + s_{j+1} − 2μ) / h²`.

Consequences, all part of the behaviour to reproduce:

- A point exactly on the knot that starts its interval has `θ = +0`; the Horner sum then gives `η_j` when the coefficients are finite, except that a value `η_j = −0` comes out as `+0` (test vector Z1). Infinite coefficients give NaN there.
- The last knot `ξ_{n−1}` evaluates on the last interval when that interval is already current (test vector E1), but gives no result when it is the point that would open a new interval (test vector E2).
- A later point below `ξ_0` is extrapolated on the current interval (E7, E3); a later point above `ξ_{n−1}` that exceeds `R` gives no result (E6).
- If `τ_0 ⊖ 1` rounds back to `τ_0` (always for `|τ_0| ≥ 2^54` and for infinite `τ_0`, and for some values with `2^53 < |τ_0| < 2^54`), or `τ_0` is NaN, the first point does not exceed `R`, no interval is chosen, and the point is evaluated with the all-zero start coefficients and `λ = +0` (giving `+0` for finite `τ_0`; test vector B1). The interval is then first chosen at the first later point that exceeds `R` (never, if `τ_0` is NaN).
- Comparisons with a NaN point are false, so a NaN point never opens an interval; its value is NaN.

### 3.6 Degenerate inputs

- `n < 3` or no evaluation points: no result.
- `n = 3`: the system is singular in exact arithmetic. In floating point the second pivot is often exactly 0 and the solve fails (no result); when rounding leaves it non-zero, the solve continues and its result is numerically meaningless (test vectors T1 to T5). This is the existing behaviour (parity ledger A46, "Copy for 3 knots").
- Repeated knots (`h_i = 0`): divisions by zero give infinities or NaNs; the first pivot is not tested, later pivots only for exact zero; the output is typically NaN, not "no result" (test vector D1).
- Knots that are not increasing: not checked; the same arithmetic applies.

---

## 4. Summary of the details on which bit-identity depends

For a reviewer, every place where a stated detail determines the last bit:

Filtering:

1. `b_k = c_k ⊗ g`, one rounding (2.1, 2.7).
2. Extension length `E = 3m` and the reflection formula `(2 ⊗ x_end) ⊖ x_k`, with the end sample excluded (2.3).
3. Acceptance `L > 3m` (2.2).
4. The initial-state matrix `M` with `M_{0,0} = 1 ⊕ a_1`, from unnormalised coefficients (2.4).
5. The right-hand side fused as `fma(a_k, −b_0, b_k)` by default, unfused `(a_k ⊗ (−b_0)) ⊕ b_k` only for the test entry points (2.4, section 5).
6. The inverse by the exact procedure of Appendix B (closed forms for `m ≤ 3`, partial-pivoting elimination for `m ≥ 4`, absolute pivot threshold, zero matrix when singular).
7. The product `W ρ` summed left to right, unfused, then `−0 → +0` (2.4).
8. The state scaled as `ζ_k ⊗ u_0` for the forward pass and `ζ_k ⊗ g_0` for the backward pass, the same `ζ` both times (2.6).
9. The recursion's accumulation order: highest lag first, each term `(β_k ⊗ v) ⊖ (α_k ⊗ w)` then `⊕` the accumulator, state entry innermost, `+0` innermost for `i ≥ m`, and `w_i = (β_0 ⊗ v_i) ⊕ s` (2.5).
10. Normalisation by `a_0` only if `a_0 ≠ 1` (2.5).
11. Reversal and trimming indices (2.6).

Spline:

12. `h_i` by subtraction and `δ_i` by true division (3.3).
13. The exact parenthesisation of the end rows' entries and right-hand sides, and of the interior right-hand side `3 ⊗ (… ⊕ …)` (3.3).
14. Thomas elimination with `γ_i = e_{i−1} ⊘ π_{i−1}`, `π_i = d_i ⊖ (l_i ⊗ γ_i)`, `q'_i = (q_i ⊖ (q'_{i−1} ⊗ l_i)) ⊘ π_i`, back-substitution `s_i = q'_i ⊖ (s_{i+1} ⊗ γ_{i+1})`; first pivot untested, later pivots tested for exact zero (3.4).
15. Interval coefficients from the reciprocal `ω = 1 ⊘ width` and `μ = Δη ⊗ ω`, with `c_2`, `c_3` parenthesised as in 3.5.
16. Horner evaluation in `θ = t ⊖ ξ_j` (3.5).
17. The interval-selection rules: new interval only when `t > R`, persistent forward-moving index, on-knot rule, start value `R = τ_0 ⊖ 1`, the "no result" conditions (3.5).
18. Range check on the first and last points only (3.2).

---

## 5. Callers' contract

All functions are in `xma_vision::filter`; their signatures are listed in Appendix A and must not change. Indices are 0-based; filtered values are in whatever unit the samples are in (positions in project units, rotation components in radians; the filter does not care).

- **`ButterworthLowPass::filter(&self, input)`.** Input: the samples of one contiguous run, one component, in frame order. Output: `None` if `input.len() ≤ 3·order` (including empty); otherwise `Some` of a vector of the same length, the zero-phase filtered samples (section 2, fused initial state). `order` is `d.len() − 1` for a designed filter. The coefficient vectors come from the struct's public fields as in 2.7: `b_k = c_k ⊗ scale` from the integer field, `a` = the denominator field.
- **`filter_unfused_zi(f, input)`.** As `filter`, with the unfused right-hand side in 2.4. For tests only (marked hidden).
- **`initial_conditions_for(f, fma)`.** Returns `ζ` of 2.4 (length `order`, before scaling by any sample), with `b` formed as in 2.7; `fma = true` gives the fused right-hand side, `false` the unfused. For tests only (marked hidden).
- **`filter_run(filter, values)`.** One run of one component: if `values.len() ≤ 12` (the constant `MIN_FILTERED_RUN`, which is `3 × FILTER_ORDER`) the values are returned unchanged (a copy); otherwise the result of `filter` on them. With a filter of order above 4 a run of length 13 to `3·order` would make `filter` return `None`; the existing code then aborts (Rust panic, "runs longer than MIN_FILTERED_RUN are long enough for filtfilt"). XMALab only uses order 4.
- **`filter_runs(filter, values, runs, out)`.** `values` and `out` have one entry per frame and must be the same length (the existing code aborts otherwise). For each range in `runs` (half-open frame ranges, normally from `runs()`), `out[range]` is overwritten with `filter_run` of `values[range]`. Frames outside every range keep what the caller put in `out`. A range outside the slices aborts.
- **`cubic_spline(x, y, at)`.** `x` are the knot positions (in XMALab, the 0-based frame numbers of frames with known values, increasing), `y` the known values (one coordinate), `at` the positions to fill (the frames of a gap, increasing, strictly inside the knot range). Output: `None` in the cases of section 3; otherwise `Some` of one value per entry of `at`, in order. `x` and `y` must be the same length (the existing code aborts otherwise). XMALab's caller always has at least 4 knots and evaluates strictly inside them in increasing order.

These functions are pure: no global state, no allocation beyond the output and working vectors proportional to the input lengths.

### 5.1 Order bound (added 16 November 2026)

There is a filter only for orders 1 to 29 (`MAX_FILTER_ORDER`). 29 is the largest order whose numerator integers the design computes exactly in 32-bit integers, as XMALab does: the intermediate product `i · C(n, i)` stays below 2^31 − 1 up to order 29 (largest value 14 · C(29, 14) = 1 085 822 640) and exceeds it at order 30 (15 · C(30, 15) = 2 326 762 800). Outside that range:

- `ButterworthLowPass::new` and `zero_phase` return the filter of order 0 (`d = [1]`, `c = [1]`, `scale = 1`), which filters nothing; `design_cutoff` returns the entered cutoff unchanged.
- A filter whose order `m` (section 2.2: the longer coefficient vector's length less one) is 0 or above 29, including one built through the public fields, gives no result from `filter` and `filter_unfused_zi` whatever the signal length, an empty vector from `initial_conditions_for`, and its values unchanged from `filter_run` and `filter_runs`.

This replaces the aborts of section 2.8 for `m = 0`. The results of sections 2 and 6 for orders 1 to 29 are unchanged; the matrix of section 2.4 and Appendix B is at most 29 × 29.

---

## 6. Test vectors

Computed with the existing code (`crates/xma-vision/src/filter.rs`, 16 November 2026) on x86-64 Linux. Every double is given as its IEEE 754 bit pattern in hexadecimal and as the shortest decimal that reads back to the same double. Inputs written in decimal (for example `0.1`) mean the nearest double to that decimal. Filter designs are named by the arguments of `ButterworthLowPass::new(order, cutoff, rate)`; the coefficients are listed so that they can be used directly.

### 6.1 Coefficient sets and initial states

Fused and unfused initial states agree for F4, F2, F1, F3 and FA, and differ for F4B (entry 0, by one unit in the last place).

**F4: order 4, cutoff 10, rate 100**. Scale factor `0x3f73c2b0cf231623` = `0.004824343357716229`.

| k | b_k (hex) | b_k | a_k (hex) | a_k |
|---|---|---|---|---|
| 0 | `0x3f73c2b0cf231623` | `0.004824343357716229` | `0x3ff0000000000000` | `1.0` |
| 1 | `0x3f93c2b0cf231623` | `0.019297373430864916` | `0xc002f4c33c4a5904` | `-2.3695130071820376` |
| 2 | `0x3f9da40936b4a134` | `0.028946060146297373` | `0x4002830c5b99e9ad` | `2.31398841441588` |
| 3 | `0x3f93c2b0cf231623` | `0.019297373430864916` | `0xbff0dfe8d5278946` | `-1.0546654058785676` |
| 4 | `0x3f73c2b0cf231623` | `0.004824343357716229` | `0x3fc7fc0d1bd4cab3` | `0.18737949236818494` |

Initial state (unscaled):

| k | fused (hex) | fused | unfused (hex) | unfused |
|---|---|---|---|---|
| 0 | `0x3fefd87a9e61b9b8` | `0.9951756566422807` | `0x3fefd87a9e61b9b8` | `0.9951756566422807` |
| 1 | `0xbff64c53eca06163` | `-1.3936347239706144` | `0xbff64c53eca06163` | `-1.3936347239706144` |
| 2 | `0x3fec86694b713ea3` | `0.891407630298961` | `0x3fec86694b713ea3` | `0.891407630298961` |
| 3 | `0xbfc75df7955bb1ed` | `-0.18255514901046813` | `0xbfc75df7955bb1ed` | `-0.18255514901046813` |

**F2: order 2, cutoff 30, rate 250**. Scale factor `0x3fb76069cf0267ac` = `0.09131490043583196`.

| k | b_k (hex) | b_k | a_k (hex) | a_k |
|---|---|---|---|---|
| 0 | `0x3fb76069cf0267ac` | `0.09131490043583196` | `0x3ff0000000000000` | `1.0` |
| 1 | `0x3fc76069cf0267ac` | `0.18262980087166392` | `0xbfef6fde4619a5e4` | `-0.9824057931083954` |
| 2 | `0x3fb76069cf0267ac` | `0.09131490043583196` | `0x3fd640265b35b376` | `0.3476653948517233` |

Initial state (unscaled):

| k | fused (hex) | fused | unfused (hex) | unfused |
|---|---|---|---|---|
| 0 | `0x3fed13f2c61fb309` | `0.9086850995641679` | `0x3fed13f2c61fb309` | `0.9086850995641679` |
| 1 | `0xbfd0680be7751989` | `-0.25635049441589125` | `0xbfd0680be7751989` | `-0.25635049441589125` |

**F4B: order 4, cutoff 15, rate 100**. Scale factor `0x3f93022e8e3b3629` = `0.018563010626897167`.

| k | b_k (hex) | b_k | a_k (hex) | a_k |
|---|---|---|---|---|
| 0 | `0x3f93022e8e3b3629` | `0.018563010626897167` | `0x3ff0000000000000` | `1.0` |
| 1 | `0x3fb3022e8e3b3629` | `0.07425204250758867` | `0xbff9205a8bbb3728` | `-1.5703988512281715` |
| 2 | `0x3fbc8345d558d13e` | `0.11137806376138301` | `0x3ff468e9849251f2` | `1.2756133249832797` |
| 3 | `0x3fb3022e8e3b3629` | `0.07425204250758867` | `0xbfdf0076fc449262` | `-0.48440336833508557` |
| 4 | `0x3f93022e8e3b3629` | `0.018563010626897167` | `0x3fb381a69c8d75a9` | `0.07619706461033239` |

Initial state (unscaled):

| k | fused (hex) | fused | unfused (hex) | unfused |
|---|---|---|---|---|
| 0 | `0x3fef67ee8b8e2644` | `0.9814369893731016` | `0x3fef67ee8b8e2645` | `0.9814369893731018` |
| 1 | `0xbfe5390c5dafaec0` | `-0.6632139043626566` | `0xbfe5390c5dafaec0` | `-0.6632139043626566` |
| 2 | `0x3fe0085df0c9daee` | `0.5010213568592385` | `0x3fe0085df0c9daee` | `0.5010213568592385` |
| 3 | `0xbfad8235f1fd5032` | `-0.05763405398343514` | `0xbfad8235f1fd5032` | `-0.05763405398343514` |

**F1: order 1, cutoff 20, rate 100**. Scale factor `0x3fdaee83c1138c1b` = `0.4208077798377318`.

| k | b_k (hex) | b_k | a_k (hex) | a_k |
|---|---|---|---|---|
| 0 | `0x3fdaee83c1138c1b` | `0.4208077798377318` | `0x3ff0000000000000` | `1.0` |
| 1 | `0x3fdaee83c1138c1b` | `0.4208077798377318` | `0xbfc445f0fbb1cf92` | `-0.1583844403245363` |

Initial state (unscaled):

| k | fused (hex) | fused | unfused (hex) | unfused |
|---|---|---|---|---|
| 0 | `0x3fe288be1f7639f2` | `0.5791922201622681` | `0x3fe288be1f7639f2` | `0.5791922201622681` |

**F3: order 3, cutoff 15, rate 100**. Scale factor `0x3fa95c638edac5b1` = `0.049532996357253174`.

| k | b_k (hex) | b_k | a_k (hex) | a_k |
|---|---|---|---|---|
| 0 | `0x3fa95c638edac5b1` | `0.049532996357253174` | `0x3ff0000000000000` | `1.0` |
| 1 | `0x3fc3054aab241445` | `0.14859898907175953` | `0xbff29736c99054ce` | `-1.1619174836717323` |
| 2 | `0x3fc3054aab241445` | `0.14859898907175953` | `0x3fe64529be002638` | `0.6959427557896509` |
| 3 | `0x3fa95c638edac5b1` | `0.049532996357253174` | `0xbfc1a2298dc86701` | `-0.13776130125989286` |

Initial state (unscaled):

| k | fused (hex) | fused | unfused (hex) | unfused |
|---|---|---|---|---|
| 0 | `0x3fee6a39c712539d` | `0.950467003642746` | `0x3fee6a39c712539d` | `0.950467003642746` |
| 1 | `0xbfd70b0cedaeb60e` | `-0.36004946910074487` | `0xbfd70b0cedaeb60e` | `-0.36004946910074487` |
| 2 | `0x3fc7f942717f1868` | `0.1872942976171459` | `0x3fc7f942717f1868` | `0.1872942976171459` |

**FA: public fields, denominator (2, −0.5, 0.25), integers (1, 2, 1), scale 0.5**. Scale factor `0x3fe0000000000000` = `0.5`.

| k | b_k (hex) | b_k | a_k (hex) | a_k |
|---|---|---|---|---|
| 0 | `0x3fe0000000000000` | `0.5` | `0x4000000000000000` | `2.0` |
| 1 | `0x3ff0000000000000` | `1.0` | `0xbfe0000000000000` | `-0.5` |
| 2 | `0x3fe0000000000000` | `0.5` | `0x3fd0000000000000` | `0.25` |

Initial state (unscaled):

| k | fused (hex) | fused | unfused (hex) | unfused |
|---|---|---|---|---|
| 0 | `0x4001555555555555` | `2.1666666666666665` | `0x4001555555555555` | `2.1666666666666665` |
| 1 | `0xbfc5555555555554` | `-0.16666666666666663` | `0xbfc5555555555554` | `-0.16666666666666663` |

### 6.2 Filter outputs

The shortest accepted lengths are 13 (order 4), 7 (order 2), 4 (order 1) and 10 (order 3); lengths 12 for order 4 and 6 for order 2 are one too short. FA uses the struct's public fields directly (denominator `(2, −0.5, 0.25)`, integers `(1, 2, 1)`, scale `0.5`) to exercise 2.8.

**F4, S13** (length 13): input `[1.0, 2.5, -0.75, 3.0, 4.25, 2.0, 0.5, -1.5, 0.0, 2.75, 3.5, 1.25, -0.5]`

Fused and unfused initial state give the same output.

| i | output (hex) | output |
|---|---|---|
| 0 | `0x3fef0d1214be6dd2` | `0.9703455357698039` |
| 1 | `0x3ff5992b7d4dde44` | `1.3498949904150246` |
| 2 | `0x3ffa5d0c58daff4d` | `1.6477168532291728` |
| 3 | `0x3ffcf2b7e3ba4cda` | `1.8092574020018248` |
| 4 | `0x3ffd3e4c223159b4` | `1.82770932537606` |
| 5 | `0x3ffbe2b700a73637` | `1.742850306083154` |
| 6 | `0x3ff9d8fdec5d7a2c` | `1.615476535130834` |
| 7 | `0x3ff7c4f0f8bbdda2` | `1.485581370941738` |
| 8 | `0x3ff56e035274bff7` | `1.3393586369456878` |
| 9 | `0x3ff1ca0ab62b2fd4` | `1.1118266216244495` |
| 10 | `0x3fe75aa0531a4447` | `0.7298127768609753` |
| 11 | `0x3fc5a2f02341271b` | `0.16903497430239942` |
| 12 | `0xbfe077bd9cd87f2f` | `-0.5146167815582653` |

**F4 S12, one too short** (length 12): input `[1.0, 2.5, -0.75, 3.0, 4.25, 2.0, 0.5, -1.5, 0.0, 2.75, 3.5, 1.25]`

No result (both variants).

**F4 S20** (length 20): input `[0.5, -1.625, -2.75, -2.875, -4.25, -4.625, -4.0, -4.625, -4.25, -2.875, -2.75, -1.625, 0.5, 1.375, 3.25, 6.125, 7.75, 10.375, 14.0, 16.375]`

Fused and unfused initial state give the same output.

| i | output (hex) | output |
|---|---|---|
| 0 | `0x3fddf0c073bf2603` | `0.4678193216646777` |
| 1 | `0xbfeedb080af8e490` | `-0.9642372335721507` |
| 2 | `0xc00209dadd549332` | `-2.254811982291721` |
| 3 | `0xc00a70bc77f6dfb7` | `-3.3050469753470355` |
| 4 | `0xc01038a47dacd81d` | `-4.055314983057568` |
| 5 | `0xc011f1607d477ca2` | `-4.485719640243731` |
| 6 | `0xc0126cfca79744a1` | `-4.606432551009818` |
| 7 | `0xc011c5cfe18d5789` | `-4.443175815822436` |
| 8 | `0xc01017dccc2a8cd9` | `-4.023303213212068` |
| 9 | `0xc00aede2bf1a2d65` | `-3.36615514086377` |
| 10 | `0xc003d4ed4a472a14` | `-2.4789682200400964` |
| 11 | `0xbff5b97606c1c218` | `-1.3577785743656303` |
| 12 | `0x3f80d6e0b96db3c6` | `0.008222346934155921` |
| 13 | `0x3ffa1762c0db36d6` | `1.6307094129037316` |
| 14 | `0x400c2124c3b266a3` | `3.516183403856148` |
| 15 | `0x4016a4de89bfd45f` | `5.661005165418003` |
| 16 | `0x402018ba1c267d48` | `8.048294906328593` |
| 17 | `0x40254b40d0a73ef3` | `10.646978874603041` |
| 18 | `0x402ad33d3610ec4c` | `13.412576379361916` |
| 19 | `0x403049e47bfd160a` | `16.288642644203755` |

**F2 S7** (length 7): input `[1.0, 2.5, -0.75, 3.0, 4.25, 2.0, 0.5]`

Fused and unfused initial state give the same output.

| i | output (hex) | output |
|---|---|---|
| 0 | `0x3ff085e56ac08f39` | `1.0326894922263377` |
| 1 | `0x3ff7681606655bd2` | `1.4629116296616655` |
| 2 | `0x3ffe7e41d0b3c615` | `1.905824485057001` |
| 3 | `0x4001f6f662942674` | `2.245587129732309` |
| 4 | `0x40019a7ef6cd38ba` | `2.2004374772432955` |
| 5 | `0x3ff937ebfcf1a157` | `1.5761527901286934` |
| 6 | `0x3fe0f9de8c3320b3` | `0.5305016268039452` |

**F2 S6, one too short** (length 6): input `[1.0, 2.5, -0.75, 3.0, 4.25, 2.0]`

No result (both variants).

**F2 S13** (length 13): input `[1.0, 2.5, -0.75, 3.0, 4.25, 2.0, 0.5, -1.5, 0.0, 2.75, 3.5, 1.25, -0.5]`

Fused and unfused initial state give the same output.

| i | output (hex) | output |
|---|---|---|
| 0 | `0x3ff04c2f9a448b23` | `1.0186000848850163` |
| 1 | `0x3ff69a9a239165af` | `1.4127446545956948` |
| 2 | `0x3ffcdf9c331fc079` | `1.8045923230202148` |
| 3 | `0x4000d4a4c3af9b8e` | `2.1038298881498525` |
| 4 | `0x4000ba775137c5f7` | `2.0910478921075497` |
| 5 | `0x3ffacbef32438883` | `1.6747886622838741` |
| 6 | `0x3ff1fb5436c2eb49` | `1.1238596095197002` |
| 7 | `0x3febd48fd71cf0c9` | `0.8696974946429005` |
| 8 | `0x3ff13cede0e0a682` | `1.077375296047109` |
| 9 | `0x3ff6f8685e1e8419` | `1.4356464077502606` |
| 10 | `0x3ff682355ac5e2fc` | `1.4067891641445263` |
| 11 | `0x3fe6766d5078ba64` | `0.7019564220152685` |
| 12 | `0xbfe013e5aeadb60c` | `-0.5024288570108992` |

**F4B S13** (length 13): input `[1.0, 2.5, -0.75, 3.0, 4.25, 2.0, 0.5, -1.5, 0.0, 2.75, 3.5, 1.25, -0.5]`

| i | fused (hex) | fused | unfused (hex) | unfused |
|---|---|---|---|---|
| 0 | `0x3fefe095a8935b7a` | `0.996165112733123` | `0x3fefe095a8935b82` | `0.9961651127331239` |
| 1 | `0x3ff9d15d789f4f53` | `1.6136145316895252` | `0x3ff9d15d789f4f56` | `1.6136145316895258` |
| 2 | `0x400174f9807b175b` | `2.182116512066313` | `0x400174f9807b175a` | `2.1821165120663126` |
| 3 | `0x4003cf1437ffe167` | `2.476112782951691` | `0x4003cf1437ffe165` | `2.4761127829516902` |
| 4 | `0x4001d37c34d59534` | `2.228264248626738` | `0x4001d37c34d59533` | `2.2282642486267377` |
| 5 | `0x3ff76194fc62a0ec` | `1.4613237246322184` | `0x3ff76194fc62a0ed` | `1.4613237246322186` |
| 6 | `0x3fe4affbd5ae9645` | `0.6464823888279133` | `0x3fe4affbd5ae9646` | `0.6464823888279134` |
| 7 | `0x3fd9e0d26b3f7621` | `0.4043470427839982` | `0x3fd9e0d26b3f761d` | `0.40434704278399797` |
| 8 | `0x3fee3df268c6d1bf` | `0.9450618788164816` | `0x3fee3df268c6d1b9` | `0.945061878816481` |
| 9 | `0x3ffc8cbec3d3d28b` | `1.7843616151341475` | `0x3ffc8cbec3d3d28a` | `1.7843616151341473` |
| 10 | `0x4000804527fb52ac` | `2.0626319049427675` | `0x4000804527fb52ad` | `2.062631904942768` |
| 11 | `0x3ff3af6017d3a789` | `1.2303162508710292` | `0x3ff3af6017d3a78c` | `1.23031625087103` |
| 12 | `0xbfdee50347b01c20` | `-0.48272783280157405` | `0xbfdee50347b01c18` | `-0.4827278328015736` |

**F4B S20** (length 20): input `[0.5, -1.625, -2.75, -2.875, -4.25, -4.625, -4.0, -4.625, -4.25, -2.875, -2.75, -1.625, 0.5, 1.375, 3.25, 6.125, 7.75, 10.375, 14.0, 16.375]`

| i | fused (hex) | fused | unfused (hex) | unfused |
|---|---|---|---|---|
| 0 | `0x3fe04043fbfd0d2d` | `0.5078449174547238` | `0x3fe04043fbfd0d39` | `0.5078449174547252` |
| 1 | `0xbff1cbe99607ea0e` | `-1.1122833118994353` | `0xbff1cbe99607ea0c` | `-1.1122833118994349` |
| 2 | `0xc003e8946c968dee` | `-2.488564346642071` | `0xc003e8946c968df0` | `-2.4885643466420717` |
| 3 | `0xc00bde140c89f6d9` | `-3.4834366778949115` | `0xc00bde140c89f6dd` | `-3.4834366778949133` |
| 4 | `0xc01062c01f243ba8` | `-4.096436010920037` | `0xc01062c01f243ba8` | `-4.096436010920037` |
| 5 | `0xc0119ca5f2ad8532` | `-4.4029767912300475` | `0xc0119ca5f2ad8530` | `-4.402976791230046` |
| 6 | `0xc011e65199fadfbb` | `-4.474920659958964` | `0xc011e65199fadfb8` | `-4.474920659958961` |
| 7 | `0xc0115ad1e1b38ac0` | `-4.338691259940504` | `0xc0115ad1e1b38abe` | `-4.338691259940502` |
| 8 | `0xc00fd7cba67b5f43` | `-3.980368900909441` | `0xc00fd7cba67b5f44` | `-3.9803689009094416` |
| 9 | `0xc00afed7d90cea0c` | `-3.3744351346078982` | `0xc00afed7d90cea0e` | `-3.374435134607899` |
| 10 | `0xc0040e514cae1389` | `-2.5069910040496706` | `0xc0040e514cae1388` | `-2.50699100404967` |
| 11 | `0xbff616cbb3b39443` | `-1.3805653590764144` | `0xbff616cbb3b3943b` | `-1.3805653590764126` |
| 12 | `0xbf7484e8979b5128` | `-0.005009563983282818` | `0xbf7484e8979b4b58` | `-0.005009563983281527` |
| 13 | `0x3ff9d8cf17e019b0` | `1.6154318745262621` | `0x3ff9d8cf17e019ad` | `1.6154318745262615` |
| 14 | `0x400be7f0f1335a9b` | `3.488252529506395` | `0x400be7f0f1335a95` | `3.488252529506392` |
| 15 | `0x401680ca69544fa3` | `5.625772138377132` | `0x401680ca69544fa1` | `5.625772138377131` |
| 16 | `0x40200eba6bb11e01` | `8.028766026850464` | `0x40200eba6bb11e02` | `8.028766026850466` |
| 17 | `0x4025570a72ce0666` | `10.67000159039453` | `0x4025570a72ce066a` | `10.670001590394538` |
| 18 | `0x402af99c0b2317cf` | `13.487518642457784` | `0x402af99c0b2317d3` | `13.487518642457792` |
| 19 | `0x4030647426b69c76` | `16.39239732702523` | `0x4030647426b69c76` | `16.39239732702523` |

**F1 S4** (length 4): input `[1.0, 2.5, -0.75, 3.0]`

Fused and unfused initial state give the same output.

| i | output (hex) | output |
|---|---|---|
| 0 | `0x3ff00f6784a5ea40` | `1.0037608319822908` |
| 1 | `0x3ff454a070e37c86` | `1.2706608209335015` |
| 2 | `0x3ff59b71c9894722` | `1.3504503128537597` |
| 3 | `0x4007fae1ef5cab46` | `2.9975012493281694` |

**F3 S10** (length 10): input `[1.0, 2.5, -0.75, 3.0, 4.25, 2.0, 0.5, -1.5, 0.0, 2.75]`

Fused and unfused initial state give the same output.

| i | output (hex) | output |
|---|---|---|
| 0 | `0x3fefd89d149cf37c` | `0.9951920893287105` |
| 1 | `0x3ff7ad23c7118117` | `1.479770448313564` |
| 2 | `0x400067a2deca2f1c` | `2.0506036191417376` |
| 3 | `0x4003eaf8e335561d` | `2.489732527795026` |
| 4 | `0x4002fb4be35beec4` | `2.3727033388548495` |
| 5 | `0x3ff8a1cfb940018e` | `1.5395047413186123` |
| 6 | `0x3fde49e6cfac48a4` | `0.47326059608998094` |
| 7 | `0x3fb164cec448f655` | `0.06794445316423865` |
| 8 | `0x3fed2850689d8083` | `0.9111711543337296` |
| 9 | `0x4005e6df56bb1ede` | `2.737730672440974` |

**F4, empty** (length 0): input `[]`

No result (both variants).

**FA S7** (length 7): input `[1.0, 2.5, -0.75, 3.0, 4.25, 2.0, 0.5]`

Fused and unfused initial state give the same output.

| i | output (hex) | output |
|---|---|---|
| 0 | `0x3ff4e24a1e6b1358` | `1.3052464664662775` |
| 1 | `0x3ff67ce91a1ac8ba` | `1.4054957408630302` |
| 2 | `0x3ffe5a4d40dcf6b1` | `1.8970463308063865` |
| 3 | `0x400905795871d3c8` | `3.1276728543984724` |
| 4 | `0x400f42450d6fcccb` | `3.9073582696670193` |
| 5 | `0x40076d071f50fb54` | `2.9282362410153073` |
| 6 | `0x3fe4e39b648e4146` | `0.6527840579965776` |

### 6.3 Spline

Each case gives knots, values and evaluation points, then the result or "No result". The column "point" is the evaluation point.

- E1: evaluation at the first knot, between knots, on interior knots and at the last knot when it is reached inside the current interval.
- E2: the last knot alone gives no result (it would open a new interval).
- E3: non-increasing evaluation points reuse the current interval.
- E4, E5: first or last point outside the knot range.
- E6: a later point above the range, followed by a point inside, gives no result.
- E7: a later point below the range is extrapolated on the current interval.
- U1: unevenly spaced knots. Q1: four knots, the fewest that XMALab's caller uses.
- T1 to T5: three knots (the minimum accepted); T5 is a case where rounding leaves the singular system solvable.
- N2, N0: too few knots, no evaluation points.
- D1: a repeated knot. B1: first evaluation point so large that `τ_0 ⊖ 1 = τ_0`. Z1: a `−0` value at the knot evaluated.

**E1, even knots**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[1.0, 3.0, 2.0, -1.0, 0.5, 4.0]`, evaluation points `[0.0, 0.5, 1.0, 2.25, 3.0, 4.75, 5.0]`

| point | result (hex) | result |
|---|---|---|
| 0.0 | `0x3ff0000000000000` | `1.0` |
| 0.5 | `0x4002777777777778` | `2.3083333333333336` |
| 1.0 | `0x4008000000000000` | `3.0` |
| 2.25 | `0x3ff3044444444444` | `1.1885416666666666` |
| 3.0 | `0xbff0000000000000` | `-1.0` |
| 4.75 | `0x4009cddddddddddf` | `3.225520833333334` |
| 5.0 | `0x4010000000000000` | `4.0` |

**E2, last knot alone**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[1.0, 3.0, 2.0, -1.0, 0.5, 4.0]`, evaluation points `[5.0]`

No result.

**E3, non-increasing points**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[1.0, 3.0, 2.0, -1.0, 0.5, 4.0]`, evaluation points `[2.5, 0.5, 2.75, -0.0]`

| point | result (hex) | result |
|---|---|---|
| 2.5 | `0x3fd3333333333330` | `0.2999999999999998` |
| 0.5 | `0xc012aaaaaaaaaaa8` | `-4.666666666666664` |
| 2.75 | `0xbfdf444444444440` | `-0.48854166666666643` |
| -0.0 | `0xc02f11111111110e` | `-15.533333333333328` |

**E4, first point below range**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[1.0, 3.0, 2.0, -1.0, 0.5, 4.0]`, evaluation points `[-0.5, 1.0]`

No result.

**E5, last point above range**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[1.0, 3.0, 2.0, -1.0, 0.5, 4.0]`, evaluation points `[1.0, 5.5]`

No result.

**E6, later point above range**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[1.0, 3.0, 2.0, -1.0, 0.5, 4.0]`, evaluation points `[1.0, 5.5, 2.0]`

No result.

**E7, later point below range**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[1.0, 3.0, 2.0, -1.0, 0.5, 4.0]`, evaluation points `[1.5, -1.0, 4.0]`

| point | result (hex) | result |
|---|---|---|
| 1.5 | `0x4007888888888888` | `2.9416666666666664` |
| -1.0 | `0xc007777777777780` | `-2.933333333333337` |
| 4.0 | `0x3fe0000000000000` | `0.5` |

**U1, uneven knots**: knots `[0.0, 1.0, 3.0, 4.0, 7.0, 8.0]`, values `[0.0, 2.0, 1.0, 1.5, -2.0, 0.25]`, evaluation points `[0.5, 2.0, 3.0, 5.5, 6.5, 7.75]`

| point | result (hex) | result |
|---|---|---|
| 0.5 | `0x3ff741c21c21c21c` | `1.4535542582417582` |
| 2.0 | `0x3ff8c72c72c72c74` | `1.548626373626374` |
| 3.0 | `0x3ff0000000000002` | `1.0000000000000004` |
| 5.5 | `0xbfdb39ab9ab9ab98` | `-0.42539491758241743` |
| 6.5 | `0xbffe6f78f78f78f8` | `-1.9022149725274726` |
| 7.75 | `0xbfe61c654654654a` | `-0.6909662603021982` |

**Q1, four knots**: knots `[10.0, 11.0, 13.0, 16.0]`, values `[0.3, 0.7, -0.2, 1.1]`, evaluation points `[10.5, 12.0, 14.0, 15.0]`

| point | result (hex) | result |
|---|---|---|
| 10.5 | `0x3fe3cccccccccccc` | `0.6187499999999999` |
| 12.0 | `0x3fd851eb851eb852` | `0.38` |
| 14.0 | `0xbfe28f5c28f5c290` | `-0.5800000000000001` |
| 15.0 | `0xbfd3333333333339` | `-0.3000000000000003` |

**T1, three even knots**: knots `[0.0, 1.0, 2.0]`, values `[1.0, 3.0, 2.0]`, evaluation points `[0.5, 1.5]`

No result.

**T2, three uneven knots**: knots `[0.0, 1.0, 3.0]`, values `[1.0, 3.0, 2.0]`, evaluation points `[0.5, 2.0]`

No result.

**T3, three uneven knots**: knots `[0.1, 0.7, 2.3]`, values `[1.0, -3.0, 2.0]`, evaluation points `[0.5, 2.0]`

No result.

**T4, three uneven knots**: knots `[1.0, 2.0, 5.0]`, values `[0.3, 0.7, 0.2]`, evaluation points `[1.5, 4.0]`

No result.

**T5, three knots, solvable after rounding**: knots `[0.0, 0.1, 0.17]`, values `[1.0, 3.0, 2.0]`, evaluation points `[0.05, 0.12]`

| point | result (hex) | result |
|---|---|---|
| 0.05 | `0x4009c313e75783c5` | `3.220252807005155` |
| 0.12 | `0x4005094f613455d6` | `2.6295459360214677` |

**N2, two knots**: knots `[0.0, 1.0]`, values `[1.0, 2.0]`, evaluation points `[0.5]`

No result.

**N0, no evaluation points**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[1.0, 3.0, 2.0, -1.0, 0.5, 4.0]`, evaluation points `[]`

No result.

**D1, repeated knot**: knots `[0.0, 1.0, 1.0, 2.0, 3.0]`, values `[0.0, 1.0, 1.0, 0.0, 2.0]`, evaluation points `[0.5]`

| point | result (hex) | result |
|---|---|---|
| 0.5 | `0xfff8000000000000` | `NaN` |

**B1, huge first point**: knots `[0.0, 1e17, 2e17, 3e17]`, values `[1.0, 2.0, 0.0, 1.0]`, evaluation points `[1e17]`

| point | result (hex) | result |
|---|---|---|
| 1e17 | `0x0000000000000000` | `0.0` |

**Z1, negative zero value at a knot**: knots `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]`, values `[-0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`, evaluation points `[0.0]`

| point | result (hex) | result |
|---|---|---|
| 0.0 | `0x0000000000000000` | `0.0` |

---

## 7. Published sources

Pointers for the reader; nothing here is quoted from them.

- **Zero-phase filtering with odd edge extension and step-response initial state.** The documentation of MATLAB's `filtfilt` (Signal Processing Toolbox, The MathWorks) and of SciPy's `scipy.signal.filtfilt` with `padtype='odd'` and `scipy.signal.lfilter_zi` (the steady-state state as the solution of `(I − A)ζ = B − a·b_0`, with `A` the companion matrix). F. Gustafsson, "Determining the initial states in forward-backward filtering", IEEE Transactions on Signal Processing 44(4), 988–992, 1996, for the problem of choosing initial states in forward-backward filtering. The transposed direct form II structure and the difference equation are in any signal-processing text, for example A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*.
- **Cubic spline with not-a-knot ends and Hermite form.** G. Wolberg, "Cubic spline interpolation: a review", Technical Report CUCS-389-88, Department of Computer Science, Columbia University, 1988. C. de Boor, *A Practical Guide to Splines* (Springer), chapter on cubic spline interpolation (slope formulation and the not-a-knot condition).
- **Tridiagonal solve and Hermite interpolation.** Any standard numerical-analysis text, for example R. L. Burden and J. D. Faires, *Numerical Analysis*, or W. H. Press et al., *Numerical Recipes*, for the Thomas algorithm (Gaussian elimination without pivoting on a tridiagonal matrix) and cubic Hermite interpolation.
- **Matrix inverse (Appendix B).** The procedure is that of OpenCV's `cv::invert` with `DECOMP_LU` (OpenCV 4.13, Apache-2.0), already ported in this workspace.

---

## Appendix A. Public signatures to keep

```rust
// crates/xma-vision/src/filter.rs
impl ButterworthLowPass {
    #[must_use]
    pub fn filter(&self, input: &[f64]) -> Option<Vec<f64>>;
}

#[doc(hidden)]
#[must_use]
pub fn filter_unfused_zi(f: &ButterworthLowPass, input: &[f64]) -> Option<Vec<f64>>;

#[doc(hidden)]
#[must_use]
pub fn initial_conditions_for(f: &ButterworthLowPass, fma: bool) -> Vec<f64>;

#[must_use]
pub fn filter_run(filter: &ButterworthLowPass, values: &[f64]) -> Vec<f64>;

pub fn filter_runs(
    filter: &ButterworthLowPass,
    values: &[f64],
    runs: &[Range<usize>],
    out: &mut [f64],
);

#[must_use]
pub fn cubic_spline(x: &[f64], y: &[f64], at: &[f64]) -> Option<Vec<f64>>;
```

The struct these use, unchanged:

```rust
pub struct ButterworthLowPass {
    pub d: Vec<f64>,   // denominator coefficients, order + 1, d[0] = 1
    pub c: Vec<i32>,   // numerator integers (binomial coefficients), order + 1
    pub scale: f64,    // scale factor
}
```

## Appendix B. The matrix inverse used for the initial state

The inverse `W` of the `m × m` matrix `M` in 2.4 is exactly what the workspace's existing routine `xma_vision::linalg::invert` returns (a port of OpenCV's `cv::invert` with `DECOMP_LU`, outside the scope of this rewrite); an implementation should call it and use its matrix whether or not it reports success. For completeness, that routine computes, with `M` read by rows and columns as written in 2.4:

- **m = 1:** if `M_{00} = 0`, singular; otherwise `W_{00} = 1 ⊘ M_{00}`.
- **m = 2:** `Δ = (M_{00} ⊗ M_{11}) ⊖ (M_{01} ⊗ M_{10})`; singular if `Δ = 0` exactly. Otherwise `r = 1 ⊘ Δ` and `W_{00} = M_{11} ⊗ r`, `W_{01} = 0 ⊖ (M_{01} ⊗ r)`, `W_{10} = 0 ⊖ (M_{10} ⊗ r)`, `W_{11} = M_{00} ⊗ r`.
- **m = 3:** `Δ = ((M_{00} ⊗ ((M_{11} ⊗ M_{22}) ⊖ (M_{12} ⊗ M_{21}))) ⊖ (M_{01} ⊗ ((M_{10} ⊗ M_{22}) ⊖ (M_{12} ⊗ M_{20})))) ⊕ (M_{02} ⊗ ((M_{10} ⊗ M_{21}) ⊖ (M_{11} ⊗ M_{20})))`; singular if `Δ = 0` exactly. Otherwise `r = 1 ⊘ Δ` and each entry is a 2×2 cofactor difference times `r`, `W_{ij} = ((M_{pq} ⊗ M_{st}) ⊖ (M_{uv} ⊗ M_{wz})) ⊗ r` with the index pairs:

  | entry | first product | second product |
  |---|---|---|
  | W00 | M11·M22 | M12·M21 |
  | W01 | M02·M21 | M01·M22 |
  | W02 | M01·M12 | M02·M11 |
  | W10 | M12·M20 | M10·M22 |
  | W11 | M00·M22 | M02·M20 |
  | W12 | M02·M10 | M00·M12 |
  | W20 | M10·M21 | M11·M20 |
  | W21 | M01·M20 | M00·M21 |
  | W22 | M00·M11 | M01·M10 |

- **m ≥ 4** (order 4, the one XMALab uses): Gaussian elimination with partial pivoting applied to `[M | I]`, then back-substitution:
  1. For each column `i = 0, …, m−1`: the pivot row `k` is the row `≥ i` whose entry in column `i` has the largest absolute value, the earliest such row on ties. If that absolute value is below `100 × 2^−52` (an absolute threshold, independent of the matrix's scale), `M` is singular. If `k ≠ i`, rows `i` and `k` are exchanged (in the right-hand block in full; in the left block at least in columns `i` onwards, which are the only ones read later). Let `δ = (−1) ⊘ M_{ii}`. For each row `j = i+1, …, m−1` in order: `α = M_{ji} ⊗ δ`; for each column `c = i+1, …, m−1`: `M_{jc} ← M_{jc} ⊕ (α ⊗ M_{ic})`; for each column `c` of the right-hand block: `B_{jc} ← B_{jc} ⊕ (α ⊗ B_{ic})` (unfused).
  2. Back-substitution, rows `i = m−1, …, 0` in decreasing order, each column `c` of the right-hand block: `σ = B_{ic}`; for `k = i+1, …, m−1` in increasing order `σ ← σ ⊖ (M_{ik} ⊗ B_{kc})`; `B_{ic} = σ ⊘ M_{ii}` (the updated `M`).
  3. `W = B`.
- **Singular:** `W` is the `m × m` zero matrix.
