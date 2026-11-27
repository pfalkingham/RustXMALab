//! The C library's `srand` / `rand` as glibc implements them, so that the cube calibration's
//! random draws (`processing/CubeCalibration.cpp:292`, `:300`) are the ones XMALab 3 makes on
//! Linux, on every platform.
//!
//! XMALab never calls `srand` (parity ledger A5), so its first calibration of a session
//! starts from glibc's default state, which is the state `srand(1)` sets, and later
//! calibrations continue the same sequence. The reference runs set the seed with `srand`
//! (`oracle/FORMAT.md`, "Calibration runs", operation `Seed`).
//!
//! Ported from glibc `stdlib/random_r.c` (`__srandom_r`, `__random_r`) for the default
//! generator `TYPE_3`: an additive lagged Fibonacci generator `r[i] = r[i-3] + r[i-31]`
//! (mod 2^32) over 31 words, seeded by the Park-Miller "minimal standard" generator
//! (`16807 * r mod (2^31 - 1)`, Schrage's method), with the first 310 outputs discarded;
//! each output is the new word shifted right by one bit (0 to `RAND_MAX` = 2^31 - 1).
//! Checked against glibc's `rand()` (test `tests/crand.rs`, values printed by a C program
//! compiled with the system compiler).

/// Number of state words of `TYPE_3` (`DEG_3`).
const DEG: usize = 31;
/// Distance between the two taps (`SEP_3`).
const SEP: usize = 3;

/// glibc's `rand()` state for one sequence. [`CRand::new`] is `srand(seed)`;
/// [`CRand::next`] is `rand()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CRand {
    state: [u32; DEG],
    /// `fptr` as an index into `state`.
    front: usize,
    /// `rptr` as an index into `state`.
    rear: usize,
}

impl Default for CRand {
    /// The state of a program that never called `srand`: the same as `srand(1)`.
    fn default() -> Self {
        CRand::new(1)
    }
}

impl CRand {
    /// `srand(seed)`.
    ///
    /// Ported from glibc `stdlib/random_r.c`, `__srandom_r` (`TYPE_3`). A seed of 0 is
    /// replaced by 1, as glibc does.
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let seed = if seed == 0 { 1 } else { seed };
        let mut state = [0u32; DEG];
        // state[0] = seed; word = seed; both int32_t, so a seed of 2^31 or more is negative.
        state[0] = seed;
        #[allow(clippy::cast_possible_wrap)]
        let mut word: i32 = seed as i32;
        for slot in state.iter_mut().skip(1) {
            // long int hi = word / 127773, lo = word % 127773 (truncating division).
            let hi = i64::from(word) / 127_773;
            let lo = i64::from(word) % 127_773;
            // word = 16807 * lo - 2836 * hi, computed in long, stored into the int32_t.
            #[allow(clippy::cast_possible_truncation)]
            {
                word = (16_807 * lo - 2_836 * hi) as i32;
            }
            if word < 0 {
                word = word.wrapping_add(2_147_483_647);
            }
            // *++dst = word;
            #[allow(clippy::cast_sign_loss)]
            {
                *slot = word as u32;
            }
        }
        let mut g = CRand {
            state,
            front: SEP,
            rear: 0,
        };
        // kc = rand_deg * 10; while (--kc >= 0) random_r(&discard);
        for _ in 0..DEG * 10 {
            g.next();
        }
        g
    }

    /// `rand()`: the next value, from 0 to 2^31 - 1.
    ///
    /// Ported from glibc `stdlib/random_r.c`, `__random_r` (`TYPE_3`).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> i32 {
        // val = *fptr += (uint32_t) *rptr; *result = val >> 1;
        let val = self.state[self.front].wrapping_add(self.state[self.rear]);
        self.state[self.front] = val;
        let result = val >> 1;
        self.front += 1;
        if self.front >= DEG {
            self.front = 0;
            self.rear += 1;
        } else {
            self.rear += 1;
            if self.rear >= DEG {
                self.rear = 0;
            }
        }
        #[allow(clippy::cast_possible_wrap)]
        {
            result as i32
        }
    }
}
