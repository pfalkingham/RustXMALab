//! Seconds since the Unix epoch to a UTC calendar date, for the build date the About dialog
//! shows. Standard library only: `build.rs` includes this file (`#[path]`), and the library
//! compiles it for its unit tests.
//!
//! The day-to-date conversion is Howard Hinnant's `civil_from_days` ("chrono-Compatible
//! Low-Level Date Algorithms"), for the proleptic Gregorian calendar.

/// The UTC date of `secs` seconds after 1970-01-01T00:00:00Z, as `YYYY-MM-DD`.
#[must_use]
pub fn utc_date(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(i64::MAX / 2);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// (year, month 1 to 12, day 1 to 31) of the day `z` days after 1970-01-01.
#[must_use]
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March first
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: u64 = 86_400;

    #[test]
    fn the_epoch_and_known_days() {
        assert_eq!(utc_date(0), "1970-01-01");
        assert_eq!(utc_date(DAY - 1), "1970-01-01");
        assert_eq!(utc_date(DAY), "1970-01-02");
        // 2000-03-01T00:00:00Z.
        assert_eq!(utc_date(951_868_800), "2000-03-01");
        // 2026-11-26T12:00:00Z.
        assert_eq!(utc_date(1_795_694_400), "2026-11-26");
    }

    #[test]
    fn the_29th_of_february() {
        // 2024-02-29T00:00:00Z and its last second, then 1 March.
        assert_eq!(utc_date(1_709_164_800), "2024-02-29");
        assert_eq!(utc_date(1_709_164_800 + DAY - 1), "2024-02-29");
        assert_eq!(utc_date(1_709_164_800 + DAY), "2024-03-01");
        // 2000 is a leap year (divisible by 400): 2000-02-29.
        assert_eq!(utc_date(951_782_400), "2000-02-29");
        // 2100 is not: the day after 28 February is 1 March.
        let feb28_2100 = 4_107_456_000;
        assert_eq!(utc_date(feb28_2100), "2100-02-28");
        assert_eq!(utc_date(feb28_2100 + DAY), "2100-03-01");
        // 2023 is not: 28 February, then 1 March.
        assert_eq!(utc_date(1_677_542_400), "2023-02-28");
        assert_eq!(utc_date(1_677_542_400 + DAY), "2023-03-01");
    }

    #[test]
    fn the_turn_of_a_year() {
        // 2025-12-31T23:59:59Z and 2026-01-01T00:00:00Z.
        assert_eq!(utc_date(1_767_225_599), "2025-12-31");
        assert_eq!(utc_date(1_767_225_600), "2026-01-01");
        // 1999 to 2000.
        assert_eq!(utc_date(946_684_799), "1999-12-31");
        assert_eq!(utc_date(946_684_800), "2000-01-01");
    }

    #[test]
    fn every_day_follows_the_one_before() {
        // Four hundred years (one whole Gregorian cycle) of consecutive days.
        let mut prev = civil_from_days(-1);
        assert_eq!(prev, (1969, 12, 31));
        for z in 0..146_097 {
            let (y, m, d) = civil_from_days(z);
            let next_day = (prev.0, prev.1, prev.2 + 1);
            let next_month = (prev.0, prev.1 + 1, 1);
            let next_year = (prev.0 + 1, 1, 1);
            assert!(
                (y, m, d) == next_day || (y, m, d) == next_month || (y, m, d) == next_year,
                "{prev:?} then {:?}",
                (y, m, d)
            );
            if m == 2 && d == 29 {
                assert!(y % 4 == 0 && (y % 100 != 0 || y % 400 == 0), "{y}-02-29");
            }
            prev = (y, m, d);
        }
        assert_eq!(prev, (2369, 12, 31));
    }

    #[test]
    fn a_huge_number_does_not_panic() {
        let _ = utc_date(u64::MAX);
    }
}
