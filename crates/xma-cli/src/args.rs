//! Hand-written command-line parsing: options with and without a value, positionals, `--`.

use std::collections::BTreeMap;

use xma_engine::SaveTime;

use crate::CliError;

/// One option a command accepts.
#[derive(Clone, Copy, Debug)]
pub struct Opt {
    /// Long name without the dashes, for example `out`.
    pub long: &'static str,
    /// Short name, for example `o`.
    pub short: Option<char>,
    /// Whether the option takes a value (`--out DIR` or `--out=DIR`).
    pub value: bool,
}

impl Opt {
    pub const fn flag(long: &'static str) -> Self {
        Self {
            long,
            short: None,
            value: false,
        }
    }

    pub const fn value(long: &'static str) -> Self {
        Self {
            long,
            short: None,
            value: true,
        }
    }

    pub const fn short(mut self, c: char) -> Self {
        self.short = Some(c);
        self
    }
}

/// Parsed arguments of one command.
#[derive(Debug, Default)]
pub struct Parsed {
    /// Positional arguments, in order.
    pub positionals: Vec<String>,
    /// Options seen, by long name; a flag has an empty value. A repeated option keeps every
    /// value in order.
    options: BTreeMap<&'static str, Vec<String>>,
}

impl Parsed {
    /// True if the flag or option was given.
    pub fn has(&self, long: &str) -> bool {
        self.options.contains_key(long)
    }

    /// The last value of an option.
    pub fn value(&self, long: &str) -> Option<&str> {
        self.options
            .get(long)
            .and_then(|v| v.last())
            .map(String::as_str)
    }

    /// Takes a positional at `i`, or fails with `what` named.
    pub fn positional(&self, i: usize, what: &str) -> Result<&str, CliError> {
        self.positionals
            .get(i)
            .map(String::as_str)
            .ok_or_else(|| CliError::usage(format!("missing {what}")))
    }

    /// Fails when more than `n` positionals were given.
    pub fn at_most(&self, n: usize) -> Result<(), CliError> {
        match self.positionals.get(n) {
            Some(extra) => Err(CliError::usage(format!("unexpected argument '{extra}'"))),
            None => Ok(()),
        }
    }
}

/// Parses `args` against the options of one command. `-h`/`--help` is always accepted as a
/// flag named `help`.
pub fn parse(args: &[String], opts: &[Opt]) -> Result<Parsed, CliError> {
    const HELP: Opt = Opt::flag("help").short('h');
    let find_long = |name: &str| {
        opts.iter()
            .chain(std::iter::once(&HELP))
            .find(|o| o.long == name)
    };
    let find_short = |c: char| {
        opts.iter()
            .chain(std::iter::once(&HELP))
            .find(|o| o.short == Some(c))
    };
    let mut parsed = Parsed::default();
    let mut i = 0;
    let mut only_positionals = false;
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if only_positionals || arg == "-" || !arg.starts_with('-') {
            parsed.positionals.push(arg.clone());
            continue;
        }
        if arg == "--" {
            only_positionals = true;
            continue;
        }
        let (opt, inline) = if let Some(long) = arg.strip_prefix("--") {
            let (name, inline) = match long.split_once('=') {
                Some((n, v)) => (n, Some(v.to_owned())),
                None => (long, None),
            };
            let opt = find_long(name)
                .ok_or_else(|| CliError::usage(format!("unknown option '--{name}'")))?;
            (opt, inline)
        } else {
            let mut chars = arg[1..].chars();
            let c = chars.next().unwrap_or('-');
            let rest: String = chars.collect();
            let opt =
                find_short(c).ok_or_else(|| CliError::usage(format!("unknown option '-{c}'")))?;
            (opt, (!rest.is_empty()).then_some(rest))
        };
        let value = if opt.value {
            match inline {
                Some(v) => v,
                None => match args.get(i) {
                    Some(v) => {
                        i += 1;
                        v.clone()
                    }
                    None => {
                        return Err(CliError::usage(format!(
                            "option '--{}' needs a value",
                            opt.long
                        )));
                    }
                },
            }
        } else {
            if inline.is_some() {
                return Err(CliError::usage(format!(
                    "option '--{}' takes no value",
                    opt.long
                )));
            }
            String::new()
        };
        parsed.options.entry(opt.long).or_default().push(value);
    }
    Ok(parsed)
}

/// Parses a number with a message naming the option.
pub fn number<T: std::str::FromStr>(text: &str, what: &str) -> Result<T, CliError> {
    text.trim()
        .parse()
        .map_err(|_| CliError::usage(format!("{what}: '{text}' is not a valid number")))
}

/// `on`/`off` (also `true`/`false`, `1`/`0`, `yes`/`no`).
pub fn switch(text: &str, what: &str) -> Result<bool, CliError> {
    match text.to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        _ => Err(CliError::usage(format!(
            "{what}: '{text}' is not on or off"
        ))),
    }
}

/// `--save-time` (`None` when absent). With `needs_output`, refused unless `--output` is
/// given too.
pub fn save_time(p: &Parsed, needs_output: bool) -> Result<Option<SaveTime>, CliError> {
    let save_time = p
        .value("save-time")
        .map(|v| {
            SaveTime::parse(v).ok_or_else(|| {
                CliError::usage(format!(
                    "--save-time '{v}': expected YYYY-MM-DDTHH:MM:SS, years 1980 to 2107"
                ))
            })
        })
        .transpose()?;
    if needs_output && save_time.is_some() && !p.has("output") {
        return Err(CliError::usage("--save-time needs --output"));
    }
    Ok(save_time)
}

/// The 1-based number given with `--<long>`, returned 0-based (`None` when absent); `0` is a
/// usage error saying that `what` (for example `trials`) are numbered from 1.
pub fn one_based(p: &Parsed, long: &str, what: &str) -> Result<Option<usize>, CliError> {
    p.value(long)
        .map(|v| {
            number::<usize>(v, &format!("--{long}"))?
                .checked_sub(1)
                .ok_or_else(|| CliError::usage(format!("--{long}: {what} are numbered from 1")))
        })
        .transpose()
}

/// A 1-based inclusive range `FROM-TO` (or a single `N`), returned 1-based, with
/// `1 <= FROM <= TO` (review 11: an end of 0 or before the start is refused, so callers
/// can subtract 1 from either end and add 1 to the 0-based end).
pub fn range(text: &str, what: &str) -> Result<(usize, usize), CliError> {
    let (a, b) = match text.split_once(['-', ':']) {
        Some((a, b)) => (a, b),
        None => (text, text),
    };
    let from: usize = number(a, what)?;
    let to: usize = number(b, what)?;
    if from == 0 || to == 0 {
        return Err(CliError::usage(format!(
            "{what}: frames are numbered from 1"
        )));
    }
    if to < from {
        return Err(CliError::usage(format!(
            "{what}: the range {from}-{to} ends before it starts"
        )));
    }
    Ok((from, to))
}

/// A comma-separated list of 1-based indices, returned 0-based.
pub fn indices(text: &str, what: &str) -> Result<Vec<usize>, CliError> {
    text.split(',')
        .map(|s| {
            let n: usize = number(s, what)?;
            n.checked_sub(1)
                .ok_or_else(|| CliError::usage(format!("{what}: numbering starts at 1")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_owned()).collect()
    }

    const OPTS: &[Opt] = &[
        Opt::value("out").short('o'),
        Opt::flag("all"),
        Opt::value("trial"),
    ];

    #[test]
    fn options_and_positionals() {
        let p = parse(
            &strings(&["a.xma", "-o", "x", "--all", "--trial=2", "--", "--b"]),
            OPTS,
        )
        .unwrap();
        assert_eq!(p.positionals, ["a.xma", "--b"]);
        assert_eq!(p.value("out"), Some("x"));
        assert!(p.has("all"));
        assert_eq!(p.value("trial"), Some("2"));
        let p = parse(&strings(&["-ox"]), OPTS).unwrap();
        assert_eq!(p.value("out"), Some("x"));
        assert!(parse(&strings(&["-h"]), OPTS).unwrap().has("help"));
    }

    #[test]
    fn errors() {
        for bad in [&["--nope"][..], &["-z"], &["--out"], &["--all=1"]] {
            let e = parse(&strings(bad), OPTS).unwrap_err();
            assert_eq!(e.code, 2, "{bad:?}");
        }
    }

    #[test]
    fn values() {
        assert_eq!(range("3-7", "f").unwrap(), (3, 7));
        assert_eq!(range("5", "f").unwrap(), (5, 5));
        assert!(range("0-3", "f").is_err());
        // Review 11: an end of 0 or before the start.
        for bad in ["2-0", "0", "0-0", "7-3", "2:1"] {
            let e = range(bad, "f").unwrap_err();
            assert_eq!(e.code, 2, "{bad}");
        }
        assert!(
            range("2-0", "f")
                .unwrap_err()
                .to_string()
                .contains("frames are numbered from 1")
        );
        assert_eq!(range("4-4", "f").unwrap(), (4, 4));
        let max = usize::MAX.to_string();
        assert_eq!(range(&format!("1-{max}"), "f").unwrap(), (1, usize::MAX));
        assert_eq!(range(&max, "f").unwrap(), (usize::MAX, usize::MAX));
        assert!(range("a-3", "f").is_err());
        assert_eq!(indices("1,3", "m").unwrap(), [0, 2]);
        assert!(indices("0", "m").is_err());
        assert!(switch("ON", "x").unwrap());
        assert!(switch("maybe", "x").is_err());
    }
}
