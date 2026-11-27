//! `rustxmalab`: the graphical program RustXMALab.
//!
//! ```text
//! rustxmalab [project.xma] [--reset-layout] [--theme system|light|dark]
//!        [--workspace undistortion|calibration|tracking]
//!        [--screenshot <file.png> --quit-after <seconds>] [--window-size <W>x<H>]
//! ```
//!
//! eframe with the wgpu renderer, as chosen by spike 1 (`spikes/camera-view/README.md`):
//! `WgpuConfiguration::default()` (high-performance adapter, Vulkan/Metal/DX12 with the GL
//! fallback, `WGPU_BACKEND` and `WGPU_POWER_PREF` honoured), vsync on.
//!
//! On Windows a release build is a window program, not a console one, so starting it opens
//! no console window; what it prints (`--help`, an error in the options) is then not shown.
//! Debug builds keep the console for their output.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::ExitCode;

use xma_app::state::Workspace;
use xma_app::theme::ThemeChoice;
use xma_app::{EframeApp, Options, XmaApp};

const USAGE: &str = "\
usage: rustxmalab [project.xma] [options]

options:
  --reset-layout              ignore the saved panel layout (first-run arrangement)
  --theme system|light|dark   theme for this run (saved as the Theme setting)
  --workspace undistortion|calibration|tracking
                              switch to this workspace once the project is loaded
  --screenshot <file.png>     save a screenshot of the main window shortly before
                              --quit-after (settings are not saved in this mode)
  --quit-after <seconds>      close the window after this many seconds
  --window-size <W>x<H>       open the window at this size in points (for comparisons
                              with reference screenshots)
  -h, --help                  show this help
";

fn parse(args: &[String]) -> Result<(Options, Option<[f32; 2]>), String> {
    let mut o = Options {
        settings_path: xma_app::settings::settings_path(),
        native_file_dialogs: true,
        system_links: true,
        ..Options::default()
    };
    let mut size = None;
    let mut i = 0;
    let value = |i: &mut usize, name: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or_else(|| format!("{name} needs a value"))
    };
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-h" | "--help" => return Err(String::new()),
            "--reset-layout" => o.reset_layout = true,
            "--theme" => {
                let v = value(&mut i, a)?;
                o.theme = Some(match v.as_str() {
                    "system" | "light" | "dark" => ThemeChoice::from_setting(&v),
                    _ => return Err(format!("unknown theme '{v}'")),
                });
            }
            "--workspace" => {
                let v = value(&mut i, a)?;
                o.workspace = Some(
                    Workspace::from_cli(&v).ok_or_else(|| format!("unknown workspace '{v}'"))?,
                );
            }
            "--screenshot" => o.screenshot = Some(PathBuf::from(value(&mut i, a)?)),
            "--quit-after" => {
                let v = value(&mut i, a)?;
                o.quit_after = Some(
                    v.parse()
                        .map_err(|_| format!("bad number of seconds '{v}'"))?,
                );
            }
            "--window-size" => {
                let v = value(&mut i, a)?;
                let bad = || format!("bad window size '{v}' (expected for example 1280x1024)");
                let (w, h) = v.split_once('x').ok_or_else(bad)?;
                let w: f32 = w.parse().map_err(|_| bad())?;
                let h: f32 = h.parse().map_err(|_| bad())?;
                if !(200.0..=10000.0).contains(&w) || !(200.0..=10000.0).contains(&h) {
                    return Err(bad());
                }
                size = Some([w, h]);
            }
            _ if a.starts_with('-') => return Err(format!("unknown option '{a}'")),
            _ => {
                if o.project.is_some() {
                    return Err("more than one project given".into());
                }
                o.project = Some(PathBuf::from(a));
            }
        }
        i += 1;
    }
    if o.screenshot.is_some() && o.quit_after.is_none() {
        return Err("--screenshot needs --quit-after".into());
    }
    Ok((o, size))
}

/// Keeps the default panic message on standard error and also appends it, with a
/// backtrace, to `crash.log` in the settings folder, so a panic is not lost when the program
/// was started without a terminal. A panic while drawing a panel or the camera views is
/// caught by the shell (the panel shows the error, the Console gets it); any other panic
/// ends the program after writing this log.
fn install_panic_log() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);
        let Some(dir) = xma_app::settings::config_dir() else {
            return;
        };
        let _ = std::fs::create_dir_all(&dir);
        let bt = std::backtrace::Backtrace::force_capture();
        let text = format!(
            "rustxmalab {} panicked: {info}\n{bt}\n\n",
            env!("CARGO_PKG_VERSION")
        );
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("crash.log"))
        {
            let _ = f.write_all(text.as_bytes());
        }
    }));
}

fn main() -> ExitCode {
    xma_app::startup::mark_start();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (opts, size) = match parse(&args) {
        Ok(o) => o,
        Err(e) => {
            if e.is_empty() {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            eprintln!("rustxmalab: {e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    install_panic_log();
    let app = XmaApp::new(opts);
    // Default geometry when none is saved: 1280 × 1024 at (50, 50) (MainWindow.cpp:158-161).
    let mut vp = egui::ViewportBuilder::default()
        .with_title(app.title())
        .with_app_id(xma_app::APP_ID)
        .with_icon(xma_app::icons::Icon::window_icon())
        .with_inner_size(xma_app::shell::DEFAULT_MAIN_SIZE)
        .with_position(xma_app::shell::DEFAULT_MAIN_POS);
    if let Some(g) = app.shell.main_window {
        vp = vp.with_inner_size(g.size);
        if let Some(p) = g.pos {
            vp = vp.with_position(p);
        }
    }
    if let Some(s) = size {
        vp = vp.with_inner_size(s);
    }
    let options = eframe::NativeOptions {
        viewport: vp,
        renderer: eframe::Renderer::Wgpu,
        // The program keeps its own settings (settings.rs); egui's are not wanted.
        persist_window: false,
        ..Default::default()
    };
    match eframe::run_native(
        xma_app::APP_ID,
        options,
        Box::new(move |_cc| Ok(Box::new(EframeApp(app)))),
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rustxmalab: {e}");
            ExitCode::FAILURE
        }
    }
}
