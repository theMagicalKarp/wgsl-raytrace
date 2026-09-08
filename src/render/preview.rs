//! The live preview: the accumulator, drawn into the terminal as it converges.
//!
//! The trick this rests on is that the accumulator holds
//! `vec4f(summed_radiance, sample_count)` per pixel, so `sum.rgb / sum.w` is a
//! correct frame at *any* point in the render, with no extra state and no second
//! buffer. Drawing a preview is then only three things the tracer already does —
//! read the sums back, [`resolve`](super::resolve) them, and put the pixels
//! somewhere.
//!
//! "Somewhere" is the terminal, through the [kitty graphics
//! protocol](super::kitty). That reuses `resolve` verbatim, so the preview and
//! the PNG cannot drift the way they would if the tone curve were reimplemented
//! in WGSL for a swapchain; it needs no windowing dependency; and it works over
//! SSH. What it costs is that only some terminals can show it, and that each
//! frame drains the sample queue to read the accumulator.

use super::Render;
use super::Renderer;
use super::downsample;
use super::kitty;
use super::progress;
use super::resolve;
use crate::config::Config;
use crate::scene::Scene;
use colored::Colorize;
use image::ImageEncoder;
use image::codecs::png::CompressionType;
use image::codecs::png::FilterType;
use image::codecs::png::PngEncoder;
use rustix::termios::tcgetwinsize;
use signal_hook::SigId;
use signal_hook::consts::signal::SIGINT;
use signal_hook::flag;
use signal_hook::low_level;
use std::error::Error;
use std::io::IsTerminal;
use std::io::Write;
use std::io::stdout;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::time::Instant;

/// Lower bound on the gap between preview frames, for the same reason
/// `REDRAW_INTERVAL` throttles the progress bar: five a second is enough to
/// watch noise fall away, and more is work nobody sees.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(200);

/// Rows the preview may occupy at most.
///
/// A taller block is a bigger PNG down the pty every frame for a picture nobody
/// is inspecting at the pixel level. Twenty-four rows is about 480 pixels, which
/// is a legible preview of any of the example scenes.
const MAX_ROWS: u32 = 24;

/// Cell size to assume when the terminal reports its size in cells but not in
/// pixels, which is allowed and not rare.
const ASSUMED_CELL: (u32, u32) = (8, 16);

/// Renders `scene`, drawing the frame into the terminal as it converges.
///
/// Falls back to a plain [`render`](super::render) — with a `warning:` line
/// saying why — on a terminal that cannot show it. That is deliberately a
/// warning and not an error: the render is still the thing the user asked for,
/// and refusing to produce it because the preview cannot be drawn would be the
/// tail wagging the dog.
///
/// `Ctrl-C` stops sampling and resolves what has converged so far, which is what
/// makes the preview useful for iterating on a scene: you stop when the frame
/// looks right rather than when the budget runs out.
pub fn preview(config: &Config, scene: &Scene) -> Result<Render, Box<dyn Error>> {
    if let Err(reason) = supported() {
        eprintln!("{}{} {}", "warning".bold().yellow(), ":".bold(), reason);
        return fallback(config, scene);
    }

    // Both checks come before the renderer, because the fallback builds one of
    // its own: fitting needs only the frame's size, which the camera already
    // knows, and asking the renderer for it instead would cost an adapter
    // request and the whole scene uploaded twice on the way to the same answer.
    let Some(block) = fitted(config.camera.get_dimensions()) else {
        eprintln!(
            "{}{} the terminal is too small to draw a preview in",
            "warning".bold().yellow(),
            ":".bold(),
        );
        return fallback(config, scene);
    };

    let mut renderer = Renderer::new(config, scene)?;

    // Scroll the block into view and leave the cursor on the line below it,
    // which is where `Progress` draws. Every frame from here on climbs back up
    // to the top of the block and returns.
    print!("{}", "\n".repeat(block.rows as usize));
    let _ = stdout().flush();

    // Turning the sums into an image and pushing them at the terminal touches
    // nothing the tracer owns, so it happens on its own thread while the GPU
    // keeps tracing. What is left on the sample loop is a copy out of the mapped
    // buffer — everything else in a frame is over here.
    // Capacity zero, so a `try_send` succeeds only while the painter is sitting
    // in `recv` waiting for one. A queue would buy nothing — a frame waiting in
    // it is already stale by the time it is painted — and would cost the end of
    // the render, where the last frame has to wait out everything ahead of it.
    let (frames, incoming) = mpsc::sync_channel::<Frame>(0);
    let painter = thread::spawn(move || {
        let mut failure: Option<String> = None;

        for frame in incoming {
            // The render is not worth abandoning over a terminal that will not
            // take a picture, and the loop must not be left blocked on a channel
            // nobody is reading — so the first failure stops the drawing, the
            // frames behind it are drained rather than redrawn, and the whole
            // thing is reported once at the end.
            if failure.is_some() {
                continue;
            }
            if let Err(error) = draw(frame) {
                failure = Some(error.to_string());
            }
        }

        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    });

    let interrupts = Interrupts::install()?;
    // Asked for *after* a sample rather than before the loop: a copy queued
    // ahead of every dispatch reads an accumulator nothing has written yet, and
    // a zero sum count resolves to black. The first frame on the screen should
    // be the first frame traced.
    let mut requested: Option<Instant> = None;

    while renderer.remaining() > 0 && !interrupts.interrupted() {
        renderer.sample()?;

        // Never waits: `None` means the copy is still in the queue, and the next
        // pass round the loop will ask again.
        if let Some(sums) = renderer.snapshot()? {
            // `try_send` rather than `send`: a painter still working on the last
            // frame is a reason to drop this one, not to stall the tracer. The
            // next request is 200ms away and will carry more samples anyway.
            let frame = Frame::new(Arc::new(sums), renderer.dimensions(), block);
            let _ = frames.try_send(frame);
        }

        if requested.is_none_or(|at| at.elapsed() >= PREVIEW_INTERVAL) {
            renderer.request();
            requested = Some(Instant::now());
        }
    }

    // Whatever stopped the loop, the last frame is behind the samples that have
    // since landed. Wait for them and paint once more from the finished sums, so
    // the picture left on screen is the picture that gets written. This one is
    // sent rather than offered, and waited on, because it is the frame that
    // stays on the screen.
    renderer.drain()?;
    let sums = Arc::new(renderer.sums()?);
    let frame = Frame::new(Arc::clone(&sums), renderer.dimensions(), block);
    let _ = frames.send(frame);
    drop(frames);
    let painted = painter.join();

    // Only now: `resolved` closes off the progress bar's line, which moves the
    // cursor the block is placed against. It resolves the sums just painted
    // rather than reading the accumulator again — the renderer is drained, so
    // the second read would return the same buffer for a second drain and
    // another 130MB copied at 1080p.
    //
    // Clearing first, because resolving and writing a PNG is what the `Ctrl-C`
    // that stopped the loop was *for*: see `Interrupts::clear`.
    interrupts.clear();
    let render = renderer.resolved(&sums)?;

    if let Ok(Err(error)) = painted {
        eprintln!("{}{} {}", "warning".bold().yellow(), ":".bold(), error);
    }

    // Hands `Ctrl-C` back, and would have done so on any of the `?` above too.
    drop(interrupts);

    Ok(render)
}

/// The plain render a terminal that cannot show a preview falls back to, still
/// under the preview's `Ctrl-C`.
///
/// [`render`](super::render) on its own runs to the end of the budget, and that
/// is the right shape for a run nobody asked to watch. But `--preview` promises
/// an early stop, and the promise is about the flag rather than about the
/// picture — `mise run example` passes it unconditionally, so redirecting the
/// output to a file lands here, and killing *that* run with nothing written is
/// the outcome the flag exists to prevent.
fn fallback(config: &Config, scene: &Scene) -> Result<Render, Box<dyn Error>> {
    let mut renderer = Renderer::new(config, scene)?;
    let interrupts = Interrupts::install()?;

    while renderer.remaining() > 0 && !interrupts.interrupted() {
        renderer.sample()?;
    }

    interrupts.clear();
    let render = renderer.finish()?;

    // Hands `Ctrl-C` back, and would have done so on the `?` above too.
    drop(interrupts);

    Ok(render)
}

/// One accumulator snapshot on its way to the terminal.
///
/// Holds its sums behind an [`Arc`] so the painter thread can work on them while
/// the sample loop carries on filling the accumulator they were copied out of —
/// shared rather than owned only for the last frame, which is also the one the
/// PNG is resolved from.
struct Frame {
    sums: Arc<Vec<[f32; 4]>>,
    source: (u32, u32),
    block: Block,
}

impl Frame {
    fn new(sums: Arc<Vec<[f32; 4]>>, source: (u32, u32), block: Block) -> Frame {
        Frame {
            sums,
            source,
            block,
        }
    }
}

/// Shrinks a snapshot to the block and transmits it.
fn draw(frame: Frame) -> Result<(), Box<dyn Error>> {
    let Frame {
        sums,
        source,
        block,
    } = frame;
    let size = transmitted(source, block.pixels);
    let scaled = downsample(&sums, source, size);
    let pixels = resolve(&scaled);

    // `resolve` writes a hardcoded 255 into every alpha, so a quarter of what it
    // returns is a constant this does not need to send.
    let opaque: Vec<u8> = pixels
        .chunks_exact(4)
        .flat_map(|pixel| &pixel[..3])
        .copied()
        .collect();

    // The filter matters far more than the compression level here, and not by a
    // little. Measured on a 640x480 preview of the teapot, worst frame (noisy)
    // and best (converged):
    //
    // | compression | filter   | bytes     | encode      |
    // | ---         | ---      | ---       | ---         |
    // | Fast        | NoFilter | 900KB     | 1.0ms       |
    // | Fast        | Up       | 333-143KB | 0.9-0.6ms   |
    // | Default     | Up       | 128-116KB | 11.5-23.3ms |
    // | Best        | Up       | 123-112KB | 29.3-81.3ms |
    //
    // Unfiltered is barely compressed at all — the first row is the raw bytes to
    // within a rounding error — because deflate finds nothing to repeat in an
    // image until a filter has turned it into differences. `Up` costs nothing
    // and takes 6x off what the terminal has to read. Spending twenty times the
    // CPU on `Default` for another 15% is the wrong trade for a picture nobody
    // is inspecting at the pixel level.
    let mut png = Vec::new();
    PngEncoder::new_with_quality(&mut png, CompressionType::Fast, FilterType::Up).write_image(
        &opaque,
        size.0,
        size.1,
        image::ExtendedColorType::Rgb8,
    )?;

    // Save the cursor, climb to the top of the reserved block, draw, and drop
    // back to where `Progress` is writing. DEC save/restore (`\x1b7`/`\x1b8`)
    // rather than the SCO `\x1b[s`/`\x1b[u`, which fewer terminals honour.
    //
    // Under `TERMINAL` for the whole burst, so the bar cannot land in the middle
    // of it. The bar skips its turn rather than waiting, which is what keeps a
    // terminal slow to read this off the sample loop's back.
    let _writing = progress::TERMINAL.lock();
    let mut out = stdout().lock();
    write!(out, "\x1b7\x1b[{}A\r", block.rows)?;
    out.write_all(kitty::encode(&png, (block.columns, block.rows)).as_bytes())?;
    write!(out, "\x1b8")?;
    out.flush()?;

    Ok(())
}

/// The pixel size to send a frame at, given the block it is going into.
///
/// Normally the block: a render is larger than the terminal, and shrinking it to
/// exactly the pixels its cells cover means the terminal scales it by one and
/// nothing is resampled twice.
///
/// A render *smaller* than the block is the other way round, and is sent at its
/// own size — there is nothing to be gained by magnifying it here when the
/// terminal is going to scale it into the same cells regardless, and
/// [`downsample`] would decline to magnify it anyway, which is how this came to
/// be a function rather than an assumption. Compared per axis rather than as a
/// pair because rounding the block down to whole cells can put one axis either
/// side of the render's.
fn transmitted(image: (u32, u32), block: (u32, u32)) -> (u32, u32) {
    (image.0.min(block.0), image.1.min(block.1))
}

/// Where the preview goes: how many cells, and the pixels they cover.
#[derive(Clone, Copy)]
struct Block {
    columns: u32,
    rows: u32,
    pixels: (u32, u32),
}

/// Fits `image` into the terminal, or `None` if there is no room worth using.
///
/// The block is measured once, at the start. A terminal resized mid-render will
/// have a preview that no longer lines up with its own reserved space, which is
/// cosmetic and repairs itself on the next run.
fn fitted(image: (u32, u32)) -> Option<Block> {
    let (cells, cell) = winsize()?;

    // A column of margin either side, and enough rows left over for the progress
    // bar and the two summary lines that follow the render.
    let columns = cells.0.checked_sub(2)?;
    let rows = cells.1.checked_sub(6)?.min(MAX_ROWS);
    if columns == 0 || rows == 0 {
        return None;
    }

    let ((columns, rows), pixels) = kitty::fit(image, (columns, rows), cell);
    Some(Block {
        columns,
        rows,
        pixels,
    })
}

/// The terminal's size in cells, and the size of one cell in pixels.
///
/// `TIOCGWINSZ` answers both at once — `ws_col`/`ws_row` in cells and
/// `ws_xpixel`/`ws_ypixel` in pixels — so a cell is the one divided by the
/// other, which [`cell`] does.
///
/// `rustix` wraps the `ioctl` rather than this crate doing it itself, which is
/// what lets the crate [forbid `unsafe`](crate).
fn winsize() -> Option<((u32, u32), (u32, u32))> {
    let size = tcgetwinsize(stdout()).ok()?;

    if size.ws_col == 0 || size.ws_row == 0 {
        return None;
    }
    let cells = (u32::from(size.ws_col), u32::from(size.ws_row));
    let pixels = (u32::from(size.ws_xpixel), u32::from(size.ws_ypixel));

    Some((cells, cell(cells, pixels)))
}

/// One cell's size, from a terminal's two answers about its own.
///
/// The division is where the care is needed, and the zero it can land on is not
/// only the documented one. A terminal is allowed to leave the pixel fields at
/// zero, and several do; a terminal may also report a pixel size too small to
/// divide into its cell count. Both come out of the division as a zero-width or
/// zero-height cell, which walks through [`kitty::fit`] as a `0.0 / 0.0` scale
/// and comes back as a block with no pixels in it — an empty PNG, a blank
/// reserved block for the whole render, and a warning only once it is over.
///
/// [`ASSUMED_CELL`] instead, in both cases: the aspect ratio of the preview is
/// the only thing the guess affects.
fn cell(cells: (u32, u32), pixels: (u32, u32)) -> (u32, u32) {
    match (pixels.0 / cells.0, pixels.1 / cells.1) {
        (0, _) | (_, 0) => ASSUMED_CELL,
        cell => cell,
    }
}

/// Why this terminal cannot show a preview, if it cannot.
///
/// The checks are on the environment rather than on the terminal itself. The
/// rigorous alternative is the protocol's own `a=q` query, but reading the reply
/// means putting stdin into raw mode with a timeout, and a wrong guess here
/// costs a warning rather than a broken render. Worth revisiting only if the
/// guess turns out to be wrong in practice.
fn supported() -> Result<(), String> {
    if !stdout().is_terminal() {
        return Err("output is not a terminal, so there is nowhere to draw a preview".into());
    }
    // Both multiplexers filter the escape sequences the protocol rides on, and
    // both are invisible to the checks below: a kitty running screen still has
    // `KITTY_WINDOW_ID` in the environment, so the terminal answers for a
    // window that is not the one being drawn into.
    if std::env::var_os("TMUX").is_some() {
        return Err(
            "tmux needs explicit passthrough for terminal graphics, so the preview is off".into(),
        );
    }
    if std::env::var_os("STY").is_some() {
        return Err(
            "GNU screen does not pass terminal graphics through, so the preview is off".into(),
        );
    }

    let variable = |name: &str| std::env::var(name).unwrap_or_default().to_lowercase();
    let term = variable("TERM");
    let program = variable("TERM_PROGRAM");
    let known = ["kitty", "ghostty", "wezterm", "konsole"]
        .iter()
        .any(|name| term.contains(name) || program.contains(name));

    // Set-but-empty is how a variable gets unset in practice — `env FOO= cmd`,
    // or a shell that exports it before it has a value — and it says nothing
    // about the terminal, so it does not count as a flag.
    let flagged = ["KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR", "WEZTERM_PANE"]
        .iter()
        .any(|name| !variable(name).is_empty());

    match known || flagged {
        true => Ok(()),
        false => Err(format!(
            "{term} does not appear to support terminal graphics — kitty, Ghostty, WezTerm and Konsole do"
        )),
    }
}

/// The `SIGINT` handling a preview runs under, undone when it is dropped.
///
/// Two actions on the one signal, and the order they go on in is the whole
/// design. The first `Ctrl-C` finds the flag clear, so the shutdown action does
/// nothing and the setter arms it; the sample loop sees it on its next pass and
/// stops with what has converged. A second one — arriving while the flag is
/// still set — finds it armed and leaves immediately, because a render that has
/// got itself stuck should still be killable from the keyboard.
///
/// Which is why [`Interrupts::clear`] exists: the work that *follows* the loop
/// is the work the first `Ctrl-C` asked for, and it must not be run under an
/// armed flag.
struct Interrupts {
    /// Set from the handler, read by the sample loop.
    interrupted: Arc<AtomicBool>,
    /// The setter, which is the half [`Drop`] takes back off the signal.
    setter: SigId,
}

impl Interrupts {
    /// Installs both actions. The shutdown goes on first so that it reads the
    /// flag as the *previous* signal left it, not as this one is about to.
    fn install() -> Result<Interrupts, Box<dyn Error>> {
        let interrupted = Arc::new(AtomicBool::new(false));
        flag::register_conditional_shutdown(SIGINT, 130, Arc::clone(&interrupted))?;
        let setter = flag::register(SIGINT, Arc::clone(&interrupted))?;

        Ok(Interrupts {
            interrupted,
            setter,
        })
    }

    /// Whether a `Ctrl-C` has arrived since this was installed.
    fn interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Relaxed)
    }

    /// Puts the flag back the way it was installed, so that resolving the
    /// accumulator and writing the PNG gets the same two strikes the sample
    /// loop had.
    ///
    /// Without it the flag stays armed from the `Ctrl-C` that stopped the loop
    /// until the render returns, and a second one anywhere in that window exits
    /// with nothing written — which is precisely the frame the early stop was
    /// asked to save. The window is not the 17ms it looks like either: what
    /// runs in it is a GPU drain, a readback and a `resolve` over every pixel,
    /// which is seconds at 4K, and hitting `Ctrl-C` twice is a habit rather
    /// than an accident.
    ///
    /// Killability survives, one strike further along: the `Ctrl-C` that lands
    /// in the window arms the flag again, and the one after it exits.
    fn clear(&self) {
        self.interrupted.store(false, Ordering::SeqCst);
    }
}

impl Drop for Interrupts {
    /// Hands `Ctrl-C` back before the preview returns: the caller still has a
    /// PNG to write, and it should stay interruptible.
    ///
    /// Which is why this takes off the setter and arms the flag rather than
    /// taking off both actions. Unregistering the last action for a signal does
    /// not put the previous handler back — it leaves the signal *ignored* — so
    /// removing both is how a render becomes unkillable. Arming the shutdown
    /// instead leaves one action holding the signal and exiting on it, which is
    /// the behaviour being restored.
    fn drop(&mut self) {
        low_level::unregister(self.setter);
        self.interrupted.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block is normally smaller than the render, and the frame is shrunk to
    /// fill it exactly.
    #[test]
    fn a_large_render_is_sent_at_the_block_size() {
        assert_eq!(transmitted((800, 600), (640, 480)), (640, 480));
    }

    /// The case that panicked before it was a function: a 32x32 golden scene in
    /// a terminal with room for 480x480. Sending it at the block size would have
    /// claimed 480x480 pixels for a buffer holding 32x32, since [`downsample`]
    /// declines to magnify. The terminal scales it into the cells instead.
    #[test]
    fn a_small_render_is_sent_at_its_own_size() {
        assert_eq!(transmitted((32, 32), (480, 480)), (32, 32));
    }

    /// Rounding the block down to whole cells can leave one axis above the
    /// render's and the other below, so the two are compared separately.
    #[test]
    fn each_axis_is_capped_on_its_own() {
        assert_eq!(transmitted((100, 600), (120, 480)), (100, 480));
        assert_eq!(transmitted((800, 40), (640, 60)), (640, 40));
    }

    /// A terminal that reports no pixel size at all, which is allowed and
    /// common enough that the preview guesses rather than declining to draw.
    #[test]
    fn a_terminal_that_reports_no_pixels_gets_the_assumed_cell() {
        assert_eq!(cell((80, 24), (0, 0)), ASSUMED_CELL);
    }

    /// And the case a check on the reported fields alone misses: pixels given,
    /// but fewer of them than there are cells, so it is the division that lands
    /// on zero. A zero-width cell reached `fit` as a NaN scale and came back as
    /// a frame with no pixels in it.
    #[test]
    fn a_pixel_size_too_small_to_divide_gets_the_assumed_cell() {
        assert_eq!(cell((80, 24), (40, 384)), ASSUMED_CELL);
        assert_eq!(cell((80, 24), (640, 12)), ASSUMED_CELL);
    }

    #[test]
    fn a_reported_pixel_size_divides_into_cells() {
        assert_eq!(cell((80, 24), (640, 384)), (8, 16));
    }

    /// The first `Ctrl-C` has to leave the process alive — the whole point is
    /// that the samples taken so far are resolved and written. `raise` delivers
    /// to the calling thread before it returns, so the flag is readable straight
    /// after it.
    ///
    /// This test arms the shutdown when it drops, which is the behaviour
    /// [`Interrupts`] is documented to have and costs the rest of the run
    /// nothing: a `Ctrl-C` out of `cargo test` exits 130 rather than dying by
    /// the signal, which no shell distinguishes.
    #[test]
    fn a_first_interrupt_stops_the_loop_rather_than_the_process() {
        let interrupts = Interrupts::install().expect("SIGINT takes a handler");
        assert!(!interrupts.interrupted());

        low_level::raise(SIGINT).expect("raising SIGINT at ourselves");
        assert!(interrupts.interrupted());
    }

    /// The `Ctrl-C` that stops the loop is a request for the PNG, so the work
    /// that writes it runs with the flag clear and gets a strike of its own: a
    /// second `Ctrl-C` while the accumulator is being resolved has to arm the
    /// flag rather than exit on it.
    ///
    /// Raising twice is the whole test, so it cannot run in a process shared
    /// with the other tests — a raise in here fires every handler they have
    /// installed too. Same shape as the test below: the child is the branch
    /// under the environment variable, and surviving to the end of it is the
    /// assertion.
    #[test]
    fn clearing_gives_the_work_after_the_loop_its_own_strike() {
        const CHILD: &str = "WGSL_RAYTRACE_CLEARED_CHILD";

        if std::env::var_os(CHILD).is_some() {
            let interrupts = Interrupts::install().expect("SIGINT takes a handler");

            low_level::raise(SIGINT).expect("raising SIGINT at ourselves");
            assert!(interrupts.interrupted());

            interrupts.clear();
            assert!(!interrupts.interrupted());

            // The shutdown action reads the flag before the setter writes it,
            // so this one arms rather than exits — as the first one did.
            low_level::raise(SIGINT).expect("raising SIGINT at ourselves");
            assert!(interrupts.interrupted());
            return;
        }

        let binary = std::env::current_exe().expect("the test binary's own path");
        let status = std::process::Command::new(binary)
            .args([
                "--exact",
                "render::preview::tests::clearing_gives_the_work_after_the_loop_its_own_strike",
            ])
            .env(CHILD, "1")
            .status()
            .expect("re-running the test binary");

        assert!(status.success(), "the cleared flag exited with {status}");
    }

    /// And the second one leaves, whatever the render is stuck on — including
    /// after the preview has handed the signal back, which is what [`Drop`]
    /// arming the flag rather than unregistering both actions buys.
    ///
    /// Exiting is not something a test can do in-process, so this re-runs the
    /// test binary against this one test and reads the status off it. The child
    /// is the branch below the environment variable.
    #[test]
    fn a_second_interrupt_leaves_with_130() {
        const CHILD: &str = "WGSL_RAYTRACE_INTERRUPT_CHILD";

        if std::env::var_os(CHILD).is_some() {
            drop(Interrupts::install().expect("SIGINT takes a handler"));
            low_level::raise(SIGINT).expect("raising SIGINT at ourselves");
            unreachable!("the armed shutdown should have exited");
        }

        let binary = std::env::current_exe().expect("the test binary's own path");
        let status = std::process::Command::new(binary)
            .args([
                "--exact",
                "render::preview::tests::a_second_interrupt_leaves_with_130",
            ])
            .env(CHILD, "1")
            .status()
            .expect("re-running the test binary");

        assert_eq!(status.code(), Some(130));
    }
}
