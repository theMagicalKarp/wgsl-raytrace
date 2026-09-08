use std::io::Write;
use std::io::stdout;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

/// Held by whoever is writing to the terminal.
pub static TERMINAL: Mutex<()> = Mutex::new(());

/// Lower bound on the gap between redraws, so a slow terminal can never become
/// the bottleneck.
const REDRAW_INTERVAL: Duration = Duration::from_millis(100);

const BAR: usize = 24;

/// Counts finished samples, redrawing in place with a carriage return.
pub struct Progress {
    total: u32,
    completed: u32,
    started: Instant,
    drawn: Instant,
}

impl Progress {
    pub fn new(total: u32) -> Self {
        let started = Instant::now();
        Progress {
            total,
            completed: 0,
            started,
            drawn: started,
        }
    }

    /// Records one finished sample, redrawing only if enough time has passed —
    /// or if this is the first or the last one, which are always worth showing.
    pub fn advance(&mut self) {
        self.completed += 1;

        let due = self.drawn.elapsed() >= REDRAW_INTERVAL;
        if (due || self.completed == 1 || self.completed == self.total) && self.draw() {
            self.drawn = Instant::now();
        }
    }

    /// Redraws the bar, or does nothing and says so if [`TERMINAL`] is busy.
    ///
    /// The timer is left alone on a skipped draw, so the next sample tries again
    /// rather than waiting out another interval.
    fn draw(&self) -> bool {
        let Ok(_writing) = TERMINAL.try_lock() else {
            return false;
        };

        let fraction = self.completed as f32 / self.total as f32;
        let filled = (fraction * BAR as f32).round() as usize;
        let elapsed = self.started.elapsed().as_secs_f32();
        // Extrapolated from the average sample time so far. `completed` is at
        // least one by the time this is called, so it never divides by zero.
        let remaining = elapsed / fraction - elapsed;

        print!(
            "\r[{:#<filled$}{:.<empty$}] {}/{} samples  {elapsed:.0}s elapsed  ~{remaining:.0}s left",
            "",
            "",
            self.completed,
            self.total,
            filled = filled,
            empty = BAR - filled,
        );
        let _ = stdout().flush();

        true
    }

    /// Closes off the line so whatever is printed next starts fresh.
    ///
    /// Borrows rather than consumes: the bar lives inside the renderer that
    /// drives it, and an interrupted preview finishes without giving that up.
    ///
    /// This one waits for [`TERMINAL`] rather than skipping: it is the newline
    /// everything printed after the render sits below.
    pub fn finish(&self) {
        let _writing = TERMINAL.lock();
        println!();
    }
}
