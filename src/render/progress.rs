use std::io::Write;
use std::io::stdout;
use std::time::Duration;
use std::time::Instant;

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
        if due || self.completed == 1 || self.completed == self.total {
            self.draw();
            self.drawn = Instant::now();
        }
    }

    fn draw(&self) {
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
    }

    /// Closes off the line so whatever is printed next starts fresh.
    pub fn finish(self) {
        println!();
    }
}
