use std::error::Error;
use std::fmt;
use std::sync::mpsc;
use std::time::Duration;

/// Dispatches a render records, capped by the query set it has to fit in: two
/// timestamps each, against wgpu's ceiling of [`wgpu::QUERY_SET_MAX_QUERIES`].
const MAX_TIMED: u32 = wgpu::QUERY_SET_MAX_QUERIES / 2;

/// The timestamp apparatus for one render: a query set with a begin/end pair
/// per timed dispatch, and the two buffers it is read back through.
pub struct Timer {
    query_set: wgpu::QuerySet,
    /// Where `resolve_query_set` writes. A query set cannot be resolved
    /// straight into a mappable buffer, so this is the hop in between.
    staging: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period: f32,
    /// Every `stride`-th sample is timed.
    stride: u32,
    /// Pairs the query set holds.
    timed: u32,
    /// Dispatches the render will issue in total.
    dispatches: u32,
}

impl Timer {
    /// `None` when the device cannot write timestamps, which is not an error —
    /// it is a render that reports no timings.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, dispatches: u32) -> Option<Timer> {
        if dispatches == 0 || !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }

        let (stride, timed) = plan(dispatches);
        let size = 2 * timed as u64 * wgpu::QUERY_SIZE as u64;

        Some(Timer {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("dispatch timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * timed,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            period: queue.get_timestamp_period(),
            stride,
            timed,
            dispatches,
        })
    }

    /// The `timestamp_writes` for this sample's compute pass, or `None` when
    /// the stride skips it.
    pub fn writes(&self, sample: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let pair = entry(sample, self.stride)?;

        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(2 * pair),
            end_of_pass_write_index: Some(2 * pair + 1),
        })
    }

    /// Queues the whole query set for reading. Belongs in the same encoder as
    /// the accumulator's readback — after the loop has waited on every sample,
    /// which is what makes the resolve read live counters rather than stale
    /// ones.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.query_set, 0..2 * self.timed, &self.staging, 0);
        encoder.copy_buffer_to_buffer(&self.staging, 0, &self.readback, 0, self.readback.size());
    }

    /// Reads the timestamps back and reduces them to a summary.
    ///
    /// Call once the submission holding [`Timer::resolve`] has been made; the
    /// map resolves the same way the accumulator's does, by polling until the
    /// queued work ahead of it has run.
    pub fn timings(&self, device: &wgpu::Device) -> Result<Option<Timings>, Box<dyn Error>> {
        let slice = self.readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??;

        let mapped = slice.get_mapped_range()?;
        let stamps: &[u64] = bytemuck::cast_slice(&mapped);
        let deltas: Vec<u64> = stamps
            .chunks_exact(2)
            .map(|pair| pair[1].saturating_sub(pair[0]))
            .collect();
        drop(mapped);
        self.readback.unmap();

        Ok(summarize(&deltas, self.period, self.dispatches))
    }
}

/// How a render of `dispatches` samples is recorded: the stride between timed
/// samples, and how many pairs that leaves for the query set to hold.
fn plan(dispatches: u32) -> (u32, u32) {
    let stride = dispatches.div_ceil(MAX_TIMED).max(1);
    (stride, dispatches.div_ceil(stride))
}

/// Which pair of queries a sample writes, or `None` when the stride skips it.
///
/// Samples are counted from one, so the arithmetic runs on `sample - 1`.
fn entry(sample: u32, stride: u32) -> Option<u32> {
    let index = sample.checked_sub(1)?;
    match index % stride {
        0 => Some(index / stride),
        _ => None,
    }
}

/// What a render's dispatches cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timings {
    /// Dispatches actually measured, which is fewer than `dispatches` when the
    /// sample count ran past [`MAX_TIMED`].
    pub timed: u32,
    /// Dispatches the render issued.
    pub dispatches: u32,
    pub mean: Duration,
    pub min: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub max: Duration,
    /// The whole render's GPU time: `mean` over every dispatch issued.
    pub traced: Duration,
}

/// Reduces raw tick deltas to a summary, or `None` if none of them are usable.
///
/// A zero delta is a query nothing ever wrote — resolving an unwritten query is
/// allowed to return anything — and averaging those in would report a render as
/// faster than it was.
fn summarize(deltas: &[u64], period: f32, dispatches: u32) -> Option<Timings> {
    let mut ticks: Vec<u64> = deltas.iter().copied().filter(|&delta| delta > 0).collect();
    if ticks.is_empty() {
        return None;
    }
    ticks.sort_unstable();

    let nanos = |ticks: f64| Duration::from_nanos((ticks * period as f64) as u64);
    let at = |quantile: f64| {
        let last = (ticks.len() - 1) as f64;
        nanos(ticks[(last * quantile).round() as usize] as f64)
    };

    let total: u64 = ticks.iter().sum();
    let mean = total as f64 / ticks.len() as f64;

    Some(Timings {
        timed: ticks.len() as u32,
        dispatches,
        mean: nanos(mean),
        min: nanos(ticks[0] as f64),
        p50: at(0.50),
        p95: at(0.95),
        max: nanos(ticks[ticks.len() - 1] as f64),
        traced: nanos(mean * dispatches as f64),
    })
}

/// The unit a duration reads most naturally in, as a divisor and its suffix.
fn magnitude(duration: Duration) -> (f64, &'static str) {
    match duration.as_secs_f64() {
        seconds if seconds >= 1.0 => (1.0, "s"),
        seconds if seconds >= 1e-3 => (1e-3, "ms"),
        _ => (1e-6, "µs"),
    }
}

impl fmt::Display for Timings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The spread is quoted in the mean's unit rather than each in its own,
        // so the five numbers can be read against each other at a glance.
        let (divisor, unit) = magnitude(self.mean);
        let scaled = |duration: Duration| duration.as_secs_f64() / divisor;

        let (total_divisor, total_unit) = magnitude(self.traced);
        write!(
            f,
            "{:.2}{unit}/dispatch  ·  {:.1}{total_unit} traced  ·  \
             min {:.2}  p50 {:.2}  p95 {:.2}  max {:.2}",
            scaled(self.mean),
            self.traced.as_secs_f64() / total_divisor,
            scaled(self.min),
            scaled(self.p50),
            scaled(self.p95),
            scaled(self.max),
        )?;

        if self.timed != self.dispatches {
            write!(f, "  ·  sampled {} of {}", self.timed, self.dispatches)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A period of one nanosecond per tick, so a tick count reads as nanoseconds.
    const NS: f32 = 1.0;

    /// Every pair a render of `dispatches` samples would write.
    fn recorded(dispatches: u32) -> Vec<u32> {
        let (stride, _) = plan(dispatches);
        (1..=dispatches).filter_map(|s| entry(s, stride)).collect()
    }

    #[test]
    fn summarizes_a_known_spread() {
        let timings = summarize(&[10, 20, 30, 40], NS, 4).unwrap();

        assert_eq!(timings.mean, Duration::from_nanos(25));
        assert_eq!(timings.min, Duration::from_nanos(10));
        assert_eq!(timings.max, Duration::from_nanos(40));
        assert_eq!(timings.p50, Duration::from_nanos(30));
        assert_eq!(timings.traced, Duration::from_nanos(100));
        assert_eq!(timings.timed, 4);
    }

    #[test]
    fn scales_ticks_by_the_period() {
        // Half a nanosecond a tick halves every figure it reports.
        let timings = summarize(&[10, 20, 30, 40], 0.5, 4).unwrap();

        assert_eq!(timings.mean, Duration::from_nanos(12));
        assert_eq!(timings.max, Duration::from_nanos(20));
    }

    #[test]
    fn drops_queries_that_were_never_written() {
        let timings = summarize(&[0, 20, 0, 40], NS, 4).unwrap();

        assert_eq!(timings.timed, 2, "the zeroes should not be averaged in");
        assert_eq!(timings.mean, Duration::from_nanos(30));
        // The render still issued four dispatches, so that is what the total is
        // extrapolated over.
        assert_eq!(timings.dispatches, 4);
        assert_eq!(timings.traced, Duration::from_nanos(120));
    }

    #[test]
    fn nothing_usable_is_no_timing_rather_than_a_zero() {
        assert!(summarize(&[], NS, 0).is_none());
        assert!(summarize(&[0, 0, 0], NS, 3).is_none());
    }

    #[test]
    fn a_short_render_times_every_dispatch() {
        assert_eq!(plan(100), (1, 100));
        assert_eq!(recorded(100), (0..100).collect::<Vec<u32>>());
    }

    #[test]
    fn a_long_render_strides_across_its_samples() {
        let dispatches = 10 * MAX_TIMED + 7;
        let (stride, timed) = plan(dispatches);

        // The pairs have to fit the query set, and fill it densely enough that
        // the readback can be walked straight through.
        assert!(2 * timed <= wgpu::QUERY_SET_MAX_QUERIES, "{timed} pairs");
        assert_eq!(recorded(dispatches), (0..timed).collect::<Vec<u32>>());

        // The stride reaches the end of the render rather than exhausting the
        // query set on its opening, which is what makes the summary describe
        // the whole run.
        let last = (timed - 1) * stride + 1;
        assert!(
            dispatches - last < stride,
            "the last timed sample is {last} of {dispatches}",
        );
    }

    #[test]
    fn sample_zero_is_not_an_entry() {
        // Samples count from one; a zero would underflow the index.
        assert_eq!(entry(0, 1), None);
    }

    #[test]
    fn a_render_of_nothing_has_no_stride_to_divide_by() {
        assert_eq!(plan(0), (1, 0));
    }

    #[test]
    fn reports_the_spread_in_the_mean_unit() {
        let timings = summarize(&[3_710_000, 3_920_000, 4_180_000, 5_300_000], NS, 4).unwrap();

        let line = timings.to_string();
        assert!(line.contains("ms/dispatch"), "{line}");
        assert!(line.contains("min 3.71"), "{line}");
        assert!(line.contains("max 5.30"), "{line}");
        assert!(!line.contains("sampled"), "nothing was skipped: {line}");
    }

    #[test]
    fn says_so_when_it_only_sampled_some() {
        let timings = summarize(&[10, 20], NS, 10_000).unwrap();

        assert!(timings.to_string().contains("sampled 2 of 10000"));
    }
}
