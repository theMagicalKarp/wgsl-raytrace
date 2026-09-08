//! The [kitty graphics protocol][spec], enough of it to put one image on the
//! screen and keep replacing it.
//!
//! [spec]: https://sw.kovidgoyal.net/kitty/graphics-protocol/

/// The image id every frame transmits under.
///
/// Reusing one id — with the placement id below — is what makes a frame
/// *replace* its predecessor. Transmitting under a fresh id each time would
/// leave the terminal holding every frame of the render in memory, and stack
/// placements down the screen.
const IMAGE: u32 = 0x7261_7974;

/// The placement id, for the same reason. A placement is identified by the pair,
/// so re-placing under both replaces rather than adds.
const PLACEMENT: u32 = 1;

/// Payload bytes per escape sequence. The protocol's own limit: a chunk carries
/// at most 4096 bytes of base64.
const CHUNK: usize = 4096;

/// Escape sequences that transmit `png` and display it in a `cells` box.
///
/// `a=T` transmits and places in one go, `f=100` says the payload is a PNG, and
/// `c`/`r` are the columns and rows to draw it across. The image is downsampled
/// to exactly the pixels those cells cover before it gets here (see
/// [`fit`]), so the terminal scales it by one and nothing is resampled twice.
///
/// Two flags carry more weight than they look like they do:
///
/// - **`q=2`** suppresses the terminal's replies. Without it every frame leaves
///   an `\x1b_G...;OK\x1b\\` on the terminal's input, which the shell reads back
///   as keystrokes once the render exits.
/// - **`m`** is the continuation flag: 1 on every chunk but the last, 0 on the
///   last. The control keys ride on the first chunk only; the rest carry `m`
///   alone, since the terminal is still filling in the image the first one
///   opened.
pub fn encode(png: &[u8], cells: (u32, u32)) -> String {
    let payload = base64(png);
    let (columns, rows) = cells;

    let mut out = String::with_capacity(payload.len() + payload.len() / CHUNK * 32 + 64);
    let mut chunks = payload.as_bytes().chunks(CHUNK).peekable();
    let mut first = true;

    while let Some(chunk) = chunks.next() {
        let more = u8::from(chunks.peek().is_some());
        out.push_str("\x1b_G");
        if first {
            out.push_str(&format!(
                "a=T,i={IMAGE},p={PLACEMENT},f=100,c={columns},r={rows},q=2,"
            ));
            first = false;
        }
        out.push_str(&format!("m={more};"));
        out.push_str(std::str::from_utf8(chunk).expect("base64 is ascii"));
        out.push_str("\x1b\\");
    }

    out
}

/// The largest whole number of cells `image` fits in `cells` at its own aspect
/// ratio, and the pixels those cells cover.
///
/// Returned as a pair because the caller needs both: the cell count goes in the
/// escape sequence, and the pixel size is what the frame is downsampled to. They
/// are the same rectangle counted two ways, which is what keeps the terminal
/// from scaling anything.
///
/// `cell` is the size of one character cell in pixels. A terminal that reports
/// zeros there is handled by the caller, not here.
pub fn fit(image: (u32, u32), cells: (u32, u32), cell: (u32, u32)) -> ((u32, u32), (u32, u32)) {
    let (image_width, image_height) = image;
    let (columns, rows) = cells;
    let (cell_width, cell_height) = cell;

    if image_width == 0 || image_height == 0 || columns == 0 || rows == 0 {
        return ((0, 0), (0, 0));
    }

    // Scale to whichever axis runs out first, in pixels, then round down to
    // whole cells — a cell is the smallest thing the protocol can place.
    let available_width = (columns * cell_width) as f64;
    let available_height = (rows * cell_height) as f64;
    let scale = (available_width / image_width as f64).min(available_height / image_height as f64);

    let columns = ((image_width as f64 * scale) / cell_width as f64).floor() as u32;
    let rows = ((image_height as f64 * scale) / cell_height as f64).floor() as u32;
    let (columns, rows) = (columns.max(1), rows.max(1));

    ((columns, rows), (columns * cell_width, rows * cell_height))
}

/// Standard base64, [RFC 4648] — the only encoding the protocol's escape
/// sequences carry.
///
/// Hand-rolled rather than pulled in: it is fifteen lines against a dependency,
/// and the RFC ships its own test vectors to check it with.
///
/// [RFC 4648]: https://datatracker.ietf.org/doc/html/rfc4648#section-4
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let packed = group
            .iter()
            .enumerate()
            .fold(0u32, |packed, (index, &byte)| {
                packed | (byte as u32) << (16 - 8 * index)
            });

        for index in 0..=group.len() {
            out.push(ALPHABET[(packed >> (18 - 6 * index) & 0x3f) as usize] as char);
        }
        // A group of one encodes to two characters and a group of two to three;
        // either way the quantum is four.
        for _ in group.len()..3 {
            out.push('=');
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vectors from RFC 4648 section 10, which exist to catch exactly the
    /// off-by-one in the padding that a hand-rolled encoder gets wrong.
    #[test]
    fn base64_matches_the_rfc() {
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(input.as_bytes()), expected, "encoding {input:?}");
        }
    }

    /// Every byte value, so a high bit lost somewhere in the shifting shows up.
    #[test]
    fn base64_survives_every_byte() {
        let bytes: Vec<u8> = (0..=255u8).collect();
        let encoded = base64(&bytes);

        assert_eq!(encoded.len(), 344, "256 bytes is 86 quanta");
        assert!(encoded.ends_with('='), "256 is not a multiple of three");
        assert!(encoded.starts_with("AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8g"));
    }

    /// Splits an encoded frame back into (control keys, payload) per escape.
    fn escapes(encoded: &str) -> Vec<(String, String)> {
        encoded
            .strip_suffix("\x1b\\")
            .expect("the last escape is terminated")
            .split("\x1b\\")
            .map(|escape| {
                let body = escape.strip_prefix("\x1b_G").expect("an APC introducer");
                let (keys, payload) = body.split_once(';').expect("keys and payload");
                (keys.to_string(), payload.to_string())
            })
            .collect()
    }

    #[test]
    fn a_small_image_is_one_escape() {
        let escapes = escapes(&encode(b"tiny", (4, 2)));

        assert_eq!(escapes.len(), 1);
        assert!(escapes[0].0.contains("a=T"), "{}", escapes[0].0);
        assert!(escapes[0].0.contains("f=100"), "{}", escapes[0].0);
        assert!(escapes[0].0.contains("c=4,r=2"), "{}", escapes[0].0);
        assert!(escapes[0].0.contains("q=2"), "the terminal must not reply");
        assert!(escapes[0].0.ends_with("m=0"), "nothing follows it");
        assert_eq!(escapes[0].1, base64(b"tiny"));
    }

    /// The chunking is the part with an off-by-one in it: control keys on the
    /// first escape only, the continuation flag on every one, and `m=0` on the
    /// last to tell the terminal the image is complete.
    #[test]
    fn a_large_image_chunks_and_reassembles() {
        let png: Vec<u8> = (0..10_000u32).map(|byte| byte as u8).collect();
        let encoded = encode(&png, (40, 20));
        let escapes = escapes(&encoded);

        let expected = base64(&png);
        assert_eq!(escapes.len(), expected.len().div_ceil(CHUNK));
        assert!(escapes.len() > 2, "worth chunking at all");

        assert!(escapes[0].0.starts_with("a=T"), "{}", escapes[0].0);
        for (index, (keys, _)) in escapes.iter().enumerate().skip(1) {
            assert!(
                keys.starts_with("m="),
                "escape {index} should carry the continuation flag alone, got {keys:?}"
            );
        }

        let last = escapes.len() - 1;
        for (index, (keys, _)) in escapes.iter().enumerate() {
            let flag = match index == last {
                true => "m=0",
                false => "m=1",
            };
            assert!(keys.ends_with(flag), "escape {index} should end {flag}");
        }

        let rejoined: String = escapes
            .iter()
            .map(|(_, payload)| payload.as_str())
            .collect();
        assert_eq!(rejoined, expected, "the payload should survive the split");
    }

    #[test]
    fn fit_pillarboxes_a_wide_image() {
        // 800x600 into 80x24 cells of 10x20 — 800x480 available, so height binds.
        let (cells, pixels) = fit((800, 600), (80, 24), (10, 20));

        assert_eq!(cells, (64, 24));
        assert_eq!(pixels, (640, 480));
        assert!(cells.0 < 80, "it should not fill the width");
    }

    #[test]
    fn fit_letterboxes_a_tall_image() {
        // 600x800 into the same box: width binds at 800 pixels only if the image
        // were wider than tall, so here the 480-pixel height is what caps it.
        let (cells, pixels) = fit((600, 800), (80, 24), (10, 20));

        assert_eq!(cells, (36, 24));
        assert_eq!(pixels, (360, 480));
    }

    /// The pixel size has to be a whole number of cells or the terminal rescales
    /// what was already scaled, which is the one thing this is trying to avoid.
    #[test]
    fn fit_always_lands_on_whole_cells() {
        let cell = (7, 15);
        for width in [17u32, 100, 801, 1920] {
            for height in [13u32, 99, 600, 1081] {
                let (cells, pixels) = fit((width, height), (73, 19), cell);

                assert_eq!(pixels.0, cells.0 * cell.0, "{width}x{height}");
                assert_eq!(pixels.1, cells.1 * cell.1, "{width}x{height}");
                assert!(
                    cells.0 <= 73 && cells.1 <= 19,
                    "{width}x{height} overflowed"
                );
            }
        }
    }

    /// A terminal that reports nothing, or a scene with no pixels, has to come
    /// back with something rather than divide by zero.
    #[test]
    fn fit_handles_a_degenerate_box() {
        assert_eq!(fit((800, 600), (0, 0), (10, 20)), ((0, 0), (0, 0)));
        assert_eq!(fit((0, 0), (80, 24), (10, 20)), ((0, 0), (0, 0)));
    }

    /// One cell is the floor, however lopsided the image: a placement of zero
    /// columns draws nothing at all.
    #[test]
    fn fit_never_rounds_down_to_nothing() {
        let (cells, pixels) = fit((4000, 4), (80, 24), (10, 20));

        assert_eq!(cells.0.min(cells.1), 1, "{cells:?}");
        assert_eq!(pixels, (cells.0 * 10, cells.1 * 20));
    }
}
