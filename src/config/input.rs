//! Per-field material inputs: the patterns a scene writes in place of a number.
//!
//! Every patternable field of a [`Principled`](super::Principled) surface holds
//! an [`Operand`] — a plain value, or a tree of [`Input`]s evaluated once per
//! hit on the GPU. The tree is data here and nothing else; `scene::program`
//! flattens it into the postfix program the shader runs.
//!
//! A pattern is written as an expression in a string —
//! `roughness = "remap(uv.r, [0.05, 1.0])"` — which
//! [`config::expression`](super::expression) parses into the tree below. This
//! module owns the tree and what can be known about it without running it; the
//! grammar and its errors live next door.
//!
//! An input **replaces** the value it is written in place of rather than
//! multiplying it, which is Blender's rule and the only one that reads without
//! surprises: `roughness = "..."` is "this pattern", not "half this pattern"
//! because the field happens to default to 0.5.

use super::expression;
use serde::Deserialize;
use serde::Deserializer;
use serde::de;
use serde::de::SeqAccess;
use serde::de::Visitor;
use serde::de::value::SeqAccessDeserializer;
use std::fmt;

/// The coordinates a pattern can be laid out in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    /// The mesh's own texture coordinates. Zero on a file with no `vt` lines,
    /// which is every mesh here but melee's.
    Uv,

    /// The space the `.obj` was authored in, recovered by undoing the object's
    /// transform. A pattern written here stays put when the object moves.
    Object,

    /// World space. A pattern written here is fixed to the scene rather than to
    /// the surface, so an object moving through it swims.
    World,
}

impl Space {
    /// The name a scene writes it under, in both forms.
    pub fn name(self) -> &'static str {
        match self {
            Space::Uv => "uv",
            Space::Object => "object",
            Space::World => "world",
        }
    }
}

/// One component of a vector input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    R,
    G,
    B,
    A,
}

impl Channel {
    /// The canonical name of the four this parses under: a colour's, since a
    /// channel is a channel whatever the space it was taken from.
    pub fn name(self) -> &'static str {
        match self {
            Channel::R => "r",
            Channel::G => "g",
            Channel::B => "b",
            Channel::A => "a",
        }
    }

    pub fn index(self) -> u32 {
        match self {
            Channel::R => 0,
            Channel::G => 1,
            Channel::B => 2,
            Channel::A => 3,
        }
    }
}

/// How a [`Input::Mix`] combines its two sides. The names and the arithmetic are
/// Blender's Mix node, restricted to the four modes worth having before there
/// are images to blend. Each is the name of a call: `overlay(a, b, factor)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Blend {
    #[default]
    Mix,
    Multiply,
    Add,
    Overlay,
}

/// How deeply a pattern may nest, in either direction: the parser recurses once
/// per level building the tree and `scene::program` recurses once per level
/// walking it, and a Rust stack is not a thing either one may run off.
///
/// Far past anything a scene writes by hand — [`crate::scene::program`] refuses
/// a tree needing more than eight values in flight long before this — and it is
/// here to turn "deeper than anyone meant" into an error with a line number
/// rather than an abort with a stack trace.
pub const MAX_NESTING: u32 = 64;

/// A material field: a number, a color, or a pattern.
///
/// Untagged, so a field reads the way it always has when it holds a constant —
/// `roughness = 0.5`, `base_color = [0.8, 0.8, 0.8]` — and reads as a string
/// when it holds a pattern:
/// `base_color = "mix([0.75, 0.15, 0.1], [0.1, 0.2, 0.7], object.z)"`.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    Scalar(f32),
    Color([f32; 3]),
    Input(Input),
}

/// Written out rather than derived as `#[serde(untagged)]`, which reports every
/// malformed field as "data did not match any variant" and throws the real
/// error away. Dispatching on the shape first means the expression parser's
/// own error is what a bad pattern reports, and a two-element color is still
/// "invalid length 2".
impl<'de> Deserialize<'de> for Operand {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Operand, D::Error> {
        deserializer.deserialize_any(OperandVisitor)
    }
}

struct OperandVisitor;

impl<'de> Visitor<'de> for OperandVisitor {
    type Value = Operand;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a number, three of them, or a pattern expression")
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Operand, E> {
        Ok(Operand::Scalar(value as f32))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Operand, E> {
        Ok(Operand::Scalar(value as f32))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Operand, E> {
        Ok(Operand::Scalar(value as f32))
    }

    /// A pattern written as an expression. The parser's error already names
    /// the column it stopped at, so it is passed through as the whole message.
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Operand, E> {
        expression::parse(value).map_err(E::custom)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Operand, A::Error> {
        Deserialize::deserialize(SeqAccessDeserializer::new(seq)).map(Operand::Color)
    }
}

impl Operand {
    /// The constant this holds, broadcast to three channels, or `None` for a
    /// pattern. What a field's GPU slot is filled with when nothing patterns it.
    pub fn constant(&self) -> Option<[f32; 3]> {
        match self {
            Operand::Scalar(value) => Some([*value; 3]),
            Operand::Color(value) => Some(*value),
            Operand::Input(_) => None,
        }
    }

    /// Whether a pattern drives this field.
    pub fn is_input(&self) -> bool {
        matches!(self, Operand::Input(_))
    }

    /// Whether this is constantly zero in every channel.
    ///
    /// The one value that decides a product on its own: a surface whose
    /// emission colour or strength is this emits nothing wherever it is hit,
    /// however unbounded the other half of it is.
    pub fn is_zero(&self) -> bool {
        self.constant()
            .is_some_and(|value| value.iter().all(|channel| *channel == 0.0))
    }

    /// The smallest and largest value this can take, per channel, or `None`
    /// when it is not bounded at all.
    ///
    /// Only emission asks. The light table weights a triangle by what it emits,
    /// and a field that is decided per hit has no one value to weight by — so
    /// the table uses an upper bound instead, which keeps light sampling
    /// unbiased however loose it is (see the note on `light_density`). A bound
    /// that does not exist is the one case that cannot be papered over, and it
    /// is reported rather than guessed at.
    ///
    /// Bounds are computed at the corners of each operand's range. Every op
    /// here is linear in each input separately, so an extreme of the whole is
    /// an extreme of the parts — with one exception: `overlay` changes branch
    /// at `a = 0.5`, and a side reaching past `[0, 1]` (emission routinely
    /// does) puts the extreme at that kink rather than at either end. The kink
    /// is evaluated alongside the corners, so this stays exact.
    pub fn range(&self) -> Option<([f32; 3], [f32; 3])> {
        match self {
            Operand::Scalar(value) => Some(([*value; 3], [*value; 3])),
            Operand::Color(value) => Some((*value, *value)),
            Operand::Input(input) => input.range(),
        }
    }
}

/// One node of a pattern.
///
/// The argument fields are themselves [`Operand`]s, so any of them can be a
/// plain number — `invert(0.25)` is a long way to write 0.75, and
/// `mix(0.9, 0.6, uv.r)` is the shape that actually gets written.
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    /// The shading point, in one of the three spaces. A `channel` keeps one
    /// component and broadcasts it; without one the whole vector is carried,
    /// which is what a color field wants.
    Coordinates {
        space: Space,
        channel: Option<Channel>,
    },

    /// One component of `input`, broadcast to every channel.
    ///
    /// The same thing a coordinate's own `channel` does, for everything that is
    /// not a coordinate: `remap(uv, [0, 4]).r` is how a pattern says which of
    /// its channels a scalar field should read, and the fold in
    /// `scene::program` is what makes a patterned `emission_strength` brighten
    /// rather than tint.
    Channel {
        input: Box<Operand>,
        channel: Channel,
    },

    /// `1 - input`, per channel.
    Invert { input: Box<Operand> },

    /// `input` taken from the `from` range onto the `to` range, clamped to it.
    ///
    /// The clamp is what makes this the way to give an unbounded pattern — a
    /// raw coordinate — a range it is allowed to drive a field with.
    Remap {
        input: Box<Operand>,
        from: [f32; 2],
        to: [f32; 2],
    },

    /// `a` and `b` blended by `factor`, which is clamped to `[0, 1]`.
    Mix {
        a: Box<Operand>,
        b: Box<Operand>,
        factor: Box<Operand>,
        mode: Blend,
    },
}

impl Blend {
    /// The name of the call that spells this mode.
    pub fn name(self) -> &'static str {
        match self {
            Blend::Mix => "mix",
            Blend::Multiply => "multiply",
            Blend::Add => "add",
            Blend::Overlay => "overlay",
        }
    }
}

/// Blends one channel the way the shader does. Kept here so the host's bounds
/// and the GPU's arithmetic cannot drift apart without a test noticing.
pub fn blend(mode: Blend, a: f32, b: f32, factor: f32) -> f32 {
    let f = factor.clamp(0.0, 1.0);
    let mixed = match mode {
        Blend::Mix => b,
        Blend::Multiply => a * b,
        Blend::Add => a + b,
        Blend::Overlay => match a < 0.5 {
            true => 2.0 * a * b,
            false => 1.0 - 2.0 * (1.0 - a) * (1.0 - b),
        },
    };

    a + (mixed - a) * f
}

impl Input {
    fn range(&self) -> Option<([f32; 3], [f32; 3])> {
        match self {
            // A texture coordinate is whatever the `.obj` wrote, and a position
            // is wherever the object is. Neither has a bound, and inventing one
            // would be inventing a light's brightness.
            Input::Coordinates { .. } => None,

            Input::Channel { input, channel } => {
                let (low, high) = input.range()?;
                // A bound is three channels, so a pattern that reads the
                // fourth has none to report — and `get` says so rather than
                // panicking on an index the enum allows.
                let index = channel.index() as usize;
                Some(([*low.get(index)?; 3], [*high.get(index)?; 3]))
            }

            Input::Invert { input } => {
                let (low, high) = input.range()?;
                Some((high.map(|c| 1.0 - c), low.map(|c| 1.0 - c)))
            }

            // Clamped onto `to`, so this is bounded whatever it wraps — which
            // is the point of it.
            Input::Remap { to, .. } => Some(([to[0].min(to[1]); 3], [to[0].max(to[1]); 3])),

            Input::Mix { a, b, factor, mode } => {
                let (a_low, a_high) = a.range()?;
                let (b_low, b_high) = b.range()?;
                // The factor is clamped on the way in, so one with no bound
                // of its own still cannot take the result past either side.
                // That is what lets an emissive mix be driven by a raw
                // coordinate and still be weighted in the light table.
                let (f_low, f_high) = factor
                    .range()
                    .map(|(low, high)| (low[0].clamp(0.0, 1.0), high[0].clamp(0.0, 1.0)))
                    .unwrap_or((0.0, 1.0));

                let mut low = [f32::INFINITY; 3];
                let mut high = [f32::NEG_INFINITY; 3];
                for channel in 0..3 {
                    // `overlay` is the one mode with a kink rather than only
                    // corners. It is linear in `a` on each side of 0.5, so
                    // adding that point to the corners covers every extreme;
                    // without it a surface mixing 3.0 over a coordinate is
                    // bounded at 1.0 and lit as if it were that dim.
                    let kink =
                        (*mode == Blend::Overlay && a_low[channel] < 0.5 && a_high[channel] > 0.5)
                            .then_some(0.5);

                    for &x in [a_low[channel], a_high[channel]].iter().chain(kink.iter()) {
                        for &y in &[b_low[channel], b_high[channel]] {
                            // The factor is a scalar on the GPU: it reads the
                            // first channel and nothing else.
                            for &f in &[f_low, f_high] {
                                let value = blend(*mode, x, y, f);
                                low[channel] = low[channel].min(value);
                                high[channel] = high[channel].max(value);
                            }
                        }
                    }
                }
                Some((low, high))
            }
        }
    }
}

/// Writes the expression form, so what a message prints is what a scene can
/// paste back in. The round-trip is a test.
impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operand::Scalar(value) => write!(f, "{value}"),
            Operand::Color([r, g, b]) => write!(f, "[{r}, {g}, {b}]"),
            Operand::Input(input) => write!(f, "{input}"),
        }
    }
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Input::Coordinates { space, channel } => match channel {
                Some(channel) => write!(f, "{}.{}", space.name(), channel.name()),
                None => f.write_str(space.name()),
            },
            Input::Channel { input, channel } => write!(f, "{input}.{}", channel.name()),
            Input::Invert { input } => write!(f, "invert({input})"),
            // Both ranges, always: a `from` left implicit is the one part of a
            // pattern that is easy to misread as the identity it usually is.
            Input::Remap { input, from, to } => write!(
                f,
                "remap({input}, [{}, {}], [{}, {}])",
                from[0], from[1], to[0], to[1]
            ),
            Input::Mix { a, b, factor, mode } => {
                write!(f, "{}({a}, {b}, {factor})", mode.name())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Operand {
        #[derive(Deserialize)]
        struct Holder {
            field: Operand,
        }

        toml::from_str::<Holder>(source)
            .unwrap_or_else(|error| panic!("{source} should parse: {error}"))
            .field
    }

    /// A pattern, through the string a scene writes it in rather than through
    /// the parser directly.
    fn pattern(expression: &str) -> Operand {
        parse(&format!("field = {expression:?}"))
    }

    #[test]
    fn a_number_is_still_a_number() {
        assert_eq!(parse("field = 0.5"), Operand::Scalar(0.5));
        assert_eq!(parse("field = 2"), Operand::Scalar(2.0));
        assert_eq!(
            parse("field = [0.1, 0.2, 0.3]"),
            Operand::Color([0.1, 0.2, 0.3])
        );
    }

    #[test]
    fn a_string_is_a_pattern() {
        let operand = pattern("remap(uv.r, [0.1, 0.9])");

        let Operand::Input(Input::Remap { input, from, to }) = &operand else {
            panic!("expected a remap, got {operand:?}");
        };
        assert_eq!(*from, [0.0, 1.0], "the default range is the unit one");
        assert_eq!(*to, [0.1, 0.9]);
        assert_eq!(
            **input,
            Operand::Input(Input::Coordinates {
                space: Space::Uv,
                channel: Some(Channel::R),
            })
        );
    }

    #[test]
    fn a_mix_takes_constants_on_either_side() {
        let operand = pattern("multiply(0.9, [0.25, 0.18, 0.1], world.y)");

        let Operand::Input(Input::Mix { a, b, mode, .. }) = &operand else {
            panic!("expected a mix, got {operand:?}");
        };
        assert_eq!(**a, Operand::Scalar(0.9));
        assert_eq!(**b, Operand::Color([0.25, 0.18, 0.1]));
        assert_eq!(*mode, Blend::Multiply);
    }

    #[test]
    fn a_mix_defaults_to_blending() {
        let operand = pattern("mix(0.0, 1.0, 0.25)");

        let Operand::Input(Input::Mix { mode, .. }) = &operand else {
            panic!("expected a mix, got {operand:?}");
        };
        assert_eq!(*mode, Blend::Mix);
    }

    /// A constant is its own range, so a tree of them is exact.
    #[test]
    fn constants_bound_themselves() {
        assert_eq!(
            Operand::Color([0.1, 0.2, 0.3]).range(),
            Some(([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]))
        );
        assert_eq!(
            pattern("invert(0.25)").range(),
            Some(([0.75; 3], [0.75; 3]))
        );
    }

    /// The one unbounded thing, and the thing that makes everything holding it
    /// unbounded too.
    #[test]
    fn a_raw_coordinate_has_no_bound() {
        assert_eq!(pattern("uv").range(), None);
        assert_eq!(pattern("invert(uv)").range(), None);
        assert_eq!(
            pattern("mix(0.0, 1.0, uv)").range(),
            Some(([0.0; 3], [1.0; 3])),
            "a factor is clamped to the unit range, so it bounds a mix even unbounded",
        );
    }

    /// And the way to give one a bound.
    #[test]
    fn a_remap_bounds_whatever_it_wraps() {
        let remap = pattern("remap(world, [-4.0, 4.0], [0.0, 12.0])");

        assert_eq!(remap.range(), Some(([0.0; 3], [12.0; 3])));
    }

    /// A descending `to` is a legal way to write an inverted remap, and its
    /// bounds are still the right way up.
    #[test]
    fn a_descending_remap_is_bounded_the_same_way() {
        let remap = pattern("remap(uv, [1.0, 0.0])");

        assert_eq!(remap.range(), Some(([0.0; 3], [1.0; 3])));
    }

    /// Every blend mode, at the two ends of its factor, against the arithmetic
    /// written out by hand.
    #[test]
    fn a_blend_at_a_factor_of_zero_is_the_first_side() {
        for mode in [Blend::Mix, Blend::Multiply, Blend::Add, Blend::Overlay] {
            assert_eq!(blend(mode, 0.3, 0.7, 0.0), 0.3, "{mode:?}");
        }
    }

    #[test]
    fn a_blend_at_a_factor_of_one_is_the_mode_itself() {
        assert_eq!(blend(Blend::Mix, 0.3, 0.7, 1.0), 0.7);
        assert!((blend(Blend::Multiply, 0.3, 0.7, 1.0) - 0.21).abs() < 1e-6);
        assert_eq!(blend(Blend::Add, 0.3, 0.7, 1.0), 1.0);
        // Below a half, overlay multiplies and doubles.
        assert!((blend(Blend::Overlay, 0.3, 0.7, 1.0) - 0.42).abs() < 1e-6);
        // Above it, it screens.
        assert!((blend(Blend::Overlay, 0.7, 0.7, 1.0) - 0.82).abs() < 1e-6);
    }

    /// Overlay is the one mode whose extreme is not at a corner: with a side
    /// past one it peaks where the branch changes. The corners here read 0 and
    /// 1, and the surface reaches 3 — a light bounded at a third of what it
    /// emits is sampled as if it were that dim.
    #[test]
    fn an_overlay_is_bounded_at_the_branch_it_turns_on() {
        let overlay = pattern("overlay(remap(uv, [0.0, 1.0]), 3.0, 1.0)");

        let (low, high) = overlay.range().expect("the remap bounds it");
        assert_eq!(low, [0.0; 3]);
        assert_eq!(high, [3.0; 3], "6a at a = 0.5, which is neither corner");
        assert_eq!(
            blend(Blend::Overlay, 0.5, 3.0, 1.0),
            3.0,
            "and the shader's arithmetic agrees it is reachable"
        );
    }

    /// The kink only counts when the first side actually crosses it.
    #[test]
    fn an_overlay_that_stays_on_one_side_is_bounded_by_its_corners() {
        let overlay = pattern("overlay(remap(uv, [0.6, 0.9]), 3.0, 1.0)");

        let (low, high) = overlay.range().expect("the remap bounds it");
        // Past the branch and with `b` over one, overlay falls as `a` rises,
        // so the ends arrive the other way round.
        assert_eq!(low, [blend(Blend::Overlay, 0.9, 3.0, 1.0); 3]);
        assert_eq!(high, [blend(Blend::Overlay, 0.6, 3.0, 1.0); 3]);
    }

    /// What a field says when its pattern is wrong: TOML names the line, the parser names
    /// the column within it.
    /// A channel of a bounded pattern is bounded by that channel, and one of an
    /// unbounded pattern is not bounded at all.
    #[test]
    fn a_channel_is_bounded_by_the_channel_it_reads() {
        let operand = pattern("remap(uv, [0.0, 1.0], [0.25, 0.75]).g");
        assert_eq!(operand.range(), Some(([0.25; 3], [0.75; 3])));

        assert_eq!(pattern("invert(uv).r").range(), None);
    }

    /// Alpha is outside the three channels a bound is written in, so a pattern
    /// reading it reports none rather than a made-up one.
    #[test]
    fn a_bound_does_not_reach_the_fourth_channel() {
        assert_eq!(pattern("[0.1, 0.2, 0.3].a").range(), None);
    }

    #[test]
    fn a_malformed_expression_is_placed_within_its_string() {
        #[derive(Deserialize, Debug)]
        struct Holder {
            #[allow(dead_code)]
            field: Operand,
        }

        let error = toml::from_str::<Holder>(r#"field = "mix(uv, 1)""#)
            .expect_err("a mix takes three arguments")
            .to_string();

        assert!(error.contains("expected `,`"), "{error}");
        assert!(error.contains("at column 10"), "{error}");
    }

    /// An untagged enum reports "data did not match any variant" and throws the
    /// real error away, which is the whole reason this one is written out: the
    /// parser's own error survives, and so does the length of a short colour.
    #[test]
    fn a_malformed_field_says_what_is_wrong_with_it() {
        let error = |source: &str| {
            #[derive(Deserialize, Debug)]
            struct Holder {
                #[allow(dead_code)]
                field: Operand,
            }

            toml::from_str::<Holder>(source)
                .expect_err("{source} should not parse")
                .to_string()
        };

        assert!(
            error(r#"field = "remap(uv.r)""#).contains("expected `,`"),
            "a missing argument is named"
        );
        assert!(
            error(r#"field = "rempa(0.5)""#).contains("rempa"),
            "and so is an unknown pattern"
        );
        assert!(
            error("field = [0.8, 0.8]").contains("length 2"),
            "and a color that is short is still short"
        );
        assert!(
            error(r#"field = { type = "remap", input = 0.5 }"#).contains("pattern expression"),
            "and a table says what a field does take"
        );
    }

    /// The factor is clamped, so a pattern that wanders outside the unit range
    /// cannot extrapolate past either side — which is what lets a mix be
    /// bounded by its two sides.
    #[test]
    fn a_blend_clamps_its_factor() {
        assert_eq!(blend(Blend::Mix, 0.0, 1.0, 4.0), 1.0);
        assert_eq!(blend(Blend::Mix, 0.0, 1.0, -4.0), 0.0);
    }
}
