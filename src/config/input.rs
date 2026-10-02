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
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;

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

/// How an image's 8-bit texels are turned into numbers.
///
/// A colour a person painted is stored gamma-encoded, and reading it as though
/// it were linear leaves every midtone too bright. Anything else — a
/// roughness, a normal, a mask — is stored as the number it is, and decoding
/// that as sRGB darkens it into something it never said. Which one a map is
/// cannot be read off the file, so it defaults by the field the image drives
/// ([`ColorSpace::default_for`]) and a scene can say otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

impl ColorSpace {
    /// The name a scene writes it under.
    pub fn name(self) -> &'static str {
        match self {
            ColorSpace::Srgb => "srgb",
            ColorSpace::Linear => "linear",
        }
    }
}

/// Which way up a tangent-space normal map's green channel points.
///
/// OpenGL (and Blender) store `+v` as green; DirectX stores `-v`. The two
/// render the same surface lit from opposite sides, which is the mistake a
/// normal map is most often wrong by and the hardest one to see in a still.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Convention {
    #[default]
    OpenGl,
    DirectX,
}

impl Convention {
    pub fn name(self) -> &'static str {
        match self {
            Convention::OpenGl => "opengl",
            Convention::DirectX => "directx",
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

/// How many nodes one field may come to once its named patterns are written
/// out. Far past anything written by hand — the densest field in the examples
/// is a few dozen — and here so that a handful of patterns each using the one
/// before twice is an error rather than a tree that doubles with every name.
pub const MAX_PATTERN_NODES: u32 = 4096;

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

    /// Every input directly under this one, which is what the walks below
    /// recurse through.
    fn children_mut(&mut self) -> Vec<&mut Operand> {
        let Operand::Input(input) = self else {
            return Vec::new();
        };
        match input {
            Input::Coordinates { .. } | Input::Image { .. } | Input::Pattern { .. } => Vec::new(),
            Input::Channel { input, .. }
            | Input::Invert { input }
            | Input::Remap { input, .. }
            | Input::Noise { input, .. }
            | Input::Voronoi { input, .. }
            | Input::Ramp { input, .. }
            | Input::NormalMap { input, .. }
            | Input::Bump { input, .. } => vec![input.as_mut()],
            Input::Mix { a, b, factor, .. } => vec![a.as_mut(), b.as_mut(), factor.as_mut()],
        }
    }

    fn children(&self) -> Vec<&Operand> {
        let Operand::Input(input) = self else {
            return Vec::new();
        };
        match input {
            Input::Coordinates { .. } | Input::Image { .. } | Input::Pattern { .. } => Vec::new(),
            Input::Channel { input, .. }
            | Input::Invert { input }
            | Input::Remap { input, .. }
            | Input::Noise { input, .. }
            | Input::Voronoi { input, .. }
            | Input::Ramp { input, .. }
            | Input::NormalMap { input, .. }
            | Input::Bump { input, .. } => vec![input.as_ref()],
            Input::Mix { a, b, factor, .. } => vec![a.as_ref(), b.as_ref(), factor.as_ref()],
        }
    }

    /// Rewrites every image path so it is relative to the process rather than
    /// to the scene, and fails on the first one that is not there — the same
    /// pass, for the same reason, that a mesh's path gets in `Config::validate`.
    pub fn resolve_images(&mut self, dir: &Path) -> Result<(), String> {
        if let Operand::Input(Input::Image { file, .. }) = self {
            *file = dir.join(&*file);
            if !file.is_file() {
                return Err(format!("image file does not exist: {}", file.display()));
            }
        }
        self.children_mut()
            .into_iter()
            .try_for_each(|child| child.resolve_images(dir))
    }

    /// This, with every image that did not name a colour space given `space`.
    /// Which is the default is the field's to say, so it is filled in by
    /// whoever knows the field.
    pub fn with_space(&self, space: ColorSpace) -> Operand {
        let mut operand = self.clone();
        operand.fill_space(space);
        operand
    }

    fn fill_space(&mut self, space: ColorSpace) {
        if let Operand::Input(Input::Image { space: slot, .. }) = self {
            slot.get_or_insert(space);
        }
        for child in self.children_mut() {
            child.fill_space(space);
        }
    }

    /// Whether a `normal_map` or a `bump` appears anywhere in this: the two
    /// inputs that produce a direction rather than a value.
    pub fn has_normal(&self) -> bool {
        matches!(
            self,
            Operand::Input(Input::NormalMap { .. } | Input::Bump { .. })
        ) || self.children().into_iter().any(Operand::has_normal)
    }

    /// Replaces every reference to a named pattern with a copy of the pattern
    /// it names, recursively, so what is left is a tree with no names in it —
    /// the only kind `scene::program` compiles.
    ///
    /// A pattern that reaches itself, however far round, is an error naming
    /// the loop, and so is a name nothing defines. Inlining copies, so a
    /// pattern used twice is two subtrees, and [`MAX_PATTERN_NODES`] caps
    /// how many nodes the copies may add up to.
    pub fn resolve_patterns(&mut self, patterns: &BTreeMap<String, Operand>) -> Result<(), String> {
        let mut budget = MAX_PATTERN_NODES;
        self.inline(patterns, &mut Vec::new(), &mut budget, 0)
    }

    fn inline(
        &mut self,
        patterns: &BTreeMap<String, Operand>,
        within: &mut Vec<String>,
        budget: &mut u32,
        nesting: u32,
    ) -> Result<(), String> {
        if nesting > MAX_NESTING {
            return Err(format!(
                "a pattern nests deeper than the {MAX_NESTING} levels this compiles \
                 once its named patterns are written out"
            ));
        }
        *budget = budget.checked_sub(1).ok_or_else(|| {
            String::from(
                "the named patterns write out to more nodes than a field may hold; \
                 a pattern that uses another more than once copies it each time",
            )
        })?;

        if let Operand::Input(Input::Pattern { name }) = self {
            if let Some(start) = within.iter().position(|seen| seen == name) {
                let cycle = within[start..]
                    .iter()
                    .chain([&*name])
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(" -> ");
                return Err(format!("pattern `{name}` refers to itself: {cycle}"));
            }
            let Some(definition) = patterns.get(name.as_str()) else {
                return Err(format!(
                    "unknown name `{name}`: not a coordinate space, and no entry \
                     in [patterns] is called that"
                ));
            };

            let name = name.clone();
            *self = definition.clone();
            within.push(name);
            // The copy is itself a tree of names, and is not counted twice:
            // this node became its root.
            *budget += 1;
            let inlined = self.inline(patterns, within, budget, nesting);
            within.pop();
            return inlined;
        }

        for child in self.children_mut() {
            child.inline(patterns, within, budget, nesting + 1)?;
        }
        Ok(())
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

    /// An image file, read at the hit's texture coordinates times `scale` plus
    /// `offset`, and repeated past the unit square.
    ///
    /// `space` is `None` until something decides it: a scene that does not
    /// say gets the default of the field the image drives, which only the
    /// field knows (`scene::program` asks [`ColorSpace::default_for`]).
    ///
    /// The path is relative to the scene file as written, and resolved against
    /// it by [`Operand::resolve_images`] before anything reads it.
    Image {
        file: PathBuf,
        space: Option<ColorSpace>,
        scale: [f32; 2],
        offset: [f32; 2],
    },

    /// A tangent-space normal map, decoded: `input`'s colour taken from
    /// `[0, 1]` to `[-1, 1]`, its green flipped for DirectX, and leaned back
    /// toward the unperturbed normal by `strength`.
    ///
    /// Only the `normal` field takes one, and only at its root. What it
    /// produces is a direction and not a value, and the one thing that knows
    /// what to do with a direction is the shading frame it is turned into.
    NormalMap {
        input: Box<Operand>,
        strength: f32,
        convention: Convention,
    },

    /// Gradient noise in `[0, 1]`: Perlin's improved noise summed over
    /// octaves (fBm), read at `input`'s first three channels times `scale`.
    ///
    /// The knobs are Blender's Noise Texture's and mean what they mean there —
    /// `detail` is how many octaves past the first, fractional ones blending
    /// the last in; `roughness` how much each octave keeps of the one before;
    /// `lacunarity` how much finer each is; `distortion` how far the lookup is
    /// pushed around by three more noises first — but the lattice hash is not
    /// Blender's, so the same numbers draw a different pattern of the same
    /// character.
    Noise {
        input: Box<Operand>,
        scale: f32,
        detail: f32,
        roughness: f32,
        lacunarity: f32,
        distortion: f32,
    },

    /// Worley's cellular pattern: feature points jittered by `randomness`
    /// inside a lattice of cells `scale` to the unit, and either the distance
    /// to the nearest one or a random colour for the cell it belongs to.
    Voronoi {
        input: Box<Operand>,
        scale: f32,
        randomness: f32,
        output: Cell,
    },

    /// `input`'s first channel mapped through two to four colour stops,
    /// linearly between them and held flat past either end. How a scalar
    /// pattern — a noise, most often — becomes a colour.
    Ramp {
        input: Box<Operand>,
        stops: Vec<(f32, [f32; 3])>,
    },

    /// A reference to an entry of `[patterns]`, by name. Only ever seen between
    /// parsing and [`Operand::resolve_patterns`], which writes the definition
    /// in its place; nothing downstream of `Config::validate` meets one.
    Pattern { name: String },

    /// The shading normal tilted to follow the slope of a height field, as
    /// Blender's Bump node does: `input`'s first channel is the height,
    /// `distance` scales it into world units, and `strength` blends the result
    /// with the unbumped normal.
    ///
    /// Like a normal map, a direction and not a value, and only the root of
    /// `normal` takes one.
    Bump {
        input: Box<Operand>,
        strength: f32,
        distance: f32,
    },
}

/// Which of a Voronoi cell's three answers a pattern reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Cell {
    /// The distance to the nearest feature point, in cells.
    #[default]
    Distance,
    /// A colour fixed per cell, each channel uniform in `[0, 1)`.
    Color,
    /// The distance to the nearest edge of the cell the point is in, in cells:
    /// Blender's Distance to Edge. Zero along every border, so a ramp over it
    /// draws the cells' outlines.
    Edge,
}

impl Cell {
    pub fn name(self) -> &'static str {
        match self {
            Cell::Distance => "distance",
            Cell::Color => "color",
            Cell::Edge => "edge",
        }
    }
}

/// The farthest a point can be from the nearest of its Voronoi feature points,
/// in cells: the diagonal of one. The feature point of the point's own cell is
/// somewhere inside that cell whatever the randomness, and the nearest one is at
/// least that near.
pub const MAX_CELL_DISTANCE: f32 = 1.732_050_8;

/// The farthest a point can be from the nearest edge of its Voronoi cell, in
/// cells. The edge between the nearest point and any other is no farther than
/// half their two distances added, and the points of the point's own lattice
/// cell and of its neighbour across the nearer face are both within
/// `sqrt(1.5² + 1 + 1)`, so one of them is not the nearest and the edge is
/// within `(√3 + √4.25) / 2`, under two.
pub const MAX_CELL_EDGE_DISTANCE: f32 = 2.0;

/// The most stops a ramp holds. Each is four words the shader walks, and four
/// is Blender's default ramp with two to spare.
pub const MAX_RAMP_STOPS: usize = 4;

/// A noise's knobs when a scene does not turn them: Blender's Noise Texture's
/// defaults.
pub struct NoiseDefaults {
    pub scale: f32,
    pub detail: f32,
    pub roughness: f32,
    pub lacunarity: f32,
    pub distortion: f32,
}

pub const NOISE_DEFAULTS: NoiseDefaults = NoiseDefaults {
    scale: 5.0,
    detail: 2.0,
    roughness: 0.5,
    lacunarity: 2.0,
    distortion: 0.0,
};

/// The most octaves past the first a noise sums, which is Blender's ceiling.
/// Each is a lattice lookup per hit, and past this they are finer than a pixel
/// at any scale worth rendering.
pub const MAX_NOISE_DETAIL: f32 = 15.0;

/// A Voronoi pattern's, likewise Blender's.
pub struct VoronoiDefaults {
    pub scale: f32,
    pub randomness: f32,
}

pub const VORONOI_DEFAULTS: VoronoiDefaults = VoronoiDefaults {
    scale: 5.0,
    randomness: 1.0,
};

impl ColorSpace {
    /// What an image in `field` is read as when the scene does not say: a
    /// colour for the two fields that are colours, and a number for every
    /// other one.
    pub fn default_for(field: &str) -> ColorSpace {
        match field {
            "base_color" | "emission_color" => ColorSpace::Srgb,
            _ => ColorSpace::Linear,
        }
    }
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

            // Eight bits a channel, normalized. The brightest texel would be
            // tighter, but the image is not read until the scene loads, and a
            // loose bound only costs an emissive image some noise.
            Input::Image { .. } => Some(([0.0; 3], [1.0; 3])),

            // A direction, which is not a thing a bound is asked of.
            Input::NormalMap { .. } | Input::Bump { .. } => None,

            // Clamped onto the unit range by the shader, which is what makes a
            // noise a thing emission can be driven by without a remap.
            Input::Noise { .. } => Some(([0.0; 3], [1.0; 3])),

            Input::Voronoi { output, .. } => match output {
                Cell::Distance => Some(([0.0; 3], [MAX_CELL_DISTANCE; 3])),
                Cell::Color => Some(([0.0; 3], [1.0; 3])),
                Cell::Edge => Some(([0.0; 3], [MAX_CELL_EDGE_DISTANCE; 3])),
            },

            // Piecewise linear between the stops and flat past them, so every
            // value it takes is between two stops' and the extremes are stops.
            Input::Ramp { stops, .. } => {
                let mut low = [f32::INFINITY; 3];
                let mut high = [f32::NEG_INFINITY; 3];
                for (_, color) in stops {
                    for channel in 0..3 {
                        low[channel] = low[channel].min(color[channel]);
                        high[channel] = high[channel].max(color[channel]);
                    }
                }
                Some((low, high))
            }

            // Written out before anything asks; a name still standing here has
            // no definition to be bounded by.
            Input::Pattern { .. } => None,

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
            // Only what differs from the defaults, so a plain image prints as
            // the one argument it was written with.
            Input::Image {
                file,
                space,
                scale,
                offset,
            } => {
                write!(f, "image(\"{}\"", file.display())?;
                if let Some(space) = space {
                    write!(f, ", color: \"{}\"", space.name())?;
                }
                if *scale != [1.0; 2] {
                    match scale[0] == scale[1] {
                        true => write!(f, ", scale: {}", scale[0])?,
                        false => write!(f, ", scale: [{}, {}]", scale[0], scale[1])?,
                    }
                }
                if *offset != [0.0; 2] {
                    write!(f, ", offset: [{}, {}]", offset[0], offset[1])?;
                }
                f.write_str(")")
            }
            Input::NormalMap {
                input,
                strength,
                convention,
            } => {
                write!(f, "normal_map({input}")?;
                if *strength != 1.0 {
                    write!(f, ", strength: {strength}")?;
                }
                if *convention != Convention::OpenGl {
                    write!(f, ", convention: \"{}\"", convention.name())?;
                }
                f.write_str(")")
            }
            Input::Noise {
                input,
                scale,
                detail,
                roughness,
                lacunarity,
                distortion,
            } => {
                write!(f, "noise({input}")?;
                let defaults = NOISE_DEFAULTS;
                for (name, value, default) in [
                    ("scale", scale, defaults.scale),
                    ("detail", detail, defaults.detail),
                    ("roughness", roughness, defaults.roughness),
                    ("lacunarity", lacunarity, defaults.lacunarity),
                    ("distortion", distortion, defaults.distortion),
                ] {
                    if *value != default {
                        write!(f, ", {name}: {value}")?;
                    }
                }
                f.write_str(")")
            }
            Input::Voronoi {
                input,
                scale,
                randomness,
                output,
            } => {
                write!(f, "voronoi({input}")?;
                if *scale != VORONOI_DEFAULTS.scale {
                    write!(f, ", scale: {scale}")?;
                }
                if *randomness != VORONOI_DEFAULTS.randomness {
                    write!(f, ", randomness: {randomness}")?;
                }
                if *output != Cell::Distance {
                    write!(f, ", output: \"{}\"", output.name())?;
                }
                f.write_str(")")
            }
            Input::Ramp { input, stops } => {
                write!(f, "ramp({input}")?;
                for (at, [r, g, b]) in stops {
                    write!(f, ", {at}, [{r}, {g}, {b}]")?;
                }
                f.write_str(")")
            }
            Input::Pattern { name } => f.write_str(name),
            Input::Bump {
                input,
                strength,
                distance,
            } => {
                write!(f, "bump({input}")?;
                if *strength != 1.0 {
                    write!(f, ", strength: {strength}")?;
                }
                if *distance != 1.0 {
                    write!(f, ", distance: {distance}")?;
                }
                f.write_str(")")
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

    /// A noise is clamped into the unit range, and a voronoi's distance is at
    /// most a cell's diagonal, so either can drive emission unwrapped.
    #[test]
    fn a_generated_pattern_is_bounded() {
        assert_eq!(
            pattern("noise(world, scale: 400)").range(),
            Some(([0.0; 3], [1.0; 3]))
        );
        assert_eq!(
            pattern("voronoi(world)").range(),
            Some(([0.0; 3], [MAX_CELL_DISTANCE; 3]))
        );
        assert_eq!(
            pattern(r#"voronoi(world, output: "color")"#).range(),
            Some(([0.0; 3], [1.0; 3]))
        );
        assert_eq!(
            pattern(r#"voronoi(world, output: "edge")"#).range(),
            Some(([0.0; 3], [MAX_CELL_EDGE_DISTANCE; 3]))
        );
    }

    /// A ramp only ever takes values between two of its stops, per channel,
    /// however wide its input.
    #[test]
    fn a_ramp_is_bounded_by_its_stops() {
        assert_eq!(
            pattern("ramp(world.x, 0, [0.2, 0.9, 0.5], 0.5, [0.8, 0.1, 0.5], 1, 0.4)").range(),
            Some(([0.2, 0.1, 0.4], [0.8, 0.9, 0.5]))
        );
    }

    fn patterns(entries: &[(&str, &str)]) -> BTreeMap<String, Operand> {
        entries
            .iter()
            .map(|(name, source)| (name.to_string(), pattern(source)))
            .collect()
    }

    /// A name is replaced by what it names, all the way down, so a field that
    /// uses one is the tree it would have been written out by hand.
    #[test]
    fn a_pattern_name_is_written_out_where_it_is_used() {
        let table = patterns(&[
            (
                "dirt",
                "remap(noise(object, scale: 4), [0.45, 0.55], [0, 1])",
            ),
            ("patchy", "mix(0.9, 0.6, dirt)"),
        ]);
        let mut field = pattern("invert(patchy)");
        field.resolve_patterns(&table).expect("should resolve");

        assert_eq!(
            field,
            pattern("invert(mix(0.9, 0.6, remap(noise(object, scale: 4), [0.45, 0.55], [0, 1])))")
        );
    }

    #[test]
    fn a_pattern_that_reaches_itself_is_named_with_its_loop() {
        let table = patterns(&[("a", "invert(b)"), ("b", "mix(0, 1, c)"), ("c", "a.r")]);
        let error = pattern("a")
            .resolve_patterns(&table)
            .expect_err("a loop never ends");
        assert!(error.contains("a -> b -> c -> a"), "{error}");

        let error = pattern("invert(me)")
            .resolve_patterns(&patterns(&[("me", "me")]))
            .expect_err("nor does a pattern that is itself");
        assert!(error.contains("me -> me"), "{error}");
    }

    #[test]
    fn a_name_nothing_defines_is_an_error() {
        let error = pattern("mix(0, 1, drit)")
            .resolve_patterns(&patterns(&[("dirt", "uv.r")]))
            .expect_err("a misspelling");
        assert!(error.contains("unknown name `drit`"), "{error}");
    }

    /// Each pattern using the one before twice doubles the tree every name,
    /// which is a tree the size of memory by the thirtieth.
    #[test]
    fn a_pattern_that_writes_out_too_large_is_an_error() {
        let mut entries = vec![(String::from("p0"), String::from("uv.r"))];
        for level in 1..30 {
            let previous = &entries[level - 1].0;
            entries.push((
                format!("p{level}"),
                format!("mix({previous}, {previous}, 0.5)"),
            ));
        }
        let table: Vec<(&str, &str)> = entries
            .iter()
            .map(|(name, source)| (name.as_str(), source.as_str()))
            .collect();

        let error = pattern("p29")
            .resolve_patterns(&patterns(&table))
            .expect_err("too large to write out");
        assert!(error.contains("more nodes"), "{error}");
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
