//! The expression form of a material input: `mix([0.75, 0.15, 0.1], …)`.
//!
//! An [`Operand`](super::Operand) that is not a constant is written as a call,
//! in a string, and parsed here into the [`Input`] tree the rest of the code
//! holds:
//!
//! ```text
//! base_color = "mix([0.75, 0.15, 0.1], [0.1, 0.2, 0.7], remap(object.z, [-6, 6], [0, 1]))"
//! ```
//!
//! This was a tree of `type`-tagged tables, one table per node, which is exact
//! and unreadable by the third level: the line above ran past two hundred
//! characters and its shape was entirely punctuation. A scene is written by
//! hand, so it is spelled the way it is read out loud.
//!
//! The grammar is small enough to state in full:
//!
//! ```text
//! expression  := term {('+' | '-') term}
//! term        := operand {('*' | '/') operand}
//! operand     := ['-'] primary {'.' channel}
//! primary     := number | color | coordinates | call | pattern | '(' expression ')'
//! number      := -1.5, 2, 1e-3 …
//! color       := '[' number ',' number ',' number ']'
//! coordinates := 'uv' | 'object' | 'world'
//! channel     := 'r' | 'g' | 'b' | 'a' | 'x' | 'y' | 'z' | 'w'
//! call        := 'invert' '(' expression ')'
//!              | 'remap' '(' expression ',' [range ','] range ')'
//!              | ('mix' | 'multiply' | 'add' | 'overlay')
//!                '(' expression ',' expression ',' expression ')'
//!              | 'blackbody' '(' number ')'
//!              | 'image' '(' string {',' keyword} ')'
//!              | 'normal_map' '(' expression {',' keyword} ')'
//!              | ('noise' | 'voronoi' | 'bump') '(' expression {',' keyword} ')'
//!              | 'ramp' '(' expression stop stop [stop [stop]] ')'
//! stop        := ',' constant ',' constant
//! pattern     := a name from the scene's [patterns] table
//! range       := '[' number ',' number ']'
//! string      := '"' any character but '"' … '"'
//! keyword     := name ':' (number | range | string)
//! ```
//!
//! Every argument a call cannot do without is positional, and every one it
//! can is a `name: value` keyword after them, in any order and each at most
//! once. The keywords a call takes, and what it does without each:
//!
//! ```text
//! image       color: "srgb" | "linear"    the field's own default
//!             scale: number | range       1
//!             offset: range               [0, 0]
//! normal_map  strength: number            1
//!             convention: "opengl" | "directx"   "opengl"
//! noise       scale: number               5
//!             detail: number in [0, 15]   2
//!             roughness: number in [0, 1] 0.5
//!             lacunarity: number          2
//!             distortion: number          0
//! voronoi     scale: number               5
//!             randomness: number in [0, 1]   1
//!             output: "distance" | "color" | "edge"   "distance"
//! bump        strength: number in [0, 1]  1
//!             distance: number            1
//! ```
//!
//! A ramp's stops are a position and then a colour, each an expression that
//! works out to a constant — `ramp(dirt, 0.3, blackbody(2000) * 0.2, 0.7,
//! [0.4, 0.5, 0.2])` — with the positions in ascending order.
//!
//! Any other name is a reference to an entry of the scene's `[patterns]`
//! table, which [`Config::validate`](super::Config::validate) writes in place of
//! it. A name followed by `(` is always a call, so a misspelled one is still
//! reported as the unknown call it is.
//!
//! A channel is still the `.r` suffix rather than a keyword, so the roughness
//! in the green of a packed map is `image("orm.png").g`.
//!
//! A channel is taken off whatever precedes it rather than off a coordinate
//! only, so `remap(uv, [0, 4]).r` is a way of saying which channel a scalar
//! field should read. A bare coordinate carries its own, which is what keeps
//! `uv.r` one node and one word of program.
//!
//! A blend mode is the name of the call rather than an argument to it, so
//! `overlay(a, b, f)` is the `mode = "overlay"` mix — four names for one node,
//! which is how the modes read at the call site and how [`Display`](fmt::Display)
//! writes them back out.
//!
//! The four arithmetic operators are those two modes at a factor of one, and
//! are written back out that way: `uv * 2` prints as `multiply(uv, 2, 1)`.
//! There is no subtract or divide node, so `a - b` is `a + b * -1` and `a / b`
//! is `a * (1 / b)` — which is why `b` may be anything in the first and only a
//! constant in the second. `-a` is `a * -1` the same way. Between two constants
//! each is folded on the spot instead, so `blackbody(6500) * 300` is the
//! colour it works out to and never a pattern the shader runs. That is also
//! why `blackbody` takes a number and not a pattern: it is a way of writing a
//! colour, and is printed as one.

use super::input::Blend;
use super::input::Cell;
use super::input::Channel;
use super::input::ColorSpace;
use super::input::Convention;
use super::input::Input;
use super::input::MAX_NESTING;
use super::input::MAX_NOISE_DETAIL;
use super::input::MAX_RAMP_STOPS;
use super::input::NOISE_DEFAULTS;
use super::input::Operand;
use super::input::Space;
use super::input::VORONOI_DEFAULTS;
use super::input::blend;
use std::path::PathBuf;

/// Every name the grammar already gives a meaning to, which a pattern in
/// `[patterns]` cannot be called: a reference to it would read as the call or
/// the space instead.
const RESERVED: [&str; 16] = [
    "uv",
    "object",
    "world",
    "invert",
    "remap",
    "mix",
    "multiply",
    "add",
    "overlay",
    "blackbody",
    "image",
    "normal_map",
    "noise",
    "voronoi",
    "ramp",
    "bump",
];

/// Whether `name` means something in the grammar already.
pub fn is_reserved(name: &str) -> bool {
    RESERVED.contains(&name)
}

/// Parses a whole expression, which must be all of `source`.
pub fn parse(source: &str) -> Result<Operand, String> {
    let mut parser = Parser {
        source,
        at: 0,
        depth: 0,
    };
    let operand = parser.expression()?;

    parser.space();
    match parser.rest().is_empty() {
        true => Ok(operand),
        false => Err(parser.error("trailing input after the expression")),
    }
}

struct Parser<'a> {
    source: &'a str,
    at: usize,
    /// How many operands are open above this one. Every level of it is a Rust
    /// stack frame, so it is counted and capped at [`MAX_NESTING`] rather than
    /// left to a scene file to decide.
    depth: u32,
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a str {
        &self.source[self.at..]
    }

    fn space(&mut self) {
        let trimmed = self.rest().trim_start();
        self.at = self.source.len() - trimmed.len();
    }

    /// The next character, with whitespace already behind it.
    fn peek(&mut self) -> Option<char> {
        self.space();
        self.rest().chars().next()
    }

    /// Takes `expected` if it is next, and says what it found if it is not.
    fn expect(&mut self, expected: char) -> Result<(), String> {
        match self.peek() {
            Some(found) if found == expected => {
                self.at += found.len_utf8();
                Ok(())
            }
            Some(found) => Err(self.error(&format!("expected `{expected}`, found `{found}`"))),
            None => Err(self.error(&format!("expected `{expected}`, found the end"))),
        }
    }

    /// The column is one-based and counted in characters, so it lines up with
    /// what the scene file shows between its quotes.
    fn error(&self, message: &str) -> String {
        let column = self.source[..self.at].chars().count() + 1;
        format!("{message}, at column {column} of `{}`", self.source)
    }

    /// Terms joined by `+` and `-`, which bind looser than `*` and `/`.
    fn expression(&mut self) -> Result<Operand, String> {
        self.chain(['+', '-'], Self::term)
    }

    /// Operands joined by `*` and `/`.
    fn term(&mut self) -> Result<Operand, String> {
        self.chain(['*', '/'], Self::operand)
    }

    /// `next {operator next}`, folded to the left. Each link is a level of the
    /// tree that no call of `operand` counted, so it is counted here, against
    /// the same cap, instead: a long enough chain is as deep a tree as the
    /// nesting it stands in for.
    fn chain(
        &mut self,
        operators: [char; 2],
        next: fn(&mut Self) -> Result<Operand, String>,
    ) -> Result<Operand, String> {
        let depth = self.depth;
        let mut left = next(self)?;
        while let Some(operator) = self.peek().filter(|c| operators.contains(c)) {
            let at = self.at;
            self.at += 1;
            self.space();
            let start = self.at;
            let right = next(self)?;
            self.depth += 1;
            if self.depth > MAX_NESTING {
                return Err(self.error(&format!(
                    "a pattern nests deeper than the {MAX_NESTING} levels this parses"
                )));
            }

            left = match operator {
                '+' => combine(left, right, Blend::Add),
                '-' => combine(left, negate(right), Blend::Add),
                '*' => combine(left, right, Blend::Multiply),
                _ => {
                    let reciprocal = reciprocal(&right).map_err(|message| {
                        self.at = start;
                        self.error(message)
                    })?;
                    combine(left, reciprocal, Blend::Multiply)
                }
            };

            // Folding two finite constants can still overflow, and a field
            // holding infinity is what `number` refuses to parse for a reason.
            if left
                .constant()
                .is_some_and(|c| c.iter().any(|v| !v.is_finite()))
            {
                self.at = at;
                return Err(self.error(&format!("`{operator}` overflows to infinity")));
            }
        }
        self.depth = depth;
        Ok(left)
    }

    fn operand(&mut self) -> Result<Operand, String> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            return Err(self.error(&format!(
                "a pattern nests deeper than the {MAX_NESTING} levels this parses"
            )));
        }

        // A sign in front of a digit is the number's own. In front of anything
        // else it negates what follows, channel and all: `-uv.r` is `-(uv.r)`.
        let negated = self.peek() == Some('-')
            && !self.rest()[1..].starts_with(|c: char| c.is_ascii_digit() || c == '.');
        if negated {
            self.at += 1;
            let operand = negate(self.operand()?);
            self.depth -= 1;
            return Ok(operand);
        }

        let mut operand = self.primary()?;
        while self.rest().starts_with('.') {
            operand = self.channel(operand)?;
        }

        self.depth -= 1;
        Ok(operand)
    }

    /// An operand with nothing taken off it yet.
    fn primary(&mut self) -> Result<Operand, String> {
        match self.peek() {
            Some('[') => self.color(),
            Some('(') => {
                self.at += 1;
                let inner = self.expression()?;
                self.expect(')')?;
                Ok(inner)
            }
            Some(c) if c == '-' || c == '+' || c == '.' || c.is_ascii_digit() => {
                self.number().map(Operand::Scalar)
            }
            Some(c) if c.is_alphabetic() || c == '_' => self.named(),
            Some(found) => Err(self.error(&format!(
                "expected a number, a colour or a pattern, found `{found}`"
            ))),
            None => Err(self.error("expected a number, a colour or a pattern, found the end")),
        }
    }

    fn number(&mut self) -> Result<f32, String> {
        self.space();
        let start = self.at;
        let mut end = 0;
        let mut previous = '\0';
        for (offset, c) in self.rest().char_indices() {
            let signed = (c == '-' || c == '+') && (previous == 'e' || previous == 'E');
            let part = c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E';
            // A leading sign is part of the number; a later one is only part of
            // an exponent, so `1-2` ends at the dash rather than failing to
            // parse as one number.
            if !(part || signed || (offset == 0 && (c == '-' || c == '+'))) {
                break;
            }
            previous = c;
            end = offset + c.len_utf8();
        }

        let text = &self.rest()[..end];
        match text.parse::<f32>() {
            Ok(value) if value.is_finite() => {
                self.at = start + end;
                Ok(value)
            }
            // `inf` never reaches here — it is not made of these characters —
            // but `1e40` is, and a field holding infinity is the one value the
            // shader turns into a NaN it accumulates forever.
            Ok(_) => Err(self.error(&format!("`{text}` is not a finite number"))),
            Err(_) => Err(self.error("expected a number")),
        }
    }

    /// `[` numbers `]`, of exactly the length asked for. Both a colour and a
    /// remap's ranges are written this way, so the length is the caller's.
    fn numbers<const N: usize>(&mut self, what: &str) -> Result<[f32; N], String> {
        self.expect('[')?;

        let mut values = [0.0; N];
        for (index, slot) in values.iter_mut().enumerate() {
            // Not `expect(',')`: a list that is short ends at its `]`, and
            // "takes three numbers" is what is wrong with it rather than the
            // comma that would have carried on.
            if index > 0 {
                match self.peek() {
                    Some(',') => self.at += 1,
                    _ => return Err(self.error(&format!("{what} takes {N} numbers"))),
                }
            }
            *slot = self.number()?;
        }

        match self.peek() {
            Some(']') => {
                self.at += 1;
                Ok(values)
            }
            _ => Err(self.error(&format!("{what} takes {N} numbers"))),
        }
    }

    fn color(&mut self) -> Result<Operand, String> {
        self.numbers::<3>("a colour").map(Operand::Color)
    }

    fn identifier(&mut self) -> &'a str {
        let start = self.at;
        let end = self
            .rest()
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(self.rest().len());
        self.at += end;
        &self.source[start..self.at]
    }

    /// Everything that starts with a name: the three coordinate spaces, and the
    /// calls.
    fn named(&mut self) -> Result<Operand, String> {
        let start = self.at;
        let name = self.identifier();

        let space = match name {
            "uv" => Some(Space::Uv),
            "object" => Some(Space::Object),
            "world" => Some(Space::World),
            _ => None,
        };
        if let Some(space) = space {
            // The channel, if one follows, is taken by `operand` above: a
            // coordinate is only the one operand that folds it back in.
            return Ok(Operand::Input(Input::Coordinates {
                space,
                channel: None,
            }));
        }

        let blend = match name {
            "mix" => Some(Blend::Mix),
            "multiply" => Some(Blend::Multiply),
            "add" => Some(Blend::Add),
            "overlay" => Some(Blend::Overlay),
            _ => None,
        };
        if let Some(mode) = blend {
            return self.mix(mode);
        }

        match name {
            "invert" => self.invert(),
            "remap" => self.remap(),
            "blackbody" => self.blackbody(),
            "image" => self.image(),
            "normal_map" => self.normal_map(),
            "noise" => self.noise(),
            "voronoi" => self.voronoi(),
            "ramp" => self.ramp(),
            "bump" => self.bump(),
            // A misspelling is reported where the name started rather than
            // where it ended, which is where the eye is.
            _ if self.peek() == Some('(') => {
                self.at = start;
                Err(self.error(&format!(
                    "unknown pattern `{name}`; expected invert, remap, mix, \
                     multiply, add, overlay, blackbody, image, normal_map, noise, \
                     voronoi, ramp or bump"
                )))
            }
            // Not a call, so a name: one of the scene's own patterns, or a
            // misspelling of a space that validation will say is neither.
            _ => Ok(Operand::Input(Input::Pattern {
                name: name.to_string(),
            })),
        }
    }

    /// `"` anything `"`. No escapes: what goes between the quotes is a file
    /// name, and a scene that writes the expression in a TOML literal string —
    /// `'image("wood.jpg")'` — never has to escape the quotes around it either.
    fn string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let Some(end) = self.rest().find('"') else {
            return Err(self.error("a string is never closed"));
        };
        let value = self.rest()[..end].to_string();
        self.at += end + 1;
        Ok(value)
    }

    /// A number, or two of them: a scale that is the same both ways is written
    /// once.
    fn pair(&mut self, what: &str) -> Result<[f32; 2], String> {
        match self.peek() {
            Some('[') => self.numbers::<2>(what),
            _ => self.number().map(|value| [value; 2]),
        }
    }

    /// The `name: value` arguments after a call's positional ones, handed to
    /// `take` one name at a time. `take` answers whether it knew the name; one
    /// it did not is reported against `known`, and so is one given twice.
    fn keywords(
        &mut self,
        call: &str,
        known: &str,
        mut take: impl FnMut(&mut Self, &str) -> Result<bool, String>,
    ) -> Result<(), String> {
        let mut seen: Vec<String> = Vec::new();
        while self.peek() == Some(',') {
            self.at += 1;
            self.space();
            let start = self.at;
            // A name starts with a letter, so a positional argument where a
            // keyword belongs is not read as a keyword called `0`.
            let starts = self
                .rest()
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_');
            let name = match starts {
                true => self.identifier(),
                false => "",
            };
            if name.is_empty() {
                return Err(self.error(&format!("expected one of {call}'s keywords: {known}")));
            }
            if seen.iter().any(|already| already == name) {
                self.at = start;
                return Err(self.error(&format!("`{name}` is given twice")));
            }
            self.expect(':')?;
            if !take(self, name)? {
                self.at = start;
                return Err(
                    self.error(&format!("{call} has no keyword `{name}`; expected {known}"))
                );
            }
            seen.push(name.to_string());
        }
        self.expect(')')
    }

    /// `image("file", …)`. See the module's table for the keywords.
    fn image(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let start = self.at;
        let file = self.string()?;
        if file.is_empty() {
            self.at = start;
            return Err(self.error("an image needs a file name"));
        }

        let mut space = None;
        let mut scale = [1.0; 2];
        let mut offset = [0.0; 2];
        self.keywords("image", "color, scale or offset", |parser, name| {
            match name {
                "color" => {
                    let start = parser.at;
                    space = Some(match parser.string()?.as_str() {
                        "srgb" => ColorSpace::Srgb,
                        "linear" => ColorSpace::Linear,
                        other => {
                            parser.at = start;
                            return Err(parser.error(&format!(
                                "unknown color space `{other}`; expected \"srgb\" or \"linear\""
                            )));
                        }
                    });
                }
                "scale" => scale = parser.pair("an image's scale")?,
                "offset" => offset = parser.numbers::<2>("an image's offset")?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;

        Ok(Operand::Input(Input::Image {
            file: PathBuf::from(file),
            space,
            scale,
            offset,
        }))
    }

    /// `normal_map(input, …)`. See the module's table for the keywords.
    fn normal_map(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let input = self.expression()?;

        let mut strength = 1.0;
        let mut convention = Convention::OpenGl;
        self.keywords("normal_map", "strength or convention", |parser, name| {
            match name {
                "strength" => {
                    let start = parser.at;
                    strength = parser.number()?;
                    if strength < 0.0 {
                        parser.at = start;
                        return Err(parser.error("a normal map's strength cannot be negative"));
                    }
                }
                "convention" => {
                    let start = parser.at;
                    convention = match parser.string()?.as_str() {
                        "opengl" => Convention::OpenGl,
                        "directx" => Convention::DirectX,
                        other => {
                            parser.at = start;
                            return Err(parser.error(&format!(
                                "unknown convention `{other}`; expected \"opengl\" or \"directx\""
                            )));
                        }
                    };
                }
                _ => return Ok(false),
            }
            Ok(true)
        })?;

        Ok(Operand::Input(Input::NormalMap {
            input: Box::new(input),
            strength,
            convention,
        }))
    }

    /// A keyword's number, held to `range` inclusive, with the message placed
    /// at the number.
    fn bounded(&mut self, what: &str, low: f32, high: f32) -> Result<f32, String> {
        self.space();
        let start = self.at;
        let value = self.number()?;
        if !(low..=high).contains(&value) {
            self.at = start;
            let range = match high == f32::INFINITY {
                true => format!("at least {low}"),
                false => format!("between {low} and {high}"),
            };
            return Err(self.error(&format!("{what} must be {range}, not {value}")));
        }
        Ok(value)
    }

    /// `noise(input, …)`. See the module's table for the keywords.
    fn noise(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let input = self.expression()?;

        let defaults = NOISE_DEFAULTS;
        let (mut scale, mut detail, mut roughness) =
            (defaults.scale, defaults.detail, defaults.roughness);
        let (mut lacunarity, mut distortion) = (defaults.lacunarity, defaults.distortion);
        self.keywords(
            "noise",
            "scale, detail, roughness, lacunarity or distortion",
            |parser, name| {
                match name {
                    "scale" => scale = parser.number()?,
                    "detail" => {
                        detail = parser.bounded("a noise's detail", 0.0, MAX_NOISE_DETAIL)?
                    }
                    "roughness" => roughness = parser.bounded("a noise's roughness", 0.0, 1.0)?,
                    "lacunarity" => {
                        lacunarity = parser.bounded("a noise's lacunarity", 0.0, f32::INFINITY)?
                    }
                    "distortion" => distortion = parser.number()?,
                    _ => return Ok(false),
                }
                Ok(true)
            },
        )?;

        Ok(Operand::Input(Input::Noise {
            input: Box::new(input),
            scale,
            detail,
            roughness,
            lacunarity,
            distortion,
        }))
    }

    /// `voronoi(input, …)`. See the module's table for the keywords.
    fn voronoi(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let input = self.expression()?;

        let mut scale = VORONOI_DEFAULTS.scale;
        let mut randomness = VORONOI_DEFAULTS.randomness;
        let mut output = Cell::Distance;
        self.keywords("voronoi", "scale, randomness or output", |parser, name| {
            match name {
                "scale" => scale = parser.number()?,
                "randomness" => randomness = parser.bounded("a voronoi's randomness", 0.0, 1.0)?,
                "output" => {
                    let start = parser.at;
                    output = match parser.string()?.as_str() {
                        "distance" => Cell::Distance,
                        "color" => Cell::Color,
                        "edge" => Cell::Edge,
                        other => {
                            parser.at = start;
                            return Err(parser.error(&format!(
                                "unknown output `{other}`; expected \"distance\", \"color\" or \"edge\""
                            )));
                        }
                    };
                }
                _ => return Ok(false),
            }
            Ok(true)
        })?;

        Ok(Operand::Input(Input::Voronoi {
            input: Box::new(input),
            scale,
            randomness,
            output,
        }))
    }

    /// An expression that has to work out to a constant, placed where it
    /// started when it does not.
    fn constant(&mut self, what: &str) -> Result<Operand, String> {
        self.space();
        let start = self.at;
        let operand = self.expression()?;
        if operand.is_input() {
            self.at = start;
            return Err(self.error(&format!("{what} must be a constant, not a pattern")));
        }
        Ok(operand)
    }

    /// `ramp(input, position, colour, …)`: two to four stops, in order.
    fn ramp(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let input = self.expression()?;

        let mut stops: Vec<(f32, [f32; 3])> = Vec::new();
        while self.peek() == Some(',') {
            self.at += 1;
            self.space();
            let start = self.at;
            if stops.len() == MAX_RAMP_STOPS {
                return Err(self.error(&format!("a ramp takes at most {MAX_RAMP_STOPS} stops")));
            }

            let Operand::Scalar(at) = self.constant("a ramp stop's position")? else {
                self.at = start;
                return Err(self.error("a ramp stop's position is one number, not a colour"));
            };
            if stops.last().is_some_and(|(previous, _)| at < *previous) {
                self.at = start;
                return Err(self.error("a ramp's stops must be in ascending order"));
            }

            self.expect(',')?;
            let color = self.constant("a ramp stop's colour")?;
            let color = color.constant().expect("a constant has a value");
            stops.push((at, color));
        }
        if stops.len() < 2 {
            return Err(self.error("a ramp takes at least two stops, each a position and a colour"));
        }
        self.expect(')')?;

        Ok(Operand::Input(Input::Ramp {
            input: Box::new(input),
            stops,
        }))
    }

    /// `bump(height, …)`. See the module's table for the keywords.
    fn bump(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let input = self.expression()?;

        let mut strength = 1.0;
        let mut distance = 1.0;
        self.keywords("bump", "strength or distance", |parser, name| {
            match name {
                "strength" => strength = parser.bounded("a bump's strength", 0.0, 1.0)?,
                "distance" => distance = parser.number()?,
                _ => return Ok(false),
            }
            Ok(true)
        })?;

        Ok(Operand::Input(Input::Bump {
            input: Box::new(input),
            strength,
            distance,
        }))
    }

    /// `.` channel, taken off the operand just parsed. Only reached with a `.`
    /// next, and with no [`space`](Self::space) before it: `uv .r` is two
    /// things, not one.
    fn channel(&mut self, operand: Operand) -> Result<Operand, String> {
        self.at += 1;
        let start = self.at;
        let name = self.identifier();
        let channel = match name {
            "r" | "x" => Channel::R,
            "g" | "y" => Channel::G,
            "b" | "z" => Channel::B,
            "a" | "w" => Channel::A,
            _ => {
                self.at = start;
                return Err(self.error(&format!(
                    "unknown channel `{name}`; expected r, g, b, a, x, y, z or w"
                )));
            }
        };

        // A bare coordinate carries its own channel, so `uv.r` stays the one
        // node it always was — and the node that wraps anything else prints
        // and parses the same way round.
        Ok(match operand {
            Operand::Input(Input::Coordinates {
                space,
                channel: None,
            }) => Operand::Input(Input::Coordinates {
                space,
                channel: Some(channel),
            }),
            input => Operand::Input(Input::Channel {
                input: Box::new(input),
                channel,
            }),
        })
    }

    fn invert(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let input = self.expression()?;
        self.expect(')')?;

        Ok(Operand::Input(Input::Invert {
            input: Box::new(input),
        }))
    }

    /// `remap(input, to)` or `remap(input, from, to)`: the two-argument form
    /// leaves `from` the unit range, exactly as the table form's default does.
    fn remap(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let input = self.expression()?;
        self.expect(',')?;
        let first = self.numbers::<2>("a remap's range")?;

        let (from, to) = match self.peek() {
            Some(',') => {
                self.at += 1;
                (first, self.numbers::<2>("a remap's range")?)
            }
            _ => ([0.0, 1.0], first),
        };
        self.expect(')')?;

        Ok(Operand::Input(Input::Remap {
            input: Box::new(input),
            from,
            to,
        }))
    }

    /// `blackbody(kelvin)`: the colour a black body glows at that temperature,
    /// as a constant. See [`blackbody`].
    fn blackbody(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        self.space();
        let start = self.at;
        let kelvin = self.number()?;
        let color = blackbody(kelvin).ok_or_else(|| {
            self.at = start;
            self.error(&format!(
                "a blackbody's temperature must be at least {MIN_KELVIN} kelvin, not {kelvin}"
            ))
        })?;
        self.expect(')')?;
        Ok(Operand::Color(color))
    }

    fn mix(&mut self, mode: Blend) -> Result<Operand, String> {
        self.expect('(')?;
        let a = self.expression()?;
        self.expect(',')?;
        let b = self.expression()?;
        self.expect(',')?;
        let factor = self.expression()?;
        self.expect(')')?;

        Ok(Operand::Input(Input::Mix {
            a: Box::new(a),
            b: Box::new(b),
            factor: Box::new(factor),
            mode,
        }))
    }
}

/// `a` and `b` under `mode` at a factor of one, worked out on the spot when
/// both are constants. A scalar and a colour give a colour, as the shader's
/// broadcast would.
fn combine(a: Operand, b: Operand, mode: Blend) -> Operand {
    let apply = |x: f32, y: f32| blend(mode, x, y, 1.0);
    match (&a, &b) {
        (Operand::Scalar(x), Operand::Scalar(y)) => Operand::Scalar(apply(*x, *y)),
        _ => match (a.constant(), b.constant()) {
            (Some(x), Some(y)) => Operand::Color([0, 1, 2].map(|c| apply(x[c], y[c]))),
            _ => Operand::Input(Input::Mix {
                a: Box::new(a),
                b: Box::new(b),
                factor: Box::new(Operand::Scalar(1.0)),
                mode,
            }),
        },
    }
}

/// `a * -1`, which is how `-a` and the right side of `a - b` are spelled on
/// the GPU.
fn negate(operand: Operand) -> Operand {
    combine(operand, Operand::Scalar(-1.0), Blend::Multiply)
}

/// `1 / divisor`, which only a constant has: there is no divide node, and a
/// pattern that passes through zero somewhere would have no answer there.
fn reciprocal(divisor: &Operand) -> Result<Operand, &'static str> {
    let Some(value) = divisor.constant() else {
        return Err("can only divide by a constant, not a pattern");
    };
    if value.contains(&0.0) {
        return Err("division by zero");
    }
    Ok(match divisor {
        Operand::Scalar(x) => Operand::Scalar(1.0 / x),
        _ => Operand::Color(value.map(|x| 1.0 / x)),
    })
}

/// Below this Planck's law puts next to nothing in the visible range, and the
/// normalization below would be dividing by a luminance f32 cannot tell from
/// zero. Nothing glows visibly this cold anyway: a stove ring is about 900.
const MIN_KELVIN: f32 = 500.0;

/// The colour of a black body at `kelvin`, in linear Rec. 709, scaled to a
/// luminance of one — Blender's convention, so that the temperature says what
/// colour a light is and a strength says how bright, and `blackbody(t) * 300`
/// is as bright at any `t`.
///
/// Planck's law is integrated against the CIE 1931 2° observer, in the
/// piecewise-Gaussian fit of Wyman, Sloan and Shirley (2013), every nanometre
/// from 380 to 780. A temperature far from 6500 lands outside the Rec. 709
/// gamut in blue (cold) or red (hot); that channel is clamped to zero before
/// normalizing, since a negative emission is not light.
fn blackbody(kelvin: f32) -> Option<[f32; 3]> {
    if kelvin.is_nan() || kelvin < MIN_KELVIN {
        return None;
    }

    fn lobe(x: f64, mu: f64, low: f64, high: f64) -> f64 {
        let t = (x - mu) * if x < mu { low } else { high };
        (-0.5 * t * t).exp()
    }

    // Second radiation constant, in nanometre-kelvin. The first constant and
    // the powers of ten are common to every wavelength and cancel in the
    // normalization.
    const C2: f64 = 1.438_776_9e7;
    let t = f64::from(kelvin);

    let mut xyz = [0.0f64; 3];
    for nm in 380..=780 {
        let l = f64::from(nm);
        let radiance = 1.0 / (l.powi(5) * ((C2 / (l * t)).exp_m1()));
        let x = 1.056 * lobe(l, 599.8, 0.0264, 0.0323) + 0.362 * lobe(l, 442.0, 0.0624, 0.0374)
            - 0.065 * lobe(l, 501.1, 0.0490, 0.0382);
        let y = 0.821 * lobe(l, 568.8, 0.0213, 0.0247) + 0.286 * lobe(l, 530.9, 0.0613, 0.0322);
        let z = 1.217 * lobe(l, 437.0, 0.0845, 0.0278) + 0.681 * lobe(l, 459.0, 0.0385, 0.0725);
        xyz[0] += radiance * x;
        xyz[1] += radiance * y;
        xyz[2] += radiance * z;
    }

    let [x, y, z] = xyz;
    let rgb = [
        3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z,
        -0.969_266_0 * x + 1.876_010_8 * y + 0.041_556_0 * z,
        0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z,
    ]
    .map(|c| c.max(0.0));

    let luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    let color = rgb.map(|c| (c / luminance) as f32);
    color.iter().all(|c| c.is_finite()).then_some(color)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(source: &str) -> Operand {
        parse(source).unwrap_or_else(|error| panic!("`{source}` should parse: {error}"))
    }

    fn err(source: &str) -> String {
        parse(source).expect_err("`{source}` should not parse")
    }

    fn boxed(operand: Operand) -> Box<Operand> {
        Box::new(operand)
    }

    fn coordinates(space: Space, channel: Option<Channel>) -> Operand {
        Operand::Input(Input::Coordinates { space, channel })
    }

    #[test]
    fn a_constant_is_still_a_constant() {
        assert_eq!(ok("0.5"), Operand::Scalar(0.5));
        assert_eq!(ok("2"), Operand::Scalar(2.0));
        assert_eq!(ok("-1.5"), Operand::Scalar(-1.5));
        assert_eq!(ok("1e-3"), Operand::Scalar(0.001));
        assert_eq!(ok(" [0.1, 0.2, 0.3] "), Operand::Color([0.1, 0.2, 0.3]));
    }

    #[test]
    fn a_space_is_a_coordinate() {
        for (source, space) in [
            ("uv", Space::Uv),
            ("object", Space::Object),
            ("world", Space::World),
        ] {
            assert_eq!(
                ok(source),
                Operand::Input(Input::Coordinates {
                    space,
                    channel: None
                })
            );
        }
    }

    /// The two spellings of a channel, and that they mean the same component.
    #[test]
    fn a_channel_reads_either_way_round() {
        for (source, channel) in [
            ("uv.r", Channel::R),
            ("uv.x", Channel::R),
            ("object.g", Channel::G),
            ("object.y", Channel::G),
            ("world.b", Channel::B),
            ("world.z", Channel::B),
            ("uv.a", Channel::A),
            ("uv.w", Channel::A),
        ] {
            let Operand::Input(Input::Coordinates { channel: found, .. }) = ok(source) else {
                panic!("`{source}` should be a coordinate");
            };
            assert_eq!(found, Some(channel), "{source}");
        }
    }

    #[test]
    fn a_call_is_the_node_it_stands_for() {
        let u = || coordinates(Space::Uv, Some(Channel::R));

        assert_eq!(
            ok("invert(uv.r)"),
            Operand::Input(Input::Invert { input: boxed(u()) })
        );
        assert_eq!(
            ok("remap(world.x, [-1, 1], [0, 1])"),
            Operand::Input(Input::Remap {
                input: boxed(coordinates(Space::World, Some(Channel::R))),
                from: [-1.0, 1.0],
                to: [0.0, 1.0],
            })
        );
        assert_eq!(
            ok("mix([0.9, 0.1, 0.2], 0.4, uv.r)"),
            Operand::Input(Input::Mix {
                a: boxed(Operand::Color([0.9, 0.1, 0.2])),
                b: boxed(Operand::Scalar(0.4)),
                factor: boxed(u()),
                mode: Blend::Mix,
            })
        );
    }

    /// The four modes are four names rather than a `mode` argument.
    #[test]
    fn a_blend_mode_is_the_name_of_the_call() {
        for (name, expected) in [
            ("mix", Blend::Mix),
            ("multiply", Blend::Multiply),
            ("add", Blend::Add),
            ("overlay", Blend::Overlay),
        ] {
            let Operand::Input(Input::Mix { mode, .. }) = ok(&format!("{name}(0, 1, 0.5)")) else {
                panic!("`{name}` should be a mix");
            };
            assert_eq!(mode, expected);
        }
    }

    /// A two-argument remap leaves `from` the unit range, as the table's
    /// default does.
    #[test]
    fn a_remap_without_a_from_takes_the_unit_range() {
        let Operand::Input(Input::Remap { from, to, .. }) = ok("remap(uv.r, [1, 0.05])") else {
            panic!("expected a remap");
        };
        assert_eq!(from, [0.0, 1.0]);
        assert_eq!(to, [1.0, 0.05]);
    }

    /// The example from the README, nested three deep.
    #[test]
    fn a_nested_tree_nests() {
        assert_eq!(
            ok("mix([0.75, 0.15, 0.1], [0.1, 0.2, 0.7], remap(object.z, [-6, 6], [0, 1]))"),
            Operand::Input(Input::Mix {
                a: boxed(Operand::Color([0.75, 0.15, 0.1])),
                b: boxed(Operand::Color([0.1, 0.2, 0.7])),
                factor: boxed(Operand::Input(Input::Remap {
                    input: boxed(coordinates(Space::Object, Some(Channel::B))),
                    from: [-6.0, 6.0],
                    to: [0.0, 1.0],
                })),
                mode: Blend::Mix,
            })
        );
    }

    /// Whitespace is not part of the grammar, including none at all.
    #[test]
    fn spacing_is_not_load_bearing() {
        let tight = ok("mix([1,0,0],[0,0,1],remap(uv.r,[0,1],[0,1]))");
        let loose = ok("mix( [1, 0, 0] , [0, 0, 1] , remap( uv.r , [0, 1] , [0, 1] ) )");
        assert_eq!(tight, loose);
    }

    /// What every message has to carry: what was wrong, and where.
    #[test]
    fn a_mistake_is_named_and_placed() {
        let cases = [
            ("mix(0, 1)", "expected `,`"),
            ("mix(0, 1, 2, 3)", "expected `)`"),
            ("invert()", "expected a number"),
            ("invert(uv", "expected `)`"),
            ("rempa(uv)", "unknown pattern `rempa`"),
            ("uv.q", "unknown channel `q`"),
            ("[0.8, 0.8]", "a colour takes 3 numbers"),
            ("[1, 2, 3, 4]", "a colour takes 3 numbers"),
            ("remap(uv, [0, 1, 2])", "a remap's range takes 2 numbers"),
            ("remap(uv, 0, 1)", "expected `[`"),
            ("uv uv", "trailing input"),
            ("1e40", "not a finite number"),
            ("image(wood.jpg)", "expected `\"`"),
            ("image(\"wood.jpg)", "never closed"),
            ("image(\"\")", "needs a file name"),
            ("image(\"a.png\", colour: \"srgb\")", "no keyword `colour`"),
            (
                "image(\"a.png\", color: \"rgb\")",
                "unknown color space `rgb`",
            ),
            (
                "image(\"a.png\", scale: 2, scale: 3)",
                "`scale` is given twice",
            ),
            ("image(\"a.png\", scale 2)", "expected `:`"),
            ("image(\"a.png\", offset: 0.5)", "expected `[`"),
            ("image(\"a.png\", )", "expected one of image's keywords"),
            ("normal_map(uv, strength: -1)", "cannot be negative"),
            (
                "normal_map(uv, convention: \"dx\")",
                "unknown convention `dx`",
            ),
            (
                "normal_map(uv, 0.5)",
                "expected one of normal_map's keywords",
            ),
            ("blackbody(uv.r)", "expected a number"),
            ("blackbody(100)", "at least 500 kelvin"),
            ("uv *", "expected a number"),
            ("(uv + 1", "expected `)`"),
            ("1 / uv.r", "can only divide by a constant"),
            ("uv / [1, 0, 1]", "division by zero"),
            ("1e30 * 1e30", "overflows to infinity"),
            (
                "noise(object, detail: 16)",
                "detail must be between 0 and 15",
            ),
            (
                "noise(object, roughness: 2)",
                "roughness must be between 0 and 1",
            ),
            (
                "noise(object, lacunarity: -1)",
                "lacunarity must be at least 0",
            ),
            (
                "noise(object, octaves: 3)",
                "noise has no keyword `octaves`",
            ),
            ("noise(object, 4.0, 5)", "expected one of noise's keywords"),
            ("voronoi(object, randomness: 1.5)", "between 0 and 1"),
            ("voronoi(object, output: \"f2\")", "unknown output `f2`"),
            ("ramp(uv.r, 0, [1, 0, 0])", "at least two stops"),
            ("ramp(uv.r)", "at least two stops"),
            ("ramp(uv.r, 0.5, 0, 0.25, 1)", "ascending order"),
            ("ramp(uv.r, uv.g, 0, 1, 1)", "position must be a constant"),
            ("ramp(uv.r, 0, uv, 1, 1)", "colour must be a constant"),
            ("ramp(uv.r, [0, 0, 0], 0, 1, 1)", "position is one number"),
            (
                "ramp(uv.r, 0, 0, 0.25, 0, 0.5, 0, 0.75, 0, 1, 1)",
                "at most 4 stops",
            ),
            (
                "bump(uv.r, strength: 2)",
                "strength must be between 0 and 1",
            ),
            ("nosie(object)", "unknown pattern `nosie`"),
        ];

        for (source, expected) in cases {
            let message = err(source);
            assert!(
                message.contains(expected),
                "`{source}` should say {expected}, said: {message}"
            );
            assert!(
                message.contains("at column"),
                "`{source}` should say where: {message}"
            );
        }
    }

    /// A channel comes off whatever precedes it, not off a coordinate only.
    /// This is how a pattern driving a scalar field says which channel the
    /// field should read.
    #[test]
    fn a_channel_comes_off_any_operand() {
        let Operand::Input(Input::Channel { input, channel }) = ok("remap(uv, [0, 4]).r") else {
            panic!("`remap(uv, [0, 4]).r` should be a channel of a remap");
        };
        assert_eq!(channel, Channel::R);
        assert!(matches!(*input, Operand::Input(Input::Remap { .. })));

        let Operand::Input(Input::Channel { channel, .. }) = ok("[0.1, 0.2, 0.3].b") else {
            panic!("a colour should take a channel too");
        };
        assert_eq!(channel, Channel::B);
    }

    /// A bare coordinate keeps carrying its own channel, so `uv.r` is the one
    /// node and the two words of program it always was.
    #[test]
    fn a_coordinate_folds_its_channel_in() {
        assert_eq!(ok("uv.r"), coordinates(Space::Uv, Some(Channel::R)));
    }

    /// Every level of nesting is a stack frame here and another in
    /// `scene::program`, so a scene that runs away is an error with a column
    /// in it rather than an abort.
    #[test]
    fn a_pattern_deeper_than_the_cap_is_an_error() {
        let mut nested = String::from("uv.r");
        for _ in 0..MAX_NESTING + 2 {
            nested = format!("invert({nested})");
        }

        let message = err(&nested);
        assert!(message.contains("nests deeper"), "{message}");

        // And the level below it is still a pattern, not a near miss: the cap
        // counts operands, and `invert` wraps one each time.
        let mut allowed = String::from("uv.r");
        for _ in 0..MAX_NESTING - 1 {
            allowed = format!("invert({allowed})");
        }
        parse(&allowed).expect("one under the cap should parse");
    }

    /// An image takes its file positionally and everything else by name, in
    /// any order, and leaves what it is not told at its default.
    #[test]
    fn an_image_takes_its_file_and_then_keywords() {
        assert_eq!(
            ok("image(\"wood.jpg\")"),
            Operand::Input(Input::Image {
                file: PathBuf::from("wood.jpg"),
                space: None,
                scale: [1.0, 1.0],
                offset: [0.0, 0.0],
            })
        );
        assert_eq!(
            ok("image(\"maps/wood.jpg\", offset: [0.5, 0], color: \"linear\", scale: [2, 4])"),
            Operand::Input(Input::Image {
                file: PathBuf::from("maps/wood.jpg"),
                space: Some(ColorSpace::Linear),
                scale: [2.0, 4.0],
                offset: [0.5, 0.0],
            })
        );

        let Operand::Input(Input::Image { scale, space, .. }) =
            ok("image(\"a.png\", scale: 3, color: \"srgb\")")
        else {
            panic!("expected an image");
        };
        assert_eq!(scale, [3.0, 3.0], "a single number scales both ways");
        assert_eq!(space, Some(ColorSpace::Srgb));
    }

    /// A channel is the suffix it always was, so one channel of a packed map
    /// needs no keyword of its own.
    #[test]
    fn a_channel_comes_off_an_image() {
        let Operand::Input(Input::Channel { input, channel }) = ok("image(\"orm.png\").g") else {
            panic!("expected a channel of an image");
        };
        assert_eq!(channel, Channel::G);
        assert!(matches!(*input, Operand::Input(Input::Image { .. })));
    }

    #[test]
    fn a_normal_map_takes_its_input_and_then_keywords() {
        assert_eq!(
            ok("normal_map(image(\"n.png\"), convention: \"directx\", strength: 0.5)"),
            Operand::Input(Input::NormalMap {
                input: boxed(Operand::Input(Input::Image {
                    file: PathBuf::from("n.png"),
                    space: None,
                    scale: [1.0; 2],
                    offset: [0.0; 2],
                })),
                strength: 0.5,
                convention: Convention::DirectX,
            })
        );

        let Operand::Input(Input::NormalMap {
            strength,
            convention,
            ..
        }) = ok("normal_map([0.5, 0.5, 1])")
        else {
            panic!("expected a normal map");
        };
        assert_eq!((strength, convention), (1.0, Convention::OpenGl));
    }

    /// The expression is usually written in a TOML literal string, so the
    /// quotes around a file name need no escaping at all.
    #[test]
    fn an_image_reads_from_a_literal_string() {
        #[derive(serde::Deserialize)]
        struct Holder {
            field: Operand,
        }

        let holder: Holder =
            toml::from_str(r#"field = 'mix(image("a.png"), [0.25, 0.18, 0.1], 0.5)'"#)
                .expect("a literal string should hold the quotes as they are");
        assert!(matches!(holder.field, Operand::Input(Input::Mix { .. })));
    }

    /// A misspelled name is reported where it starts, not where it ends.
    #[test]
    fn a_name_is_reported_where_it_starts() {
        assert!(
            err("mix(0, 1, rempa(uv))").contains("at column 11"),
            "{}",
            err("mix(0, 1, rempa(uv))")
        );
    }

    /// `Display` writes this grammar, so anything printed can be pasted back
    /// into a scene and mean the same thing.
    #[test]
    fn display_round_trips() {
        for source in [
            "0.5",
            "[0.1, 0.2, 0.3]",
            "uv",
            "world.b",
            "invert(uv.r)",
            "remap(object.g, [-6, 6], [0, 1])",
            "overlay([0.9, 0.1, 0.5], [0.2, 0.7, 0.5], 1)",
            "mix([0.75, 0.15, 0.1], [0.1, 0.2, 0.7], remap(object.b, [-6, 6], [0, 1]))",
            "add(invert(uv), multiply(uv, world, 0.5), remap(uv.r, [0, 1], [0.25, 0.75]))",
            "remap(uv, [0, 1], [0, 4]).r",
            "invert(mix(uv, world, 0.5)).g",
            "image(\"wood.jpg\")",
            "image(\"maps/wood.jpg\", color: \"linear\", scale: 4, offset: [0.5, 0])",
            "image(\"wood.jpg\", scale: [2, 3]).g",
            "mix(image(\"grass.jpg\"), [0.25, 0.18, 0.1], image(\"mask.png\").r)",
            "normal_map(image(\"n.png\"))",
            "normal_map(image(\"n.png\"), strength: 0.5, convention: \"directx\")",
            "multiply(uv, 2, 1)",
            "noise(object)",
            "noise(uv, scale: 4, detail: 5.5, roughness: 0.75, lacunarity: 3, distortion: 0.5)",
            "voronoi(world)",
            "voronoi(object, scale: 2, randomness: 0.5, output: \"color\")",
            "voronoi(object, scale: 40, output: \"edge\")",
            "ramp(noise(object), 0.3, [0.2, 0.15, 0.1], 0.7, [0.4, 0.5, 0.2])",
            "ramp(uv.r, 0, [0, 0, 0], 0.5, [1, 0, 0], 0.5, [0, 1, 0], 1, [1, 1, 1])",
            "bump(noise(object, scale: 40))",
            "bump(dirt, strength: 0.5, distance: 0.02)",
            "mix(image(\"grass.jpg\"), [0.25, 0.18, 0.1], dirt)",
            "remap(dirt, [0.45, 0.55], [0, 1])",
        ] {
            let operand = ok(source);
            assert_eq!(operand.to_string(), source, "printed form");
            assert_eq!(ok(&operand.to_string()), operand, "and it parses back");
        }
    }

    /// The example that asked for this: a colour temperature, made brighter.
    /// Both sides are constants, so it is one colour and no pattern at all.
    #[test]
    fn a_blackbody_times_a_strength_is_a_constant_colour() {
        let Operand::Color(white) = ok("blackbody(6500)") else {
            panic!("a blackbody is a colour");
        };
        let luminance = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        assert!((luminance(white) - 1.0).abs() < 1e-4, "{white:?}");
        // 6500 K is within a few hundred of D65, which is Rec. 709's white.
        for channel in white {
            assert!((channel - 1.0).abs() < 0.1, "{white:?}");
        }

        let Operand::Color(bright) = ok("blackbody(6500) * 300") else {
            panic!("a constant times a constant is a constant");
        };
        for (b, w) in bright.iter().zip(white) {
            assert!((b - w * 300.0).abs() < 1e-3, "{bright:?}");
        }
    }

    /// Candlelight is red over blue, an overcast sky blue over red, and
    /// neither goes negative in the channel it runs out of gamut in.
    #[test]
    fn a_blackbody_warms_as_it_cools() {
        let color = |kelvin: u32| match ok(&format!("blackbody({kelvin})")) {
            Operand::Color(c) => c,
            other => panic!("{other:?}"),
        };
        let [r, _, b] = color(1900);
        assert!(r > b && b >= 0.0, "{:?}", color(1900));
        let [r, _, b] = color(12000);
        assert!(b > r && r >= 0.0, "{:?}", color(12000));
    }

    /// `*` binds tighter than `+`, both fold to the left, parentheses group,
    /// and a pattern on either side makes a mix at a factor of one.
    #[test]
    fn arithmetic_reads_the_usual_way() {
        assert_eq!(ok("1 + 2 * 3"), Operand::Scalar(7.0));
        assert_eq!(ok("(1 + 2) * 3"), Operand::Scalar(9.0));
        assert_eq!(ok("2 * [1, 2, 3]"), Operand::Color([2.0, 4.0, 6.0]));
        assert_eq!(ok("1+-2"), Operand::Scalar(-1.0));
        assert_eq!(ok("1e+1*2"), Operand::Scalar(20.0));
        assert_eq!(ok("8 - 2 - 1"), Operand::Scalar(5.0));
        assert_eq!(ok("8 / 2 / 4"), Operand::Scalar(1.0));
        assert_eq!(ok("1-2"), Operand::Scalar(-1.0));
        assert_eq!(ok("[2, 4, 8] / [2, 4, 8]"), Operand::Color([1.0; 3]));
        assert_eq!(ok("-(1 + 2)"), Operand::Scalar(-3.0));
        assert_eq!(ok("- 2"), Operand::Scalar(-2.0));

        // Subtracting and negating a pattern are adding and multiplying by -1;
        // dividing one is multiplying by the reciprocal.
        assert_eq!(ok("1 - uv.r"), ok("add(1, multiply(uv.r, -1, 1), 1)"));
        assert_eq!(ok("-uv.r"), ok("multiply(uv.r, -1, 1)"));
        assert_eq!(ok("uv / 4"), ok("multiply(uv, 0.25, 1)"));

        assert_eq!(
            ok("uv.r * 2 + 1"),
            Operand::Input(Input::Mix {
                a: boxed(Operand::Input(Input::Mix {
                    a: boxed(coordinates(Space::Uv, Some(Channel::R))),
                    b: boxed(Operand::Scalar(2.0)),
                    factor: boxed(Operand::Scalar(1.0)),
                    mode: Blend::Multiply,
                })),
                b: boxed(Operand::Scalar(1.0)),
                factor: boxed(Operand::Scalar(1.0)),
                mode: Blend::Add,
            })
        );

        // Inside a call's arguments too, and a channel off a group.
        assert_eq!(
            ok("remap((uv + 1).g * 2, [0, 4])"),
            ok("remap(multiply(add(uv, 1, 1).g, 2, 1), [0, 4])")
        );
    }

    /// Every knob of a noise is a keyword, left at Blender's default when it
    /// is not given.
    #[test]
    fn a_noise_takes_its_coordinates_and_then_keywords() {
        assert_eq!(
            ok("noise(object)"),
            Operand::Input(Input::Noise {
                input: boxed(coordinates(Space::Object, None)),
                scale: 5.0,
                detail: 2.0,
                roughness: 0.5,
                lacunarity: 2.0,
                distortion: 0.0,
            })
        );
        assert_eq!(
            ok("noise(world * 2, distortion: 0.25, detail: 5, scale: 4)"),
            Operand::Input(Input::Noise {
                input: boxed(ok("world * 2")),
                scale: 4.0,
                detail: 5.0,
                roughness: 0.5,
                lacunarity: 2.0,
                distortion: 0.25,
            })
        );
    }

    #[test]
    fn a_voronoi_says_which_answer_it_reads() {
        let Operand::Input(Input::Voronoi { output, .. }) = ok("voronoi(object)") else {
            panic!("expected a voronoi");
        };
        assert_eq!(output, Cell::Distance);

        let Operand::Input(Input::Voronoi {
            output, randomness, ..
        }) = ok("voronoi(object, output: \"color\", randomness: 0.25)")
        else {
            panic!("expected a voronoi");
        };
        assert_eq!((output, randomness), (Cell::Color, 0.25));

        let Operand::Input(Input::Voronoi { output, .. }) = ok("voronoi(object, output: \"edge\")")
        else {
            panic!("expected a voronoi");
        };
        assert_eq!(output, Cell::Edge);
    }

    /// A stop's position and colour are anything that works out to a
    /// constant, so a colour temperature or a grey written as one number is a
    /// stop like any other.
    #[test]
    fn a_ramp_folds_its_stops_to_constants() {
        let Operand::Input(Input::Ramp { stops, .. }) =
            ok("ramp(uv.r, 0.1 * 2, 0.5, 1 - 0.2, blackbody(6500) * 0)")
        else {
            panic!("expected a ramp");
        };
        assert_eq!(stops, vec![(0.2, [0.5; 3]), (0.8, [0.0; 3])]);
    }

    /// A name that is not a call is a reference to `[patterns]`, and one that
    /// is followed by `(` is still an unknown call.
    #[test]
    fn a_bare_name_is_a_pattern() {
        assert_eq!(
            ok("dirt"),
            Operand::Input(Input::Pattern {
                name: String::from("dirt")
            })
        );
        let Operand::Input(Input::Mix { factor, .. }) = ok("mix(0.9, 0.6, dirt_2.r)") else {
            panic!("expected a mix");
        };
        assert!(matches!(*factor, Operand::Input(Input::Channel { .. })));

        assert!(err("dirt(uv)").contains("unknown pattern `dirt`"));
        for name in RESERVED {
            assert!(is_reserved(name), "{name}");
        }
        assert!(!is_reserved("dirt"));
    }

    /// A chain of operators nests the tree without nesting the parser, so it
    /// is counted against the same cap.
    #[test]
    fn a_long_chain_is_held_to_the_nesting_cap() {
        let chain = vec!["uv.r"; MAX_NESTING as usize + 2].join(" * ");
        assert!(err(&chain).contains("nests deeper"));
    }
}
