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
//! operand     := primary {'.' channel}
//! primary     := number | color | coordinates | call
//! number      := -1.5, 2, 1e-3 …
//! color       := '[' number ',' number ',' number ']'
//! coordinates := 'uv' | 'object' | 'world'
//! channel     := 'r' | 'g' | 'b' | 'a' | 'x' | 'y' | 'z' | 'w'
//! call        := 'invert' '(' operand ')'
//!              | 'remap' '(' operand ',' [range ','] range ')'
//!              | ('mix' | 'multiply' | 'add' | 'overlay')
//!                '(' operand ',' operand ',' operand ')'
//! range       := '[' number ',' number ']'
//! ```
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

use super::input::Blend;
use super::input::Channel;
use super::input::Input;
use super::input::MAX_NESTING;
use super::input::Operand;
use super::input::Space;

/// Parses a whole expression, which must be all of `source`.
pub fn parse(source: &str) -> Result<Operand, String> {
    let mut parser = Parser {
        source,
        at: 0,
        depth: 0,
    };
    let operand = parser.operand()?;

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

    fn operand(&mut self) -> Result<Operand, String> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            return Err(self.error(&format!(
                "a pattern nests deeper than the {MAX_NESTING} levels this parses"
            )));
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
    /// six calls.
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
            // A misspelling is reported where the name started rather than
            // where it ended, which is where the eye is.
            _ => {
                self.at = start;
                Err(self.error(&format!(
                    "unknown pattern `{name}`; expected uv, object, world, \
                     invert, remap, mix, multiply, add or overlay"
                )))
            }
        }
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
        let input = self.operand()?;
        self.expect(')')?;

        Ok(Operand::Input(Input::Invert {
            input: Box::new(input),
        }))
    }

    /// `remap(input, to)` or `remap(input, from, to)`: the two-argument form
    /// leaves `from` the unit range, exactly as the table form's default does.
    fn remap(&mut self) -> Result<Operand, String> {
        self.expect('(')?;
        let input = self.operand()?;
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

    fn mix(&mut self, mode: Blend) -> Result<Operand, String> {
        self.expect('(')?;
        let a = self.operand()?;
        self.expect(',')?;
        let b = self.operand()?;
        self.expect(',')?;
        let factor = self.operand()?;
        self.expect(')')?;

        Ok(Operand::Input(Input::Mix {
            a: Box::new(a),
            b: Box::new(b),
            factor: Box::new(factor),
            mode,
        }))
    }
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
        ] {
            let operand = ok(source);
            assert_eq!(operand.to_string(), source, "printed form");
            assert_eq!(ok(&operand.to_string()), operand, "and it parses back");
        }
    }
}
