//! Flattens a material's input trees into the one program buffer the shader
//! evaluates.
//!
//! Everything a scene writes as a pattern ends up here as a postfix program: a
//! list of ops that push and pop `vec4f`s on a small fixed stack, with the
//! answer left on top. One flat `array<u32>` holds all of it — every material's
//! field table and every program those tables point at — because a storage
//! binding is a scarce thing on this pipeline and words are not.
//!
//! The encoding is deliberately dull. Each op is its code followed by its
//! immediate words; floats ride as their bit patterns and come back through
//! `bitcast`. The shader's decoder and [`Program::push`] below are the two
//! halves of one format, and the tests in `render::shader_tests` run the same
//! trees through both.

use crate::config::Blend;
use crate::config::Channel;
use crate::config::Input;
use crate::config::MAX_NESTING;
use crate::config::Operand;
use crate::config::Principled;
use crate::config::Space;
use crate::config::emission_bound;

/// A field table slot, or a material, with nothing patterned. Also what the
/// shader tests against before it evaluates anything at all.
pub const NO_PROGRAM: u32 = u32::MAX;

/// Op codes, matching `shader.wgsl`.
const OP_END: u32 = 0;
const OP_CONSTANT: u32 = 1;
const OP_COORDINATES: u32 = 2;
const OP_CHANNEL: u32 = 3;
const OP_INVERT: u32 = 4;
const OP_REMAP: u32 = 5;
const OP_MIX: u32 = 6;

/// How deep the register stack is, on the host and on the GPU. Eight is
/// generous — the deepest tree anyone writes by hand is three or four — and it
/// is a fixed array in the shader, so it is a budget rather than a limit to
/// grow later without thinking about occupancy.
pub const MAX_STACK: u32 = 8;

/// The patternable fields, in the order the shader reads their table.
///
/// Emission is one slot rather than two: `emission_color` and
/// `emission_strength` are only ever multiplied, and the host folds them into
/// one program the same way it folds them into one constant, so the shader has
/// one number to resolve instead of two and a multiply.
pub const FIELD_BASE_COLOR: usize = 0;
pub const FIELD_ROUGHNESS: usize = 1;
pub const FIELD_METALLIC: usize = 2;
pub const FIELD_IOR: usize = 3;
pub const FIELD_TRANSMISSION: usize = 4;
pub const FIELD_SUBSURFACE_WEIGHT: usize = 5;
pub const FIELD_EMISSION: usize = 6;
pub const FIELDS: usize = 7;

/// Every material's programs, concatenated.
#[derive(Debug, Default)]
pub struct Programs {
    words: Vec<u32>,
}

impl Programs {
    /// The buffer as the shader reads it.
    ///
    /// Never empty: a binding cannot be, and a scene with no patterns in it is
    /// the common case. The one word is never read — every material's `inputs`
    /// is [`NO_PROGRAM`] alongside it — and it is the same fallback the light
    /// table and the sky distribution take.
    pub fn words(&self) -> &[u32] {
        match self.words.is_empty() {
            true => &[NO_PROGRAM],
            false => &self.words,
        }
    }

    /// Compiles every patterned field of `principled` and returns the offset of
    /// its field table, or [`NO_PROGRAM`] when nothing about the surface is
    /// patterned — which is the answer for every scene written before this
    /// existed, and the flag that lets the shader skip all of it.
    ///
    /// `subsurface` is false when the host has already decided the walk has
    /// nowhere to go, in which case a patterned weight is dropped along with
    /// the constant one: the shader divides by the radius, and a weight arriving
    /// per hit would be dividing by a mean free path of zero.
    pub fn compile<'a>(
        &mut self,
        principled: &'a Principled,
        subsurface: bool,
    ) -> Result<u32, String> {
        // A field is compiled only when a pattern actually drives it. The
        // subsurface weight is the one that can be driven and still dropped,
        // and emission is the one that is folded before it is asked.
        let emission = emission(principled);
        let mut fields: [Option<&Operand>; FIELDS] = [None; FIELDS];
        let patterned = |operand: &'a Operand| operand.is_input().then_some(operand);

        fields[FIELD_BASE_COLOR] = patterned(&principled.base_color);
        fields[FIELD_ROUGHNESS] = patterned(&principled.roughness);
        fields[FIELD_METALLIC] = patterned(&principled.metallic);
        fields[FIELD_IOR] = patterned(&principled.ior);
        fields[FIELD_TRANSMISSION] = patterned(&principled.transmission);
        fields[FIELD_SUBSURFACE_WEIGHT] = match subsurface {
            true => patterned(&principled.subsurface_weight),
            false => None,
        };
        fields[FIELD_EMISSION] = emission.as_ref();

        if fields.iter().all(Option::is_none) {
            return Ok(NO_PROGRAM);
        }

        // The table is written first and patched afterwards, so a field's
        // program sits right behind the table that names it rather than being
        // laid out in a second pass.
        let table = self.words.len() as u32;
        self.words.extend([NO_PROGRAM; FIELDS]);

        for (slot, operand) in fields.into_iter().enumerate() {
            let Some(operand) = operand else { continue };

            let offset = self.program(operand)?;
            self.words[table as usize + slot] = offset;
        }

        Ok(table)
    }

    /// One operand's program, with no material around it.
    ///
    /// The tests hold the shader's evaluator against [`evaluate`] op by op, and
    /// which field a program happens to drive says nothing about how it runs.
    #[cfg(test)]
    pub fn compile_one(&mut self, operand: &Operand) -> Result<u32, String> {
        self.program(operand)
    }

    /// One field's program: its ops, then [`OP_END`].
    fn program(&mut self, operand: &Operand) -> Result<u32, String> {
        let start = self.words.len() as u32;
        let depth = self.push(operand, 0, 0)?;
        self.words.push(OP_END);

        debug_assert_eq!(depth, 1, "a program leaves exactly its answer behind");
        Ok(start)
    }

    /// Emits `operand` in postfix order and returns how deep the stack is
    /// afterwards, which is what bounds it against [`MAX_STACK`].
    ///
    /// `depth` is what is already on the stack when this operand starts. Every
    /// op below pushes its arguments first and consumes them immediately, so
    /// the high-water mark is reached inside the last argument of the widest
    /// node — a `mix`, three deep — and checking after each push is enough to
    /// catch it.
    ///
    /// `nesting` is the other measure of the same tree, and the two do not
    /// track each other: an `invert` of an `invert` leaves the stack where it
    /// found it and still costs a frame of this function's own recursion. A
    /// tree built in code rather than parsed has had nothing counting them, so
    /// the cap is here as well as in `config::expression`.
    fn push(&mut self, operand: &Operand, depth: u32, nesting: u32) -> Result<u32, String> {
        if nesting > MAX_NESTING {
            return Err(format!(
                "an input nests deeper than the {MAX_NESTING} levels this compiles"
            ));
        }

        let constant = |slot: &mut Vec<u32>, value: [f32; 4]| {
            slot.push(OP_CONSTANT);
            slot.extend(value.map(f32::to_bits));
        };

        match operand {
            // A scalar broadcasts to every channel, so mixing one with a color
            // means what it looks like it means. A color's fourth channel is
            // one and is never read.
            Operand::Scalar(value) => constant(&mut self.words, [*value; 4]),
            Operand::Color(value) => constant(&mut self.words, [value[0], value[1], value[2], 1.0]),

            Operand::Input(Input::Coordinates { space, channel }) => {
                self.words.push(OP_COORDINATES);
                self.words.push(match space {
                    Space::Uv => 0,
                    Space::Object => 1,
                    Space::World => 2,
                });
                if let Some(channel) = channel {
                    self.words.push(OP_CHANNEL);
                    self.words.push(channel.index());
                }
            }

            Operand::Input(Input::Channel { input, channel }) => {
                self.push(input, depth, nesting + 1)?;
                self.words.push(OP_CHANNEL);
                self.words.push(channel.index());
            }

            Operand::Input(Input::Invert { input }) => {
                self.push(input, depth, nesting + 1)?;
                self.words.push(OP_INVERT);
            }

            Operand::Input(Input::Remap { input, from, to }) => {
                // `from` is the one range nothing else looks at: `range()`
                // reports a remap by its `to`, so `Material::validate` never
                // sees these numbers. Checked here, where every other float in
                // a scene is checked for being a number at all.
                if !from.iter().chain(to).all(|end| end.is_finite()) {
                    return Err(format!(
                        "a remap's ranges must be finite, not {from:?} -> {to:?}"
                    ));
                }
                if from[0] == from[1] {
                    return Err(format!(
                        "a remap's `from` range must have two ends, not {from:?}"
                    ));
                }
                self.push(input, depth, nesting + 1)?;
                self.words.push(OP_REMAP);
                self.words
                    .extend([from[0], from[1], to[0], to[1]].map(f32::to_bits));
            }

            Operand::Input(Input::Mix { a, b, factor, mode }) => {
                self.push(a, depth, nesting + 1)?;
                self.push(b, depth + 1, nesting + 1)?;
                self.push(factor, depth + 2, nesting + 1)?;
                self.words.push(OP_MIX);
                self.words.push(match mode {
                    Blend::Mix => 0,
                    Blend::Multiply => 1,
                    Blend::Add => 2,
                    Blend::Overlay => 3,
                });
            }
        }

        let after = depth + 1;
        match after <= MAX_STACK {
            true => Ok(after),
            false => Err(format!(
                "an input nests deeper than the {MAX_STACK} values the shader's \
                 stack holds"
            )),
        }
    }
}

/// The one operand that resolves to a surface's emitted radiance, or `None`
/// when both halves of it are constants and [`crate::scene::GpuMaterial`] can
/// carry the product on its own.
///
/// Written as a multiply blend at full factor, which is the op the shader
/// already has, rather than a fourth kind of node nothing else would use.
fn emission(principled: &Principled) -> Option<Operand> {
    let color = &principled.emission_color;
    let strength = &principled.emission_strength;
    if !color.is_input() && !strength.is_input() {
        return None;
    }

    // A half that is constantly zero makes the product zero wherever the hit
    // is, so there is no program worth running and nothing for it to say that
    // the zero in `emission_constant` does not.
    if color.is_zero() || strength.is_zero() {
        return None;
    }

    Some(Operand::Input(Input::Mix {
        a: Box::new(color.clone()),
        b: Box::new(scalar(strength)),
        factor: Box::new(Operand::Scalar(1.0)),
        mode: Blend::Multiply,
    }))
}

/// `operand` read the way a scalar field reads one: its first channel,
/// broadcast to the rest.
///
/// Every other scalar field gets this for free — the shader takes `.r` off
/// whatever `resolve_field` hands back. Emission is the one that does not,
/// because its two halves are multiplied into a single field before the shader
/// sees either, and a multiply is per-channel. Without this a strength written
/// as `remap(uv, [0, 4])` tints the colour it was only asked to brighten: the
/// green channel scaled by `v` and the blue by nothing at all.
///
/// A constant already broadcasts, so only a pattern needs the node.
fn scalar(operand: &Operand) -> Operand {
    match operand.is_input() {
        true => Operand::Input(Input::Channel {
            input: Box::new(operand.clone()),
            channel: Channel::R,
        }),
        false => operand.clone(),
    }
}

/// The constant a field's GPU slot carries. For a patterned field this is only
/// ever a placeholder — the shader overwrites it — except for emission, where
/// it is the upper bound the light table is built from and has to stay.
pub fn constant(operand: &Operand, default: [f32; 3]) -> [f32; 3] {
    operand.constant().unwrap_or(default)
}

/// The radiance the light table weights a surface by: the exact product when
/// both halves are constants, and the upper bound when either is a pattern.
///
/// Validation has already rejected a surface with no bound at all, so this only
/// falls back when a material is built outside a validated config.
pub fn emission_constant(principled: &Principled) -> [f32; 3] {
    emission_bound(principled).unwrap_or([0.0; 3])
}

/// Which slot of the field table drives `field`. Exposed for the tests, which
/// read a compiled table back.
#[cfg(test)]
pub fn table(programs: &Programs, offset: u32, field: usize) -> u32 {
    programs.words()[offset as usize + field]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn principled(source: &str) -> Principled {
        toml::from_str(source).unwrap_or_else(|error| panic!("{source}: {error}"))
    }

    #[test]
    fn a_surface_with_no_patterns_compiles_to_nothing() {
        let mut programs = Programs::default();
        let material = principled("roughness = 0.25\nbase_color = [1.0, 0.0, 0.0]");

        assert_eq!(programs.compile(&material, true), Ok(NO_PROGRAM));
        assert_eq!(programs.words(), [NO_PROGRAM], "and nothing is uploaded");
    }

    #[test]
    fn only_the_patterned_fields_get_a_slot() {
        let mut programs = Programs::default();
        let material = principled(
            r#"
roughness = "uv.r"
metallic = 0.5
"#,
        );

        let table = programs.compile(&material, true).expect("should compile");
        assert_ne!(table, NO_PROGRAM);
        assert_ne!(super::table(&programs, table, FIELD_ROUGHNESS), NO_PROGRAM);
        for field in [
            FIELD_BASE_COLOR,
            FIELD_METALLIC,
            FIELD_IOR,
            FIELD_TRANSMISSION,
            FIELD_SUBSURFACE_WEIGHT,
            FIELD_EMISSION,
        ] {
            assert_eq!(super::table(&programs, table, field), NO_PROGRAM, "{field}");
        }
    }

    /// Two materials share one buffer, and the second's table has to name its
    /// own programs rather than the first's.
    #[test]
    fn two_materials_keep_their_own_tables() {
        let mut programs = Programs::default();
        let first = programs
            .compile(&principled(r#"roughness = "uv""#), true)
            .expect("should compile");
        let second = programs
            .compile(&principled(r#"metallic = "world""#), true)
            .expect("should compile");

        assert_ne!(first, second);
        assert_eq!(super::table(&programs, first, FIELD_METALLIC), NO_PROGRAM);
        assert_eq!(super::table(&programs, second, FIELD_ROUGHNESS), NO_PROGRAM);
        assert_ne!(super::table(&programs, second, FIELD_METALLIC), NO_PROGRAM);
    }

    /// A patterned emission is folded into one field whichever half of it
    /// carries the pattern, and the constant left behind is the bound.
    #[test]
    fn either_half_of_emission_compiles_to_one_field() {
        for source in [
            r#"emission_color = "remap(uv, [0.0, 2.0])"
emission_strength = 3.0"#,
            r#"emission_color = [0.0, 1.0, 0.0]
emission_strength = "remap(uv, [0.0, 4.0])""#,
        ] {
            let mut programs = Programs::default();
            let material = principled(source);

            let table = programs.compile(&material, true).expect("should compile");
            assert_ne!(super::table(&programs, table, FIELD_EMISSION), NO_PROGRAM);
            assert!(emission_constant(&material).iter().any(|c| *c > 0.0));
        }
    }

    /// A walk with no radius has no weight to resolve either, so the field is
    /// dropped rather than left to divide by a mean free path of zero.
    #[test]
    fn a_subsurface_weight_is_dropped_when_the_walk_has_nowhere_to_go() {
        let mut programs = Programs::default();
        let material = principled(r#"subsurface_weight = "uv""#);

        assert_eq!(programs.compile(&material, false), Ok(NO_PROGRAM));

        let table = programs.compile(&material, true).expect("should compile");
        assert_ne!(
            super::table(&programs, table, FIELD_SUBSURFACE_WEIGHT),
            NO_PROGRAM
        );
    }

    #[test]
    fn a_remap_with_no_range_is_an_error() {
        let mut programs = Programs::default();
        let material = principled(r#"roughness = "remap(uv, [0.5, 0.5], [0.0, 1.0])""#);

        let error = programs
            .compile(&material, true)
            .expect_err("a zero-width range divides by nothing");
        assert!(error.contains("two ends"), "{error}");
    }

    /// A remap is where a NaN or an infinity would otherwise reach the GPU:
    /// the clamp there is not specified to sanitize a NaN, and one in `from`
    /// divides straight into the field it drives. The expression parser
    /// already refuses to spell either, so this is the second line of the two
    /// and the tree is built here by hand to reach it at all.
    #[test]
    fn a_remap_with_a_range_that_is_not_a_number_is_an_error() {
        for (from, to) in [
            ([f32::NAN, 1.0], [0.0, 1.0]),
            ([0.0, f32::INFINITY], [0.0, 1.0]),
            ([0.0, 1.0], [0.0, f32::NAN]),
        ] {
            let mut programs = Programs::default();
            let material = Principled {
                roughness: Operand::Input(Input::Remap {
                    input: Box::new(Operand::Input(Input::Coordinates {
                        space: Space::Uv,
                        channel: None,
                    })),
                    from,
                    to,
                }),
                ..Principled::default()
            };

            let error = programs
                .compile(&material, true)
                .expect_err("a range that is not a number divides into the field");
            assert!(error.contains("finite"), "{from:?} -> {to:?}: {error}");
        }
    }

    /// `emission_strength` is a scalar field, and a pattern driving one is read
    /// the way every other scalar field reads one: first channel, broadcast.
    /// The fold multiplies the two halves per channel, so without the `.r` a
    /// strength written across `uv` tints the colour instead of brightening it.
    #[test]
    fn a_patterned_emission_strength_is_read_as_a_scalar() {
        let material = principled(
            r#"
emission_color = [1.0, 1.0, 1.0]
emission_strength = "remap(uv, [0.0, 4.0])"
"#,
        );
        let folded = emission(&material).expect("a patterned strength folds into one operand");

        // `uv` is `(u, v, 0, 1)`: a per-channel multiply would scale green by
        // `v` and blue by nothing, which is the bug this is here for.
        let at = Coordinates {
            uv: [0.25, 0.75],
            object: [0.0; 3],
            world: [0.0; 3],
        };
        let emitted = evaluate(&folded, at);

        assert_eq!(emitted[0], 1.0, "u of 0.25 over [0, 4]");
        assert_eq!(emitted[1], emitted[0], "green: {emitted:?}");
        assert_eq!(emitted[2], emitted[0], "blue: {emitted:?}");
    }

    /// And the bound the light table is built from has to be the same number,
    /// which means reading the same channel.
    #[test]
    fn the_emission_bound_reads_the_strength_the_same_way() {
        let material = principled(
            r#"
emission_color = [1.0, 0.5, 0.25]
emission_strength = "remap(uv, [0.0, 4.0])"
"#,
        );

        assert_eq!(emission_constant(&material), [4.0, 2.0, 1.0]);
    }

    /// A half that is constantly zero settles the product, so there is nothing
    /// to compile and nothing the light table needs bounded.
    #[test]
    fn an_emission_with_a_zero_half_compiles_to_nothing() {
        let mut programs = Programs::default();
        let material = principled(r#"emission_color = "uv""#);

        assert_eq!(programs.compile(&material, true), Ok(NO_PROGRAM));
        assert_eq!(emission_constant(&material), [0.0; 3]);
    }

    /// The stack is one measure of a tree and the recursion here is another: an
    /// `invert` chain leaves the stack where it found it and still costs a
    /// frame a level, so a tree built in code rather than parsed is capped
    /// here too.
    #[test]
    fn a_tree_that_nests_past_the_cap_is_an_error() {
        let mut operand = Operand::Input(Input::Coordinates {
            space: Space::Uv,
            channel: None,
        });
        for _ in 0..MAX_NESTING + 2 {
            operand = Operand::Input(Input::Invert {
                input: Box::new(operand),
            });
        }

        let mut programs = Programs::default();
        let error = programs
            .compile_one(&operand)
            .expect_err("the recursion should be capped");
        assert!(error.contains("nests deeper"), "{error}");
    }

    /// The stack is a fixed array in the shader, so a tree that would outrun it
    /// has to be caught here rather than clamped there.
    #[test]
    fn a_tree_deeper_than_the_stack_is_an_error() {
        let mut nested = String::from("0.5");
        for _ in 0..MAX_STACK + 2 {
            nested = format!("mix(0.0, 1.0, {nested})");
        }

        let mut programs = Programs::default();
        let error = programs
            .compile(&principled(&format!("roughness = {nested:?}")), true)
            .expect_err("the stack should run out");
        assert!(error.contains("stack"), "{error}");
    }
}

/// Where a pattern is being evaluated, as the shader's `Coordinates` carries it.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub struct Coordinates {
    pub uv: [f32; 2],
    pub object: [f32; 3],
    pub world: [f32; 3],
}

/// What a tree resolves to, evaluated straight off the tree.
///
/// The reference the GPU's flat program is held against. Deliberately a second
/// implementation and not a shared one: the whole risk in a compiled format is
/// that the compiler and the decoder agree with each other about something the
/// scene did not ask for, and only an evaluator that never sees the words can
/// catch that.
#[cfg(test)]
pub fn evaluate(operand: &Operand, at: Coordinates) -> [f32; 4] {
    use crate::config::blend;

    match operand {
        Operand::Scalar(value) => [*value; 4],
        Operand::Color(value) => [value[0], value[1], value[2], 1.0],

        Operand::Input(Input::Coordinates { space, channel }) => {
            let value = match space {
                Space::Uv => [at.uv[0], at.uv[1], 0.0, 1.0],
                Space::Object => [at.object[0], at.object[1], at.object[2], 1.0],
                Space::World => [at.world[0], at.world[1], at.world[2], 1.0],
            };
            match channel {
                Some(channel) => [value[channel.index() as usize]; 4],
                None => value,
            }
        }

        Operand::Input(Input::Channel { input, channel }) => {
            [evaluate(input, at)[channel.index() as usize]; 4]
        }

        Operand::Input(Input::Invert { input }) => evaluate(input, at).map(|c| 1.0 - c),

        Operand::Input(Input::Remap { input, from, to }) => evaluate(input, at).map(|c| {
            let t = ((c - from[0]) / (from[1] - from[0])).clamp(0.0, 1.0);
            to[0] + t * (to[1] - to[0])
        }),

        Operand::Input(Input::Mix { a, b, factor, mode }) => {
            let (a, b) = (evaluate(a, at), evaluate(b, at));
            let factor = evaluate(factor, at)[0];
            let mut out = [0.0; 4];
            for (channel, slot) in out.iter_mut().enumerate() {
                *slot = blend(*mode, a[channel], b[channel], factor);
            }
            out
        }
    }
}
