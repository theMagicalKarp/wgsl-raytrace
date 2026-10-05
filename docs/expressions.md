# Expression reference

Most material fields in the [config](config.md#principled) take a number, _or_ a
pattern evaluated at every hit, written as an expression in a string:

```toml
[[objects]]
shape = "wavefront"
file = "scene.obj"
material = "principled"

# Polished at one edge of the mesh's texture coordinates, rough at the other.
roughness = "remap(uv.r, [0.05, 1.0])"

# Red at the front of the object, blue at the back, in the object's own space
# so it does not slide about when the object is moved.
base_color = "mix([0.75, 0.15, 0.1], [0.1, 0.2, 0.7], remap(object.z, [-6, 6], [0, 1]))"
```

The whole grammar:

```text
expression  := term {(+ | -) term}
term        := operand {(* | /) operand}
operand     := [-] primary {. channel}
primary     := number | color | coordinates | call | name | (expression)
color       := [r, g, b]
coordinates := uv | object | world
channel     := r | g | b | a   (or x | y | z | w, for the same four)
call        := invert(expression)
             | remap(expression, [to0, to1])
             | remap(expression, [from0, from1], [to0, to1])
             | (mix | multiply | add | overlay)(expression, expression, expression)
             | blackbody(number)
             | image("file", keyword: value, …)
             | normal_map(expression, keyword: value, …)
             | noise(expression, keyword: value, …)
             | voronoi(expression, keyword: value, …)
             | ramp(expression, position, color, position, color, …)
             | bump(expression, keyword: value, …)
name        := an entry of [patterns]
```

Any argument written `expression` takes a whole expression, arithmetic included:
`invert(uv * 2)`.

Arguments a call can't do without are positional; optional ones follow as
`name: value` keywords, in any order.

A blend mode is the name of the call rather than an argument to it, so
`overlay(a, b, f)` is the overlay mix. `+`, `-`, `*` and `/` work with the usual
precedence and parentheses; between two constants they are worked out when the
scene loads, so `blackbody(6500) * 300` is just a colour. Whitespace is free,
numbers are plain decimals, optionally signed or with an exponent (`-6`, `0.5`,
`1e-3`), and an error names the column it stopped at inside the string, on top
of the line TOML names.

A pattern **replaces** the field it is written in place of; it does not multiply
it. Where a pattern produces three channels and the field wants one, the first
is used; where it produces one and the field wants three, it is broadcast.
`.channel` after any operand — not only after a coordinate — picks one component
and broadcasts it, which is how a pattern says which of its channels a scalar
field should read: `remap(uv, [0, 4]).g`.

**Patternable**: `base_color`, `roughness`, `metallic`, `ior`, `transmission`,
`specular_ior_level`, `emission_color`, `emission_strength`,
`subsurface_weight`, a `light`'s `emit`, and `normal` (which only takes a
`normal_map` or a `bump`). **Not**: `alpha`, `subsurface_radius`,
`subsurface_scale`, `subsurface_anisotropy`.

| Expression                                                            | What it is                                                                                                                                                                                                                                                                                         |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv`, `object`, `world`                                               | Where the hit is. `uv` is the mesh's own texture coordinates, zero on a `.obj` with no `vt` lines. `object` is the space the file was authored in, recovered by undoing the object's transform, so a pattern written in it stays put when the object moves. `world` is fixed to the scene instead. |
| `.r`/`.g`/`.b`/`.a` (or `.x`/`.y`/`.z`/`.w`) after any operand        | Keeps that one component and broadcasts it to the rest.                                                                                                                                                                                                                                            |
| `invert(input)`                                                       | `1 - input`, per channel.                                                                                                                                                                                                                                                                          |
| `remap(input, [from], [to])`, `from` defaulting to `[0, 1]`           | `input` taken from one range onto another, **clamped** to `to`. The way to give an unbounded pattern a range. `from` may not have two equal ends; `to` may descend.                                                                                                                                |
| `mix(a, b, factor)`, and `multiply`, `add`, `overlay` the same way    | `a` and `b` blended by `factor`, which is clamped to `[0, 1]`. Blender's Mix node arithmetic.                                                                                                                                                                                                      |
| `a + b`, `a - b`, `a * b`, `a / b`, `-a`                              | Arithmetic, per channel. Shorthand for `add(a, b, 1)` and `multiply(a, b, 1)`: `a - b` is `a + b × -1`, `-a` is `a × -1`, and `a / b` is `a × (1 / b)`, so `b` must be a non-zero constant when dividing.                                                                                          |
| `blackbody(kelvin)`                                                   | The colour of a black body at that temperature (at least 500 K), in linear Rec. 709 and scaled to a luminance of 1 as Blender's Blackbody node is, so it sets a light's colour and a strength sets its brightness. A constant: `kelvin` is a number, not a pattern.                                |
| `image("file", color:, scale:, offset:)`                              | An image, sampled at `uv × scale + offset`. See [Image maps](#image-maps).                                                                                                                                                                                                                         |
| `normal_map(input, strength:, convention:)`                           | A tangent-space normal map, for the `normal` field only. See [Image maps](#image-maps).                                                                                                                                                                                                            |
| `noise(input, scale:, detail:, roughness:, lacunarity:, distortion:)` | Gradient noise in `[0, 1]`, read at `input`'s first three channels. See [Generated patterns](#generated-patterns).                                                                                                                                                                                 |
| `voronoi(input, scale:, randomness:, output:)`                        | Cells: the distance to the nearest feature point or to the nearest border, or a random colour per cell. See [Generated patterns](#generated-patterns).                                                                                                                                             |
| `ramp(input, position, color, …)`                                     | `input`'s first channel through two to four colour stops. See [Generated patterns](#generated-patterns).                                                                                                                                                                                           |
| `bump(height, strength:, distance:)`                                  | The normal tilted along the slope of a height, for the `normal` field only. See [Generated patterns](#generated-patterns).                                                                                                                                                                         |
| a name                                                                | An entry of the scene's `[patterns]` table. See [Named patterns](#named-patterns).                                                                                                                                                                                                                 |

Any of `input`, `a`, `b` and `factor` may itself be a plain number, a colour, or
another pattern, up to eight values in flight and 64 levels of nesting.

Emission is the one field with a rule of its own. Emissive surfaces are sampled
as lights, and the light table has to weigh each one before the render starts —
so a patterned emission needs an **upper bound** the host can compute:

- Constants are their own bound, and an image is bounded by `[0, 1]`.
- `remap` is bounded by its `to` range, whatever it wraps.
- `mix` and the other blends clamp their factor, so they are bounded whenever
  `a` and `b` are, even when the factor is a coordinate.
- `invert`, arithmetic and `.r`/`.g`/`.b` are bounded when their inputs are.
- A coordinate (`uv`, `object`, `world`, with or without a channel) has no
  bound, and neither does `.a`/`.w`, even on an image.

An emission with no bound is a scene error naming the fix: wrap the unbounded
part in a `remap`. So `uv.r * 4` can't emit, but `remap(uv.r, [0, 4])` can, and
so can `mix([0, 0, 0], [1, 1, 1], uv.r)`. The rule only applies where the
surface can emit at all: `emission_strength` defaults to zero, and a half that
is constantly zero settles the product whatever the other half does. The bound
is only used to decide how often to aim at the surface; what it actually emits
is evaluated at the point that was drawn, so a loose bound costs noise and
nothing else.

## Image maps

```toml
[[objects]]
shape = "wavefront"
file = "scene.obj"
material = "principled"
base_color = 'image("oak/base.jpg", scale: 2)'
roughness = 'image("oak/roughness.jpg", scale: 2).r'
metallic = 'image("oak/metalic.jpg", scale: 2).r'
normal = 'normal_map(image("oak/normal.png", scale: 2))'
```

Write these in TOML literal strings (`'…'`) so the quotes around the file name
need no escaping.

| Keyword  | Default  | Meaning                                                                                                                                                                                                                                                                                                 |
| -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `color`  | by field | `"srgb"` decodes the texels as gamma-encoded colour; `"linear"` takes them as the numbers they are. `base_color` and `emission_color` default to `"srgb"`, every other field (and `normal`) to `"linear"`. A roughness map read as sRGB comes out too dark, and a colour map read as linear too bright. |
| `scale`  | `1`      | Multiplies the texture coordinates: `4` repeats the image four times each way, `[2, 1]` twice across and once up.                                                                                                                                                                                       |
| `offset` | `[0, 0]` | Added after the scale.                                                                                                                                                                                                                                                                                  |

- The image is sampled at the mesh's `uv`, so the mesh needs `vt` lines. Past
  the unit square it repeats. It is bilinearly filtered, with no mipmaps: each
  sample jitters within its pixel, which antialiases a distant texture over
  enough samples.
- A channel is the usual suffix: `image("orm.png").g` reads a packed map's
  green.
- Paths are relative to the scene file. PNG, JPEG, EXR and HDR are read, all
  converted to 8 bits a channel; TIFF is not.
- Every image is a layer of one of two texture arrays (sRGB and linear), and
  every layer of an array is resized to the largest image in it, capped at 4096
  on a side. A file used both ways is loaded twice. Budget about 16 MB per 2048²
  image.
- An emissive image is bounded by its strength, so it needs no `remap` — unless
  it reads `.a`, which has no bound.

`normal_map(input, strength: 1, convention: "opengl")` bends the shading normal
by a tangent-space normal map. The tangent follows the direction `u` runs across
each triangle, so mirrored UV islands read correctly. `convention: "directx"`
flips green for maps authored the other way up (if a normal-mapped surface looks
lit from the wrong side, this is why). `strength` leans the result back toward
the mesh's own normal: `0` is no effect, and values above `1` exaggerate the
map. Where the bent normal would reflect a ray into the surface, near
silhouettes, it is bent back as Cycles does, instead of turning black. A mesh
with no texture coordinates ignores the map.

## Generated patterns

```toml
# Marble: a distorted noise, dark only in a thin band around the middle.
base_color = "ramp(noise(object, scale: 5, detail: 8, distortion: 3), 0.42, [0.92, 0.9, 0.86], 0.5, [0.3, 0.28, 0.33], 0.58, [0.92, 0.9, 0.86])"

# Cobbles: a shade of stone per cell, each cell domed.
base_color = 'ramp(voronoi(object, scale: 9, output: "color").r, 0, [0.22, 0.2, 0.18], 1, [0.6, 0.55, 0.47])'
normal = "bump(voronoi(object, scale: 9), distance: -0.025)"
```

`noise(input, …)` is Perlin's improved gradient noise summed over octaves (fBm),
in `[0, 1]`. Its knobs are Blender's Noise Texture's and mean the same thing,
but the lattice hash is not Blender's, so the same numbers draw a different
pattern of the same character. `input` is usually `object`; a scalar input is
read along the diagonal, and `uv` is read at `z = 0`.

| Keyword      | Default | Meaning                                                                                                             |
| ------------ | ------- | ------------------------------------------------------------------------------------------------------------------- |
| `scale`      | `5`     | Multiplies `input`: features per unit.                                                                              |
| `detail`     | `2`     | Octaves past the first, `0` to `15`. A fraction blends the last one in.                                             |
| `roughness`  | `0.5`   | How much of the one before each octave keeps, `0` to `1`. At `1` many octaves average each other back toward `0.5`. |
| `lacunarity` | `2`     | How much finer each octave is than the one before.                                                                  |
| `distortion` | `0`     | How far three more noises push the lookup around first: swirls.                                                     |

`voronoi(input, …)` is Worley's cellular pattern (F1): one feature point per
cell, `scale` cells per unit (default `5`), jittered by `randomness` (`0` to
`1`, default `1`; `0` is a regular grid). `output: "distance"` (the default) is
the distance to the nearest point, in cells, at most `√3`; `output: "color"` is
a random colour fixed per cell; `output: "edge"` is the distance to the nearest
border between two cells, in cells, which is zero along every border — a ramp
over it outlines the cells.

`ramp(input, position, color, …)` takes two to four stops, each a position and a
colour, positions ascending. Between two stops it is linear, and past either end
it holds the end's colour; two stops at one position make a hard edge. Every
position and colour has to work out to a constant (`blackbody(2000) * 0.5` is
fine, a pattern is not).

`bump(height, strength: 1, distance: 1)` tilts the shading normal along the
slope of `height`'s first channel, as Blender's Bump node does: `distance` is
how many world units a height of one stands for (negative inverts it), and
`strength` (`0` to `1`) blends the result back toward the unbumped normal. The
slope is a finite difference, so the height is evaluated three times per hit.
Its step is a thousandth of a unit, scaled up for hits far from the origin. Only
`normal` takes one, and unlike a normal map it needs no texture coordinates.

Each of these is bounded — a noise by `[0, 1]`, a distance by `√3`, an edge
distance by `2`, a ramp by its stops — so any of them can drive emission without
a `remap`. All of them are pure functions of where they are asked: the same on
every sample, bounce and run.

## Named patterns

A pattern that drives several fields, or several objects, can be written once
under `[patterns]` and used by name:

```toml
[patterns]
ground = "noise(object, scale: 1.4, detail: 6, roughness: 0.6)"
dirt = "remap(ground, [0.47, 0.55], [0, 1])"

[[objects]]
shape = "wavefront"
file = "scene.obj"
material = "principled"
base_color = 'mix(image("oak/base.jpg"), [0.07, 0.05, 0.035], dirt)'
roughness = "mix(0.4, 0.9, dirt)"
normal = "bump(dirt, distance: 0.004)"
```

- A name is anything a pattern can't otherwise mean: letters, digits and
  underscores, starting with a letter, and not one of the grammar's own words
  (`uv`, `noise`, …). Names may use other names.
- A name is written out in full wherever it is used, before anything else about
  the scene is checked, so a field using a name is exactly the field with the
  pattern pasted in — including the colour space an image inside it defaults to,
  which is still the field's. It is evaluated once per field that uses it.
- A loop (`a` using `b` using `a`), or a name nothing defines, is an error, as
  is any entry that is wrong whether or not anything uses it.
- A field may come to at most 4096 nodes once its names are written out, so a
  chain of patterns that each use the one before twice is an error rather than a
  tree that doubles with every name.
