# wgsl-raytrace (wip)

A GPU path tracer in Rust, running headless on [wgpu](https://wgpu.rs/) with the
tracing kernel written in [WGSL](https://www.w3.org/TR/WGSL/). It is the GPU
counterpart to [raytrace](https://github.com/theMagicalKarp/raytrace), my CPU
tracer: same TOML scene format, same CLI, a very different back end.

## Requirements

- [mise](https://mise.jdx.dev/) — installs the pinned
  [Rust](https://www.rust-lang.org/) toolchain and runs the project tasks

```Bash
mise install     # install the pinned Rust toolchain
mise run         # test, lint and build a release binary
mise run check   # run the tests and check formatting and linting
mise run fix     # auto-apply formatter and linter suggestions
mise tasks       # list every available task
```

Running the tracer will additionally need a GPU with a working
[WebGPU](https://www.w3.org/TR/webgpu/) backend — Metal, Vulkan, DX12 or GL.

## Usage

```Bash
$ wgsl-raytrace --help
Usage: wgsl-raytrace [OPTIONS] --config <CONFIG>

Options:
  -c, --config <CONFIG>    Path of toml configuration file
  -o, --output <OUTPUT>    Path of file to save the render to [default: render.png]
  -s, --samples <SAMPLES>  Directly override the sample count listed in the configuration file
  -h, --help               Print help
  -V, --version            Print version

$ wgsl-raytrace --config examples/teapot/render.toml --output render.png
┌─── Render Settings ────────────────────────────────────────────────────────────┐
│    Dimensions: 800x600                                                         │
│  Aspect Ratio: 4:3                                                             │
│       Samples: 10000                                                           │
│   Max Bounces: 64                                                              │
│ Field of View: 45                                                              │
│     Look From: [1.55, 0  , 1.9]                                                │
│       Look At: [0  , -0.5, 0  ]                                                │
│           Vup: [0  , 1  , 0  ]                                                 │
│ Defocus Angle: 0                                                               │
│Focus Distance: 1                                                               │
│    Background: [0  , 0  , 0  ]                                                 │
│       Objects: 2                                                               │
│             0: examples/teapot/teapot.obj [Teapot] · metal[0.42, 0.2, 0.7] rou…│
│             1: examples/teapot/teapot.obj [Plane] · lambertian[0.72, 0.72, 0.7…│
└────────────────────────────────────────────────────────────────────────────────┘
```

A malformed scene is reported against the line that caused it:

```Bash
$ wgsl-raytrace --config examples/teapot/render.toml
error: unknown variant `ultrawide`, expected one of `widescreen`, `square`, ...
  --> examples/teapot/render.toml:24:35
   2 | aspect_ratio = "ultrawide"
```

## Scene Configuration Specification

### Camera Configuration

The camera settings define how the scene is viewed and rendered. Below are the
parameters that control the camera's behavior:

```toml
[camera]
aspect_ratio = "standard"
image_width = 800
samples = 10000
max_bounces = 64
fov = 45
look_from = [1.55, 0.0, 1.9]
look_at = [0.0, -0.5, 0.0]
vup = [0.0, 1.0, 0.0]

background = [0.70, 0.80, 0.99]

defocus_angle = 0.6
focus_dist = 10.0
```

- `aspect_ratio`: Specifies the aspect ratio of the rendered image.
  - `widescreen` _(16:9)_
  - `square` _(1:1)_
  - `smartphone` _(9:16)_
  - `standard` _(4:3)_
  - `cinema` _(1.85:1)_
- `image_width`: The width of the rendered image.
- `samples`: The number of samples per pixel, controlling the quality of the
  image. Each sample is a separate GPU dispatch, so this trades render time
  against noise directly.
- `max_bounces`: Limits the number of light bounces for each ray.
- `fov`: The camera's
  [field of view](https://en.wikipedia.org/wiki/Field_of_view) in degrees.
- `look_from`: The coordinates from which the camera views the scene.
- `look_at`: The point the camera is focused on.
- `vup`: Defines the camera's orientation. _(Defaults to `[0.0, 1.0, 0.0]`,
  where y positive is "up")_
- `background`: Set the rgb values for the default color when a ray misses every
  object. _(Defaults to black `[0.0, 0.0, 0.0]`)_
- `defocus_angle`: Variation angle of rays through each pixel _(Defaults to
  being disabled)_
- `focus_dist`: Distance from the camera `look_from` point to the plane of
  perfect focus _(Defaults to being disabled)_

### Objects

A scene is a list of objects, each one a combination of "geometry" and
"material".

### Geometry

#### Wavefront _(.obj file)_

Wavefront meshes are the **only** geometry this tracer supports. Triangles are
the single primitive the shader intersects, so anything that would be a sphere
or a quad in the CPU tracer is authored as a mesh here instead.

```toml
[[objects]]
shape = "wavefront"
file = "teapot.obj"
group = "Teapot"
```

- `file`: Path to the `.obj` file _(relative to the config location)_
- `group`: Optional name of a single `o`/`g` block to load from the file. When
  omitted the whole file is used. This is how one `.obj` gets several materials
  — list it once per block, as `examples/teapot` does.

### Transforms

Each object may carry an ordered list of transforms, applied first to last and
baked into the mesh before it reaches the GPU, so the tracer works entirely in
world space.

```toml
[[objects.transform]]
type = "scale"
scalar = [0.5, 0.5, 0.5]

[[objects.transform]]
type = "rotate"
axis = "y"
degrees = 31.5

[[objects.transform]]
type = "translate"
offset = [0.0, 1.0, 0.0]
```

- `scale`: `scalar` — per-axis scale factors.
- `rotate`: `axis` (`x`, `y` or `z`) and `degrees`.
- `translate`: `offset` — the vector to move by.

### Materials

Materials define the visual properties of the objects. The set is smaller than
the CPU tracer's: the shader carries one scalar per material, so the
texture-backed materials (checkered, image, noise) are absent until there is
somewhere to put them.

#### Lambertian

```toml
[[objects]]
material = "lambertian"
albedo = [1.0, 0.2, 0.3] # red
```

- `albedo`: The diffuse reflection color as an RGB array.

#### Metal

```toml
[[objects]]
material = "metal"
albedo = [0.7, 0.7, 0.7]
roughness = 0.13
```

- `albedo`: The reflective color of the metal.
- `roughness`: Controls the scattering of reflected light. _(The higher, the
  more scattering)_

#### Dielectric

```toml
[[objects]]
material = "dielectric"
refraction_index = 1.5
```

- `refraction_index`: The index of refraction for the material.

#### Glass

```toml
[[objects]]
material = "glass"
```

_This is a dielectric with an index of refraction of `1.5`_

#### Water

```toml
[[objects]]
material = "water"
```

_This is a dielectric with an index of refraction of `1.33`_

#### Light

```toml
[[objects]]
material = "light"
emit = [7.0, 7.0, 7.0]
```

- `emit`: The RGB color of the emitted light.

## Layout

```
src/
  main.rs            CLI entry point: read scene, validate, report
  config/mod.rs      the TOML scene format and its CLI arguments
examples/
  teapot/            a two-object scene, mesh included
```

`config` is deliberately pure data — deserializing a scene touches nothing but
the config file, and `Config::validate` is the separate pass that resolves
object paths against the config's directory. Keeping those apart is what lets
the whole scene format be tested without a GPU, or a mesh, in sight.
