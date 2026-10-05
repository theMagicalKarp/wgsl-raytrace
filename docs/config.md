# Config reference

A scene is one TOML file. Paths inside it are resolved relative to the file
itself. Unknown keys are rejected.

## `[camera]`

```toml
[camera]
aspect_ratio = "standard"
image_width = 800
samples = 1000
max_bounces = 64
fov = 30
look_from = [-1.89, 2.29, 10.68]
look_at = [-1.68, 2.39, 4.63]
```

| Key             | Default     | Description                                                                                                                            |
| --------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `aspect_ratio`  | required    | `widescreen` (16:9), `square` (1:1), `smartphone` (9:16), `standard` (4:3), `cinema` (1.85:1).                                         |
| `image_width`   | required    | Width in pixels. Height comes from the aspect ratio.                                                                                   |
| `samples`       | required    | Samples per pixel. Each sample is one GPU dispatch.                                                                                    |
| `max_bounces`   | required    | Path length cap. Russian roulette usually ends paths earlier.                                                                          |
| `fov`           | required    | Vertical field of view, in degrees.                                                                                                    |
| `look_from`     | required    | Camera position.                                                                                                                       |
| `look_at`       | required    | Point the camera faces.                                                                                                                |
| `vup`           | `[0, 1, 0]` | Up vector.                                                                                                                             |
| `defocus_angle` | `0.0`       | Lens cone angle in degrees. `0` is a pinhole (everything in focus).                                                                    |
| `focus_dist`    | `1.0`       | Distance to the plane of perfect focus.                                                                                                |
| `exposure`      | `1.0`       | Linear gain applied just before tone mapping (`2.0` = +1 stop). It has no effect on convergence, the denoiser or the `noise:` figures. |

## `[environment]`

Controls the sky. Rays that escape the scene pick up this radiance. The table is
optional, and without it the sky is black.

```toml
[environment]
file = "sky.exr"
rotation = -90.0
```

| Key         | Default     | Description                                                                                                                                 |
| ----------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `color`     | `[0, 0, 0]` | A flat sky colour.                                                                                                                          |
| `file`      | —           | Equirectangular map (`.exr`, `.hdr`, `.jpg`, `.png`). Overrides `color`. Use HDR formats for real lighting, because LDR maps clip at white. |
| `intensity` | `1.0`       | Multiplier on whichever of `color` or `file` is in use.                                                                                     |
| `rotation`  | `0.0`       | Yaw about +Y in degrees, for moving the sun.                                                                                                |

When `file` is set, the tracer builds a brightness-weighted distribution over
the map's texels and samples it directly (see [Render pipeline](rendering.md)).
A flat `color` is not importance-sampled, because cosine-weighted bounces
already sample a uniform sky exactly.

## `[denoise]`

These settings only take effect with `--denoise`. For how each one is used, see
[Denoising](rendering.md#denoising).

| Key               | Default | Description                                                                          |
| ----------------- | ------- | ------------------------------------------------------------------------------------ |
| `iterations`      | `5`     | Number of à-trous passes. Each pass doubles the tap stride. Maximum `16`.            |
| `sigma_normal`    | `128.0` | Exponent on the normal similarity. Larger is **stricter**.                           |
| `sigma_depth`     | `1.0`   | Depth tolerance, as a multiple of the local depth gradient. Larger is looser.        |
| `sigma_luminance` | `4.0`   | Brightness tolerance, as a multiple of the pixel's standard error. Larger is looser. |

## `[outliers]`

Firefly rejection, applied during tracing. Each sample is checked as it goes
into the accumulator. This is the only setting that can bias the image.

| Key      | Default | Description                                                                                                                     |
| -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `k`      | `0.0`   | A sample whose luminance exceeds `mean + k · max(σ, mean) · n^¼` is scaled down to that threshold, keeping its hue. `0` is off. |
| `warmup` | `32`    | Number of samples a pixel must gather before `k` applies.                                                                       |

Because the threshold widens as `n^¼`, the render still converges to the
unbiased result. It is off by default because testing showed it wasn't worth it.
On `examples/glass` it cut raw error by about 20% but tripled the bias. With
`--denoise`, `k = 0` had both the lowest error and the lowest bias.

## `[patterns]`

```toml
[patterns]
ground = "noise(object, scale: 1.4, detail: 6, roughness: 0.6)"
dirt = "remap(ground, [0.47, 0.55], [0, 1])"
```

Optional. Each entry names a pattern that material fields can then use by name;
see [Named patterns](expressions.md#named-patterns).

## `[[objects]]`

Each object pairs a mesh with a material. Wavefront `.obj` is the only geometry,
and triangles are the only primitive.

```toml
[[objects]]
shape = "wavefront"
file = "scene.obj"
group = "Teapot"
material = "metal"
albedo = [0.7, 0.7, 0.7]
roughness = 0.13

[[objects.transform]]
type = "rotate"
axis = "y"
degrees = 31.5
```

| Key         | Description                                                                                                                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shape`     | Always `"wavefront"`.                                                                                                                                                  |
| `file`      | Path to the `.obj`.                                                                                                                                                    |
| `group`     | Optional name of one `o`/`g` block. Leave it out to load the whole file. To give one file several materials, list it once per group. Group names can't contain spaces. |
| `material`  | The surface model, plus that material's own fields beside it. See [Materials](#materials).                                                                             |
| `transform` | Optional list of `[[objects.transform]]` entries. See [Transforms](#transforms).                                                                                       |

Faces with more than three vertices are split into triangles, and vertices
without normals get the face normal.

### Transforms

`[[objects.transform]]` entries are applied in order, first to last, and baked
into world space at load time.

| `type`      | Fields                                |
| ----------- | ------------------------------------- |
| `scale`     | `scalar = [x, y, z]`                  |
| `rotate`    | `axis = "x" \| "y" \| "z"`, `degrees` |
| `translate` | `offset = [x, y, z]`                  |

### Materials

`material` picks the surface model. Everything is a `principled` surface on the
GPU; the other names are shorthand for common settings of it.

| `material`   | Fields                            | Shorthand for                                                                                                                                                                                                                                        |
| ------------ | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `principled` | See [Principled](#principled)     | —                                                                                                                                                                                                                                                    |
| `lambertian` | `albedo = [r, g, b]`              | `base_color = albedo`, `roughness = 1`, `metallic = 0`, `ior = 1`. Pure diffuse.                                                                                                                                                                     |
| `metal`      | `albedo = [r, g, b]`, `roughness` | `base_color = albedo`, `metallic = 1`.                                                                                                                                                                                                               |
| `dielectric` | `refraction_index`                | `base_color = [1, 1, 1]`, `roughness = 0`, `metallic = 0`, `ior = refraction_index`, `transmission = 1`. Clear glass.                                                                                                                                |
| `glass`      | —                                 | `dielectric` with IOR 1.5.                                                                                                                                                                                                                           |
| `water`      | —                                 | `dielectric` with IOR 1.33.                                                                                                                                                                                                                          |
| `light`      | `emit = [r, g, b]`                | `base_color = [0, 0, 0]`, `ior = 1`, `emission_color = emit`, `emission_strength = 1`. Emits from its front face and reflects nothing. Values above 1 are normal, and `emit` is patternable like `emission_color`: `emit = "blackbody(6500) * 300"`. |

### Principled

A physically based surface modelled on Blender's Principled BSDF: GGX microfacet
reflection over a diffuse base, blended toward rough or smooth glass by
`transmission` and toward a conductor by `metallic`. Every field is optional and
defaults to Blender's, so `material = "principled"` alone is a grey, half-rough
plastic.

| Field                   | Default           | Range     | Patternable                 |
| ----------------------- | ----------------- | --------- | --------------------------- |
| `base_color`            | `[0.8, 0.8, 0.8]` | ≥ 0       | yes                         |
| `roughness`             | `0.5`             | `[0, 1]`  | yes                         |
| `metallic`              | `0.0`             | `[0, 1]`  | yes                         |
| `ior`                   | `1.5`             | ≥ 1       | yes                         |
| `transmission`          | `0.0`             | `[0, 1]`  | yes                         |
| `specular_ior_level`    | `0.5`             | `[0, 1]`  | yes                         |
| `normal`                | none              | —         | `normal_map` or `bump` only |
| `alpha`                 | `1.0`             | `[0, 1]`  | no                          |
| `emission_color`        | `[1, 1, 1]`       | ≥ 0       | yes                         |
| `emission_strength`     | `0.0`             | ≥ 0       | yes                         |
| `subsurface_weight`     | `0.0`             | `[0, 1]`  | yes                         |
| `subsurface_radius`     | `[1, 0.2, 0.1]`   | ≥ 0       | no                          |
| `subsurface_scale`      | `0.05`            | ≥ 0       | no                          |
| `subsurface_anisotropy` | `0.0`             | `[-1, 1]` | no                          |

Patternable fields take a pattern in place of a number; see the
[Expression reference](expressions.md).

- **`base_color`** tints diffuse light, metal reflections and transmitted light.
  Reflections off the dielectric coat are not tinted.
- **`specular_ior_level`** scales the opaque part's reflectance at normal
  incidence as Blender 4 does: `0.5` is exactly what `ior` gives, `0` is no
  specular coat, and `1` doubles it. Glass keeps `ior`.
- **`normal`** takes a [normal map](expressions.md#image-maps) or a
  [bump](expressions.md#generated-patterns).
- **`alpha`** is coverage, not refraction: a ray passes straight through with
  probability `1 - alpha`, at most 32 times per path.
- **Emission** is `emission_color × emission_strength`, given off from the front
  face only (the side its winding faces), on top of whatever the surface
  reflects. Emissive surfaces are sampled as lights.
- **`subsurface_weight`** replaces that much of the diffuse base with a random
  walk under the surface; see [Subsurface scattering](#subsurface-scattering).

### Subsurface scattering

`subsurface_weight` above zero turns that much of the diffuse base into light
that goes _into_ the surface instead of bouncing off it — skin, wax, marble,
milk. It is a volumetric random walk, the same one Cycles runs by default, and
not a diffusion profile: the path really does step around inside the object
until it finds a way out, through the real geometry, so a thin edge glows and a
thick middle does not without either being written down anywhere.

- `subsurface_radius × subsurface_scale` is the mean free path per channel, in
  world units — how far light of that channel travels inside before it scatters.
  Blender's default `[1, 0.2, 0.1]` lets red travel ten times as far as blue,
  which is what makes skin red at its edges. At a scale of zero the walk has
  nowhere to go and the surface is the plain diffuse it replaced.
- `base_color` is the colour the surface has to end up, not the medium's own:
  the shader inverts it (Chiang et al. 2016) to find what the walk has to
  scatter with, because a medium loses a great deal over the dozens of
  scattering events one path takes.
- `subsurface_anisotropy` leans each of those events: back the way it came at
  -1, every direction alike at 0, straight on at 1.

Two limits worth knowing. The mesh has to be **closed** — a path that dives into
an open surface walks away under it and never comes back, so an open mesh loses
what goes in (it never gains any). And a walk is given 256 steps, so a mean free
path far smaller than the object is biased dark; the fix is a larger
`subsurface_scale`, which is also faster.
