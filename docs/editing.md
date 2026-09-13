# Editing a stacked nightscape

A stack is cleaner than any single frame, but it still looks flat: a brown or grey glow near the
horizon, a washed-out sky, stars that compete with the Milky Way. Astro editing is a handful of
fixed moves, applied in a fixed order. This page explains each one and which tool does it.

## The moves, in order

| # | Move | What it fixes | Why the order matters |
|-|-|-|-|
| 1 | **Gradient removal** (background extraction) - `astroclaw blend --gradient` | Light pollution and airglow: the sky brightens and changes colour towards the horizon or one corner | Every later step amplifies whatever gradient is left |
| 2 | **Colour balance** | A sky that is orange, green or grey instead of neutral dark blue | Needs a flat background to judge against |
| 3 | **Noise reduction** | Grain left after stacking | Do it before the stretch pulls the noise up with the signal |
| 4 | **Star separation** | Splits the image into a starless layer and a stars-only layer (StarNet) | Lets steps 5 and 6 treat nebula and stars differently |
| 5 | **Stretch** | Makes faint Milky Way structure visible by lifting the midtones without blowing out stars | Applied to the starless layer, it lifts dust lanes without bloating stars |
| 6 | **Star reduction** | Shrinks or dims stars so the Milky Way core reads first | Recombine the star layer at 50-80 % strength |
| 7 | **Local contrast and saturation** | Dust lanes and the core colour pop | Last, on the final look |
| 8 | **Foreground** - `astroclaw blend --foreground` for noise | Lift shadows, reduce noise, match colour temperature to the sky | Separate from the sky, through the blend mask |

**Only the sky gets moves 1-7.** Gradient removal and stretches assume everything in frame is sky;
run them over a mountain and they treat it as a gradient to flatten.

## Caveat for JPG stacks

Most astro tutorials start from **linear** data (RAW, before any tone curve). A camera JPG is already
gamma-encoded with white balance and contrast baked in. In practice:

- Gradient removal, noise reduction, StarNet and star reduction work normally.
- Stretches need a much lighter hand: the data is already stretched once. Start with curves.
- Photometric colour calibration (`pcc` / `spcc` in Siril) is meaningless on non-linear data. Balance
  colour by eye against a neutral patch of dark sky instead.

## Which tool does what

| Tool | Cost | Good at | Limits for nightscapes |
|-|-|-|-|
| **Siril** (already installed) | free | 1, 3-6 as scriptable commands: `subsky`, `denoise`, `starnet`, `ght`, `satu`, `pm` | Whole-image by default; no masking UI in scripts |
| **GraXpert** | free | AI background extraction and denoise, best-in-class gradient removal; Siril 1.4 can call it | Separate install; processes the whole image |
| **StarNet** | free | Star removal; Siril's `starnet` calls its CLI | Separate install, path set in Siril preferences |
| **AstroWizard** (Lukomatico) | free, beta | Nine guided steps over a stacked TIFF/FITS: crop, gradient removal, colour, sharpen, denoise, stretch, star tools via GraXpert and StarNet | Built for telescope images with no foreground: it will process the ground as sky. Use it on the **sky stack**, not on the blend |
| **Affinity Photo** | paid | Layers and masks: combine a processed sky with the foreground, curves, local contrast, final look | Manual; no star-aware tools of its own |

There is no "RemasterWizard". Lukomatico's two tools are AstroWizard (editing) and StackingWizard
(stacking FITS from smart telescopes; it does not read JPG or camera RAW).

## A workflow that fits astroclaw

1. `astroclaw stack` + `blend` + `finish` → the unedited TIFF, plus `sky_stack.fit` in the workdir.
2. Sky only: open `sky_stack.fit` in AstroWizard or Siril and apply moves 1-7. Export a 16-bit TIFF.
   (The sky stack includes a smeared foreground; ignore it, it gets masked away.)
3. Affinity Photo: put the processed sky over the astroclaw TIFF, mask it with the same horizon or
   object outline (`preview_mask.jpg` shows where it runs), then do move 8 on the foreground layer.

## Siril commands worth knowing

Syntax as printed by `siril-cli` 1.4.4 `help <command>`:

```
subsky { -rbf | degree } [-dither] [-samples=20] [-tolerance=1.0] [-smooth=0.5] [-existing]
denoise [-nocosmetic] [-mod=m] [ -vst | -da3d | -sos=n [-rho=r] ] [-indep]
starnet [-stretch] [-upscale] [-stride=value] [-nostarmask]
ght -D= [-B=] [-LP=] [-SP=] [-HP=] [-clipmode=] [-human | -even | -independent | -sat] [channels]
satu amount [background_factor [hue_range_index]]
pm "expression" [-rescale [low] [high]] [-nosum]
savetif filename [-astro] [-deflate]
```

- `subsky 1` (a degree-1 polynomial) is a safe first gradient removal; `-rbf` follows complex
  gradients but can eat the Milky Way's own glow.
- `ght` strength `-D` between 0 and 10; `-HP` below 1 protects stars from bloating.
- `denoise -mod=0.7` blends 70 % of the denoised result back, which keeps the image from looking plastic.
