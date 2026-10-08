# localgen

**Image → 3D and image generation on a 10 GB GPU, from the command line.**

`localgen` drives [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to run
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (image → textured GLB) and
FLUX.2 klein 4B (text → image, image edit) on a consumer card. It was built
for an indie game's asset pipeline on an RTX 3080 (10 GB) to replace paid
cloud generators, and ships the settings that actually fit in 10 GB.

```bash
localgen gen   concept.png "a wooden drawbridge, three-quarter view, plain background"
localgen edit  turned.png  "turn the object to a three-quarter view" concept.png
localgen trellis turned.png drawbridge.glb
```

Every output gets a `<file>.localgen.json` sidecar recording the generator,
model, seed, settings, timing and licence, so each asset has a provenance trail.

## Why this exists

- **The example workflows don't fit 10 GB.** TRELLIS.2's texturing example
  (1024 conditioning, 2048 texture, quad reconstruction at 1024, 500k faces)
  runs out of memory. Worse, on Windows the driver spills to system RAM before
  failing, so a run crawls for 16 minutes and *then* dies. `localgen` defaults
  to settings that finish: 512 conditioning, 1024 texture, reconstruction at the
  shape's resolution, 200k faces, FP8 weights, low-VRAM loading.
- **No hand-copied node graphs.** ComfyUI workflows saved from the UI are not
  API prompts. `localgen` converts them with the server's own `/object_info`,
  so widget order, seed-control slots and linked widgets are resolved
  correctly instead of being copied by hand and breaking on the next update.
- **Scriptable.** One command per job, a Python API for pipelines, errors that
  name the missing node or file.
- **Commercial-friendly models only.** TRELLIS.2 (MIT) and FLUX.2 klein 4B
  (Apache 2.0). Models with non-commercial or region-restricted licences were
  deliberately left out.

## Measured on an RTX 3080 10 GB

| job | time | notes |
|---|---|---|
| `gen` / `edit`, klein 4B distilled (4 steps) | ~12 s | ~8 GB VRAM |
| `edit`, klein 4B base (20 steps, cfg 5) | ~100 s | slower, follows prompts more closely |
| `trellis` at 512, textured | ~250 s | 185k tris, 7 MB GLB |

Don't run klein while a TRELLIS job is on the card.

### Flat objects come back as cubes

Give TRELLIS.2 a **front-on** picture of a flat object (a door, a shield, a
sign) and it often returns a box: there's no depth cue. What fixed it: turn the
concept to a three-quarter view first with `localgen edit`, then run
`trellis` on the result. A `--side <depth/width>` option that synthesises a
side view is included but is **experimental** — in testing it did not fix the
problem.

### Baked lighting

TRELLIS.2's albedo textures include baked shading. If you relight the asset in
a game engine, plan to grade the texture down.

## Install

Requirements: an NVIDIA GPU with 10 GB+, Python 3.12, and ComfyUI.

1. **ComfyUI** in its own venv. Pin **torch 2.8.0** (`torch==2.8.0+cu128`,
   `xformers==0.0.32.post2`): the prebuilt TRELLIS.2 wheels break on newer torch.
2. **[ComfyUI-Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2)** in
   `ComfyUI/custom_nodes` (it has Windows wheels for torch 2.8.0 / cp312).
3. **Models** under `ComfyUI/models`:

   | path | what |
   |---|---|
   | `visualbruno/TRELLIS.2-4B-FP8/` | TRELLIS.2 FP8, the 10 GB variant (~8 GB) |
   | `microsoft/TRELLIS-image-large/ckpts/` | sparse-structure decoder |
   | `facebook/dinov3-vitl16-pretrain-lvd1689m/` | image encoder |
   | `diffusion_models/flux-2-klein-4b-fp8.safetensors` | klein distilled |
   | `diffusion_models/flux-2-klein-base-4b-fp8.safetensors` | klein base (optional) |
   | `text_encoders/qwen_3_4b.safetensors` | klein text encoder |
   | `vae/flux2-vae.safetensors` | klein VAE |

4. **localgen** itself:

   ```bash
   pip install .
   ```

5. Start ComfyUI with `scripts/start_comfyui.cmd` (Windows) or
   `scripts/start_comfyui.sh`, leave it running, and point `localgen` at it:

   | variable | default | meaning |
   |---|---|---|
   | `COMFY_URL` | `http://127.0.0.1:8188` | the running server |
   | `COMFY_DIR` | `./ComfyUI` | your ComfyUI folder (needed for `trellis`) |
   | `TRELLIS_MODEL` | `visualbruno/TRELLIS.2-4B-FP8` | `microsoft/TRELLIS.2-4B` for bf16 on bigger cards |
   | `TRELLIS_ATTN` / `TRELLIS_SPARSE_ATTN` | `sdpa` / `xformers` | attention backends |

The first `trellis` run downloads rembg's background cutter (~1 GB).

## Usage

```
localgen trellis <image.png> <out.glb> [--res 512|1024|1024_cascade|1536_cascade]
                 [--seed N] [--faces N] [--no-texture] [--tex 1024] [--texres 512]
                 [--back b.png] [--left l.png] [--right r.png] [--side 0.23]
localgen edit <out.png> "<prompt>" <subject.png> [<reference.png> ...] [--base] [--seed N] [--neg "..."]
localgen gen  <out.png> "<prompt>" [--w 1024 --h 1024] [--base] [--seed N] [--neg "..."]
```

- `edit`: the first image is the subject and sets the output size; further
  images are references the prompt can call "image 2", "image 3"...
- `--base`: the 20-step base model instead of the 4-step distilled one.
- PNGs with an alpha channel skip background removal.

From Python:

```python
import localgen
localgen.klein_generate("a stone well, plain background", "well.png", seed=1)
localgen.trellis("well.png", "well.glb", faces=100_000)
```

## Development

```bash
pip install pytest
pytest
```

The tests cover workflow conversion and prompt building and need no GPU or
server.

## Licences

`localgen` is MIT (see `LICENSE`). The models have their own licences:
TRELLIS.2 is MIT; the DINOv3 encoder is under the DINOv3 License; FLUX.2 klein
4B is Apache 2.0. Check them for your use. Whether AI-generated output is
copyrightable depends on your jurisdiction and how much human authorship goes
into the final asset.
