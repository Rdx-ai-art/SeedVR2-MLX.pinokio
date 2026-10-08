# SeedVR2-7B MLX Image Upscaler 🍎

<div align="center">

![Logo](logo.png)

</div>

A one-click Pinokio app that runs the **SeedVR2-7B** one-step diffusion super-resolution
model **natively on Apple Silicon Macs** using **MLX** (through [mflux](https://github.com/mflux-community/mflux)).

>**Fastest and most memory efficient Seedvr2 implementation on Apple Silicon.**

Upload an image, pick a model variant and a few diffusion settings, and get a high-quality upscale — with a live **before/after comparison slider** (zoomable).

>Advanced Options: Custom memory limiter, Vae tile settings.

No PyTorch/CUDA, no Docker. Everything runs on the Apple GPU via MLX.
The web UI is a lightweight **FastAPI + vanilla HTML/JS** app (no Gradio).

## What it does

SeedVR2 is ByteDance's single-step diffusion restoration/super-resolution model. Given a
low-resolution image it produces a sharp, high-resolution result in **one denoising step**
(no text prompt, no multi-step schedule). This app:

- Loads the pre-converted **MLX** weights from [`benc0`](https://huggingface.co/benc0)
- Exposes the real tunable knobs: **scale factor / target size**, **softness** (pre-downsampling), **input noise** (controlled variation, can help with certain artifacts), **seed**, and VAE **tiling** for large images
- Lets you switch between four model variants (see below)
- Controls **Metal memory usage** — an auto limit derived from your RAM by default, or a manual cap you pick in the UI
- Shows the result in an interactive **before/after slider** (drag, scroll to zoom, double-click to reset) and offers a PNG download
- Saves every result to `app/outputs/` with a **sidecar `.json`** recording the full run (model, settings, tiling, memory limit, generation time)

### Model variants

| Variant | Repo | Size | Notes |
|---|---|---|---|
| 7B (int8) — *default* | `benc0/SeedVR2-7B-mlx-int8` | ~9.3 GB | Recommended: best memory efficiency |
| 7B (fp16) | `benc0/SeedVR2-7B-mlx` | ~16.5 GB | Full precision / highest fidelity |
| 7B Sharp (int8) | `benc0/SeedVR2-7B-sharp-mlx-int8` | ~9.3 GB | Sharper detail |
| 7B Sharp (fp16) | `benc0/SeedVR2-7B-sharp-mlx` | ~16.5 GB | Sharper detail, full precision |

The int8 vs fp16 difference is detected automatically from each repo's `config.json`.
Models download **on first use** (not at install) into `app/models/<variant>/`, so a fresh
install is fast. Only one model is held in memory at a time.

> **Requires Apple Silicon (arm64 macOS).** MLX runs on the Apple GPU; this app is not
> available on Intel Macs or other platforms.

## 💻 Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Chip** | M1+ | M1+ Pro/Max or better |
| **RAM** | **16 GB** (with int8 model, upto 2k upscale, lowest mem settings) | **24 GB+** |
| **OS** | macOS 14 (Sonoma) | macOS 15+ (Sequoia) |
| **Storage** | 9 GB Free | 17 GB Free |

## 📝 NOTES
> **On an M1 max 64gb, a 2k upscale (source image 1024 x 1024 px) at default settings, takes ~1.5mins. Both int8 and fp16 models usually takes similar generation time(fp16: +2-3 secs more), difference is in memory use, and slightly better quality/details with the fp16 model**

> I made this app, after many months of torture from using seedvr2 via comfyUI on mac(mps), which is much slower, and uses absurdly high memory, pushing past my 64gb ram,using swap everytime. as per the example above (1024px source image, 2k upscale), the same generation takes around 6-7mins+, and insanely high ram( often 50+gb) on comfy. But i do notice a little better details in comfyUI seedvr renders. I'm estimating MLX fp16 to be around 97-99% similar details, compared to comfyUI seedvr fp16 model. which is a good tradeoff, considering the huge benefits in speed and ram usage.

> Drawthings for mac also offers Seedvr2 upscaler, but i noticed the details are far less quality than this mlx implementation .

## 📦 Installation (One-Click)

1. **Download Pinokio:** [pinokio.computer](https://pinokio.computer)
2. **Copy this Repository URL:** `https://github.com/Rdx-ai-art/seedvr2-mlx.pinokio.git`
3. **Paste into Pinokio:** Discover > Download from URL > paste and select Download.

		>**OR** search directly inside pinokio's app section for this app, Select install.
    
4. **Click Install:** Installs all files, dependencies, selected MLX models will be downloaded during first generation (~9 to ~16 GB)
5. **Click Start:** Launches the web UI

## 📸 Screenshots
!(screenshots/start.png)

!(screenshots/run.png)

!(screenshots/compare.png)

## 🚀 How to use

1. **Install** — Pinokio installs the Python dependencies (mflux + MLX + FastAPI). This
   does *not* download any model weights.
2. **Start** — launches the web UI.
3. In the UI:
   - (Optional) Set the **Metal memory limit** — defaults to **Auto** (derived from your RAM);
     pick a manual value to cap usage. Best set *before* loading the model.
   - Choose a **model** and click **Load Model** (downloads on first use, then loads).
   - Upload an image.
   - Set **Output size mode** (scale factor like `2×`/`3×`, or a target shortest-side in px).
   - Tune **Softness** (`0` off → `1` max pre-downsampling), **Input noise** (`0.01` steps), **Seed**, and (for large images) the **tiling** tile size.
   - Click **Upscale**.
4. Go to the **Compare** tab: drag anywhere to move the slider, **scroll to zoom** (1–8×),
   drag to pan when zoomed, double-click to reset. Then **Download PNG**.
5. All generated images are auto-saved in `app/outputs/`, each with a `.json` file next to
   it containing the settings used (resolution, model, tiling, memory limit, generation time).
   Uploaded source images in `app/uploads/` are wiped on every app start.

### 🕹️ Tips

- **Softness** controls an internal down-then-up sampling that can reduce artifacts on
  very blurry/noisy inputs. Start at `0.5`; raise it for low-quality source images.
- **Input noise** adds controlled variation to the encoded latent. Can reduce certain artifacts.
- **Tiling** (advanced) reduces peak memory for large images. Smaller **tile size** = safer
  peak memory, but slower generation; the default **512** is as fast as no-tiling at normal
  resolutions and bounds memory for large ones. **Tile overlap** blends adjacent tiles to hide
  seams — raise it if you see seams, or set it to `0` for maximum speed.
- **Memory** is kept lean automatically: only one model is held in memory at a time, and the
  VAE encoder/decoder activation buffers are returned to the OS after every stage and request.
  The Metal limit is a *ceiling*, not a target — a single large allocation (e.g. 1024 VAE
  tiles) can briefly exceed it.
- **int8** is the right default for most machines; choose **fp16** only if you have ample
  unified memory and want maximum fidelity.

## 💾 Memory management

The app caps MLX's (per-stream) Metal memory with `mx.set_memory_limit`:

- **Auto** (default): 50% of system RAM, capped at 24 GB. The floor depends on the loaded
  model's precision — **10 GB minimum for int8** (weights ~8.7 GB model + 500mb vae), **18 GB minimum for fp16** (weights ~16.4 GB model) —
  and is re-applied automatically whenever you (re)load a model.
- **Manual**: pick any value (10 GB and up) in the UI's Memory section; it applies instantly
  and stays until you switch back to Auto.

Note the limit is *soft*: MLX reclaims cached buffers when it hits the cap, but a single
allocation larger than the remaining headroom goes through anyway — so peak usage can run
higher over the cap (most visibly during VAE decode).

## 💡 How it works

`app/app.py` is a **FastAPI** server that serves the static UI and a small JSON API. Because
MLX streams are *thread-local*, all MLX work (model load, upscale, memory-limit changes,
cache clears) is serialized onto **one dedicated worker thread** via a work queue — the async
event loop stays free for serving the UI.

`app/seedvr2_mlx.py` is a thin MLX-native wrapper that reuses mflux's building blocks
(`SeedVR2Transformer`, `SeedVR2VAE`, the `seedvr2_euler` scheduler, latent/condition
creators, and image utilities) while loading the pre-converted benc0 weights directly:

```
transformer  ←  nn.quantize(int8) + mx.load("transformer.safetensors")   [or fp16, no quantize]
vae          ←  mx.load("vae.safetensors")
text emb     ←  mx.load("pos_emb.safetensors")  (fixed prompt; mflux bundles a copy)
```

The upscale pipeline mirrors mflux's own `SeedVR2.generate_image`:
preprocess (normalize → pad to /16) → VAE-encode → one denoise step → VAE-decode →
crop → color-correct. Optional VAE tiling (encode + decode) bounds peak memory for
large images.

## 🔧 API Documentation

The web UI is the HTML app above; the same server exposes a JSON API at the base URL.
Assume the server is running at `http://127.0.0.1:7860` (adjust the port).

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/api/models` | GET | — | model list with download/load status |
| `/api/models/load` | POST | form: `slug` | `{"status": "ready", "label": ...}` |
| `/api/system` | GET | — | `ram_gb`, current + auto `mem_limit_gb`, `mode` |
| `/api/memlimit` | POST | JSON: `{"gb": 12}` | `{"mem_limit_gb": 12}` |
| `/api/upscale` | POST | multipart (below) | `filename`, `width`, `height`, `elapsed`, before/after/download URLs |
| `/api/download/{filename}` | GET | — | the file (from `outputs/` or `uploads/`) |

### 🐍 Python

```python
import requests

BASE = "http://127.0.0.1:7860"

# 1) List models / check status
print(requests.get(f"{BASE}/api/models").json())

# 2) Load a model (downloads on first use, idempotent)
r = requests.post(f"{BASE}/api/models/load", data={"slug": "SeedVR2-7B-mlx-int8"})
print(r.json())  # {"status": "ready", "label": "..."}

# 3) Upscale (multipart form)
with open("input.png", "rb") as f:
    r = requests.post(
        f"{BASE}/api/upscale",
        files={"file": ("input.png", f, "image/png")},
        data={
            "model": "SeedVR2-7B-mlx-int8",
            "scale_mode": "factor", "scale_factor": 2.0, "target_px": 2048,
            "softness": 0.0, "input_noise": 0.0, "seed": 42,
            "use_tiling": True, "tile_size": 512, "tile_overlap": 64,
        },
    )
    print(r.json())  # filename, width, height, elapsed, before/after/download URLs
```

### ⌨️ JavaScript

```javascript
const BASE = "http://127.0.0.1:7860";

async function upscale(file, { model = "SeedVR2-7B-mlx-int8",
                               scale = 2, softness = 0, seed = 42, tile = 512 } = {}) {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("scale_mode", "factor");
  form.append("scale_factor", scale);
  form.append("target_px", 2048);
  form.append("softness", softness);
  form.append("input_noise", 0);
  form.append("seed", seed);
  form.append("use_tiling", true);
  form.append("tile_size", tile);
  form.append("tile_overlap", 64);
  const res = await fetch(`${BASE}/api/upscale`, { method: "POST", body: form });
  return res.json();
}
```

### cURL

```bash
# List models
curl -s http://127.0.0.1:7860/api/models | jq

# Load a model
curl -s -X POST http://127.0.0.1:7860/api/models/load -d "slug=SeedVR2-7B-mlx-int8"

# Upscale
curl -s -X POST http://127.0.0.1:7860/api/upscale \
  -F "file=@input.png" -F "model=SeedVR2-7B-mlx-int8" \
  -F "scale_mode=factor" -F "scale_factor=2.0" -F "seed=42" \
  -F "use_tiling=true" -F "tile_size=512" -F "tile_overlap=64"
```

## 📁 Project structure

```
.
├── app/
│   ├── app.py              # FastAPI server: API, static UI, dedicated MLX worker thread
│   ├── seedvr2_mlx.py      # MLX-native SeedVR2 loader + upscale()
│   ├── requirements.txt    # MLX + FastAPI deps (mflux is installed --no-deps in install.js)
│   ├── static/             # web UI (index.html, app.js, style.css)
│   ├── models/             # downloaded weights, per variant
│   ├── uploads/            # uploaded images (auto-deleted on app start)
│   └── outputs/            # generated results + sidecar .json metadata
├── install.js  start.js  reset.js  update.js
├── pinokio.js  pinokio.json
├── README.md   .gitignore
└── icon.png
```

## 📄 License

- **Model weights:** upstream license (derived from bytedance/seedrv2)
- **This app:** Apache-2.0


## 🙏 Credits

- **Model weights**: [`benc0/SeedVR2-7B-mlx*`](https://huggingface.co/benc0) (MLX conversions)
- **Engine**: [mflux](https://github.com/mflux-community/mflux) (filipstrand) — MLX-native model implementations
- **Upstream**: ByteDance SeedVR2 (Apache-2.0) (https://github.com/ByteDance-Seed/SeedVR)
- **Pinokio:** [pinokio.computer](https://pinokio.computer)
