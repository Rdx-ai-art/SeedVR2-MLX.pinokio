"""SeedVR2-7B MLX upscaler — web UI (HTML/FastAPI).

A lightweight FastAPI server with a vanilla HTML/JS/CSS frontend for the SeedVR2-7B
one-step super-resolution model running natively on Apple Silicon via MLX (mflux).

Run:  python app.py --port 7860
"""

import argparse
import asyncio
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Optional

import huggingface_hub as hf
import mlx.core as mx
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# mflux's __init__.py chain eagerly imports torch, transformers, etc. (for its
# PyTorch backend). We only use the MLX path, so inject permissive stubs for all
# PyTorch-backend-only deps. Attribute access returns dummy objects (satisfies
# annotations, class inheritance); actual *calls* raise a clear error.
import types

class _StubAttr:
    def __init__(self, module: str, name: str):
        self._module = module
        self._name = name
    def __getattr__(self, name):
        return _StubAttr(self._module, f"{self._name}.{name}")
    def __call__(self, *a, **kw):
        raise RuntimeError(f"{self._module}.{self._name}() called — not installed (MLX-only mode)")
    def __mro_entries__(self, bases):
        return (object,)
    def __getitem__(self, idx):
        return _StubAttr(self._module, f"{self._name}[{idx}]")

def _install_stub(module_name: str):
    if module_name not in sys.modules:
        mod = types.ModuleType(module_name)
        mod.__getattr__ = lambda name: _StubAttr(module_name, name)
        sys.modules[module_name] = mod

for _mod in ("torch", "transformers", "matplotlib", "sentencepiece", "tokenizers"):
    _install_stub(_mod)

from mflux.models.common.vae.tiling_config import TilingConfig

from seedvr2_mlx import SeedVR2MLX

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
MODELS = {
    "SeedVR2-7B-mlx":           {"repo": "benc0/SeedVR2-7B-mlx",            "label": "SeedVR2-7B (fp16)"},
    "SeedVR2-7B-mlx-int8":      {"repo": "benc0/SeedVR2-7B-mlx-int8",       "label": "SeedVR2-7B (int8) · recommended"},
    "SeedVR2-7B-sharp-mlx":     {"repo": "benc0/SeedVR2-7B-sharp-mlx",      "label": "SeedVR2-7B Sharp (fp16)"},
    "SeedVR2-7B-sharp-mlx-int8":{"repo": "benc0/SeedVR2-7B-sharp-mlx-int8", "label": "SeedVR2-7B Sharp (int8)"},
}
DEFAULT_MODEL = "SeedVR2-7B-mlx-int8"

MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "outputs"
UPLOAD_DIR = Path(__file__).parent / "uploads"
STATIC_DIR = Path(__file__).parent / "static"
REQUIRED_FILES = ("config.json", "transformer.safetensors", "vae.safetensors")

# --------------------------------------------------------------------------- #
# Memory limit (auto-calculate from system RAM, before thread is created)
# --------------------------------------------------------------------------- #
def _total_ram_gb() -> float:
    """Total system RAM in GB (best-effort, defaults to 24 if undetectable)."""
    try:
        if sys.platform == "darwin":
            total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        elif sys.platform == "linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024  # kB -> bytes
                        break
                else:
                    total = 24 * 1024**3
        else:
            total = 24 * 1024**3
    except Exception:
        total = 24 * 1024**3
    return total / 1024**3


def _auto_mem_limit_gb(model: Optional[str] = None) -> int:
    """50% of system RAM, capped at 24 GB.

    The floor depends on model precision: 16 GB for fp16 (weights are
    ~14 GB) and 10 GB for int8 (weights are ~7.5 GB). Override with the
    SEEDVR2_MEM_LIMIT_GB env var (e.g. 12) for testing.
    """
    override = os.environ.get("SEEDVR2_MEM_LIMIT_GB")
    if override:
        return max(1, int(override))
    floor = 16 if (model and "fp16" in MODELS[model]["repo"]) else 10
    return max(floor, min(int(_total_ram_gb() * 0.5), 24))


# Current Metal memory limit, in GB. The MLX worker thread applies it at
# startup; /api/memlimit switches it to manual. "mode" tracks whether the
# value is auto (re-derived per model on load) or user-set.
_mem_limit = {"gb": _auto_mem_limit_gb(), "mode": "auto"}

# --------------------------------------------------------------------------- #
# Dedicated MLX thread
# --------------------------------------------------------------------------- #
# MLX streams are thread-local. Gradio's worker pool caused "no Stream in current
# thread" errors. We solve this definitively: ALL MLX work runs on one known thread
# that we initialize at startup. The async event loop stays free for serving the UI.

_mlx_work: Queue[Callable] = Queue()
_mlx_done: Queue[tuple] = Queue()


def _mlx_thread_main():
    """Runs forever on a single thread. All MLX ops are serialized here."""
    mx.eval(mx.array([0.0]))  # force Metal stream creation on THIS thread
    # Set the memory limit ON this thread's stream (it's per-stream, not global).
    limit = _mem_limit["gb"]
    mx.set_memory_limit(limit * 1024**3)
    print(f"Metal memory limit: {limit} GB")
    while True:
        fn = _mlx_work.get()
        try:
            result = fn()
            _mlx_done.put((True, result))
        except Exception as e:
            _mlx_done.put((False, e))


_mlx_thread = threading.Thread(target=_mlx_thread_main, daemon=True, name="mlx-worker")
_mlx_thread.start()


def mlx_run(fn: Callable) -> Any:
    """Submit `fn` to the dedicated MLX thread. Blocks the caller until done."""
    _mlx_work.put(fn)
    ok, value = _mlx_done.get()
    if not ok:
        raise value
    return value


# --------------------------------------------------------------------------- #
# Model management
# --------------------------------------------------------------------------- #
_model_lock = threading.Lock()
_model_state: dict[str, Any] = {"ref": None, "slug": None, "status": "unloaded"}
_output_counter = 0


def model_is_downloaded(slug: str) -> bool:
    d = MODELS_DIR / slug
    return all((d / f).exists() for f in REQUIRED_FILES)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="SeedVR2-7B MLX Upscaler")


@app.middleware("http")
async def add_no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
async def startup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Uploaded source images are transient — wipe them on every startup.
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/models")
async def list_models():
    """Return available models with download/load status."""
    models = []
    for slug, info in MODELS.items():
        models.append({
            "slug": slug,
            "label": info["label"],
            "downloaded": model_is_downloaded(slug),
            "loaded": _model_state["slug"] == slug,
        })
    return {"models": models, "current": _model_state["slug"], "status": _model_state["status"]}


@app.post("/api/models/load")
async def load_model(slug: str = Form(...)):
    """Download (if needed) then load a model.

    Download is async I/O (event loop). Only the MLX model construction
    runs on the dedicated MLX thread.
    """
    if slug not in MODELS:
        raise HTTPException(400, f"Unknown model: {slug}")

    with _model_lock:
        if _model_state["slug"] == slug and _model_state["ref"] is not None:
            return {"status": "ready", "label": MODELS[slug]["label"]}

        # Free any previously loaded model.
        if _model_state["ref"] is not None:
            del _model_state["ref"]
            _model_state["ref"] = None
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, mx.clear_cache)

        # Download (I/O bound — run in executor, NOT on the MLX thread).
        d = MODELS_DIR / slug
        if not model_is_downloaded(slug):
            d.mkdir(parents=True, exist_ok=True)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: hf.snapshot_download(
                    MODELS[slug]["repo"], local_dir=str(d), ignore_patterns=["*.md"]
                ),
            )

        # Load model on the dedicated MLX thread (it respects the active
        # memory limit, so set the limit BEFORE loading for a clean state).
        def _do_load():
            # In auto mode, re-derive the limit for THIS model's precision
            # (fp16 needs a 16 GB floor; int8 can go to 10 GB).
            if _mem_limit["mode"] == "auto":
                limit = _auto_mem_limit_gb(slug)
                mx.set_memory_limit(limit * 1024**3)
                _mem_limit["gb"] = limit
                print(f"Metal memory limit (auto, {slug}): {limit} GB")
            return SeedVR2MLX(str(d))

        loop = asyncio.get_event_loop()
        inst = await loop.run_in_executor(None, lambda: mlx_run(_do_load))

        _model_state["ref"] = inst
        _model_state["slug"] = slug
        _model_state["status"] = "ready"

    return {"status": "ready", "label": MODELS[slug]["label"]}


@app.get("/api/system")
async def system_info():
    """System RAM and current Metal memory limit (for the UI memory picker)."""
    return {
        "ram_gb": round(_total_ram_gb(), 1),
        "mem_limit_gb": _mem_limit["gb"],
        "auto_mem_limit_gb": _auto_mem_limit_gb(),
        "mode": _mem_limit["mode"],
    }


@app.post("/api/memlimit")
async def set_mem_limit(gb: int = Body(..., embed=True)):
    """Set the Metal memory limit at runtime (applied on the MLX thread).

    Affects new allocations only — best set before loading the model.
    """
    gb = max(10, min(gb, 256))

    def _do():
        mx.set_memory_limit(gb * 1024**3)
        mx.eval(mx.array([0.0]))

    mlx_run(_do)
    _mem_limit["gb"] = gb
    _mem_limit["mode"] = "manual"
    return {"mem_limit_gb": gb}


@app.post("/api/upscale")
async def upscale(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    scale_mode: str = Form("factor"),
    scale_factor: float = Form(2.0),
    target_px: int = Form(2048),
    softness: float = Form(0.0),
    input_noise: float = Form(0.0),
    seed: int = Form(42),
    use_tiling: bool = Form(True),
    tile_size: int = Form(512),
    tile_overlap: int = Form(64),
):
    """Run the upscale pipeline. Returns the result as a PNG download."""
    global _output_counter

    if _model_state["slug"] != model or _model_state["ref"] is None:
        raise HTTPException(400, f"Model '{model}' not loaded. Load it first.")
    ref = _model_state["ref"]

    # Save uploaded file (kept for the compare view).
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"upload_{int(time.time()*1000)}{suffix}"
    content = await file.read()
    upload_path.write_bytes(content)

    try:
        # Determine resolution.
        if scale_mode == "target":
            resolution = int(target_px)
        else:
            resolution = f"{scale_factor:g}x"

        # Tiling config.
        tiling = None
        if use_tiling:
            ov = int(tile_overlap)
            tiling = TilingConfig(
                vae_encode_tiled=True,
                vae_encode_tile_size=int(tile_size),
                vae_encode_tile_overlap=ov,
                vae_decode_tiles_per_dim=8,
                vae_decode_tile_size=int(tile_size),
                vae_decode_overlap=ov // 8,
            )

        # Run on the dedicated MLX thread.
        loop = asyncio.get_event_loop()
        t0 = time.time()

        def _do_upscale():
            return ref.upscale(
                str(upload_path),
                resolution=resolution,
                softness=float(softness),
                input_noise_scale=float(input_noise),
                seed=int(seed),
                tiling_config=tiling,
            )

        result = await loop.run_in_executor(None, lambda: mlx_run(_do_upscale))
        elapsed = time.time() - t0
        mx.clear_cache()

        # Save output: input_stem_seedvr_seed_runN.png
        _output_counter += 1
        stem = Path(file.filename or "image").stem
        out_path = OUTPUT_DIR / f"{stem}_seedvr_s{int(seed)}_{_output_counter}.png"
        after = result.image.convert("RGB")
        after.save(out_path)

        # Run metadata as a sidecar JSON next to the image (keeps the PNG clean).
        meta = {
            "software": "SeedVR2 MLX",
            "image": out_path.name,
            "model": {
                "label": MODELS[model]["label"],
                "slug": model,
                "repo": MODELS[model]["repo"],
            },
            "resolution": {"width": after.width, "height": after.height},
            "settings": {
                "scale": str(resolution),
                "softness": float(softness),
                "input_noise": float(input_noise),
                "seed": int(seed),
            },
            "vae_tiling": {
                "enabled": bool(use_tiling),
                "tile_size": int(tile_size) if use_tiling else None,
                "tile_overlap": int(tile_overlap) if use_tiling else None,
            },
            "mem_limit_gb": _mem_limit["gb"],
            "generation_time_s": round(elapsed, 1),
        }
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

        # For the compare view: serve the original upload directly (no extra file).
        before_name = upload_path.name
        return {
            "filename": out_path.name,
            "width": after.width,
            "height": after.height,
            "elapsed": round(elapsed, 1),
            "before_url": f"/api/download/{before_name}",
            "after_url": f"/api/download/{out_path.name}",
            "download_url": f"/api/download/{out_path.name}",
        }
    finally:
        mx.clear_cache()


@app.get("/api/download/{filename}")
async def download(filename: str):
    """Serve a saved file (output or upload) for download/viewing."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        path = UPLOAD_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path, media_type="image/png", filename=filename)


@app.post("/api/download_all")
async def download_all(files: list = Body(..., embed=True)):
    """Zip the given output filenames (from outputs/) into a single download.

    Names are validated to live directly inside outputs/ (no path traversal).
    """
    base = OUTPUT_DIR.resolve()
    buf = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in files:
            if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in (".", ".."):
                continue
            path = (OUTPUT_DIR / name).resolve()
            if path.parent != base or not path.is_file():
                continue
            zf.write(path, arcname=name)
            count += 1
    if count == 0:
        raise HTTPException(404, "No files to download")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="seedvr2_results.zip"'},
    )


@app.get("/api/outputs")
async def outputs_info():
    """Total size + file count of the outputs folder (results + sidecar JSONs)."""
    total = 0
    count = 0
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            total += f.stat().st_size
            count += 1
    return {"size_bytes": total, "count": count}


@app.post("/api/outputs/clear")
async def outputs_clear():
    """Delete every file in outputs/ (all sessions, not just the current one)."""
    deleted = 0
    freed = 0
    for f in OUTPUT_DIR.iterdir():
        if f.is_file():
            freed += f.stat().st_size
            f.unlink()
            deleted += 1
    return {"deleted": deleted, "freed_bytes": freed}


# Static frontend (mounted last so /api/* routes take priority)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f" http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
