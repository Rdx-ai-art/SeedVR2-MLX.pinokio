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
import types
import zipfile
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Optional

import huggingface_hub as hf
import mlx.core as mx
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from tqdm import tqdm

# mflux's __init__.py chain eagerly imports torch, transformers, etc. (for its
# PyTorch backend). We only use the MLX path, so inject permissive stubs for all
# PyTorch-backend-only deps.
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
    "SeedVR2-7B-mlx":            {"repo": "benc0/SeedVR2-7B-mlx",            "label": "SeedVR2-7B (fp16)"},
    "SeedVR2-7B-mlx-int8":       {"repo": "benc0/SeedVR2-7B-mlx-int8",       "label": "SeedVR2-7B (int8) · recommended"},
    "SeedVR2-7B-sharp-mlx":      {"repo": "benc0/SeedVR2-7B-sharp-mlx",      "label": "SeedVR2-7B Sharp (fp16)"},
    "SeedVR2-7B-sharp-mlx-int8": {"repo": "benc0/SeedVR2-7B-sharp-mlx-int8", "label": "SeedVR2-7B Sharp (int8)"},
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
    """50% of system RAM, capped at 24 GB."""
    override = os.environ.get("SEEDVR2_MEM_LIMIT_GB")
    if override:
        return max(1, int(override))
    floor = 16 if (model and "fp16" in MODELS[model]["repo"]) else 10
    return max(floor, min(int(_total_ram_gb() * 0.5), 24))


_mem_limit = {"gb": _auto_mem_limit_gb(), "mode": "auto"}

# --------------------------------------------------------------------------- #
# Dedicated MLX thread
# --------------------------------------------------------------------------- #
_mlx_work: Queue[Callable] = Queue()
_mlx_done: Queue[tuple] = Queue()


def _mlx_thread_main():
    """Runs forever on a single thread. All MLX ops are serialized here."""
    mx.eval(mx.array([0.0]))
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

# --------------------------------------------------------------------------- #
# Model download progress (mirrored from huggingface_hub + live disk delta)
# --------------------------------------------------------------------------- #
_download_lock = threading.Lock()
_download_progress: dict[str, Any] = {}  # keyed by model slug (download phase only)

_speed_window: list[tuple[float, int]] = []

# Module-level global (NOT threading.local) so the download thread running _run
# can read the slug in _mirror_bar.  Set before run_in_executor, cleared after.
_current_download_slug = None


# --------------------------------------------------------------------------- #
# Background model download manager
# Downloads run in a thread-pool executor so they never block the request
# loop and never touch the _model_state (the currently-loaded model stays
# loaded and usable while another model downloads in the background).
# --------------------------------------------------------------------------- #
_download_managers: dict[str, object] = {}


def _start_background_download(slug: str, repo_id: str, loop) -> None:
    """Start a background download for `slug`. The download persists in the
    model cache directory (see `MODELS_DIR`) and huggingface_hub will resume
    it automatically if it is interrupted or the app is quit mid-download."""

    global _current_download_slug  # module-level global so _mirror_bar sees it

    def _run():
        d = MODELS_DIR / slug
        d.mkdir(parents=True, exist_ok=True)

        # Fetch metadata so the progress bar knows the total size.
        try:
            info = hf.HfApi().model_info(repo_id, files_metadata=True)
            siblings = [
                s for s in (info.siblings or [])
                if not s.rfilename.endswith(".md")
            ]
            total_bytes = sum((s.size or 0) for s in siblings)
            with _download_lock:
                _download_progress.setdefault(slug, {
                    "phase": "download",
                    "slug": slug,
                    "bytes_done": 0,
                    "bytes_total": 0,
                    "files_done": 0,
                    "files_total": 0,
                    "speed_bps": 0,
                })
                if total_bytes > 0:
                    _download_progress[slug]["bytes_total"] = total_bytes
                if siblings:
                    _download_progress[slug]["files_total"] = len(siblings)
        except Exception:
            # Still start the download even if we can't fetch metadata.
            with _download_lock:
                _download_progress.setdefault(slug, {
                    "phase": "download",
                    "slug": slug,
                    "bytes_done": 0,
                    "bytes_total": 0,
                    "files_done": 0,
                    "files_total": 0,
                    "speed_bps": 0,
                })

        import huggingface_hub.utils as hf_utils
        orig_tqdm = getattr(hf_utils, "tqdm", None)
        hf_utils.tqdm = _HFTqdm
        try:
            hf.snapshot_download(
                repo_id,
                local_dir=str(d),
                ignore_patterns=["*.md"],
                tqdm_class=_HFTqdm,  # type: ignore[arg-type]
            )
        finally:
            if orig_tqdm is not None:
                hf_utils.tqdm = orig_tqdm

        # Download finished (or failed).  Keep the entry in _download_progress
        # so the UI can show "ready" — the load phase is tracked separately.
        if model_is_downloaded(slug):
            with _download_lock:
                _download_managers.pop(slug, None)

    _current_download_slug = slug  # set BEFORE run_in_executor so the download
    # thread sees the right slug in _mirror_bar (it may start before this line
    # returns)
    future = loop.run_in_executor(None, _run)
    _download_managers[slug] = future


def _update_speed(now_bytes: int) -> int:
    """Compute rolling 3-second download speed in bytes/sec (call inside _download_lock)."""
    now = time.time()
    _speed_window.append((now, now_bytes))
    cutoff = now - 3.0
    while len(_speed_window) > 2 and _speed_window[0][0] < cutoff:
        _speed_window.pop(0)
    if len(_speed_window) >= 2:
        dt = _speed_window[-1][0] - _speed_window[0][0]
        db = _speed_window[-1][1] - _speed_window[0][1]
        if dt > 0.1 and db >= 0:
            return int(db / dt)
    return _download_progress.get("speed_bps", 0)


def _dir_bytes(path: Path) -> int:
    """Sum all file sizes inside `path` (including .cache temp files during download)."""
    if not path.exists():
        return 0
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _mirror_bar(bar: "tqdm", delta: int = 0) -> None:
    """Copy a huggingface_hub progress bar's counters into _download_progress."""
    slug = _current_download_slug
    if not slug:
        return
    desc = (getattr(bar, "desc", None) or "").strip()
    unit = (getattr(bar, "unit", None) or "").strip()
    n = int(getattr(bar, "n", 0) or 0)
    total = int(getattr(bar, "total", 0) or 0)

    with _download_lock:
        entry = _download_progress.get(slug)
        if entry is None:
            return
        if desc.startswith("Fetching"):
            entry["files_done"] = n
            if total > 0:
                entry["files_total"] = total
        elif desc.startswith("Downloading bytes"):
            if n > entry["bytes_done"]:
                entry["bytes_done"] = n
                entry["speed_bps"] = _update_speed(n)
        elif desc.startswith("Reconstructing"):
            if total > entry["bytes_total"]:
                entry["bytes_total"] = total
            if n > entry["bytes_done"]:
                entry["bytes_done"] = n
                entry["speed_bps"] = _update_speed(n)
        elif unit in ("B", "iB") or "B" in unit:
            if delta > 0:
                new_done = entry["bytes_done"] + int(delta)
                if entry["bytes_total"] > 0:
                    new_done = min(new_done, entry["bytes_total"])
                entry["bytes_done"] = new_done
                entry["speed_bps"] = _update_speed(new_done)


class _HFTqdm(tqdm):
    """tqdm subclass passed to hf.snapshot_download so the UI can poll progress."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _mirror_bar(self, 0)

    def update(self, n: Any = 1) -> Any:
        ret = super().update(n)
        _mirror_bar(self, int(n or 0))
        return ret

    def set_description(self, desc=None, refresh=True) -> Any:
        ret = super().set_description(desc, refresh=refresh)
        _mirror_bar(self, 0)
        return ret


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


@app.get("/api/models/progress")
async def model_download_progress():
    """Live model download/load progress, polled by the UI for its progress bar.

    ``_download_progress`` is keyed by model slug, so a download for one model
    is never clobbered when the user loads a different model.  The UI shows the
    entry for the currently selected model, or whichever download is still
    running in the background if the user switched away from it.
    """
    with _download_lock:
        progress = dict(_download_progress)

    # Sync live disk bytes for the in-flight download.
    slug = next((s for s, p in progress.items() if p.get("phase") == "download"), None)
    if slug:
        disk_b = _dir_bytes(MODELS_DIR / slug)
        with _download_lock:
            entry = _download_progress.get(slug)
            if entry is not None and disk_b > entry["bytes_done"]:
                entry["bytes_done"] = disk_b
                entry["speed_bps"] = _update_speed(disk_b)
                progress = dict(_download_progress)

    return progress


@app.post("/api/models/load")
async def load_model(slug: str = Form(...)):
    """Download (in background) then load a model.

    The download phase runs in a thread-pool executor: it never holds
    _model_lock and never touches _model_state, so the currently-loaded
    model stays loaded and usable while another model downloads in the
    background.
    """
    if slug not in MODELS:
        raise HTTPException(400, f"Unknown model: {slug}")

    # ------------------------------------------------------------------ #
    # Download phase: non-blocking, runs in the background.
    # ------------------------------------------------------------------ #
    if not model_is_downloaded(slug):
        repo_id = MODELS[slug]["repo"]
        mgr = _download_managers.get(slug)
        if mgr is not None and not mgr.done():  # type: ignore[union-attr]
            # Already downloading — just report progress.
            return {"status": "downloading", "slug": slug}
        # Download finished (or was never started) — fall through to load.

        # Start a new background download.
        loop = asyncio.get_event_loop()
        _start_background_download(slug, repo_id, loop)

        return {"status": "downloading", "slug": slug, "label": MODELS[slug]["label"]}

    # ------------------------------------------------------------------ #
    # Load phase: model is already downloaded — load it now (this may
    # unload the currently-loaded model, since the user explicitly asked
    # to switch to a fully-available model).
    # ------------------------------------------------------------------ #

    with _model_lock:
        if _model_state["slug"] == slug and _model_state["ref"] is not None:
            return {"status": "ready", "label": MODELS[slug]["label"]}

        if _model_state["ref"] is not None:
            del _model_state["ref"]
            _model_state["ref"] = None
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, mx.clear_cache)

        try:
            d = MODELS_DIR / slug
            if not model_is_downloaded(slug):
                raise HTTPException(500, "Model files missing. Please re-download.")

            def _do_load():
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
        finally:
            with _download_lock:
                _download_managers.pop(slug, None)

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
    """Set the Metal memory limit at runtime (applied on the MLX thread)."""
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

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"upload_{int(time.time()*1000)}{suffix}"
    content = await file.read()
    upload_path.write_bytes(content)

    with Image.open(upload_path) as im:
        in_width, in_height = im.size

    try:
        if scale_mode == "target":
            resolution = int(target_px)
        else:
            resolution = f"{scale_factor:g}x"

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

        _output_counter += 1
        stem = Path(file.filename or "image").stem
        out_path = OUTPUT_DIR / f"{stem}_seedvr_s{int(seed)}_{_output_counter}.png"
        after = result.image.convert("RGB")
        after.save(out_path)

        meta = {
            "software": "SeedVR2 MLX",
            "image": out_path.name,
            "model": {
                "label": MODELS[model]["label"],
                "slug": model,
                "repo": MODELS[model]["repo"],
            },
            "resolution": {"width": after.width, "height": after.height},
            "input": {"width": in_width, "height": in_height},
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

        before_name = upload_path.name
        return {
            "filename": out_path.name,
            "width": after.width,
            "height": after.height,
            "in_width": in_width,
            "in_height": in_height,
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
    """Zip the given output filenames (from outputs/) into a single download."""
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