// SeedVR2-7B MLX Upscaler — frontend logic

const $ = (s) => document.querySelector(s);

// --- State ---
let uploadedFiles = [];  // [{ file, url }] — the batch queue
let library = [];        // completed runs this session, newest first
let batching = false;
let modelLoaded = false;

// --- Init ---
document.addEventListener("DOMContentLoaded", async () => {
  await loadModels();
  resumeModelProgress();
  setupDropzone();
  setupComparisonSlider();
  setupZoom();
  setupResultZoom();
  setupCollapsibles();
  setupSliderLabels();
  setupTargetSync();
  setupRangeFill();
  setupMemLimit();

  $("#load-btn").addEventListener("click", loadModel);
  $("#upscale-btn").addEventListener("click", doUpscale);
  $("#download-all-btn").addEventListener("click", downloadAll);
  $("#clear-outputs-btn").addEventListener("click", clearOutputs);
  refreshOutputsSize();

  const pasteBtn = $("#paste-btn");
  if (pasteBtn) pasteBtn.addEventListener("click", pasteFromClipboard);
  window.addEventListener("paste", handlePasteEvent);

  const randEachEl = $("#random-seed-each");
  const syncSeedInputState = () => {
    if (!randEachEl) return;
    $("#seed").style.opacity = randEachEl.checked ? "0.65" : "1";
  };
  if (randEachEl) {
    randEachEl.addEventListener("change", syncSeedInputState);
    syncSeedInputState();
  }

  $("#rand-seed").addEventListener("click", () => {
    $("#seed").value = Math.floor(Math.random() * 1000000);
  });

  $("#scale-mode").addEventListener("change", (e) => {
    const isFactor = e.target.value === "factor";
    $("#factor-row").style.display = isFactor ? "" : "none";
    $("#target-row").style.display = isFactor ? "none" : "";
  });
});

// --- Tab switching (global, for inline onclick handlers) ---
function switchTab(tab) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
  document.querySelector(`[data-tab="${tab}"]`).classList.add("active");
  document.querySelectorAll(".tab-content").forEach((c) => c.style.display = "none");
  $(`#tab-${tab}`).style.display = "";
  if (tab === "compare") {
    const cmp = $("#comparison");
    if (cmp) cmp.style.display = "";
  }
}

// --- Models ---
// Client-side cache of which models are fully downloaded (used so the load
// dialog can tell "download in background" from "load now" without a round-trip).
const downloadedModels = new Set();

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const select = $("#model-select");
    const prev = select.value;
    // Cache which models are fully downloaded (used by the load dialog).
    downloadedModels.clear();
    for (const m of data.models) {
      if (m.downloaded) downloadedModels.add(m.slug);
    }

    select.innerHTML = "";
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m.slug;
      opt.textContent = m.label + (m.downloaded ? "" : " (not downloaded)") + (m.loaded ? " ✓" : "");
      select.appendChild(opt);
    }
    if (prev && [...select.options].some((o) => o.value === prev)) {
      select.value = prev;
    } else if (data.current && [...select.options].some((o) => o.value === data.current)) {
      select.value = data.current;
    }
    modelLoaded = data.status === "ready";
    updateUpscaleBtn();
  } catch (e) {
    console.error("Failed to load models:", e);
  }
}

// Client-side check: is this model fully downloaded? (avoids a round-trip to the
// server so the load dialog can decide "download in background" vs "load now".)
function model_is_downloaded_client(slug) {
  return downloadedModels.has(slug);
}

// --- Model download / load progress (polled from the server) ---
let modelProgressTimer = null;
let modelLoadActive = false;

function stopModelProgress() {
  modelLoadActive = false;
  if (modelProgressTimer) {
    clearInterval(modelProgressTimer);
    modelProgressTimer = null;
  }
  const bar = $("#model-progress");
  if (bar) bar.style.display = "none";
}

function renderModelProgress(p) {
  const bar = $("#model-progress");
  const fill = $("#model-progress-fill");
  const text = $("#model-progress-text");
  if (!bar || !fill || !text) return;

  // New shape: a map keyed by model slug { slug: {phase, slug, bytes_done, ...} }.
  // Fall back to the top-level entry for older backends.
  const isMap = p && typeof p === "object" && !p.phase && Object.keys(p).some(k => p[k] && p[k].phase);
  let entry = null;
  if (isMap) {
    const selectedSlug = $("#model-select").value;
    entry = p[selectedSlug] || null;
    // If the selected model isn't in the map, show whichever download is
    // still running in the background (the user may have switched away from it).
    if (!entry) {
      for (const k of Object.keys(p)) {
        if (p[k] && p[k].phase === "download") { entry = p[k]; break; }
      }
    }
  } else {
    entry = p; // old backend shape: top-level entry
  }

  // No download entry at all -> the download finished.  Show "ready".
  if (!entry) {
    bar.style.display = "flex";
    bar.classList.remove("indeterminate");
    fill.style.width = "100%";
    text.textContent = "✅ Model ready — click Load / Switch Model to switch.";
    return;
  }

  bar.style.display = "flex";

  if (entry.phase === "download") {
    const speedStr = entry.speed_bps > 0 ? ` · ${fmtBytes(entry.speed_bps)}/s` : "";
    const filesStr = entry.files_total > 0 ? ` · ${entry.files_done}/${entry.files_total} files` : "";
    if (entry.bytes_total > 0) {
      bar.classList.remove("indeterminate");
      const pct = Math.min(100, (entry.bytes_done / entry.bytes_total) * 100);
      fill.style.width = pct.toFixed(1) + "%";
      text.textContent = `⬇️ ${pct.toFixed(0)}% · ${fmtBytes(entry.bytes_done)} / ${fmtBytes(entry.bytes_total)}${speedStr}${filesStr}`;
    } else {
      bar.classList.add("indeterminate");
      fill.style.width = "100%";
      text.textContent = entry.bytes_done > 0
        ? `⬇️ ${fmtBytes(entry.bytes_done)}${speedStr}${filesStr}`
        : `⬇️ Preparing download…${filesStr}`;
    }
  } else if (entry.phase === "load" || p.phase === "load" || modelLoadActive) {
    bar.classList.add("indeterminate");
    fill.style.width = "100%";
    text.textContent = p.phase === "load" ? "⏳ Loading model weights into Metal…" : "⬇️ Starting download / load…";
  } else {
    stopModelProgress();
  }
}

async function pollModelProgress() {
  try {
    const res = await fetch("/api/models/progress");
    if (res.ok) {
      const p = await res.json();
      renderModelProgress(p);
    } else if (res.status === 404) {
      renderModelProgress({ phase: "unknown" });
    }
  } catch (e) { /* transient — keep polling */ }
}

async function resumeModelProgress() {
  try {
    const res = await fetch("/api/models/progress");
    if (!res.ok) return;
    const p = await res.json();
    // New shape: a map keyed by slug.  Look for any in-flight entry.
    let active = null;
    if (p && typeof p === "object" && !p.phase && Object.keys(p).some(k => p[k] && p[k].phase)) {
      for (const k of Object.keys(p)) {
        const e = p[k];
        if (e.phase === "download" || e.phase === "load") { active = e; break; }
      }
    } else if (p && p.phase) {
      active = p; // old backend shape
    }
    if (active) {
      modelLoadActive = true;
      $("#model-status").textContent = active.phase === "load"
        ? "⏳ Loading model weights into Metal…"
        : "⬇️ Downloading / loading…";
      $("#load-btn").disabled = true;
      $("#upscale-btn").disabled = true;
      $("#model-progress").style.display = "flex";
      renderModelProgress(p);
      modelProgressTimer = setInterval(pollModelProgress, 500);
    }
  } catch (e) { /* server not ready — ignore */ }
}

async function loadModel() {
  const slug = $("#model-select").value;
  const statusEl = $("#model-status");
  const loadBtn = $("#load-btn");
  const upscaleBtn = $("#upscale-btn");

  loadBtn.disabled = true;

  // --- Download phase (non-blocking, fire-and-forget) ---
  // If the model isn't downloaded yet, kick off the background download and
  // return immediately. The UI stays responsive: the currently-loaded model
  // stays loaded and the user can generate right away. The progress bar
  // keeps polling /api/models/progress until the download finishes.
  if (!model_is_downloaded_client(slug)) {
    modelLoadActive = true;
    const bar = $("#model-progress");
    if (bar) {
      bar.style.display = "flex";
      bar.classList.add("indeterminate");
      $("#model-progress-fill").style.width = "100%";
      $("#model-progress-text").textContent = "⬇️ Connecting to Hugging Face…";
    }
    if (modelProgressTimer) clearInterval(modelProgressTimer);
    modelProgressTimer = setInterval(pollModelProgress, 500);

    try {
      const form = new FormData();
      form.append("slug", slug);
      const res = await fetch("/api/models/load", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();

      if (data.status === "downloading") {
        // Download started in background — return immediately, UI stays live.
        statusEl.textContent = `⬇️ Downloading ${data.label || slug}…`;
        // Re-enable controls right away (generation continues with the
        // currently-loaded model while the download runs in the background).
        loadBtn.disabled = false;
        updateUpscaleBtn();
        return; // <-- fire-and-forget; load phase runs when download finishes
      }
      // If the server returned something else (e.g. already downloaded),
      // fall through to the load phase below.
    } catch (e) {
      statusEl.textContent = `❌ ${e.message}`;
      statusEl.className = "status";
      modelLoadActive = false;
      stopModelProgress();
      loadBtn.disabled = false;
      updateUpscaleBtn();
      return;
    }
  }

  // --- Load phase (blocking — unloads the current model) ---
  // Runs here when the model is already downloaded, or after the background
  // download finishes and the user switches to the newly-available model.
  modelLoadActive = true;
  await doModelLoad(slug, statusEl, loadBtn, upscaleBtn);
}

// Shared load-phase logic (used by loadModel after a download, and by the
// "Load" button when switching to an already-downloaded model).
async function doModelLoad(slug, statusEl, loadBtn, upscaleBtn) {
  const finish = (ok, msg) => {
    statusEl.textContent = msg;
    statusEl.className = ok ? "status ok" : "status";
    modelLoaded = ok;
    modelLoadActive = false;
    stopModelProgress();
    loadBtn.disabled = false;
    updateUpscaleBtn();
  };

  try {
    const form = new FormData();
    form.append("slug", slug);
    const res = await fetch("/api/models/load", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    await loadModels();
    finish(true, `✅ Model ready: ${data.label}`);
  } catch (e) {
    finish(false, `❌ ${e.message}`);
  }
}

// --- Dropzone (multi-file) ---
function setupDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");

  dz.addEventListener("click", (e) => {
    if (e.target.closest(".dz-remove")) return;
    input.click();
  });
  input.addEventListener("change", () => {
    addFiles([...input.files]);
    input.value = "";
  });

  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    addFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/")));
  });
}

function removeFile(i) {
  const item = uploadedFiles[i];
  if (!item) return;
  URL.revokeObjectURL(item.url);
  uploadedFiles.splice(i, 1);
  renderThumbs();
}

function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    if (uploadedFiles.some((x) => x.file.name === f.name && x.file.size === f.size)) continue;
    uploadedFiles.push({ file: f, url: URL.createObjectURL(f) });
  }
  renderThumbs();
}

// --- Clipboard Paste (Button + Cmd/Ctrl+V) ---
async function pasteFromClipboard() {
  const statusEl = $("#input-info");
  try {
    if (!navigator.clipboard || !navigator.clipboard.read) {
      throw new Error("Clipboard API not available. Press ⌘V / Ctrl+V instead.");
    }
    const items = await navigator.clipboard.read();
    const files = [];
    for (const item of items) {
      const imgType = item.types.find((t) => t.startsWith("image/"));
      if (imgType) {
        const blob = await item.getType(imgType);
        const ext = imgType.split("/")[1] || "png";
        const file = new File([blob], `clipboard_${Date.now()}.${ext}`, { type: imgType });
        files.push(file);
      }
    }
    if (!files.length) {
      if (statusEl) {
        statusEl.textContent = "⚠️ No image found in clipboard.";
        statusEl.style.display = "";
      }
      return;
    }
    addFiles(files);
  } catch (err) {
    if (statusEl) {
      statusEl.textContent = `⚠️ Paste failed (${err.message || "permission denied"}) — try pressing ⌘V / Ctrl+V.`;
      statusEl.style.display = "";
    }
  }
}

function handlePasteEvent(e) {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  const cd = e.clipboardData;
  if (!cd) return;

  const files = [];
  for (const item of cd.items || []) {
    if (item.type.startsWith("image/")) {
      const f = item.getAsFile();
      if (f) {
        const ext = item.type.split("/")[1] || "png";
        const namedFile = f.name && f.name !== "image.png"
          ? f
          : new File([f], `clipboard_${Date.now()}.${ext}`, { type: f.type });
        files.push(namedFile);
      }
    }
  }
  if (files.length) {
    e.preventDefault();
    addFiles(files);
  }
}

function renderThumbs() {
  const strip = $("#dropzone-thumbs");
  const hint = $("#dropzone-hint");
  strip.innerHTML = uploadedFiles
    .map(({ file, url }, i) => `
      <div class="dz-thumb" title="${esc(file.name)}">
        <img src="${url}" alt="" onload="thumbDims(this)">
        <span class="dz-dims"></span>
        <button type="button" class="dz-remove" onclick="removeFile(${i})" title="Remove">✕</button>
      </div>`)
    .join("");
  const has = uploadedFiles.length > 0;
  strip.style.display = has ? "flex" : "none";
  hint.style.display = has ? "none" : "";
  updateInputInfo();
  updateUpscaleBtn();
}

function thumbDims(img) {
  if (!img.naturalWidth) return;
  const cap = img.parentElement && img.parentElement.querySelector(".dz-dims");
  if (cap) cap.textContent = `${img.naturalWidth}×${img.naturalHeight}`;
  updateInputInfo();
}

function updateInputInfo() {
  const el = $("#input-info");
  if (!el) return;
  if (!uploadedFiles.length) { el.style.display = "none"; return; }
  if (uploadedFiles.length === 1) {
    const img = $("#dropzone-thumbs") && $("#dropzone-thumbs").querySelector("img");
    const dims = img && img.naturalWidth ? `${img.naturalWidth}×${img.naturalHeight} px` : "…";
    el.textContent = `Input: ${dims} · ${fmtBytes(uploadedFiles[0].file.size)}`;
  } else {
    const total = uploadedFiles.reduce((a, x) => a + x.file.size, 0);
    el.textContent = `${uploadedFiles.length} images · ${fmtBytes(total)}`;
  }
  el.style.display = "";
}

function updateUpscaleBtn() {
  const n = uploadedFiles.length;
  const btn = $("#upscale-btn");
  btn.disabled = !(n && modelLoaded) || batching;
  btn.textContent = n > 1 ? `🚀 Upscale (${n})` : "🚀 Upscale";
}

// --- Zoom/pan controller ---
function makeZoom(wrapper, rectEl, onApply) {
  const st = { level: 1, x: 0, y: 0 };

  function clamp() {
    const rect = rectEl.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const mx = ((1 - 1 / st.level) * rect.width) / 2;
    const my = ((1 - 1 / st.level) * rect.height) / 2;
    st.x = Math.max(-mx, Math.min(mx, st.x));
    st.y = Math.max(-my, Math.min(my, st.y));
  }

  function apply() {
    const s = st.level;
    wrapper.style.transform = s === 1 ? "" : `scale(${s}) translate(${st.x}px, ${st.y}px)`;
    if (onApply) onApply(s);
  }

  return {
    get level() { return st.level; },
    get panX() { return st.x; },
    get panY() { return st.y; },
    reset() { st.level = 1; st.x = 0; st.y = 0; apply(); },
    apply,
    zoomAt(clientX, clientY, delta) {
      const rect = rectEl.getBoundingClientRect();
      if (!rect || rect.width === 0) return;
      const s = st.level;
      const s2 = Math.max(1, Math.min(8, s + delta));
      if (s2 === s) return;
      const cx = clientX - rect.left, cy = clientY - rect.top;
      const ox = rect.width / 2, oy = rect.height / 2;
      const ux = (cx + (s - 1) * ox) / s;
      const uy = (cy + (s - 1) * oy) / s;
      st.x = (cx + (s2 - 1) * ox) / s2 - (ux - st.x);
      st.y = (cy + (s2 - 1) * oy) / s2 - (uy - st.y);
      st.level = s2;
      if (st.level === 1) { st.x = 0; st.y = 0; }
      clamp();
      apply();
    },
    panBy(dx, dy) {
      st.x += dx / st.level;
      st.y += dy / st.level;
      clamp();
      apply();
    },
  };
}

let cmpZoom = null;
let resZoom = null;
let sliderDragging = false;
let panDragging = false;
let panLastX = 0;
let panLastY = 0;
let panStartX = 0;
let panStartY = 0;

function cmpRect() {
  const cmp = $("#comparison");
  return cmp ? cmp.getBoundingClientRect() : null;
}

function resetZoom() {
  if (cmpZoom) cmpZoom.reset();
}

function setSliderFromClientX(clientX) {
  if (!cmpZoom) return;
  const rect = cmpRect();
  if (!rect || rect.width === 0) return;
  const s = cmpZoom.level;
  const px = (clientX - rect.left + (s - 1) * (rect.width / 2)) / s - cmpZoom.panX;
  const v = Math.max(0, Math.min(100, (px / rect.width) * 100));
  $("#cmp-range").value = v;
  $("#cmp-after").style.clipPath = `inset(0 ${100 - v}% 0 0)`;
  $("#cmp-handle").style.left = v + "%";
}

function panTo(clientX, clientY) {
  if (!cmpZoom) return;
  cmpZoom.panBy(clientX - panLastX, clientY - panLastY);
  panLastX = clientX;
  panLastY = clientY;
}

function beginDrag(clientX, clientY, target) {
  const onKnob = target && target.closest && target.closest(".cmp-handle");
  if (onKnob || !cmpZoom || cmpZoom.level <= 1) {
    sliderDragging = true;
    setSliderFromClientX(clientX);
  } else {
    panDragging = true;
    panLastX = panStartX = clientX;
    panLastY = panStartY = clientY;
  }
}

function endDrag(clientX, clientY) {
  if (panDragging && cmpZoom && cmpZoom.level > 1) {
    if (Math.hypot(clientX - panStartX, clientY - panStartY) < 5) {
      setSliderFromClientX(clientX);
    }
  }
  sliderDragging = false;
  panDragging = false;
}

function setupComparisonSlider() {
  const cmp = $("#comparison");
  if (!cmp) return;

  cmp.addEventListener("mousedown", (e) => {
    e.preventDefault();
    beginDrag(e.clientX, e.clientY, e.target);
  });
  window.addEventListener("mousemove", (e) => {
    if (sliderDragging) setSliderFromClientX(e.clientX);
    else if (panDragging) panTo(e.clientX, e.clientY);
  });
  window.addEventListener("mouseup", (e) => endDrag(e.clientX, e.clientY));

  cmp.addEventListener("touchstart", (e) => beginDrag(e.touches[0].clientX, e.touches[0].clientY, e.target), { passive: true });
  window.addEventListener("touchmove", (e) => {
    if (sliderDragging) setSliderFromClientX(e.touches[0].clientX);
    else if (panDragging) panTo(e.touches[0].clientX, e.touches[0].clientY);
  }, { passive: true });
  window.addEventListener("touchend", () => { sliderDragging = false; panDragging = false; });

  $("#cmp-range").addEventListener("input", () => {
    const v = $("#cmp-range").value;
    $("#cmp-after").style.clipPath = `inset(0 ${100 - v}% 0 0)`;
    $("#cmp-handle").style.left = v + "%";
  });
}

function setupZoom() {
  const cmp = $("#comparison");
  if (!cmp) return;

  cmpZoom = makeZoom($("#cmp-zoom"), cmp, (s) => {
    const knob = $("#cmp-knob");
    if (knob) knob.style.transform = `translate(-50%, -50%) scale(${1 / s})`;
    $("#zoom-level").textContent = `${Math.round(s * 100)}%`;
  });

  cmp.addEventListener("wheel", (e) => {
    e.preventDefault();
    cmpZoom.zoomAt(e.clientX, e.clientY, e.deltaY > 0 ? -0.25 : 0.25);
  }, { passive: false });
  cmp.addEventListener("dblclick", () => cmpZoom.reset());
}

function setupResultZoom() {
  const clip = $("#result-zoom-clip");
  if (!clip) return;

  resZoom = makeZoom($("#result-zoom"), clip, (s) => {
    $("#res-zoom-level").textContent = `${Math.round(s * 100)}%`;
  });

  clip.addEventListener("wheel", (e) => {
    e.preventDefault();
    resZoom.zoomAt(e.clientX, e.clientY, e.deltaY > 0 ? -0.25 : 0.25);
  }, { passive: false });
  clip.addEventListener("dblclick", () => resZoom.reset());

  let dragging = false, lastX = 0, lastY = 0;
  clip.addEventListener("mousedown", (e) => {
    if (resZoom.level <= 1) return;
    e.preventDefault();
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    clip.classList.add("dragging");
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    resZoom.panBy(e.clientX - lastX, e.clientY - lastY);
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    clip.classList.remove("dragging");
  });
  clip.addEventListener("touchstart", (e) => {
    if (resZoom.level <= 1 || e.touches.length !== 1) return;
    dragging = true;
    lastX = e.touches[0].clientX;
    lastY = e.touches[0].clientY;
  }, { passive: true });
  window.addEventListener("touchmove", (e) => {
    if (!dragging || e.touches.length !== 1) return;
    resZoom.panBy(e.touches[0].clientX - lastX, e.touches[0].clientY - lastY);
    lastX = e.touches[0].clientX;
    lastY = e.touches[0].clientY;
  }, { passive: true });
  window.addEventListener("touchend", () => { dragging = false; });
}

function resetResultZoom() {
  if (resZoom) resZoom.reset();
}

function toggleCollapse(el) {
  el.closest(".collapsible").classList.toggle("open");
}
function setupCollapsibles() { /* handled by inline onclick */ }

function setupSliderLabels() {
  const bindings = [
    ["#scale-factor", "#factor-val", (v) => `${parseFloat(v).toFixed(1)}`],
    ["#target-px", "#target-val", (v) => `${v} px`],
    ["#softness", "#soft-val", (v) => parseFloat(v).toFixed(2)],
    ["#input-noise", "#noise-val", (v) => parseFloat(v).toFixed(2)],
    ["#tile-overlap", "#tile-ov-val", (v) => v],
  ];
  for (const [sliderSel, labelSel, fmt] of bindings) {
    const slider = $(sliderSel);
    const label = $(labelSel);
    if (slider && label) {
      slider.addEventListener("input", () => { label.textContent = fmt(slider.value); });
    }
  }
}

async function setupMemLimit() {
  const sel = $("#mem-limit");
  const ramInfo = $("#ram-info");
  const status = $("#mem-status");
  if (!sel) return;

  let info;
  try {
    info = await (await fetch("/api/system")).json();
  } catch (e) {
    status.textContent = "Could not read system info.";
    return;
  }

  ramInfo.textContent = `System RAM: ${info.ram_gb} GB`;

  const autoGb = info.auto_mem_limit_gb;
  const candidates = [10, 12, 16, 20, 24, 32, 48, 64, 80, 96];
  const maxRam = Math.ceil(info.ram_gb);
  let options = candidates.filter((g) => g <= maxRam);
  if (options.length === 0) options = [10];
  if (!options.includes(info.mem_limit_gb)) {
    options = [...options, info.mem_limit_gb].sort((a, b) => a - b);
  }

  const isAuto = info.mode === "auto";
  sel.innerHTML =
    `<option value="auto"${isAuto ? " selected" : ""}>Auto — ${autoGb} GB</option>` +
    options
      .map((g) => `<option value="${g}"${!isAuto && g === info.mem_limit_gb ? " selected" : ""}>${g} GB</option>`)
      .join("");

  status.textContent = `Current limit: ${info.mem_limit_gb} GB (${isAuto ? "auto" : "manual"})`;

  sel.addEventListener("change", async () => {
    const gb = sel.value === "auto" ? autoGb : parseInt(sel.value, 10);
    if (gb === info.mem_limit_gb) {
      status.textContent = `Current limit: ${gb} GB (${sel.value === "auto" ? "auto" : "manual"})`;
      return;
    }
    status.textContent = `Applying ${gb} GB…`;
    try {
      const res = await fetch("/api/memlimit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gb }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      status.textContent = `Limit set to ${data.mem_limit_gb ?? gb} GB — set it before loading the model.`;
    } catch (e) {
      status.textContent = `Failed to apply the memory limit (${e.message}). Restart the app if the endpoint is missing.`;
    }
  });
}

function setupRangeFill() {
  document.querySelectorAll('input[type="range"]').forEach((r) => {
    if (r.id === "cmp-range") return;
    const update = () => {
      const min = parseFloat(r.min || 0);
      const max = parseFloat(r.max || 100);
      r.style.setProperty("--fill", (((parseFloat(r.value) - min) / (max - min)) * 100) + "%");
    };
    r.addEventListener("input", update);
    update();
  });
}

function setupTargetSync() {
  const slider = $("#target-px");
  const num = $("#target-px-num");
  const label = $("#target-val");

  slider.addEventListener("input", () => {
    num.value = slider.value;
    label.textContent = `${slider.value} px`;
  });
  num.addEventListener("input", () => {
    const v = Math.max(64, Math.min(16384, parseInt(num.value) || 512));
    slider.value = Math.min(8192, Math.max(512, v));
    label.textContent = `${v} px`;
    slider.dispatchEvent(new Event("input"));
  });
  num.addEventListener("change", () => {
    num.value = Math.max(64, Math.min(16384, parseInt(num.value) || 512));
  });
}

// --- Upscale (batch) ---
async function doUpscale() {
  if (!uploadedFiles.length || batching) return;

  const items = [...uploadedFiles];
  batching = true;
  updateUpscaleBtn();

  const progress = $("#progress");
  const progressFill = $("#progress-fill");
  const progressText = $("#progress-text");
  const statusEl = $("#result-status");

  const cfg = {
    scale: $("#scale-mode").value === "factor"
      ? `${parseFloat($("#scale-factor").value)}×`
      : `${$("#target-px-num").value || $("#target-px").value} px`,
    model: ($("#model-select").selectedOptions[0]?.textContent || "").replace(" · recommended", ""),
  };

  progress.style.display = "flex";
  progressFill.style.width = "0%";

  let fillPct = 0;
  let fillRaf = null;
  let filling = false;
  const startFill = (bandStart, bandEnd, estSeconds) => {
    stopFill();
    filling = true;
    const t0 = performance.now();
    const startPct = fillPct;
    const capPct = bandStart + (bandEnd - bandStart) * 0.98;
    const step = (now) => {
      if (!filling) return;
      const dt = (now - t0) / 1000;
      const frac = Math.min(1, dt / estSeconds);
      fillPct = Math.min(capPct, startPct + (bandEnd - startPct) * frac);
      progressFill.style.width = fillPct + "%";
      fillRaf = requestAnimationFrame(step);
    };
    fillRaf = requestAnimationFrame(step);
  };
  const stopFill = (snapPct) => {
    filling = false;
    if (fillRaf) cancelAnimationFrame(fillRaf);
    if (snapPct != null) fillPct = snapPct;
    progressFill.style.width = fillPct + "%";
  };

  const itemPx = (i) => {
    const el = $("#dropzone-thumbs").children[i]?.querySelector("img");
    return el && el.naturalWidth ? el.naturalWidth * el.naturalHeight : 0;
  };

  let ok = 0, fail = 0, lastError = null, lastData = null;
  let prevElapsed = null, prevInPx = 0;
  const DEFAULT_EST = 90;

  for (let i = 0; i < items.length; i++) {
    const f = items[i].file;
    const bandStart = (i / items.length) * 100;
    const bandEnd = ((i + 1) / items.length) * 100;
    const px = itemPx(i) || prevInPx;
    const est = prevElapsed
      ? (px && prevInPx ? Math.min(600, Math.max(5, prevElapsed * (px / prevInPx))) : prevElapsed)
      : DEFAULT_EST;

    if ($("#random-seed-each")?.checked) {
      $("#seed").value = Math.floor(Math.random() * 1000000);
    }
    const currentSeed = parseInt($("#seed").value, 10) || 42;

    progressText.textContent = `Processing ${i + 1}/${items.length}: ${f.name} (seed ${currentSeed})`;
    statusEl.textContent = `⏳ ${i + 1}/${items.length} — ${f.name}`;
    startFill(bandStart, bandEnd, est);

    try {
      const data = await upscaleOne(f, currentSeed);
      ok++;
      lastData = data;
      prevElapsed = data.elapsed;
      prevInPx = px;
      library.unshift({
        ...data,
        seed: currentSeed,
        sourceName: f.name,
        scale: cfg.scale,
        model: cfg.model,
      });
      renderLibrary();
      showResult(data);
      if (ok === 1) switchTab("result");
    } catch (e) {
      fail++;
      lastError = e;
      statusEl.textContent = `⚠️ ${f.name} failed — ${e.message}`;
    }
    stopFill(bandEnd);
  }

  stopFill(100);
  progressText.textContent = "Done!";
  if (fail === 0) {
    if (lastData) {
      const dims = lastData.in_width
        ? `${lastData.in_width}×${lastData.in_height} → ${lastData.width}×${lastData.height} · `
        : "";
      statusEl.textContent =
        (items.length > 1 ? `✅ ${ok}/${items.length} done — ` : "✅ ") +
        `${dims}${lastData.elapsed}s · seed ${$("#seed").value} → ${lastData.filename}` +
        (items.length > 1 ? " · see Library" : "");
    } else {
      statusEl.textContent = `✅ Done`;
    }
    statusEl.className = "status ok";
  } else {
    statusEl.textContent = `⚠️ ${ok} done, ${fail} failed — ${lastError?.message || ""}`;
    statusEl.className = "status";
  }

  switchTab(items.length > 1 ? "library" : "result");

  batching = false;
  updateUpscaleBtn();
  setTimeout(() => { progress.style.display = "none"; progressFill.style.width = "0%"; }, 800);
}

async function upscaleOne(file, seedOverride) {
  const seedVal = seedOverride != null ? seedOverride : $("#seed").value;
  const form = new FormData();
  form.append("file", file);
  form.append("model", $("#model-select").value);
  form.append("scale_mode", $("#scale-mode").value);
  form.append("scale_factor", $("#scale-factor").value);
  form.append("target_px", $("#target-px-num").value || $("#target-px").value);
  form.append("softness", $("#softness").value);
  form.append("input_noise", $("#input-noise").value);
  form.append("seed", seedVal);
  form.append("use_tiling", $("#use-tiling").checked);
  form.append("tile_size", $("#tile-size").value);
  form.append("tile_overlap", $("#tile-overlap").value);

  const res = await fetch("/api/upscale", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upscale failed");
  }
  return res.json();
}

function showResult(data) {
  $("#output-img").src = data.after_url;
  $("#download-link").href = data.download_url;
  $("#download-link").download = data.filename;

  $("#cmp-before").src = data.before_url;
  $("#cmp-after").src = data.after_url;
  $("#cmp-range").value = 50;
  $("#cmp-after").style.clipPath = "inset(0 50% 0 0)";
  $("#cmp-handle").style.left = "50%";
  resetZoom();
  resetResultZoom();

  $("#result-tabs").style.display = "flex";
}

function renderLibrary() {
  const grid = $("#library-grid");
  $("#library-empty").style.display = library.length ? "none" : "";
  const countEl = $("#library-count");
  const dlAll = $("#download-all-btn");
  countEl.textContent = library.length
    ? `${library.length} image${library.length > 1 ? "s" : ""} this session`
    : "";
  dlAll.disabled = !library.length || dlAll.dataset.busy === "1";
  if (dlAll.dataset.busy !== "1") {
    dlAll.textContent = library.length
      ? `⬇️ Download all (${library.length})`
      : "⬇️ Download all";
  }
  grid.innerHTML = library
    .map((r, i) => `
      <div class="lib-card" onclick="libraryCompare(${i})" title="Click to compare">
        <img src="${r.after_url}" loading="lazy" alt="">
        <div class="lib-meta">
          <span class="lib-name">${esc(r.sourceName)}</span>
          <span class="lib-sub">${r.width}×${r.height} · ${r.elapsed}s</span>
          <span class="lib-sub">${esc(r.scale || "")}${r.seed != null ? ` · s:${r.seed}` : ""} · ${esc(r.model || "")}</span>
          <div class="lib-actions" onclick="event.stopPropagation()">
            <button type="button" class="btn icon" onclick="libraryCompare(${i})" title="Compare">↔️</button>
            <a class="btn icon" href="${r.download_url}" download="${esc(r.filename)}" title="Download">⬇️</a>
          </div>
        </div>
      </div>`)
    .join("");
  refreshOutputsSize();
}

function libraryCompare(i) {
  const r = library[i];
  if (!r) return;
  $("#cmp-before").src = r.before_url;
  $("#cmp-after").src = r.after_url;
  $("#cmp-range").value = 50;
  $("#cmp-after").style.clipPath = "inset(0 50% 0 0)";
  $("#cmp-handle").style.left = "50%";
  resetZoom();
  resetResultZoom();
  $("#output-img").src = r.after_url;
  $("#download-link").href = r.download_url;
  $("#download-link").download = r.filename;
  switchTab("compare");
}

async function downloadAll() {
  if (!library.length) return;
  const btn = $("#download-all-btn");
  const statusEl = $("#result-status");
  const orig = btn.textContent;
  btn.dataset.busy = "1";
  btn.disabled = true;
  btn.textContent = "⏳ Zipping…";

  try {
    const res = await fetch("/api/download_all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ files: library.map((r) => r.filename) }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "seedvr2_results.zip";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    statusEl.textContent = `✅ Downloaded ${library.length} image${library.length > 1 ? "s" : ""} as seedvr2_results.zip`;
    statusEl.className = "status ok";
  } catch (e) {
    statusEl.textContent = `❌ Download all failed — ${e.message}`;
    statusEl.className = "status";
  } finally {
    delete btn.dataset.busy;
    btn.disabled = false;
    btn.textContent = orig;
  }
}

function fmtBytes(n) {
  if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(2) + " GB";
  if (n >= 1024 ** 2) return (n / 1024 ** 2).toFixed(1) + " MB";
  if (n >= 1024) return Math.round(n / 1024) + " KB";
  return n + " B";
}

async function refreshOutputsSize() {
  try {
    const res = await fetch("/api/outputs");
    if (!res.ok) return;
    const data = await res.json();
    $("#library-size").textContent = data.count
      ? `· ≈ ${fmtBytes(data.size_bytes)} total in outputs folder.`
      : "";
    $("#clear-outputs-btn").disabled = data.count === 0;
  } catch (e) { /* non-critical */ }
}

async function clearOutputs() {
  const btn = $("#clear-outputs-btn");
  const statusEl = $("#result-status");
  if (!confirm(
    "Delete ALL files in app/outputs/?\n\n" +
    "This removes every saved result (from ALL sessions), not just this session's library."
  )) return;

  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = "⏳ Clearing…";
  try {
    const res = await fetch("/api/outputs/clear", { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    library.length = 0;
    renderLibrary();
    statusEl.textContent = `🗑 Cleared ${data.deleted} files (${fmtBytes(data.freed_bytes)}) from outputs/`;
    statusEl.className = "status ok";
    refreshOutputsSize();
  } catch (e) {
    statusEl.textContent = `❌ Clear failed — ${e.message}`;
    statusEl.className = "status";
  } finally {
    btn.textContent = orig;
    refreshOutputsSize();
  }
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}