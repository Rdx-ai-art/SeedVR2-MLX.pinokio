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
  setupDropzone();
  setupComparisonSlider();
  setupZoom();
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
  $("#rand-seed").addEventListener("click", () => {
    $("#seed").value = Math.floor(Math.random() * 1000000);
  });
  $("#scale-mode").addEventListener("change", (e) => {
    const isFactor = e.target.value === "factor";
    $("#factor-row").style.display = isFactor ? "" : "none";
    $("#target-row").style.display = isFactor ? "none" : "";
  });
  setupCollapsibles();
  setupSliderLabels();
  setupTargetSync();
});

// --- Tab switching (global, for inline onclick handlers) ---
function switchTab(tab) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
  document.querySelector(`[data-tab="${tab}"]`).classList.add("active");
  document.querySelectorAll(".tab-content").forEach((c) => c.style.display = "none");
  $(`#tab-${tab}`).style.display = "";
  // Un-nest any inner hidden elements (e.g., #comparison inside #tab-compare)
  if (tab === "compare") {
    const cmp = $("#comparison");
    if (cmp) cmp.style.display = "";
  }
}

// --- Models ---
async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const select = $("#model-select");
    select.innerHTML = "";
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m.slug;
      opt.textContent = m.label + (m.downloaded ? "" : " (not downloaded)") + (m.loaded ? " ✓" : "");
      select.appendChild(opt);
    }
    modelLoaded = data.status === "ready";
    updateUpscaleBtn();
  } catch (e) {
    console.error("Failed to load models:", e);
  }
}

async function loadModel() {
  const slug = $("#model-select").value;
  const statusEl = $("#model-status");
  statusEl.textContent = "⬇️ Downloading / loading…";
  statusEl.className = "status";
  $("#load-btn").disabled = true;
  $("#upscale-btn").disabled = true;

  try {
    const form = new FormData();
    form.append("slug", slug);
    const res = await fetch("/api/models/load", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    statusEl.textContent = `✅ Model ready: ${data.label}`;
    statusEl.className = "status ok";
    modelLoaded = true;
  } catch (e) {
    statusEl.textContent = `❌ ${e.message}`;
    statusEl.className = "status";
  } finally {
    $("#load-btn").disabled = false;
    updateUpscaleBtn();
  }
}

// --- Dropzone (multi-file) ---
function setupDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");

  dz.addEventListener("click", (e) => {
    // Ignore clicks on the remove buttons.
    if (e.target.closest(".dz-remove")) return;
    input.click();
  });
  input.addEventListener("change", () => {
    addFiles([...input.files]);
    input.value = ""; // allow re-selecting the same file
  });

  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    addFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/")));
  });
}

// Global for inline onclick handlers.
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
    // Skip duplicates (same name + size).
    if (uploadedFiles.some((x) => x.file.name === f.name && x.file.size === f.size)) continue;
    uploadedFiles.push({ file: f, url: URL.createObjectURL(f) });
  }
  renderThumbs();
}

function renderThumbs() {
  const strip = $("#dropzone-thumbs");
  const hint = $("#dropzone-hint");
  strip.innerHTML = uploadedFiles
    .map(({ file, url }, i) => `
      <div class="dz-thumb" title="${esc(file.name)}">
        <img src="${url}" alt="">
        <button type="button" class="dz-remove" onclick="removeFile(${i})" title="Remove">✕</button>
      </div>`)
    .join("");
  const has = uploadedFiles.length > 0;
  strip.style.display = has ? "flex" : "none";
  hint.style.display = has ? "none" : "";
  updateUpscaleBtn();
}

function updateUpscaleBtn() {
  const n = uploadedFiles.length;
  const btn = $("#upscale-btn");
  btn.disabled = !(n && modelLoaded) || batching;
  btn.textContent = n > 1 ? `🚀 Upscale (${n})` : "🚀 Upscale";
}

// --- Comparison slider + zoom (integrated) ---
// The handle lives inside #cmp-zoom, which is the element that gets the zoom
// transform, so the clip edge, handle line, and images always stay aligned.
// Not zoomed: dragging anywhere moves the slider.
// Zoomed: dragging the image pans, dragging the ⇔ handle moves the slider,
// a plain click places the slider.
let zoomLevel = 1;
let zoomPanX = 0;
let zoomPanY = 0;
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

function clampPan() {
  const rect = cmpRect();
  if (!rect || rect.width === 0) return;
  const mx = ((1 - 1 / zoomLevel) * rect.width) / 2;
  const my = ((1 - 1 / zoomLevel) * rect.height) / 2;
  zoomPanX = Math.max(-mx, Math.min(mx, zoomPanX));
  zoomPanY = Math.max(-my, Math.min(my, zoomPanY));
}

function applyZoom() {
  const wrapper = $("#cmp-zoom");
  const knob = $("#cmp-knob");
  if (!wrapper) return;
  const s = zoomLevel;
  wrapper.style.transform = s === 1 ? "" : `scale(${s}) translate(${zoomPanX}px, ${zoomPanY}px)`;
  // Counter-scale the knob so it stays a constant size on screen.
  if (knob) knob.style.transform = `translate(-50%, -50%) scale(${1 / s})`;
  $("#zoom-level").textContent = `${Math.round(s * 100)}%`;
}

function resetZoom() {
  zoomLevel = 1;
  zoomPanX = 0;
  zoomPanY = 0;
  applyZoom();
}

// Map a cursor screen-X back through the zoom/pan transform into image space.
function setSliderFromClientX(clientX) {
  const rect = cmpRect();
  if (!rect || rect.width === 0) return;
  const s = zoomLevel;
  const ox = rect.width / 2;
  const px = (clientX - rect.left + (s - 1) * ox) / s - zoomPanX;
  const v = Math.max(0, Math.min(100, (px / rect.width) * 100));
  $("#cmp-range").value = v;
  $("#cmp-after").style.clipPath = `inset(0 ${100 - v}% 0 0)`;
  $("#cmp-handle").style.left = v + "%";
}

function panTo(clientX, clientY) {
  zoomPanX += (clientX - panLastX) / zoomLevel;
  zoomPanY += (clientY - panLastY) / zoomLevel;
  panLastX = clientX;
  panLastY = clientY;
  clampPan();
  applyZoom();
}

function beginDrag(clientX, clientY, target) {
  const onKnob = target && target.closest && target.closest(".cmp-handle");
  if (onKnob || zoomLevel <= 1) {
    sliderDragging = true;
    setSliderFromClientX(clientX);
  } else {
    panDragging = true;
    panLastX = panStartX = clientX;
    panLastY = panStartY = clientY;
  }
}

function endDrag(clientX, clientY) {
  if (panDragging && zoomLevel > 1) {
    // A plain click (no drag) places the slider under the cursor.
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

  cmp.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = cmpRect();
    if (!rect || rect.width === 0) return;
    const s = zoomLevel;
    const s2 = Math.max(1, Math.min(8, s + (e.deltaY > 0 ? -0.25 : 0.25)));
    if (s2 === s) return;
    // Keep the image point under the cursor fixed while zooming.
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const ox = rect.width / 2;
    const oy = rect.height / 2;
    const ux = (cx + (s - 1) * ox) / s;
    const uy = (cy + (s - 1) * oy) / s;
    zoomPanX = (cx + (s2 - 1) * ox) / s2 - (ux - zoomPanX);
    zoomPanY = (cy + (s2 - 1) * oy) / s2 - (uy - zoomPanY);
    zoomLevel = s2;
    if (zoomLevel === 1) { zoomPanX = 0; zoomPanY = 0; }
    clampPan();
    applyZoom();
  }, { passive: false });

  cmp.addEventListener("dblclick", () => resetZoom());
}

// --- Collapsibles (global, for inline onclick) ---
function toggleCollapse(el) {
  el.closest(".collapsible").classList.toggle("open");
}
function setupCollapsibles() { /* handled by inline onclick */ }

// --- Slider value labels ---
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

// --- Memory limit (Metal) ---
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

  // Candidate limits, filtered to what this machine can actually spare.
  const autoGb = info.auto_mem_limit_gb;
  const candidates = [10, 12, 16, 20, 24, 32, 48, 64, 80, 96];
  const maxRam = Math.ceil(info.ram_gb);
  let options = candidates.filter((g) => g <= maxRam);
  if (options.length === 0) options = [10];
  if (!options.includes(info.mem_limit_gb)) {
    options = [...options, info.mem_limit_gb].sort((a, b) => a - b);
  }

  // "Auto" is the default; concrete options below it.
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

// --- Range fill: keep the macOS-style filled track in sync with the value ---
function setupRangeFill() {
  document.querySelectorAll('input[type="range"]').forEach((r) => {
    if (r.id === "cmp-range") return; // invisible compare slider
    const update = () => {
      const min = parseFloat(r.min || 0);
      const max = parseFloat(r.max || 100);
      r.style.setProperty("--fill", (((parseFloat(r.value) - min) / (max - min)) * 100) + "%");
    };
    r.addEventListener("input", update);
    update();
  });
}

// --- Target size: sync slider ↔ number input ---
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
  });
  num.addEventListener("change", () => {
    num.value = Math.max(64, Math.min(16384, parseInt(num.value) || 512));
  });
}

// --- Upscale (batch) ---
// Runs every queued file through /api/upscale SEQUENTIALLY — the MLX worker
// thread serializes GPU work anyway, so parallel requests would only pile up.
async function doUpscale() {
  if (!uploadedFiles.length || batching) return;

  const items = [...uploadedFiles];
  batching = true;
  updateUpscaleBtn();

  const progress = $("#progress");
  const progressFill = $("#progress-fill");
  const progressText = $("#progress-text");
  const statusEl = $("#result-status");

  // Snapshot the settings once, so every run in the batch is identical.
  const cfg = {
    scale: $("#scale-mode").value === "factor"
      ? `${parseFloat($("#scale-factor").value)}×`
      : `${$("#target-px-num").value || $("#target-px").value} px`,
    model: ($("#model-select").selectedOptions[0]?.textContent || "").replace(" · recommended", ""),
  };

  progress.style.display = "flex";
  progressFill.style.width = "0%";

  // Dynamic fill: linear, driven by the best time estimate available —
  // the previous run's actual elapsed, scaled by pixel count (inference
  // time is deterministic per resolution/model/tiling). The fill is
  // capped at 98% of the band until the response actually arrives, so a
  // bad estimate can never make the bar look finished early.
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

  // Input pixel count from the already-decoded dropzone thumbnail.
  const itemPx = (i) => {
    const el = $("#dropzone-thumbs").children[i]?.querySelector("img");
    return el && el.naturalWidth ? el.naturalWidth * el.naturalHeight : 0;
  };

  let ok = 0, fail = 0, lastError = null, lastData = null;
  let prevElapsed = null, prevInPx = 0;
  const DEFAULT_EST = 90; // seconds, used for the first item only

  for (let i = 0; i < items.length; i++) {
    const f = items[i].file;
    const bandStart = (i / items.length) * 100;
    const bandEnd = ((i + 1) / items.length) * 100;
    const px = itemPx(i) || prevInPx;
    const est = prevElapsed
      ? (px && prevInPx ? Math.min(600, Math.max(5, prevElapsed * (px / prevInPx))) : prevElapsed)
      : DEFAULT_EST;
    progressText.textContent = `Processing ${i + 1}/${items.length}: ${f.name}`;
    statusEl.textContent = `⏳ ${i + 1}/${items.length} — ${f.name}`;
    startFill(bandStart, bandEnd, est);
    try {
      const data = await upscaleOne(f);
      ok++;
      lastData = data;
      prevElapsed = data.elapsed;
      prevInPx = px;
      library.unshift({ ...data, sourceName: f.name, scale: cfg.scale, model: cfg.model });
      renderLibrary();
      showResult(data);
      // First result: surface the Result pane so progress is visible live.
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
      statusEl.textContent =
        (items.length > 1 ? `✅ ${ok}/${items.length} done — ` : "✅ ") +
        `${lastData.width}×${lastData.height} · ${lastData.elapsed}s → ${lastData.filename}` +
        (items.length > 1 ? " · see Library" : "");
    } else {
      statusEl.textContent = `✅ Done`;
    }
    statusEl.className = "status ok";
  } else {
    statusEl.textContent = `⚠️ ${ok} done, ${fail} failed — ${lastError?.message || ""}`;
    statusEl.className = "status";
  }

  // A batch is best reviewed in the Library; a single run shows its result.
  switchTab(items.length > 1 ? "library" : "result");

  batching = false;
  updateUpscaleBtn();
  setTimeout(() => { progress.style.display = "none"; progressFill.style.width = "0%"; }, 800);
}

// One file through the API. Throws on failure.
async function upscaleOne(file) {
  const form = new FormData();
  form.append("file", file);
  form.append("model", $("#model-select").value);
  form.append("scale_mode", $("#scale-mode").value);
  form.append("scale_factor", $("#scale-factor").value);
  form.append("target_px", $("#target-px-num").value || $("#target-px").value);
  form.append("softness", $("#softness").value);
  form.append("input_noise", $("#input-noise").value);
  form.append("seed", $("#seed").value);
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

// Fill the Result + Compare views with a run's images.
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

  $("#result-tabs").style.display = "flex";
}

// --- Library (session-only: empty on load, fed by every completed run) ---
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
          <span class="lib-sub">${esc(r.scale || "")} · ${esc(r.model || "")}</span>
          <div class="lib-actions" onclick="event.stopPropagation()">
            <button type="button" class="btn icon" onclick="libraryCompare(${i})" title="Compare">↔️</button>
            <a class="btn icon" href="${r.download_url}" download="${esc(r.filename)}" title="Download">⬇️</a>
          </div>
        </div>
      </div>`)
    .join("");
  refreshOutputsSize(); // keep the on-disk size current after each run/clear
}

// Global for inline onclick handlers.
function libraryCompare(i) {
  const r = library[i];
  if (!r) return;
  $("#cmp-before").src = r.before_url;
  $("#cmp-after").src = r.after_url;
  $("#cmp-range").value = 50;
  $("#cmp-after").style.clipPath = "inset(0 50% 0 0)";
  $("#cmp-handle").style.left = "50%";
  resetZoom();
  $("#output-img").src = r.after_url;
  $("#download-link").href = r.download_url;
  $("#download-link").download = r.filename;
  switchTab("compare");
}

// Zip the whole session's results into one download (server-side).
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

// --- Outputs folder: size readout + full clear (all sessions) ---
function fmtBytes(n) {
  if (n >= 1024 ** 3) return (n / 1024 ** 3).toFixed(1) + " GB";
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
    refreshOutputsSize(); // re-enables the button if files remain
  }
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
