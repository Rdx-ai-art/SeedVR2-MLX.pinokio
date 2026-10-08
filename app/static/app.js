// SeedVR2-7B MLX Upscaler — frontend logic

const $ = (s) => document.querySelector(s);

// --- State ---
let uploadedFile = null;
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

// --- Dropzone ---
function setupDropzone() {
  const dz = $("#dropzone");
  const input = $("#file-input");

  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) setFile(input.files[0]);
  });

  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) setFile(file);
  });
}

function setFile(file) {
  uploadedFile = file;
  const preview = $("#dropzone-preview");
  const hint = $("#dropzone-hint");
  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  hint.style.display = "none";
  updateUpscaleBtn();
}

function updateUpscaleBtn() {
  $("#upscale-btn").disabled = !(uploadedFile && modelLoaded);
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

// --- Upscale ---
async function doUpscale() {
  if (!uploadedFile) return;

  const btn = $("#upscale-btn");
  const progress = $("#progress");
  const progressFill = $("#progress-fill");
  const progressText = $("#progress-text");
  const statusEl = $("#result-status");

  btn.disabled = true;
  progress.style.display = "flex";
  progressFill.style.width = "30%";
  progressText.textContent = "Uploading & processing…";
  statusEl.textContent = "⏳ Working…";

  const form = new FormData();
  form.append("file", uploadedFile);
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

  try {
    const res = await fetch("/api/upscale", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Upscale failed");
    }
    const data = await res.json();

    progressFill.style.width = "100%";
    progressText.textContent = "Done!";

    // Result tab: full-res output
    $("#output-img").src = data.after_url;
    $("#download-link").href = data.download_url;
    $("#download-link").download = data.filename;

    // Compare tab: both at output resolution (input upscaled to match)
    $("#cmp-before").src = data.before_url;
    $("#cmp-after").src = data.after_url;
    $("#cmp-range").value = 50;
    $("#cmp-after").style.clipPath = "inset(0 50% 0 0)";
    $(".cmp-handle").style.left = "50%";
    resetZoom();

    // Show tabs
    $("#result-tabs").style.display = "flex";
    $("#tab-result").style.display = "";

    statusEl.textContent = `✅ ${data.width}×${data.height} · ${data.elapsed}s → ${data.filename}`;
    statusEl.className = "status ok";
  } catch (e) {
    statusEl.textContent = `❌ ${e.message}`;
    statusEl.className = "status";
  } finally {
    btn.disabled = false;
    setTimeout(() => { progress.style.display = "none"; progressFill.style.width = "0%"; }, 800);
  }
}
