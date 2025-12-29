// WebGPU version of the CRT shader demo with HDR (extended tone mapping) support.
// Mirrors the WebGL UI/controls and pipes emulator/output canvas frames into a WebGPU texture.
import { wgslSource } from './shader.js';


(() => {
  const canvas = document.getElementById('gl');
  const statusEl = document.getElementById('webgpuStatus');
  const TEST_IMAGE_DIR = 'assets/test-images/';
  const fallbackImageFiles = [
    'smpte-ntsc-320x240.png',
    'smpte-ntsc-640x480.png',
    'ebu-pal-720x576.png',
    'ebu-pal-768x576.png',
    'crosshatch-640x480.png',
    'crosshatch-1280x720.png',
    'rec709-geometry-1920x1080.png',
    'mgs-logo-320x256.png',
  ];

  const sliderSettingIds = [
    'scanline','persistence','beamMin','beamMax','sharp','caX','caY','caEdge','energyLock',
    'bloom','bloomSigma','halation','halationSigma','chromaSpill','chromaSigma','chromaGain',
    'blackSpill','blackSigma','maskStr','maskContrast',
    'maskRound','gamma','hdrGain','hdrHighlights','curvature','vignette','postContrast','postSaturation',
    'spillBias','pixelShape','syncJitter','syncJitterSpeed','rollingBand','rollingBandSpeed','noiseStrength'
  ];

  // WebGPU objects
  let device = null;
  let context = null;
  let uniformBuffer = null;
  let sampler = null;
  let samplerLinear = null;
  let beamPipeline = null;
  let finalPipeline = null;
  let bindGroupLayout = null;
  let pipelineLayout = null;
  let bindGroups = [];
  let sourceTexture = null;
  let historyTextures = [];
  let historyViews = [];
  let historyIndex = 0;
  let historyTexW = 0;
  let historyTexH = 0;
  let historyValid = false;
  let historyFormat = 'rgba16float';
  let lastFrameTime = 0;
  let pipelinesReady = false;
  let sourceTexW = 0;
  let sourceTexH = 0;
  let configuredFormat = null;
  let usedExtendedToneMapping = false;
  const preferredFormat = navigator.gpu ? navigator.gpu.getPreferredCanvasFormat() : 'bgra8unorm';

  // Source / emulator state
  let srcW = 320, srcH = 240;
  let imageSrcW = srcW, imageSrcH = srcH;
  let emulatorSrcW = srcW, emulatorSrcH = srcH;
  let parXVal = 1, parYVal = 1;
  let fitContain = false;
  let dockVisible = true;
  let toggleDockBtn = null;
  let sourceElement = null;
  let imageElement = null;
  let sourceMode = 'image';
  let emulatorStream = null;
  let emulatorVideo = null;
  let builtinSelect = null;
  let currentBuiltinId = null;
  let pendingBuiltinId = null;
  let emulatorCanvas = null;
  let emulatorLoaded = false;
  const builtinImages = new Map();
  const biosEntries = new Map();
  const isoEntries = new Map();
  const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']);
  const BIOS_EXTS = new Set(['.bin', '.rom', '.bios']);
  const ISO_EXTS = new Set(['.bin', '.iso', '.img', '.pbp', '.exe', '.psx', '.cue', '.chd']);
  const preferDirectEmulatorCapture = true;
  let needsReconfigure = true;
  const PRESET_KEY = 'crtShader.webgpu.presets.v1';
  const LAST_KEY = 'crtShader.webgpu.last.v1';
  const AUTOSAVE_KEY = 'crtShader.webgpu.autosave.v1';
  let presets = [];
  let activePreset = null;
  let autosaveEnabled = true;
  let autosaveTimer = null;
  let defaultSettings = null;
  const gpuDebug = new URLSearchParams(window.location.search).has('gpuDebug');

  try {
    const storedAutosave = localStorage.getItem(AUTOSAVE_KEY);
    if (storedAutosave !== null) autosaveEnabled = storedAutosave === '1';
  } catch (err) {
    console.warn('Failed to read autosave preference', err);
  }

  const params = {
    scanline: 0.36,
    persistence: 0.55,
    beamMin: 0.92,
    beamMax: 1.70,
    sharp: 1.15,
    maskStr: 0.55,
    maskType: 0,
    gamma: 1.8,
    bloom: 0.25,
    bloomSigma: 3.1,
    halation: 0.2,
    halationSigma: 6.0,
    caX: 0.03,
    caY: 0.0,
    caEdge: 0.8,
    maskContrast: 1.1,
    maskRound: 0.67,
    chromaSpill: 1.68,
    chromaSigma: 2.8,
    chromaGain: 1.77,
    blackSpill: 0.72,
    blackSigma: 5.9,
    energyLock: 0.77,
    curvature: 0.12,
    vignette: 0.25,
    hdrGain: 1.0,
    hdrHighlights: 1.0,
    postContrast: 1.08,
    postSaturation: 0.69,
    spillBias: -0.06,
    syncJitter: 0.06,
    syncJitterSpeed: 1.2,
    rollingBand: 0.18,
    rollingBandSpeed: 0.35,
    noiseStrength: 0.08,
    bypass: 0.0,
    time: 0.0,
    debugHDR: 0.0,
  };

  const uniformData = new Float32Array(48); // 12 vec4 slots

  function setStatus(text, tone = '') {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.style.color = tone === 'bad' ? '#ff8080' : tone === 'warn' ? '#ffd38a' : 'var(--muted)';
  }

  function resize() {
    const stage = document.getElementById('stage');
    const dpr = Math.min(window.devicePixelRatio || 1, 2.0);
    let ww = stage.clientWidth;
    let hh = stage.clientHeight || window.innerHeight;
    const effW = srcW * parXVal;
    const effH = srcH * parYVal;

    let scale = fitContain
      ? Math.min(ww / effW, hh / effH)
      : Math.max(1, Math.floor(Math.min(ww / effW, hh / effH)));

    const outW = effW * scale;
    const outH = effH * scale;
    canvas.style.width = `${outW}px`;
    canvas.style.height = `${outH}px`;
    const displayW = Math.max(1, Math.round(outW * dpr));
    const displayH = Math.max(1, Math.round(outH * dpr));
    canvas.width = displayW;
    canvas.height = displayH;
    needsReconfigure = true;
  }

  function resizeCanvasToDisplaySize() {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      return true;
    }
    return false;
  }

  function computeHistoryDecay() {
    return [0, 0, 0];
  }

  function updateUniformBuffer(deltaTime, resetHistory) {
    const [decayR, decayG, decayB] = resetHistory ? [0, 0, 0] : computeHistoryDecay();
    uniformData[0] = srcW; // v0
    uniformData[1] = srcH;
    uniformData[2] = canvas.width;
    uniformData[3] = canvas.height;

    uniformData[4] = parXVal; // v1
    uniformData[5] = parYVal;
    uniformData[6] = params.scanline;
    uniformData[7] = params.beamMin;

    uniformData[8] = params.beamMax; // v2
    uniformData[9] = params.sharp;
    uniformData[10] = params.maskStr;
    uniformData[11] = params.gamma;

    uniformData[12] = params.bloom; // v3
    uniformData[13] = params.bloomSigma;
    uniformData[14] = params.halation;
    uniformData[15] = params.halationSigma;

    uniformData[16] = params.caX; // v4
    uniformData[17] = params.caY;
    uniformData[18] = params.caEdge;
    uniformData[19] = params.energyLock;

    uniformData[20] = params.maskContrast; // v5
    uniformData[21] = decayR;
    uniformData[22] = decayG;
    uniformData[23] = decayB;

    uniformData[24] = params.maskRound; // v6
    uniformData[25] = params.chromaSpill;
    uniformData[26] = params.chromaSigma;
    uniformData[27] = params.chromaGain;

    uniformData[28] = params.blackSpill; // v7
    uniformData[29] = params.blackSigma;
    uniformData[30] = params.curvature;
    uniformData[31] = params.vignette;

    uniformData[32] = params.postContrast; // v8
    uniformData[33] = params.postSaturation;
    uniformData[34] = params.persistence;
    uniformData[35] = params.spillBias;

    uniformData[36] = params.time; // v9
    uniformData[37] = params.syncJitter;
    uniformData[38] = params.syncJitterSpeed;
    uniformData[39] = params.rollingBand;

    uniformData[40] = params.rollingBandSpeed; // v10
    uniformData[41] = params.noiseStrength;
    uniformData[42] = params.bypass;
    uniformData[43] = params.maskType;

    uniformData[44] = params.debugHDR ? 1.0 : 0.0; // v11
    uniformData[45] = params.hdrGain;
    uniformData[46] = params.hdrHighlights;
    uniformData[47] = historyFormat === 'rgba16float' ? 0.0 : 1.0;

    device.queue.writeBuffer(uniformBuffer, 0, uniformData.buffer);
  }

  function setSrcSize(w, h, mode = sourceMode, updateInputs = true) {
    if (!w || !h) return;
    srcW = w;
    srcH = h;
    historyValid = false;
    if (mode === 'image') {
      imageSrcW = w;
      imageSrcH = h;
    } else if (mode === 'emulator') {
      emulatorSrcW = w;
      emulatorSrcH = h;
    }
    if (updateInputs) {
      const wEl = document.getElementById('srcW');
      const hEl = document.getElementById('srcH');
      if (wEl) wEl.value = w;
      if (hEl) hEl.value = h;
    }
  }

  function setStoredSrcSize(mode, w, h) {
    if (!w || !h) return;
    if (mode === 'image') {
      imageSrcW = w;
      imageSrcH = h;
    } else if (mode === 'emulator') {
      emulatorSrcW = w;
      emulatorSrcH = h;
    }
  }

  function ensureSourceTexture(w, h) {
    if (!device) return;
    if (sourceTexture && sourceTexW === w && sourceTexH === h) return;
    if (sourceTexture && sourceTexture.destroy) sourceTexture.destroy();
    sourceTexture = device.createTexture({
      size: { width: w, height: h },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    sourceTexW = w;
    sourceTexH = h;
    updateBindGroups();
  }

  function ensureHistoryTextures(w, h) {
    if (!device) return;
    if (historyTextures[0] && historyTexW === w && historyTexH === h) return;
    historyTextures.forEach((tex) => tex && tex.destroy && tex.destroy());
    historyTextures = [];
    historyViews = [];
    for (let i = 0; i < 2; i += 1) {
      const tex = device.createTexture({
        size: { width: w, height: h },
        format: historyFormat,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      historyTextures.push(tex);
      historyViews.push(tex.createView());
    }
    historyTexW = w;
    historyTexH = h;
    historyIndex = 0;
    historyValid = false;
    lastFrameTime = 0;
    updateBindGroups();
  }

  function updateBindGroups() {
    if (!device || !sourceTexture || !sampler || !samplerLinear || !bindGroupLayout) return;
    if (!historyViews[0] || !historyViews[1]) return;
    const sourceView = sourceTexture.createView();
    bindGroups = historyViews.map((view) => device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: sampler },
        { binding: 2, resource: samplerLinear },
        { binding: 3, resource: sourceView },
        { binding: 4, resource: view },
      ],
    }));
  }

  function copySourceToTexture() {
    if (!device || !sourceElement || !sourceTexture) return;
    const w = sourceElement.naturalWidth || sourceElement.videoWidth || sourceElement.width;
    const h = sourceElement.naturalHeight || sourceElement.videoHeight || sourceElement.height;
    if (!w || !h) return;
    ensureSourceTexture(w, h);
    try {
      device.queue.copyExternalImageToTexture(
        { source: sourceElement, flipY: false },
        { texture: sourceTexture },
        { width: w, height: h }
      );
    } catch (err) {
      console.warn('copyExternalImageToTexture failed', err);
    }
  }

  function uploadElement(el) {
    const w = el.naturalWidth || el.width;
    const h = el.naturalHeight || el.height;
    if (!w || !h) {
      console.warn('Source element has no dimensions');
      return;
    }
    imageElement = el;
    setStoredSrcSize('image', w, h);
    if (sourceMode === 'image') {
      sourceElement = el;
      setSrcSize(w, h, 'image');
      ensureSourceTexture(w, h);
      resize();
    }
    scheduleAutosave();
  }

  function syncDockVisibility() {
    if (emulatorDock) {
      const showDock = sourceMode === 'emulator' && dockVisible;
      emulatorDock.classList.toggle('dock-hidden', !showDock);
      emulatorDock.setAttribute('aria-hidden', showDock ? 'false' : 'true');
      if (toggleDockBtn) {
        toggleDockBtn.disabled = sourceMode !== 'emulator';
        toggleDockBtn.textContent = showDock ? 'Dock: On' : 'Dock: Off';
      }
    } else if (toggleDockBtn) {
      toggleDockBtn.disabled = true;
    }
  }

  function setSourceMode(mode) {
    if (mode !== 'image' && mode !== 'emulator') return;
    sourceMode = mode;
    document.querySelectorAll('[data-source]').forEach((btn) => {
      const isActive = btn.dataset.source === mode;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
      btn.tabIndex = isActive ? 0 : -1;
    });
    const imgPanel = document.getElementById('sourceImagePanel');
    const emuPanel = document.getElementById('sourceEmulatorPanel');
    if (imgPanel) {
      imgPanel.hidden = mode !== 'image';
      imgPanel.setAttribute('aria-hidden', mode === 'image' ? 'false' : 'true');
    }
    if (emuPanel) {
      emuPanel.hidden = mode !== 'emulator';
      emuPanel.setAttribute('aria-hidden', mode === 'emulator' ? 'false' : 'true');
    }
    syncDockVisibility();
    const srcWInput = document.getElementById('srcW');
    const srcHInput = document.getElementById('srcH');
    if (srcWInput) srcWInput.disabled = mode !== 'image';
    if (srcHInput) srcHInput.disabled = mode !== 'image';

    if (mode === 'image') {
      if (imageElement) {
        sourceElement = imageElement;
        const nextW = imageSrcW || imageElement.naturalWidth || imageElement.width || srcW;
        const nextH = imageSrcH || imageElement.naturalHeight || imageElement.height || srcH;
        setSrcSize(nextW, nextH, 'image');
      } else if (imageSrcW && imageSrcH) {
        setSrcSize(imageSrcW, imageSrcH, 'image');
      }
    } else {
      if (emulatorCanvas) {
        const dims = readEmulatorDimensions();
        if (dims && dims.w && dims.h) {
          setSrcSize(dims.w, dims.h, 'emulator');
        } else if (emulatorSrcW && emulatorSrcH) {
          setSrcSize(emulatorSrcW, emulatorSrcH, 'emulator');
        }
        const videoReady = emulatorVideo && emulatorVideo.videoWidth && emulatorVideo.videoHeight;
        sourceElement = videoReady ? emulatorVideo : emulatorCanvas;
        syncEmulatorDimensions();
      } else {
        sourceElement = null;
        if (emulatorSrcW && emulatorSrcH) {
          setSrcSize(emulatorSrcW, emulatorSrcH, 'emulator');
        }
        setEmulatorStatus('Emulator not loaded yet.', false);
      }
    }
    resize();
    if (mode === 'emulator') focusEmulatorInput();
    scheduleAutosave();
  }

  function clampPixelShape(v) {
    return Math.max(-0.9, Math.min(0.9, v));
  }

  function applyPixelShapeBias(bias) {
    const clamped = clampPixelShape(bias);
    const xVal = Math.max(0.1, 1 + clamped);
    const yVal = Math.max(0.1, 1 - clamped);
    parXVal = xVal;
    parYVal = yVal;
    const parX = document.getElementById('parX');
    const parY = document.getElementById('parY');
    if (parX) parX.value = xVal.toFixed(3);
    if (parY) parY.value = yVal.toFixed(3);
    updateRangeDisplay('pixelShape', clamped);
    resize();
    scheduleAutosave();
  }

  function applyPAR() {
    const parX = document.getElementById('parX');
    const parY = document.getElementById('parY');
    if (!parX || !parY) return;
    parXVal = parseFloat(parX.value) || 1;
    parYVal = parseFloat(parY.value) || 1;
    const bias = clampPixelShape((parXVal - parYVal) / Math.max(0.01, parXVal + parYVal));
    const pixelShapeSlider = document.getElementById('pixelShape');
    if (pixelShapeSlider) pixelShapeSlider.value = bias.toFixed(2);
    updateRangeDisplay('pixelShape', bias);
    resize();
    scheduleAutosave();
  }

  function updateRangeDisplay(id, value) {
    const span = document.getElementById(id + 'V');
    if (span) span.textContent = Number(value).toFixed(2);
  }

  function bindRange(id, on) {
    const el = document.getElementById(id);
    if (!el) return;
    const update = () => {
      const val = parseFloat(el.value);
      updateRangeDisplay(id, isFinite(val) ? val : 0);
      on(val);
      scheduleAutosave();
    };
    el.addEventListener('input', update);
    update();
  }

  function normalizeLabels() {
    document.querySelectorAll('label').forEach((label) => {
      const textNodes = Array.from(label.childNodes).filter(
        (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim() !== ''
      );
      textNodes.forEach((node) => {
        const span = document.createElement('span');
        span.className = 'label-text';
        span.textContent = node.textContent.trim();
        label.insertBefore(span, node);
        label.removeChild(node);
      });
      const firstSpan = label.querySelector('span:not([id])');
      if (firstSpan && !firstSpan.classList.contains('label-text')) {
        firstSpan.classList.add('label-text');
      }
      const hasValue = !!label.querySelector('span[id$="V"]');
      label.classList.toggle('no-value', !hasValue);
      label.classList.toggle('has-checkbox', !!label.querySelector('input[type="checkbox"]'));
    });
  }

  function getExtension(name) {
    if (!name) return '';
    const idx = name.lastIndexOf('.');
    return idx >= 0 ? name.slice(idx).toLowerCase() : '';
  }

  function labelFromFilename(filename) {
    const base = filename.replace(/^.*\//, '');
    return base || filename;
  }

  function normalizeEntry(entry, basePath = TEST_IMAGE_DIR) {
    if (!entry) return null;
    if (typeof entry === 'string') {
      const src = basePath ? `${basePath}${entry}` : entry;
      return {
        id: entry,
        label: labelFromFilename(entry),
        filename: entry,
        src,
        ext: getExtension(entry),
      };
    }
    if (entry.id || entry.src) {
      const id = entry.id || (entry.src ? entry.src.replace(/^.*\//, '') : '');
      if (!id) return null;
      const src = entry.src || (basePath ? `${basePath}${id}` : id);
      return {
        id,
        label: entry.label || labelFromFilename(id),
        filename: entry.filename || id,
        src,
        ext: getExtension(id),
        width: entry.width,
        height: entry.height,
      };
    }
    return null;
  }

  function addEntry(map, entry, allowedSet) {
    if (!entry || !entry.id) return;
    if (map.has(entry.id)) return;
    if (allowedSet && entry.ext && allowedSet.size && !allowedSet.has(entry.ext)) return;
    map.set(entry.id, entry);
  }

  function parseDirectoryListing(markup) {
    const names = new Set();
    try {
      const doc = new DOMParser().parseFromString(markup, 'text/html');
      doc.querySelectorAll('a[href]').forEach((a) => {
        const href = a.getAttribute('href') || '';
        if (!href || href.startsWith('../')) return;
        const fname = href.split('/').pop();
        if (fname && !fname.endsWith('/')) names.add(fname);
      });
    } catch (err) {
      const rx = /href\s*=\s*\"([^\"?#]+)\"/gi;
      let match;
      while ((match = rx.exec(markup)) !== null) {
        const fname = match[1].split('/').pop();
        if (fname && !fname.endsWith('/')) names.add(fname);
      }
    }
    return Array.from(names);
  }

  async function discoverBuiltinImages() {
    const collected = new Map();

    async function tryManifest() {
      try {
        const res = await fetch(`${TEST_IMAGE_DIR}manifest.json`, { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        if (!Array.isArray(data)) return;
        data.forEach((item) => {
          const normalized = normalizeEntry(item, TEST_IMAGE_DIR);
          addEntry(collected, normalized, IMAGE_EXTS);
        });
      } catch (err) {
        console.warn('Could not load test image manifest.', err);
      }
    }

    async function tryDirectory() {
      try {
        const res = await fetch(TEST_IMAGE_DIR, { cache: 'no-store' });
        if (!res.ok) return;
        const text = await res.text();
        parseDirectoryListing(text).forEach((name) => {
          const normalized = normalizeEntry(name, TEST_IMAGE_DIR);
          addEntry(collected, normalized, IMAGE_EXTS);
        });
      } catch (err) {
        console.warn('Could not enumerate test images directory; falling back.', err);
      }
    }

    await tryManifest();
    await tryDirectory();

    if (!collected.size) {
      fallbackImageFiles.forEach((name) => {
        const normalized = normalizeEntry(name, TEST_IMAGE_DIR);
        addEntry(collected, normalized, IMAGE_EXTS);
      });
    }

    const entries = Array.from(collected.values());
    entries.sort((a, b) => a.label.localeCompare(b.label));
    return entries;
  }

  async function discoverAssetFiles(directory, allowedExts = new Set(), fallbackNames = []) {
    const collected = new Map();
    const basePath = directory.endsWith('/') ? directory : `${directory}/`;

    async function tryManifest() {
      try {
        const res = await fetch(`${basePath}manifest.json`, { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        if (!Array.isArray(data)) return;
        data.forEach((item) => {
          const normalized = normalizeEntry(item, basePath);
          addEntry(collected, normalized, allowedExts);
        });
      } catch (err) {
        console.warn(`Could not load manifest for ${directory}`, err);
      }
    }

    async function tryDirectory() {
      try {
        const res = await fetch(basePath, { cache: 'no-store' });
        if (!res.ok) return;
        const text = await res.text();
        parseDirectoryListing(text).forEach((name) => {
          const normalized = normalizeEntry(name, basePath);
          addEntry(collected, normalized, allowedExts);
        });
      } catch (err) {
        console.warn(`Could not enumerate ${directory}`, err);
      }
    }

    await tryManifest();
    await tryDirectory();

    if (!collected.size) {
      fallbackNames.forEach((name) => {
        const normalized = normalizeEntry(name, basePath);
        addEntry(collected, normalized, allowedExts);
      });
    }

    const entries = Array.from(collected.values());
    entries.sort((a, b) => (a.label || a.id).localeCompare(b.label || b.id));
    return entries;
  }

  function registerBuiltin(entry) {
    if (!builtinSelect || !entry || !entry.id) return;
    if (IMAGE_EXTS.size && entry.ext && !IMAGE_EXTS.has(entry.ext)) return;
    if (builtinImages.has(entry.id)) return;
    const opt = document.createElement('option');
    opt.value = entry.id;
    opt.textContent = entry.label || entry.id;
    builtinSelect.appendChild(opt);
    builtinImages.set(entry.id, entry);
  }

  function loadBuiltinImage(id) {
    if (!id) return;
    const entry = builtinImages.get(id);
    if (!entry) {
      pendingBuiltinId = id;
      return;
    }
    const img = new Image();
    img.onload = () => {
      currentBuiltinId = id;
      uploadElement(img);
      setSourceMode('image');
      if (builtinSelect && builtinSelect.value !== id) builtinSelect.value = id;
      pendingBuiltinId = null;
    };
    img.onerror = () => console.error('Failed to load built-in image:', entry.src);
    img.src = entry.src;
  }

  async function initBuiltinSelect() {
    builtinSelect = document.getElementById('builtinImage');
    if (!builtinSelect) return;
    builtinSelect.innerHTML = '<option value="">Loading…</option>';
    builtinSelect.disabled = true;
    const entries = await discoverBuiltinImages();
    builtinSelect.innerHTML = '';
    builtinImages.clear();
    const customOption = document.createElement('option');
    customOption.value = 'custom';
    customOption.textContent = 'Custom Upload';
    customOption.hidden = true;
    builtinSelect.appendChild(customOption);
    entries.forEach((entry) => registerBuiltin(entry));
    builtinSelect.disabled = false;
    if (!builtinSelect.dataset.bound) {
      builtinSelect.addEventListener('change', (e) => {
        const id = e.target.value;
        if (!id || id === 'custom') return;
        loadBuiltinImage(id);
      });
      builtinSelect.dataset.bound = '1';
    }
    if (pendingBuiltinId === 'custom' && builtinSelect.querySelector('option[value="custom"]')) {
      builtinSelect.value = 'custom';
      pendingBuiltinId = null;
      return;
    }
    let targetId = entries.length ? entries[0].id : null;
    if (pendingBuiltinId && builtinImages.has(pendingBuiltinId)) {
      targetId = pendingBuiltinId;
    }
    if (targetId) {
      builtinSelect.value = targetId;
      loadBuiltinImage(targetId);
    }
  }

  // File upload
  function loadImageFile(file) {
    const img = new Image();
    img.onload = () => {
      uploadElement(img);
      setSourceMode('image');
      if (builtinSelect) builtinSelect.value = 'custom';
      currentBuiltinId = null;
    };
    const fr = new FileReader();
    fr.onload = () => { img.src = fr.result; };
    fr.onerror = () => console.error('FileReader failed', fr.error);
    fr.readAsDataURL(file);
  }

  // EmulatorJS wiring
  const emulatorHost = document.getElementById('emulatorHost');
  const emulatorDock = document.getElementById('emulatorDock');
  const loadEmulatorBtn = document.getElementById('loadEmulator');
  const focusEmulatorBtn = document.getElementById('focusEmulator');
  const biosSelect = document.getElementById('biosPicker');
  const isoSelect = document.getElementById('isoPicker');
  const emulatorStatus = document.getElementById('emulatorStatus');
  const emulatorCoreSelect = document.getElementById('emulatorCore');
  let emulatorJSLoaded = false;
  let emulatorObserver = null;
  let emulatorProbeUntil = 0;
  let emulatorProbeHandle = null;
  let lastEmulatorDims = { w: 0, h: 0 };

  window.EJS_ready = function () {
    setEmulatorStatus('EmulatorJS ready.');
    tryAttachEmulatorCanvas();
    scheduleEmulatorProbe(3000);
  };

  function setEmulatorStatus(message, isError = false) {
    if (!emulatorStatus) return;
    emulatorStatus.textContent = message;
    emulatorStatus.style.color = isError ? '#ff4444' : '#ffffff';
    emulatorStatus.dataset.base = message;
    emulatorStatus.dataset.error = isError ? '1' : '0';
  }

  function readEmulatorDimensions() {
    const emulator = window.EJS_emulator;
    if (emulator && emulator.gameManager && typeof emulator.gameManager.getVideoDimensions === 'function') {
      const w = emulator.gameManager.getVideoDimensions('width');
      const h = emulator.gameManager.getVideoDimensions('height');
      if (w && h) {
        const aspect = emulator.gameManager.getVideoDimensions('aspect') || (w / h);
        return { w, h, aspect };
      }
    }
    if (emulatorVideo && emulatorVideo.videoWidth && emulatorVideo.videoHeight) {
      return {
        w: emulatorVideo.videoWidth,
        h: emulatorVideo.videoHeight,
        aspect: emulatorVideo.videoWidth / emulatorVideo.videoHeight,
      };
    }
    if (emulatorCanvas && emulatorCanvas.width && emulatorCanvas.height) {
      return {
        w: emulatorCanvas.width,
        h: emulatorCanvas.height,
        aspect: emulatorCanvas.width / emulatorCanvas.height,
      };
    }
    return null;
  }

  function applyEmulatorDimensions(dims, force = false) {
    if (!dims || !dims.w || !dims.h) return false;
    const w = Math.round(dims.w);
    const h = Math.round(dims.h);
    const aspect = dims.aspect || (w / h);
    const changed = w !== lastEmulatorDims.w || h !== lastEmulatorDims.h;
    if (changed) lastEmulatorDims = { w, h };
    setStoredSrcSize('emulator', w, h);
    if (sourceMode === 'emulator' && (force || w !== srcW || h !== srcH)) {
      setSrcSize(w, h, 'emulator');
      resize();
    }
    if (emulatorHost) emulatorHost.style.setProperty('--emu-aspect', aspect.toFixed(4));
    if (emulatorStatus && emulatorLoaded && emulatorStatus.dataset.error !== '1') {
      const base = emulatorStatus.dataset.base || emulatorStatus.textContent || 'Emulator ready';
      if (!/loading/i.test(base)) {
        emulatorStatus.textContent = `${base} · ${w}x${h}`;
      }
    }
    return true;
  }

  function syncEmulatorDimensions(force = false) {
    return applyEmulatorDimensions(readEmulatorDimensions(), force);
  }

  function scheduleEmulatorProbe(durationMs = 2000) {
    emulatorProbeUntil = performance.now() + durationMs;
    if (emulatorProbeHandle) return;
    const tick = (now) => {
      if (now > emulatorProbeUntil) {
        emulatorProbeHandle = null;
        return;
      }
      syncEmulatorDimensions();
      emulatorProbeHandle = requestAnimationFrame(tick);
    };
    emulatorProbeHandle = requestAnimationFrame(tick);
  }

  function setEmulatorCore(core) {
    if (!core) return;
    window.EJS_core = core;
    setEmulatorStatus(`Core set: ${core}. Launch EmulatorJS.`);
    if (emulatorLoaded) {
      emulatorHost.innerHTML = '';
      emulatorLoaded = false;
    }
  }

  function focusEmulatorInput() {
    if (sourceMode !== 'emulator' || !emulatorCanvas) return;
    emulatorCanvas.tabIndex = 0;
    emulatorCanvas.focus({ preventScroll: true });
  }

  function tryAttachEmulatorCanvas() {
    if (!emulatorHost) return false;
    const ejsCanvas = emulatorHost.querySelector('canvas.ejs_canvas');
    if (!ejsCanvas || ejsCanvas === emulatorCanvas) return false;
    emulatorLoaded = true;
    hookEmulatorCanvas(ejsCanvas);
    setEmulatorStatus('Emulator ready. Select BIOS and ROM to start.');
    if (focusEmulatorBtn) focusEmulatorBtn.disabled = false;
    scheduleEmulatorProbe(3000);
    return true;
  }

  async function loadEmulatorJS() {
    if (emulatorJSLoaded) return Promise.resolve();
    window.EJS_startOnLoaded = true;
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'libs/emulatorjs/data/loader.js';
      script.onload = () => {
        emulatorJSLoaded = true;
        resolve();
      };
      script.onerror = () => reject(new Error('Failed to load EmulatorJS'));
      document.body.appendChild(script);
    });
  }

  let emulatorResizeObserver = null;
  function hookEmulatorCanvas(canvasEl) {
    if (!(canvasEl instanceof HTMLCanvasElement)) return;
    emulatorCanvas = canvasEl;
    emulatorCanvas.width = emulatorCanvas.width || 320;
    emulatorCanvas.height = emulatorCanvas.height || 240;
    attachEmulatorStream(canvasEl);
    setStoredSrcSize('emulator', emulatorCanvas.width || emulatorSrcW, emulatorCanvas.height || emulatorSrcH);
    syncEmulatorDimensions(true);
    if (sourceMode === 'emulator') {
      const videoReady = emulatorVideo && emulatorVideo.videoWidth && emulatorVideo.videoHeight;
      sourceElement = videoReady ? emulatorVideo : emulatorCanvas;
      if (videoReady) {
        setSrcSize(emulatorVideo.videoWidth, emulatorVideo.videoHeight, 'emulator');
      } else {
        setSrcSize(canvasEl.width || 640, canvasEl.height || 480, 'emulator');
      }
    }
    currentBuiltinId = null;
    if (builtinSelect) builtinSelect.value = 'custom';
    if (emulatorResizeObserver) emulatorResizeObserver.disconnect();
    emulatorResizeObserver = new ResizeObserver(() => {
      if (!emulatorCanvas) return;
      const w = emulatorCanvas.width || emulatorSrcW || srcW;
      const h = emulatorCanvas.height || emulatorSrcH || srcH;
      setStoredSrcSize('emulator', w, h);
      if (sourceMode === 'emulator') {
        setSrcSize(w, h, 'emulator');
        resize();
      }
    });
    emulatorResizeObserver.observe(canvasEl);
    resize();
  }

  function attachEmulatorStream(canvasEl) {
    if (emulatorStream) {
      emulatorStream.getTracks().forEach((track) => track.stop());
      emulatorStream = null;
    }
    emulatorVideo = null;
    if (!canvasEl || preferDirectEmulatorCapture || !canvasEl.captureStream) return;

    emulatorStream = canvasEl.captureStream(60);
    const track = emulatorStream.getVideoTracks()[0];
    const settings = track && track.getSettings ? track.getSettings() : null;
    if (settings && settings.width && settings.height && sourceMode === 'emulator') {
      setSrcSize(settings.width, settings.height, 'emulator');
    }
    emulatorVideo = document.createElement('video');
    emulatorVideo.muted = true;
    emulatorVideo.playsInline = true;
    emulatorVideo.autoplay = true;
    emulatorVideo.srcObject = emulatorStream;
    emulatorVideo.addEventListener('loadedmetadata', () => {
      if (sourceMode === 'emulator') {
        sourceElement = emulatorVideo;
        setSrcSize(emulatorVideo.videoWidth || canvasEl.width || srcW, emulatorVideo.videoHeight || canvasEl.height || srcH, 'emulator');
        resize();
      }
      syncEmulatorDimensions(true);
      scheduleEmulatorProbe(2000);
    }, { once: true });
    emulatorVideo.play().catch(() => {});
  }

  async function initEmulator() {
    if (emulatorLoaded) return;
    setEmulatorStatus('Loading EmulatorJS...');
    try {
      await loadEmulatorJS();
      const checkCanvas = setInterval(() => {
        if (tryAttachEmulatorCanvas()) {
          clearInterval(checkCanvas);
        }
      }, 120);
      setTimeout(() => {
        clearInterval(checkCanvas);
        if (!emulatorLoaded) {
          setEmulatorStatus('Emulator initialization timeout. Check console.', true);
        }
      }, 10000);
    } catch (err) {
      console.error('Failed to initialize emulator:', err);
      setEmulatorStatus('Failed to load emulator: ' + err.message, true);
    }
  }

  async function setEmulatorBIOS(biosUrl) {
    window.EJS_biosUrl = biosUrl;
    setEmulatorStatus(`BIOS set: ${biosUrl.split('/').pop()}`);
  }

  async function setEmulatorROM(romUrl) {
    window.EJS_gameUrl = romUrl;
    setEmulatorStatus(`Loading ROM: ${romUrl.split('/').pop()}...`);
    if (emulatorLoaded) {
      if (emulatorStream) {
        emulatorStream.getTracks().forEach((track) => track.stop());
        emulatorStream = null;
        emulatorVideo = null;
      }
      emulatorHost.innerHTML = '';
      emulatorLoaded = false;
      await initEmulator();
    }
    scheduleEmulatorProbe(4000);
  }

  async function discoverAssets(directory, extensions) {
    const entries = [];
    try {
      const response = await fetch(directory);
      if (response.ok) {
        const text = await response.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/html');
        doc.querySelectorAll('a').forEach((link) => {
          const href = link.getAttribute('href');
          if (href && !href.startsWith('..')) {
            const ext = href.slice(href.lastIndexOf('.')).toLowerCase();
            if (extensions.has(ext)) {
              entries.push({ id: href, filename: href, src: directory + href });
            }
          }
        });
      }
    } catch (err) {
      console.warn(`Could not scan ${directory}:`, err);
    }
    return entries;
  }

  async function initAssetPickers() {
    const [biosList, romList] = await Promise.all([
      discoverAssets('assets/bios/', BIOS_EXTS),
      discoverAssets('assets/isos/', ISO_EXTS),
    ]);

    if (biosSelect) {
      biosSelect.innerHTML = '<option value="">Select BIOS...</option>';
      biosList.forEach((entry) => {
        biosEntries.set(entry.id, entry);
        const opt = document.createElement('option');
        opt.value = entry.id;
        opt.textContent = entry.filename;
        biosSelect.appendChild(opt);
      });
      biosSelect.disabled = biosList.length === 0;
    }

    if (isoSelect) {
      isoSelect.innerHTML = '<option value="">Select ROM...</option>';
      romList.forEach((entry) => {
        isoEntries.set(entry.id, entry);
        const opt = document.createElement('option');
        opt.value = entry.id;
        opt.textContent = entry.filename;
        isoSelect.appendChild(opt);
      });
      isoSelect.disabled = romList.length === 0;
    }
  }

  function getLookSettings() {
    const o = {};
    sliderSettingIds.forEach((id) => {
      const el = document.getElementById(id);
      if (el) o[id] = parseFloat(el.value);
    });
    const mt = document.getElementById('maskType');
    if (mt) o.maskType = parseInt(mt.value, 10);
    return o;
  }

  function filterLookSettings(raw = {}) {
    const o = {};
    sliderSettingIds.forEach((id) => {
      if (raw[id] !== undefined) o[id] = raw[id];
    });
    if (raw.maskType !== undefined) o.maskType = raw.maskType;
    return o;
  }

  function getSessionSettings() {
    const o = getLookSettings();
    o.srcW = srcW;
    o.srcH = srcH;
    o.imageSrcW = imageSrcW;
    o.imageSrcH = imageSrcH;
    o.emulatorSrcW = emulatorSrcW;
    o.emulatorSrcH = emulatorSrcH;
    o.fitContain = fitContain;
    const parX = document.getElementById('parX');
    const parY = document.getElementById('parY');
    o.parX = parX ? parseFloat(parX.value) || 1 : 1;
    o.parY = parY ? parseFloat(parY.value) || 1 : 1;
    o.builtinId = currentBuiltinId || (builtinSelect && builtinSelect.value === 'custom' ? 'custom' : null);
    o.sourceMode = sourceMode;
    if (emulatorCoreSelect) o.emulatorCore = emulatorCoreSelect.value;
    return o;
  }

  function setControl(id, val) {
    const el = document.getElementById(id);
    if (!el || val === undefined) return;
    el.value = String(val);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  function applyLookSettings(o) {
    sliderSettingIds.forEach((id) => {
      if (id === 'pixelShape') return;
      if (o[id] !== undefined) setControl(id, o[id]);
    });
    if (o.pixelShape !== undefined) setControl('pixelShape', o.pixelShape);
    const mt = document.getElementById('maskType');
    if (mt && o.maskType !== undefined) {
      mt.value = String(o.maskType);
      mt.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }

  function applySessionSettings(o) {
    applyLookSettings(o);
    if (o.parX !== undefined || o.parY !== undefined) {
      const parX = document.getElementById('parX');
      const parY = document.getElementById('parY');
      if (parX && o.parX !== undefined) parX.value = o.parX;
      if (parY && o.parY !== undefined) parY.value = o.parY;
      applyPAR();
    }
    if (o.imageSrcW !== undefined && o.imageSrcH !== undefined) {
      imageSrcW = o.imageSrcW;
      imageSrcH = o.imageSrcH;
    }
    if (o.emulatorSrcW !== undefined && o.emulatorSrcH !== undefined) {
      emulatorSrcW = o.emulatorSrcW;
      emulatorSrcH = o.emulatorSrcH;
    }
    if (o.srcW !== undefined && o.srcH !== undefined) {
      const targetMode = o.sourceMode || sourceMode;
      if (targetMode === 'emulator' && (o.emulatorSrcW === undefined || o.emulatorSrcH === undefined)) {
        emulatorSrcW = o.srcW;
        emulatorSrcH = o.srcH;
      }
      if (targetMode === 'image' && (o.imageSrcW === undefined || o.imageSrcH === undefined)) {
        imageSrcW = o.srcW;
        imageSrcH = o.srcH;
      }
    }
    if (typeof o.fitContain === 'boolean') {
      fitContain = o.fitContain;
      const toggleBtn = document.getElementById('toggleFit');
      if (toggleBtn) toggleBtn.textContent = 'Fit: ' + (fitContain ? 'Contain' : 'Integer');
    }
    if (o.builtinId !== undefined) {
      if (o.builtinId === null || o.builtinId === 'custom') {
        currentBuiltinId = null;
        pendingBuiltinId = null;
        if (builtinSelect) builtinSelect.value = 'custom';
      } else {
        pendingBuiltinId = o.builtinId;
        loadBuiltinImage(o.builtinId);
      }
    }
    if (o.sourceMode) {
      setSourceMode(o.sourceMode);
    } else if (sourceMode === 'image' && imageSrcW && imageSrcH) {
      setSrcSize(imageSrcW, imageSrcH, 'image');
    } else if (sourceMode === 'emulator' && emulatorSrcW && emulatorSrcH) {
      setSrcSize(emulatorSrcW, emulatorSrcH, 'emulator');
    }
    if (o.emulatorCore && emulatorCoreSelect) {
      emulatorCoreSelect.value = o.emulatorCore;
      setEmulatorCore(o.emulatorCore);
    }
    resize();
  }

  function scheduleAutosave() {
    if (!autosaveEnabled) return;
    if (autosaveTimer) clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(() => {
      try {
        localStorage.setItem(LAST_KEY, JSON.stringify(getSessionSettings()));
      } catch (err) {
        console.warn('Autosave failed', err);
      }
    }, 250);
  }

  function loadPresets() {
    try {
      const raw = localStorage.getItem(PRESET_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      const normalizePreset = (preset) => ({
        name: preset.name,
        settings: filterLookSettings(preset.settings || {}),
      });
      if (Array.isArray(parsed)) {
        presets = parsed.filter((p) => p && p.name).map(normalizePreset);
      } else if (parsed && Array.isArray(parsed.presets)) {
        presets = parsed.presets.filter((p) => p && p.name).map(normalizePreset);
      }
    } catch (err) {
      console.warn('Failed to load presets', err);
    }
  }

  function persistPresets() {
    try {
      localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
    } catch (err) {
      console.warn('Failed to save presets', err);
    }
  }

  function makePresetName() {
    let idx = presets.length + 1;
    let name = `Preset ${idx}`;
    while (presets.some((p) => p.name === name)) {
      idx += 1;
      name = `Preset ${idx}`;
    }
    return name;
  }

  function renderPresetList() {
    const list = document.getElementById('presetList');
    if (!list) return;
    list.innerHTML = '';
    if (!presets.length) {
      const empty = document.createElement('div');
      empty.className = 'muted';
      empty.textContent = 'No presets saved yet.';
      list.appendChild(empty);
      return;
    }
    presets.forEach((preset) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'preset-item' + (preset.name === activePreset ? ' active' : '');
      btn.textContent = preset.name;
      btn.addEventListener('click', () => {
        activePreset = preset.name;
        const nameInput = document.getElementById('presetName');
        if (nameInput) nameInput.value = preset.name;
        applyLookSettings(preset.settings || {});
        renderPresetList();
        scheduleAutosave();
      });
      list.appendChild(btn);
    });
  }

  function savePreset(name) {
    const trimmed = (name || '').trim();
    const finalName = trimmed || makePresetName();
    const existing = presets.find((p) => p.name === finalName);
    const payload = { name: finalName, settings: getLookSettings() };
    if (existing) {
      existing.settings = payload.settings;
    } else {
      presets.push(payload);
    }
    activePreset = finalName;
    persistPresets();
    renderPresetList();
    const nameInput = document.getElementById('presetName');
    if (nameInput) nameInput.value = finalName;
  }

  function deletePreset(name) {
    const target = (name || '').trim();
    if (!target) return;
    presets = presets.filter((p) => p.name !== target);
    if (activePreset === target) activePreset = null;
    persistPresets();
    renderPresetList();
  }

  function exportPresets() {
    const blob = new Blob([JSON.stringify({ presets }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'crt-presets.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function importPresetsFile(file) {
    try {
      const text = await file.text();
      const parsed = JSON.parse(text);
      const incoming = Array.isArray(parsed) ? parsed : parsed.presets;
      if (!Array.isArray(incoming)) throw new Error('Invalid preset file');
      incoming.forEach((preset) => {
        if (!preset || !preset.name) return;
        let name = preset.name;
        let suffix = 1;
        while (presets.some((p) => p.name === name)) {
          suffix += 1;
          name = `${preset.name} ${suffix}`;
        }
        presets.push({ name, settings: filterLookSettings(preset.settings || {}) });
      });
      persistPresets();
      renderPresetList();
    } catch (err) {
      console.error('Preset import failed', err);
    }
  }

  function loadLastSettings() {
    try {
      const raw = localStorage.getItem(LAST_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') applySessionSettings(parsed);
    } catch (err) {
      console.warn('Failed to load last session', err);
    }
  }

  // WebGPU setup
  async function initWebGPU() {
    if (!('gpu' in navigator)) {
      setStatus('WebGPU not available in this browser', 'bad');
      return;
    }
    setStatus('Requesting adapter…');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      setStatus('Failed to get WebGPU adapter', 'bad');
      return;
    }
    setStatus('Requesting device…');
    const supports16f = adapter.features.has('texture-format-16-bit-float');
    const requiredFeatures = supports16f ? ['texture-format-16-bit-float'] : [];
    device = await adapter.requestDevice({ requiredFeatures });
    historyFormat = supports16f ? 'rgba16float' : 'rgba8unorm';
    device.addEventListener('uncapturederror', (event) => {
      console.error('WebGPU uncaptured error', event.error);
      setStatus(`WebGPU error: ${event.error?.message || 'Unknown'}`, 'bad');
    });
    context = canvas.getContext('webgpu');
    if (!context) {
      setStatus('Failed to get webgpu canvas context', 'bad');
      return;
    }
    uniformBuffer = device.createBuffer({
      size: uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    sampler = device.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    samplerLinear = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });
    pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    resize();
    configureContext();
    await createPipelines();
    ensureSourceTexture(srcW, srcH);
    ensureHistoryTextures(canvas.width, canvas.height);
    updateBindGroups();
    requestAnimationFrame(frame);
  }

  function configureContext() {
    const prevFormat = configuredFormat;
    try {
      configuredFormat = 'rgba16float';
      context.configure({
        device,
        format: configuredFormat,
        alphaMode: 'opaque',
        colorSpace: 'display-p3',
        toneMapping: { mode: 'extended' },
      });
      usedExtendedToneMapping = true;
    } catch (e) {
      usedExtendedToneMapping = false;
      configuredFormat = preferredFormat;
      context.configure({ device, format: configuredFormat, alphaMode: 'opaque' });
    }
    const hdrText = usedExtendedToneMapping ? 'HDR: extended tonemap (rgba16float)' : `HDR: fallback (${configuredFormat})`;
    const histText = historyFormat === 'rgba16float' ? 'history=rgba16f' : `history=${historyFormat}`;
    setStatus(`${hdrText} · ${histText} · dpr=${(window.devicePixelRatio || 1).toFixed(2)}`);
    if (prevFormat && prevFormat !== configuredFormat && pipelineLayout) {
      createPipelines().catch((err) => console.error('Pipeline rebuild failed', err));
    }
    needsReconfigure = false;
  }

  async function createPipelines() {
    const shaderModule = device.createShaderModule({ code: wgslSource });
    if (shaderModule.compilationInfo) {
      shaderModule.compilationInfo().then((info) => {
        const errors = info.messages.filter((m) => m.type === 'error');
        const warnings = info.messages.filter((m) => m.type === 'warning');
        if (errors.length) {
          const first = errors[0];
          setStatus(`Shader error L${first.lineNum}:${first.linePos} ${first.message}`, 'bad');
          console.error('WGSL errors', info.messages);
        } else if (warnings.length) {
          console.warn('WGSL warnings', warnings);
        }
      }).catch((err) => {
        console.warn('Shader compilation info unavailable', err);
      });
    }
    try {
      pipelinesReady = false;
      const createPipeline = device.createRenderPipelineAsync
        ? (desc) => device.createRenderPipelineAsync(desc)
        : (desc) => Promise.resolve(device.createRenderPipeline(desc));
      beamPipeline = await createPipeline({
        layout: pipelineLayout,
        vertex: { module: shaderModule, entryPoint: 'vs' },
        fragment: { module: shaderModule, entryPoint: 'fs_beam', targets: [{ format: historyFormat }] },
        primitive: { topology: 'triangle-list' },
      });
      finalPipeline = await createPipeline({
        layout: pipelineLayout,
        vertex: { module: shaderModule, entryPoint: 'vs' },
        fragment: { module: shaderModule, entryPoint: 'fs_final', targets: [{ format: configuredFormat }] },
        primitive: { topology: 'triangle-list' },
      });
      pipelinesReady = true;
    } catch (err) {
      pipelinesReady = false;
      console.error('Pipeline creation failed', err);
      setStatus(`Pipeline error: ${err.message || 'Unknown'}`, 'bad');
    }
  }

  function frame(now) {
    if (!device || !context || !pipelinesReady || !beamPipeline || !finalPipeline || bindGroups.length < 2) return;
    const resized = resizeCanvasToDisplaySize();
    if (resized || needsReconfigure) {
      configureContext();
      ensureHistoryTextures(canvas.width, canvas.height);
    }
    if (!historyViews[0] || !historyViews[1]) return;
    const dt = lastFrameTime ? Math.min((now - lastFrameTime) * 0.001, 0.1) : (1 / 60);
    lastFrameTime = now;
    params.time = (now * 0.001);
    const resetHistory = !historyValid;
    updateUniformBuffer(dt, resetHistory);
    historyValid = true;
    copySourceToTexture();

    const prevIndex = historyIndex;
    const nextIndex = 1 - historyIndex;

    if (gpuDebug) device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    const beamPass = encoder.beginRenderPass({
      colorAttachments: [{
        view: historyViews[nextIndex],
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    beamPass.setPipeline(beamPipeline);
    beamPass.setBindGroup(0, bindGroups[prevIndex]);
    beamPass.draw(3);
    beamPass.end();

    const view = context.getCurrentTexture().createView();
    const finalPass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    finalPass.setPipeline(finalPipeline);
    finalPass.setBindGroup(0, bindGroups[nextIndex]);
    finalPass.draw(3);
    finalPass.end();

    device.queue.submit([encoder.finish()]);
    if (gpuDebug) {
      device.popErrorScope().then((err) => {
        if (!err) return;
        console.error('WebGPU validation error', err);
        setStatus(`WebGPU validation error: ${err.message || 'Unknown'}`, 'bad');
      });
    }
    historyIndex = nextIndex;
    requestAnimationFrame(frame);
  }

  // Event listeners and wiring
  function initControls() {
    normalizeLabels();
    // Collapse toggles
    document.querySelectorAll('fieldset .collapse').forEach((btn) => {
      btn.addEventListener('click', () => {
        const fs = btn.closest('fieldset');
        fs.classList.toggle('collapsed');
        btn.textContent = fs.classList.contains('collapsed') ? '▸' : '▾';
      });
    });

    // Source mode switch
    document.querySelectorAll('[data-source]').forEach((btn) => {
      btn.addEventListener('click', () => {
        setSourceMode(btn.dataset.source);
      });
    });
    setSourceMode(sourceMode);

    const toggleLeft = document.getElementById('toggleLeft');
    const toggleRight = document.getElementById('toggleRight');
    toggleDockBtn = document.getElementById('toggleDock');
    const toggleFullscreen = document.getElementById('toggleFullscreen');
    const updatePanelButtons = () => {
      if (toggleLeft) toggleLeft.textContent = document.body.classList.contains('hide-left') ? 'Show Left' : 'Hide Left';
      if (toggleRight) toggleRight.textContent = document.body.classList.contains('hide-right') ? 'Show Right' : 'Hide Right';
    };
    if (toggleLeft) {
      toggleLeft.addEventListener('click', () => {
        document.body.classList.toggle('hide-left');
        updatePanelButtons();
      });
    }
    if (toggleRight) {
      toggleRight.addEventListener('click', () => {
        document.body.classList.toggle('hide-right');
        updatePanelButtons();
      });
    }
    if (toggleDockBtn) {
      toggleDockBtn.addEventListener('click', () => {
        dockVisible = !dockVisible;
        syncDockVisibility();
        resize();
      });
      syncDockVisibility();
    }
    if (toggleFullscreen) {
      const updateFullscreen = () => {
        const isFullscreen = !!document.fullscreenElement;
        document.body.classList.toggle('fullscreen', isFullscreen);
        toggleFullscreen.textContent = isFullscreen ? 'Exit Fullscreen' : 'Fullscreen';
        resize();
        if (isFullscreen && sourceMode === 'emulator') {
          requestAnimationFrame(() => focusEmulatorInput());
        }
      };
      toggleFullscreen.addEventListener('click', () => {
        if (document.fullscreenElement) {
          document.exitFullscreen().catch(() => {});
        } else {
          document.documentElement.requestFullscreen().catch(() => {});
        }
      });
      document.addEventListener('fullscreenchange', updateFullscreen);
      updateFullscreen();
    }
    updatePanelButtons();

    const hdrToggleBtn = document.getElementById('toggleHdrStrip');
    if (hdrToggleBtn) {
      const syncHdrLabel = () => {
        hdrToggleBtn.textContent = params.debugHDR > 0.5 ? 'HDR Strip: On (H)' : 'HDR Strip: Off (H)';
      };
      hdrToggleBtn.addEventListener('click', () => {
        params.debugHDR = params.debugHDR > 0.5 ? 0.0 : 1.0;
        syncHdrLabel();
      });
      syncHdrLabel();
    }

    // Drag & drop
    function enableDrop(target) {
      if (!target) return;
      ['dragenter', 'dragover'].forEach((evt) =>
        target.addEventListener(evt, (e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'copy';
        }),
      );
      target.addEventListener('drop', (e) => {
        e.preventDefault();
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) loadImageFile(f);
      });
    }
    enableDrop(window);
    enableDrop(document.getElementById('stage'));
    enableDrop(canvas);

    const fileInput = document.getElementById('file');
    if (fileInput) {
      fileInput.addEventListener('change', (e) => {
        const f = e.target.files[0];
        if (f) loadImageFile(f);
      });
    }

    // PAR controls
    const parX = document.getElementById('parX');
    const parY = document.getElementById('parY');
    if (parX) parX.addEventListener('input', applyPAR);
    if (parY) parY.addEventListener('input', applyPAR);
    applyPAR();

    // Sliders to params
    bindRange('scanline', (v) => (params.scanline = v));
    bindRange('persistence', (v) => (params.persistence = v));
    bindRange('beamMin', (v) => (params.beamMin = v));
    bindRange('beamMax', (v) => (params.beamMax = v));
    bindRange('sharp', (v) => (params.sharp = v));
    bindRange('maskStr', (v) => (params.maskStr = v));
    bindRange('gamma', (v) => (params.gamma = v));
    bindRange('bloom', (v) => (params.bloom = v));
    bindRange('bloomSigma', (v) => (params.bloomSigma = v));
    bindRange('halation', (v) => (params.halation = v));
    bindRange('halationSigma', (v) => (params.halationSigma = v));
    bindRange('caX', (v) => (params.caX = v));
    bindRange('caY', (v) => (params.caY = v));
    bindRange('caEdge', (v) => (params.caEdge = v));
    bindRange('maskContrast', (v) => (params.maskContrast = v));
    bindRange('maskRound', (v) => (params.maskRound = v));
    bindRange('chromaSpill', (v) => (params.chromaSpill = v));
    bindRange('chromaSigma', (v) => (params.chromaSigma = v));
    bindRange('chromaGain', (v) => (params.chromaGain = v));
    bindRange('blackSpill', (v) => (params.blackSpill = v));
    bindRange('blackSigma', (v) => (params.blackSigma = v));
    bindRange('energyLock', (v) => (params.energyLock = v));
    bindRange('curvature', (v) => (params.curvature = v));
    bindRange('vignette', (v) => (params.vignette = v));
    bindRange('postContrast', (v) => (params.postContrast = v));
    bindRange('postSaturation', (v) => (params.postSaturation = v));
    bindRange('hdrGain', (v) => (params.hdrGain = v));
    bindRange('hdrHighlights', (v) => (params.hdrHighlights = v));
    bindRange('spillBias', (v) => (params.spillBias = v));
    bindRange('pixelShape', applyPixelShapeBias);
    bindRange('syncJitter', (v) => (params.syncJitter = v));
    bindRange('syncJitterSpeed', (v) => (params.syncJitterSpeed = v));
    bindRange('rollingBand', (v) => (params.rollingBand = v));
    bindRange('rollingBandSpeed', (v) => (params.rollingBandSpeed = v));
    bindRange('noiseStrength', (v) => (params.noiseStrength = v));

    // Mask type
    const maskTypeSel = document.getElementById('maskType');
    if (maskTypeSel) {
      maskTypeSel.addEventListener('change', () => {
        params.maskType = parseInt(maskTypeSel.value || '0', 10);
        scheduleAutosave();
      });
      params.maskType = parseInt(maskTypeSel.value || '0', 10);
    }

    // Toggle fit
    const toggleBtn = document.getElementById('toggleFit');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => {
        fitContain = !fitContain;
        toggleBtn.textContent = 'Fit: ' + (fitContain ? 'Contain' : 'Integer');
        resize();
        scheduleAutosave();
      });
    }

    // Enable CRT
    const enableEl = document.getElementById('enableCRT');
    function applyBypass() {
      params.bypass = enableEl && enableEl.checked ? 0.0 : 1.0;
      scheduleAutosave();
    }
    if (enableEl) {
      enableEl.addEventListener('change', applyBypass);
      applyBypass();
    }

    // Source size override
    const srcWInput = document.getElementById('srcW');
    const srcHInput = document.getElementById('srcH');
    if (srcWInput) srcWInput.addEventListener('change', (e) => {
      const v = parseInt(e.target.value, 10);
      if (!Number.isNaN(v) && v > 0) {
        setSrcSize(v, srcH, 'image');
        resize();
        scheduleAutosave();
      }
    });
    if (srcHInput) srcHInput.addEventListener('change', (e) => {
      const v = parseInt(e.target.value, 10);
      if (!Number.isNaN(v) && v > 0) {
        setSrcSize(srcW, v, 'image');
        resize();
        scheduleAutosave();
      }
    });

    // Save PNG
    const saveBtn = document.getElementById('savePng');
    if (saveBtn) {
      saveBtn.addEventListener('click', async () => {
        try {
          const blob = await new Promise((res) => canvas.toBlob(res, 'image/png'));
          if (!blob) return;
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.download = 'crt_frame.png';
          a.href = url;
          a.rel = 'noopener';
          a.target = '_self';
          document.body.appendChild(a);
          a.click();
          a.remove();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        } catch (e) {
          console.error('Save failed', e);
        }
      });
    }

    const copyBtn2 = document.getElementById('copyPng');
    if (copyBtn2) {
      copyBtn2.addEventListener('click', async () => {
        try {
          const blob = await new Promise((res) => canvas.toBlob(res, 'image/png'));
          if (!blob) {
            alert('Could not create PNG');
            return;
          }
          if (navigator.clipboard && window.ClipboardItem) {
            await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
            copyBtn2.textContent = 'Copied ✓';
            setTimeout(() => (copyBtn2.textContent = 'Copy frame to clipboard'), 1200);
          } else {
            alert('Clipboard API not supported.');
          }
        } catch (err) {
          alert('Clipboard write blocked.');
        }
      });
    }

    // Presets
    const presetNameInput = document.getElementById('presetName');
    const presetSaveBtn = document.getElementById('presetSave');
    const presetNewBtn = document.getElementById('presetNew');
    const presetDeleteBtn = document.getElementById('presetDelete');
    const presetExportBtn = document.getElementById('presetExport');
    const presetImportBtn = document.getElementById('presetImport');
    const presetImportFile = document.getElementById('presetImportFile');
    const presetAutosave = document.getElementById('presetAutosave');
    if (presetAutosave) {
      presetAutosave.checked = autosaveEnabled;
      presetAutosave.addEventListener('change', () => {
        autosaveEnabled = presetAutosave.checked;
        localStorage.setItem(AUTOSAVE_KEY, autosaveEnabled ? '1' : '0');
        scheduleAutosave();
      });
    }
    if (presetSaveBtn) {
      presetSaveBtn.addEventListener('click', () => {
        savePreset(presetNameInput ? presetNameInput.value : '');
      });
    }
    if (presetNewBtn) {
      presetNewBtn.addEventListener('click', () => {
        if (defaultSettings) applyLookSettings(defaultSettings);
        activePreset = null;
        if (presetNameInput) presetNameInput.value = '';
        renderPresetList();
      });
    }
    if (presetDeleteBtn) {
      presetDeleteBtn.addEventListener('click', () => {
        const target = activePreset || (presetNameInput ? presetNameInput.value : '');
        deletePreset(target);
      });
    }
    if (presetExportBtn) {
      presetExportBtn.addEventListener('click', exportPresets);
    }
    if (presetImportBtn && presetImportFile) {
      presetImportBtn.addEventListener('click', () => presetImportFile.click());
      presetImportFile.addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (file) await importPresetsFile(file);
        presetImportFile.value = '';
      });
    }

    // Emulator events
    if (loadEmulatorBtn) {
      loadEmulatorBtn.addEventListener('click', () => {
        setSourceMode('emulator');
        loadEmulatorBtn.disabled = true;
        initEmulator().finally(() => {
          loadEmulatorBtn.disabled = false;
        });
      });
    }
    if (focusEmulatorBtn) {
      focusEmulatorBtn.addEventListener('click', () => {
        if (emulatorCanvas) {
          emulatorCanvas.focus();
          emulatorCanvas.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      });
    }
    if (biosSelect) {
      biosSelect.addEventListener('change', async (e) => {
        const id = e.target.value;
        if (!id) return;
        const entry = biosEntries.get(id);
        if (entry) await setEmulatorBIOS(entry.src);
      });
    }
    if (isoSelect) {
      isoSelect.addEventListener('change', async (e) => {
        const id = e.target.value;
        if (!id) return;
        const entry = isoEntries.get(id);
        if (entry) await setEmulatorROM(entry.src);
      });
    }

    if (emulatorCoreSelect) {
      emulatorCoreSelect.addEventListener('change', () => {
        setEmulatorCore(emulatorCoreSelect.value);
      });
      setEmulatorCore(emulatorCoreSelect.value || 'psx');
    }

    if (emulatorHost && !emulatorObserver) {
      emulatorObserver = new MutationObserver(() => {
        tryAttachEmulatorCanvas();
      });
      emulatorObserver.observe(emulatorHost, { childList: true, subtree: true });
    }

    const stage = document.getElementById('stage');
    if (stage) {
      stage.addEventListener('pointerdown', focusEmulatorInput);
    }
    canvas.addEventListener('pointerdown', focusEmulatorInput);
    canvas.addEventListener('focus', focusEmulatorInput);
  }

  // WGSL shader: adapted from the WebGL CRT shader, tuned for HDR highlights.
  

  // Bootstrap
  initControls();
  defaultSettings = getLookSettings();
  loadPresets();
  renderPresetList();
  loadLastSettings();
  initBuiltinSelect();
  initAssetPickers();
  initWebGPU().catch((err) => {
    console.error('WebGPU init failed', err);
    setStatus('WebGPU init failed', 'bad');
  });
  window.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'h') {
      const btn = document.getElementById('toggleHdrStrip');
      if (btn) {
        btn.click();
      } else {
        params.debugHDR = params.debugHDR > 0.5 ? 0.0 : 1.0;
        console.log(`HDR debug strip ${params.debugHDR > 0.5 ? 'on' : 'off'}`);
      }
    }
  });
  window.addEventListener('resize', resize);
})();
