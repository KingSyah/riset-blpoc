/* ============================================================
   Cross-Spectral Face Matching — POC/BLPOC Living Paper
   UI Controller — faithful to SIMBLPOC_v2.m
   ============================================================ */

// ── DOM refs ───────────────────────────────────────────────
const canvasVisual  = document.getElementById('canvasVisual');
const canvasThermal = document.getElementById('canvasThermal');
const canvasResult  = document.getElementById('canvasResult');
const canvas3D      = document.getElementById('canvas3D');
const ctxV = canvasVisual.getContext('2d');
const ctxT = canvasThermal.getContext('2d');

const pairSlider = document.getElementById('pairSlider');
const bwSlider   = document.getElementById('bwSlider');
const logSlider  = document.getElementById('logSlider');

let cvReady      = false;
let imagesLoaded = false;
let visualImg    = null;
let thermalImg   = null;
let batchResults = [];
let lastResult   = null;
let viewMode     = '2d';

// ── Logging ────────────────────────────────────────────────
function log(msg, type = '') {
  const el = document.getElementById('log');
  const ts = new Date().toLocaleTimeString();
  el.innerHTML += `<div class="${type}">[${ts}] ${msg}</div>`;
  el.scrollTop = el.scrollHeight;
}

// ── OpenCV.js bootstrap ────────────────────────────────────
function onOpenCvReady() {
  cvReady = true;
  log('OpenCV.js loaded ✓', 'ok');
  document.getElementById('btnProcess').disabled = false;
  document.getElementById('loadingOverlay').classList.add('hidden');
  loadPair(1);
}

(function loadOpenCv() {
  log('Loading OpenCV.js from CDN…', 'info');
  const s = document.createElement('script');
  s.src = 'https://docs.opencv.org/4.9.0/opencv.js';
  s.async = true;
  s.onload = () => {
    if (typeof cv !== 'undefined' && cv.Mat) return onOpenCvReady();
    if (typeof cv !== 'undefined') cv['onRuntimeInitialized'] = onOpenCvReady;
  };
  s.onerror = () => {
    log('Failed to load OpenCV.js', 'error');
    document.getElementById('loadingText').textContent = 'Failed to load OpenCV.js';
  };
  document.head.appendChild(s);
})();

// ── Image loading ──────────────────────────────────────────
async function loadPair(id) {
  const vUrl = BLPoc.visualUrl(id);
  const tUrl = BLPoc.thermalUrl(id);
  log(`Loading pair #${id}…`, 'info');
  try {
    [visualImg, thermalImg] = await Promise.all([
      BLPoc.loadImage(vUrl),
      BLPoc.loadImage(tUrl)
    ]);
    BLPoc.drawFit(canvasVisual, ctxV, visualImg);
    BLPoc.drawFit(canvasThermal, ctxT, thermalImg);

    const ctxR = canvasResult.getContext('2d');
    ctxR.fillStyle = '#000'; ctxR.fillRect(0, 0, 256, 256);
    ctxR.fillStyle = '#555'; ctxR.font = '14px sans-serif'; ctxR.textAlign = 'center';
    ctxR.fillText('Click "Process" to compute', 128, 128);

    imagesLoaded = true;
    log(`Pair #${id} loaded ✓`, 'ok');
  } catch (err) {
    log(err.message, 'error');
    imagesLoaded = false;
  }
}

// ── Slider wiring ──────────────────────────────────────────
pairSlider.addEventListener('input', () => {
  const id = parseInt(pairSlider.value);
  document.getElementById('pairVal').textContent = id;
  loadPair(id);
});

bwSlider.addEventListener('input', () => {
  document.getElementById('bwVal').textContent = parseInt(bwSlider.value);
  updateMethodLabel();
});

logSlider.addEventListener('input', () => {
  const v = parseInt(logSlider.value);
  document.getElementById('logVal').textContent = v === 0 ? 'OFF' : `σ=${v}`;
  updateMethodLabel();
});

function updateMethodLabel() {
  const bw = parseInt(bwSlider.value);
  const useLoG = parseInt(logSlider.value) > 0;
  let label = `BLPOC(${bw})`;
  if (useLoG) label = `LoG+${label}`;
  document.getElementById('methodLabel').textContent = label;
}

document.getElementById('btnPrev').addEventListener('click', () => {
  const v = Math.max(1, parseInt(pairSlider.value) - 1);
  pairSlider.value = v; pairSlider.dispatchEvent(new Event('input'));
});

document.getElementById('btnNext').addEventListener('click', () => {
  const v = Math.min(BLPoc.MAX_ID, parseInt(pairSlider.value) + 1);
  pairSlider.value = v; pairSlider.dispatchEvent(new Event('input'));
});

// ── 2D / 3D view toggle ───────────────────────────────────
document.getElementById('btnView2D').addEventListener('click', () => {
  viewMode = '2d';
  document.getElementById('btnView2D').classList.add('active');
  document.getElementById('btnView3D').classList.remove('active');
  canvasResult.classList.remove('hidden');
  canvasResult.classList.add('visible');
  canvas3D.classList.remove('visible');
  canvas3D.style.display = 'none';
});

document.getElementById('btnView3D').addEventListener('click', () => {
  viewMode = '3d';
  document.getElementById('btnView3D').classList.add('active');
  document.getElementById('btnView2D').classList.remove('active');
  canvasResult.classList.add('hidden');
  canvasResult.classList.remove('visible');
  canvas3D.classList.add('visible');
  canvas3D.style.display = 'block';
  Surface3D.init(canvas3D);
  Surface3D.resize();
  if (lastResult) {
    Surface3D.render(lastResult.blpocMap, lastResult.blpocRows, lastResult.blpocCols);
  }
});

window.addEventListener('resize', () => {
  if (viewMode === '3d') Surface3D.resize();
});

// ── Process button ─────────────────────────────────────────
document.getElementById('btnProcess').addEventListener('click', () => {
  if (!cvReady || !imagesLoaded) return log('OpenCV or images not ready', 'error');

  const bw  = parseInt(bwSlider.value);
  const loG = parseInt(logSlider.value);
  const useLoG = loG > 0;

  log(`Processing pair #${pairSlider.value} | BW=${bw} | LoG=${useLoG ? 'σ='+loG : 'OFF'}`, 'info');

  try {
    const result = BLPoc.process(canvasVisual, canvasThermal, canvasResult, {
      bandwidth: bw,
      useLoG: useLoG,
      loGsigma: loG,
      debugLog: (msg) => log('[POC] ' + msg, 'info')
    });

    lastResult = result;

    document.getElementById('peakVal').textContent  = result.blpocPeak.toFixed(6);
    document.getElementById('peakLoc').textContent  = `(${result.peakX}, ${result.peakY})`;
    document.getElementById('procTime').textContent = `${result.timeMs.toFixed(1)} ms`;

    if (viewMode === '3d') {
      Surface3D.init(canvas3D);
      Surface3D.resize();
      Surface3D.render(result.blpocMap, result.blpocRows, result.blpocCols);
    }

    log(`✓ POC peak=${result.pocPeak.toFixed(6)} | BLPOC peak=${result.blpocPeak.toFixed(6)} in ${result.timeMs.toFixed(1)}ms`, 'ok');
  } catch (err) {
    log(`Error: ${err.message || err}`, 'error');
    if (err.stack) log(err.stack.split('\n').slice(0,3).join(' | '), 'error');
    console.error(err);
  }
});

// ── Batch processing ───────────────────────────────────────
document.getElementById('btnBatch').addEventListener('click', async () => {
  if (!cvReady) return;

  const section = document.getElementById('batchSection');
  section.classList.add('visible');
  const tbody  = document.getElementById('batchBody');
  const bar    = document.getElementById('batchBar');
  const status = document.getElementById('batchStatus');
  tbody.innerHTML = '';
  batchResults = [];

  const bw  = parseInt(bwSlider.value);
  const loG = parseInt(logSlider.value);
  const useLoG = loG > 0;
  const method = (useLoG ? 'LoG+' : '') + `BLPOC(${bw})`;

  log(`Starting batch — ${BLPoc.MAX_ID} pairs…`, 'info');

  for (let i = 1; i <= BLPoc.MAX_ID; i++) {
    status.textContent = `Processing #${i} / ${BLPoc.MAX_ID}…`;
    bar.style.width = `${(i / BLPoc.MAX_ID) * 100}%`;

    await loadPair(i);
    if (!imagesLoaded) { log(`Skipping #${i}`, 'warn'); continue; }

    try {
      const r = BLPoc.process(canvasVisual, canvasThermal, canvasResult, {
        bandwidth: bw,
        useLoG: useLoG,
        loGsigma: loG
      });
      batchResults.push({ id: i, pocPeak: r.pocPeak, blpocPeak: r.blpocPeak, peakX: r.peakX, peakY: r.peakY, timeMs: r.timeMs, method });
      const row = document.createElement('tr');
      row.innerHTML = `<td>${i}</td><td>${r.blpocPeak.toFixed(6)}</td><td>(${r.peakX}, ${r.peakY})</td><td>${method}</td><td>${r.timeMs.toFixed(1)}</td>`;
      tbody.appendChild(row);
    } catch (err) {
      log(`#${i} error: ${err.message || err}`, 'error');
    }
    await new Promise(r => setTimeout(r, 30));
  }

  status.textContent = `✓ ${batchResults.length} / ${BLPoc.MAX_ID} done`;
  bar.style.width = '100%';
  log(`Batch complete: ${batchResults.length} pairs`, 'ok');
});

// ── CSV export ─────────────────────────────────────────────
document.getElementById('btnExport').addEventListener('click', () => {
  if (!batchResults.length) return log('No batch results — run batch first', 'warn');
  let csv = 'Pair,POC Peak,BLPOC Peak,Peak X,Peak Y,Method,Time (ms)\n';
  for (const r of batchResults)
    csv += `${r.id},${r.pocPeak.toFixed(6)},${r.blpocPeak.toFixed(6)},${r.peakX},${r.peakY},${r.method},${r.timeMs.toFixed(1)}\n`;

  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  a.download = `blpoc_results_${Date.now()}.csv`;
  a.click();
  log(`Exported ${batchResults.length} rows`, 'ok');
});

// ── Keyboard shortcuts ─────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft')  document.getElementById('btnPrev').click();
  if (e.key === 'ArrowRight') document.getElementById('btnNext').click();
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); document.getElementById('btnProcess').click(); }
});

log('UI ready — waiting for OpenCV.js…', 'info');
