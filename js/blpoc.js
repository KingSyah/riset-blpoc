/* ============================================================
   Cross-Spectral Face Matching — POC/BLPOC Living Paper
   Core Processing Engine (OpenCV.js)

   Faithful port of SIMBLPOC_v2.m:
   - POC:  full FFT cross-power spectrum
   - BLPOC: band-limited to first bw×bw frequencies
   - No windowing (matches MATLAB code)
   - No resize (works on actual image size)
   ============================================================ */

const BLPoc = (() => {
  const BASE_URL = 'https://kingsyah.github.io/riset-blpoc/dataset/';
  const MAX_ID   = 100;

  // DFT flag fallbacks (some OpenCV.js builds miss these)
  const DFT_INVERSE        = (typeof cv.DFT_INVERSE !== 'undefined')        ? cv.DFT_INVERSE        : 1;
  const DFT_REAL_OUTPUT    = (typeof cv.DFT_REAL_OUTPUT !== 'undefined')    ? cv.DFT_REAL_OUTPUT    : 4;
  const DFT_SCALE          = (typeof cv.DFT_SCALE !== 'undefined')          ? cv.DFT_SCALE          : 2;
  const DFT_COMPLEX_OUTPUT = (typeof cv.DFT_COMPLEX_OUTPUT !== 'undefined') ? cv.DFT_COMPLEX_OUTPUT : 0;

  // ── Helpers ──────────────────────────────────────────────
  function padId(n) { return String(n); }

  /** FFT shift for 2-channel complex mat (pixel copy) */
  function fftShift_complex(data, rows, cols) {
    const hR = rows >> 1, hC = cols >> 1;
    const tmp = new Float32Array(rows * cols * 2);
    for (let r = 0; r < rows; r++) {
      const srcR = (r < hR) ? r + hR : r - hR;
      for (let c = 0; c < cols; c++) {
        const srcC = (c < hC) ? c + hC : c - hC;
        const di = (r * cols + c) * 2;
        const si = (srcR * cols + srcC) * 2;
        tmp[di]     = data[si];
        tmp[di + 1] = data[si + 1];
      }
    }
    data.set(tmp);
  }

  /** FFT shift for single-channel real mat */
  function fftShift_real(data, rows, cols) {
    const hR = rows >> 1, hC = cols >> 1;
    const tmp = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const srcR = (r < hR) ? r + hR : r - hR;
      for (let c = 0; c < cols; c++) {
        const srcC = (c < hC) ? c + hC : c - hC;
        tmp[r * cols + c] = data[srcR * cols + srcC];
      }
    }
    data.set(tmp);
  }

  // ── URL builders ─────────────────────────────────────────
  function visualUrl(id)  { return `${BASE_URL}visual/${id}.visual_gray.jpg`; }
  function thermalUrl(id) { return `${BASE_URL}thermal/${id}.termal_gray.jpg`; }

  // ── Image loader ─────────────────────────────────────────
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'Anonymous';
      img.onload  = () => resolve(img);
      img.onerror = () => reject(new Error(`Load failed: ${url}`));
      img.src = url;
    });
  }

  function drawFit(canvas, ctx, img) {
    const S = 256;
    canvas.width = S; canvas.height = S;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, S, S);
    const scale = Math.min(S / img.width, S / img.height);
    const w = img.width * scale, h = img.height * scale;
    ctx.drawImage(img, (S - w) / 2, (S - h) / 2, w, h);
  }

  // ── Core POC / BLPOC pipeline (faithful to SIMBLPOC_v2.m) ──
  /**
   * @param {HTMLCanvasElement} canvasVis
   * @param {HTMLCanvasElement} canvasThm
   * @param {HTMLCanvasElement} canvasOut
   * @param {Object} opts
   * @param {number} opts.bandwidth    — integer, number of frequency components to keep (e.g. 20)
   * @param {boolean} opts.useLoG      — apply Laplacian of Gaussian prefilter
   * @param {number} opts.loGsigma     — LoG sigma
   * @param {function} [opts.debugLog]
   * @returns {{ peakValue, peakX, peakY, timeMs, pocMap, blpocMap, rows, cols }}
   */
  function process(canvasVis, canvasThm, canvasOut, opts) {
    const bandwidth = opts.bandwidth ?? 20;
    const useLoG    = opts.useLoG ?? false;
    const loGsigma  = opts.loGsigma ?? 2.0;
    const debugLog  = opts.debugLog ?? (() => {});
    const log = debugLog;
    const t0  = performance.now();

    // ─── 1. Read images as grayscale float ─────────────────
    log('step 1: read & greyscale');
    let srcV = cv.imread(canvasVis);
    let srcT = cv.imread(canvasThm);
    let grayV = new cv.Mat(), grayT = new cv.Mat();
    cv.cvtColor(srcV, grayV, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(srcT, grayT, cv.COLOR_RGBA2GRAY);
    srcV.delete(); srcT.delete();

    let fV = new cv.Mat(), fT = new cv.Mat();
    grayV.convertTo(fV, cv.CV_32FC1);
    grayT.convertTo(fT, cv.CV_32FC1);
    grayV.delete(); grayT.delete();

    const rows = fV.rows, cols = fV.cols;
    const N = rows * cols;
    log(`image size: ${cols}×${rows}`);

    // ─── 2. Optional LoG prefilter ─────────────────────────
    if (useLoG) {
      log('step 2: LoG prefilter (σ=' + loGsigma.toFixed(1) + ')');
      applyLoG(fV, rows, cols, loGsigma);
      applyLoG(fT, rows, cols, loGsigma);
    } else {
      log('step 2: no LoG');
    }

    // ─── 3. FFT2 (full) ───────────────────────────────────
    log('step 3: FFT2');
    let fftV = new cv.Mat(), fftT = new cv.Mat();
    cv.dft(fV, fftV, DFT_COMPLEX_OUTPUT);
    cv.dft(fT, fftT, DFT_COMPLEX_OUTPUT);
    fV.delete(); fT.delete();

    // ─── 4. POC — full cross-power spectrum ────────────────
    log('step 4: POC (full)');
    const pocResult = crossPowerSpectrum(fftV, fftT, rows, cols, null);

    // ─── 5. BLPOC — band-limited cross-power spectrum ──────
    log('step 5: BLPOC (bw=' + bandwidth + ')');
    const blpocResult = crossPowerSpectrum(fftV, fftT, rows, cols, bandwidth);

    fftV.delete(); fftT.delete();

    // ─── 6. Render BLPOC (primary result) ──────────────────
    log('step 6: render');
    renderCorrelationMap(canvasOut, blpocResult.map, blpocResult.rows, blpocResult.cols);

    const timeMs = performance.now() - t0;
    log('done in ' + timeMs.toFixed(1) + 'ms');

    return {
      peakValue: blpocResult.peak,
      peakX: blpocResult.peakX,
      peakY: blpocResult.peakY,
      timeMs: timeMs,
      pocMap: pocResult.map,
      pocRows: pocResult.rows,
      pocCols: pocResult.cols,
      pocPeak: pocResult.peak,
      blpocMap: blpocResult.map,
      blpocRows: blpocResult.rows,
      blpocCols: blpocResult.cols,
      blpocPeak: blpocResult.peak
    };
  }

  /**
   * Cross-Power Spectrum computation.
   * Matches MATLAB: R = P1.*conj(Q1)./(abs(P1.*conj(Q1)))
   * If bandwidth != null, crop FFT to first bw×bw components.
   */
  function crossPowerSpectrum(fftA, fftB, rows, cols, bandwidth) {
    const useFull = (bandwidth === null || bandwidth >= Math.min(rows, cols));
    const outRows = useFull ? rows : bandwidth;
    const outCols = useFull ? cols : bandwidth;
    const outN = outRows * outCols;

    // Get raw FFT data
    const dA = fftA.data32F;
    const dB = fftB.data32F;

    // Compute cross-power spectrum: R = P1.*conj(Q1) ./ abs(P1.*conj(Q1))
    // For each frequency (u,v):
    //   F = aRe + j*aIm,  G = bRe + j*bIm
    //   F·G* = (aRe*bRe + aIm*bIm) + j*(aIm*bRe - aRe*bIm)
    //   R = F·G* / |F·G*|
    const crossRe = new Float32Array(outN);
    const crossIm = new Float32Array(outN);

    for (let r = 0; r < outRows; r++) {
      for (let c = 0; c < outCols; c++) {
        const srcIdx = (r * cols + c) * 2;
        const aRe = dA[srcIdx], aIm = dA[srcIdx + 1];
        const bRe = dB[srcIdx], bIm = dB[srcIdx + 1];

        // F · G*
        const prodRe = aRe * bRe + aIm * bIm;
        const prodIm = aIm * bRe - aRe * bIm;

        // Normalize by magnitude
        const mag = Math.sqrt(prodRe * prodRe + prodIm * prodIm) + 1e-10;
        const dstIdx = r * outCols + c;
        crossRe[dstIdx] = prodRe / mag;
        crossIm[dstIdx] = prodIm / mag;
      }
    }

    // Inverse FFT2 → correlation map
    // Pack into 2-channel cv.Mat
    let complex = new cv.Mat(outRows, outCols, cv.CV_32FC2);
    const cData = complex.data32F;
    for (let i = 0; i < outN; i++) {
      cData[i * 2]     = crossRe[i];
      cData[i * 2 + 1] = crossIm[i];
    }

    // IDFT
    let spatial = new cv.Mat();
    cv.dft(complex, spatial, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    complex.delete();

    // Take abs() — matches MATLAB: abs(ifft2(R))
    const sData = spatial.data32F;
    const resultMap = new Float32Array(outN);
    let peak = -Infinity, peakX = 0, peakY = 0;
    for (let r = 0; r < outRows; r++) {
      for (let c = 0; c < outCols; c++) {
        const v = Math.abs(sData[r * outCols + c]);
        resultMap[r * outCols + c] = v;
        if (v > peak) { peak = v; peakX = c; peakY = r; }
      }
    }
    spatial.delete();

    return { map: resultMap, rows: outRows, cols: outCols, peak, peakX, peakY };
  }

  /**
   * Apply Laplacian of Gaussian in frequency domain.
   */
  function applyLoG(mat, rows, cols, sigma) {
    let complex = new cv.Mat();
    cv.dft(mat, complex, DFT_COMPLEX_OUTPUT);
    const cData = complex.data32F;
    const halfR = rows >> 1, halfC = cols >> 1;
    const twoPiSqSigmaSq = 2.0 * Math.PI * Math.PI * sigma * sigma;

    for (let r = 0; r < rows; r++) {
      const vr = (r <= halfR) ? r / rows : (r - rows) / rows;
      for (let c = 0; c < cols; c++) {
        const uc = (c <= halfC) ? c / cols : (c - cols) / cols;
        const freqSq = uc * uc + vr * vr;
        const logVal = -freqSq * Math.exp(-twoPiSqSigmaSq * freqSq);
        const idx = (r * cols + c) * 2;
        cData[idx]     *= logVal;
        cData[idx + 1] *= logVal;
      }
    }

    let result = new cv.Mat();
    cv.dft(complex, result, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    complex.delete();

    // Copy back
    const outData = result.data32F;
    const inData = mat.data32F;
    for (let i = 0; i < rows * cols; i++) {
      inData[i] = outData[i];
    }
    result.delete();
  }

  /**
   * Render correlation map to canvas with JET colormap.
   */
  function renderCorrelationMap(canvas, mapData, rows, cols) {
    // Find min/max
    let minV = Infinity, maxV = -Infinity;
    for (let i = 0; i < rows * cols; i++) {
      if (mapData[i] < minV) minV = mapData[i];
      if (mapData[i] > maxV) maxV = mapData[i];
    }
    const range = maxV - minV || 1;

    // Create grayscale mat
    let disp = new cv.Mat(rows, cols, cv.CV_8UC1);
    const dD = disp.data;
    for (let i = 0; i < rows * cols; i++) {
      dD[i] = ((mapData[i] - minV) / range * 255) | 0;
    }

    // Apply JET colormap
    let colour = new cv.Mat();
    cv.applyColorMap(disp, colour, cv.COLORMAP_JET);
    cv.imshow(canvas, colour);
    disp.delete(); colour.delete();
  }

  // ── Public API ──────────────────────────────────────────
  return {
    BASE_URL, MAX_ID,
    padId, visualUrl, thermalUrl,
    loadImage, drawFit, process
  };
})();
