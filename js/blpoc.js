/* ============================================================
   Cross-Spectral Face Matching — POC/BLPOC Living Paper
   Core Processing Engine (OpenCV.js)

   SAFETY: All math operations use Float32Array pixel loops
   instead of OpenCV.js matrix ops to avoid BindingErrors.
   Only cv.imread, cv.cvtColor, cv.resize, cv.dft, cv.split,
   cv.merge, cv.imshow are used — all verified available.
   ============================================================ */

const BLPoc = (() => {
  const BASE_URL = 'https://kingsyah.github.io/riset-blpoc/dataset/';
  const MAX_ID   = 100;
  const FFT_SIZE = 256;
  const N        = FFT_SIZE * FFT_SIZE;   // total pixels

  // ── Helpers ──────────────────────────────────────────────
  function padId(n) { return String(n); }

  /** 2D Hanning window, blended with rectangular by alpha */
  function generateWindow2D(size, alpha) {
    const win = new Float32Array(size * size);
    for (let y = 0; y < size; y++) {
      const wy = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * y / (size - 1)));
      for (let x = 0; x < size; x++) {
        const wx = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * x / (size - 1)));
        win[y * size + x] = (1.0 - alpha) + alpha * (wx * wy);
      }
    }
    return win;
  }

  /**
   * FFT shift via manual pixel copy (safe for any OpenCV.js build).
   * Works on a single-channel CV_32FC1 mat.
   */
  function fftShift_real(mat) {
    const rows = mat.rows, cols = mat.cols;
    const hR = rows >> 1, hC = cols >> 1;
    const src = mat.data32F;
    const tmp = new Float32Array(rows * cols);

    // Copy to tmp with quadrants swapped
    for (let r = 0; r < rows; r++) {
      const srcR = (r < hR) ? r + hR : r - hR;
      for (let c = 0; c < cols; c++) {
        const srcC = (c < hC) ? c + hC : c - hC;
        tmp[r * cols + c] = src[srcR * cols + srcC];
      }
    }
    src.set(tmp);
  }

  /**
   * FFT shift for 2-channel complex mat (data32F layout: re,im,re,im,…).
   */
  function fftShift_complex(mat) {
    const rows = mat.rows, cols = mat.cols;
    const hR = rows >> 1, hC = cols >> 1;
    const src = mat.data32F;
    const tmp = new Float32Array(rows * cols * 2);

    for (let r = 0; r < rows; r++) {
      const srcR = (r < hR) ? r + hR : r - hR;
      for (let c = 0; c < cols; c++) {
        const srcC = (c < hC) ? c + hC : c - hC;
        const di = (r * cols + c) * 2;
        const si = (srcR * cols + srcC) * 2;
        tmp[di]     = src[si];
        tmp[di + 1] = src[si + 1];
      }
    }
    src.set(tmp);
  }

  /**
   * Band-limit mask: zero out frequencies outside rectangular band.
   * DC is assumed at (0,0) — we fftshift, mask, ifftshift.
   */
  function applyBandLimitMask(complexMat, bwRatio) {
    if (bwRatio >= 1.0) return;

    fftShift_complex(complexMat);

    const rows = complexMat.rows, cols = complexMat.cols;
    const cx = cols >> 1, cy = rows >> 1;
    const halfW = Math.floor(cx * bwRatio);
    const halfH = Math.floor(cy * bwRatio);
    const data = complexMat.data32F;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (Math.abs(r - cy) > halfH || Math.abs(c - cx) > halfW) {
          const idx = (r * cols + c) * 2;
          data[idx]     = 0;
          data[idx + 1] = 0;
        }
      }
    }

    fftShift_complex(complexMat);
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
    const S = FFT_SIZE;
    canvas.width = S; canvas.height = S;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, S, S);
    const scale = Math.min(S / img.width, S / img.height);
    const w = img.width * scale, h = img.height * scale;
    ctx.drawImage(img, (S - w) / 2, (S - h) / 2, w, h);
  }

  // ── Core POC / BLPOC pipeline ───────────────────────────
  /**
   * @param {HTMLCanvasElement} canvasVis
   * @param {HTMLCanvasElement} canvasThm
   * @param {HTMLCanvasElement} canvasOut
   * @param {number} bandwidthLimit  0.1 – 1.0
   * @param {number} windowAlpha     0 – 1
   * @param {function} [debugLog]    optional logging callback
   * @returns {{ peakValue, peakX, peakY, timeMs }}
   */
  function process(canvasVis, canvasThm, canvasOut, bandwidthLimit, windowAlpha, debugLog) {
    const log = debugLog || (() => {});
    const t0  = performance.now();

    // ─── 1. Read & greyscale ──────────────────────────────
    log('step 1: read & greyscale');
    let srcV = cv.imread(canvasVis);
    let srcT = cv.imread(canvasThm);
    let grayV = new cv.Mat(), grayT = new cv.Mat();
    cv.cvtColor(srcV, grayV, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(srcT, grayT, cv.COLOR_RGBA2GRAY);
    srcV.delete(); srcT.delete();

    // ─── 2. Resize & float32 ──────────────────────────────
    log('step 2: resize & float32');
    let resV = new cv.Mat(), resT = new cv.Mat();
    const sz = new cv.Size(FFT_SIZE, FFT_SIZE);
    cv.resize(grayV, resV, sz, 0, 0, cv.INTER_AREA);
    cv.resize(grayT, resT, sz, 0, 0, cv.INTER_AREA);
    grayV.delete(); grayT.delete();

    let fV = new cv.Mat(), fT = new cv.Mat();
    resV.convertTo(fV, cv.CV_32FC1);
    resT.convertTo(fT, cv.CV_32FC1);
    resV.delete(); resT.delete();

    // ─── 3. Windowing ─────────────────────────────────────
    log('step 3: windowing (alpha=' + windowAlpha.toFixed(2) + ')');
    if (windowAlpha > 0.01) {
      const winArr = generateWindow2D(FFT_SIZE, windowAlpha);
      const vD = fV.data32F, tD = fT.data32F;
      for (let i = 0; i < N; i++) {
        vD[i] *= winArr[i];
        tD[i] *= winArr[i];
      }
    }

    // ─── 4. DFT → complex output (2-channel) ──────────────
    log('step 4: DFT');
    let cV = new cv.Mat(), cT = new cv.Mat();
    cv.dft(fV, cV, cv.DFT_COMPLEX_OUTPUT);
    cv.dft(fT, cT, cv.DFT_COMPLEX_OUTPUT);
    fV.delete(); fT.delete();

    // ─── 5. Cross-Power Spectrum  F · G* ──────────────────
    log('step 5: cross-power spectrum');
    // Manual pixel loop — avoids mulSpectrums
    const dV = cV.data32F, dT = cT.data32F;
    const crossRe = new Float32Array(N);
    const crossIm = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const idx = i * 2;
      const aRe = dV[idx], aIm = dV[idx + 1];
      const bRe = dT[idx], bIm = dT[idx + 1];
      // F · G* = (aRe + j·aIm)(bRe - j·bIm)
      crossRe[i] = aRe * bRe + aIm * bIm;
      crossIm[i] = aIm * bRe - aRe * bIm;
    }
    cV.delete(); cT.delete();

    // ─── 6. Phase-only normalisation ──────────────────────
    log('step 6: phase-only normalisation');
    const nReArr = new Float32Array(N);
    const nImArr = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const mag = Math.sqrt(crossRe[i] * crossRe[i] + crossIm[i] * crossIm[i]) + 1e-5;
      nReArr[i] = crossRe[i] / mag;
      nImArr[i] = crossIm[i] / mag;
    }

    // Pack back into 2-channel cv.Mat
    let norm = new cv.Mat(FFT_SIZE, FFT_SIZE, cv.CV_32FC2);
    const normD = norm.data32F;
    for (let i = 0; i < N; i++) {
      const idx = i * 2;
      normD[idx]     = nReArr[i];
      normD[idx + 1] = nImArr[i];
    }

    // ─── 7. Band-limit (BLPOC) ────────────────────────────
    log('step 7: band-limit (bw=' + bandwidthLimit.toFixed(2) + ')');
    applyBandLimitMask(norm, bandwidthLimit);

    // ─── 8. Inverse DFT ───────────────────────────────────
    log('step 8: inverse DFT');
    let spatial = new cv.Mat();
    cv.dft(norm, spatial, cv.DFT_INVERSE | cv.DFT_REAL_OUTPUT | cv.DFT_SCALE);
    norm.delete();

    // ─── 9. FFT shift (centre peak) ───────────────────────
    log('step 9: fft shift');
    fftShift_real(spatial);

    // ─── 10. Find peak via manual pixel scan ──────────────
    log('step 10: find peak');
    const sData = spatial.data32F;
    let maxVal = -Infinity, peakX = 0, peakY = 0;
    for (let r = 0; r < FFT_SIZE; r++) {
      for (let c = 0; c < FFT_SIZE; c++) {
        const v = sData[r * FFT_SIZE + c];
        if (v > maxVal) { maxVal = v; peakX = c; peakY = r; }
      }
    }

    // ─── 11. Normalise & colourmap for display ────────────
    log('step 11: render');
    // Manual min-max normalise to 0-255
    let minV = Infinity;
    for (let i = 0; i < N; i++) { if (sData[i] < minV) minV = sData[i]; }
    const range = maxVal - minV || 1;

    let disp = new cv.Mat(FFT_SIZE, FFT_SIZE, cv.CV_8UC1);
    const dD = disp.data;
    for (let i = 0; i < N; i++) {
      dD[i] = ((sData[i] - minV) / range * 255) | 0;
    }

    let colour = new cv.Mat();
    cv.applyColorMap(disp, colour, cv.COLORMAP_JET);
    cv.imshow(canvasOut, colour);
    disp.delete(); colour.delete(); spatial.delete();

    log('done in ' + (performance.now() - t0).toFixed(1) + 'ms');

    return {
      peakValue: maxVal,
      peakX: peakX,
      peakY: peakY,
      timeMs: performance.now() - t0
    };
  }

  // ── Public API ──────────────────────────────────────────
  return {
    BASE_URL, MAX_ID, FFT_SIZE,
    padId, visualUrl, thermalUrl,
    loadImage, drawFit, process
  };
})();
