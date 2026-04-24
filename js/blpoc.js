/* ============================================================
   Cross-Spectral Face Matching — POC/BLPOC Living Paper
   Core Processing Engine (OpenCV.js)
   ============================================================ */

const BLPoc = (() => {
  const BASE_URL = 'https://kingsyah.github.io/riset-blpoc/dataset/';
  const MAX_ID   = 100;
  const FFT_SIZE = 256;

  // ── Helpers ──────────────────────────────────────────────
  function padId(n) { return String(n); }   // dataset uses bare integers: 1, 2, … 100

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

  /** Quadrant swap for 2-channel complex matrix */
  function fftShift2D(mat) {
    const halfR = Math.floor(mat.rows / 2);
    const halfC = Math.floor(mat.cols / 2);
    const q0 = mat.roi(new cv.Rect(0, 0, halfC, halfR));
    const q3 = mat.roi(new cv.Rect(halfC, halfR, mat.cols - halfC, mat.rows - halfR));
    const q1 = mat.roi(new cv.Rect(halfC, 0, mat.cols - halfC, halfR));
    const q2 = mat.roi(new cv.Rect(0, halfR, halfC, mat.rows - halfR));
    let t;
    t = new cv.Mat(); q0.copyTo(t); q3.copyTo(q0); t.copyTo(q3); t.delete();
    t = new cv.Mat(); q1.copyTo(t); q2.copyTo(q1); t.copyTo(q2); t.delete();
    q0.delete(); q1.delete(); q2.delete(); q3.delete();
  }

  /** Quadrant swap for single-channel real matrix */
  function fftShift2D_real(mat) {
    const halfR = Math.floor(mat.rows / 2);
    const halfC = Math.floor(mat.cols / 2);
    const q0 = mat.roi(new cv.Rect(0, 0, halfC, halfR));
    const q3 = mat.roi(new cv.Rect(halfC, halfR, mat.cols - halfC, mat.rows - halfR));
    const q1 = mat.roi(new cv.Rect(halfC, 0, mat.cols - halfC, halfR));
    const q2 = mat.roi(new cv.Rect(0, halfR, halfC, mat.rows - halfR));
    let t;
    t = new cv.Mat(); q0.copyTo(t); q3.copyTo(q0); t.copyTo(q3); t.delete();
    t = new cv.Mat(); q1.copyTo(t); q2.copyTo(q1); t.copyTo(q2); t.delete();
    q0.delete(); q1.delete(); q2.delete(); q3.delete();
  }

  /** Band-limit mask on normalised cross-power spectrum */
  function applyBandLimitMask(complexMat, bwRatio) {
    if (bwRatio >= 1.0) return;
    fftShift2D(complexMat);
    const rows = complexMat.rows, cols = complexMat.cols;
    const cx = Math.floor(cols / 2), cy = Math.floor(rows / 2);
    const halfW = Math.floor(cx * bwRatio);
    const halfH = Math.floor(cy * bwRatio);
    const data = complexMat.data32F;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (Math.abs(r - cy) > halfH || Math.abs(c - cx) > halfW) {
          const idx = (r * cols + c) * 2;
          data[idx] = 0;
          data[idx + 1] = 0;
        }
      }
    }
    fftShift2D(complexMat);
  }

  // ── URL builders ─────────────────────────────────────────
  function visualUrl(id) { return `${BASE_URL}visual/${id}.visual_gray.jpg`; }
  function thermalUrl(id) { return `${BASE_URL}thermal/${id}.termal_gray.jpg`; }

  // ── Image loader (returns Promise<HTMLImageElement>) ─────
  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'Anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Load failed: ${url}`));
      img.src = url;
    });
  }

  /** Draw image fitted inside a 256×256 canvas */
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
   * @param {HTMLCanvasElement} canvasOut   ← target for correlation map
   * @param {number} bandwidthLimit  0.1 – 1.0
   * @param {number} windowAlpha     0 – 1
   * @returns {{ peakValue, peakX, peakY, timeMs }}
   */
  function process(canvasVis, canvasThm, canvasOut, bandwidthLimit, windowAlpha) {
    const t0 = performance.now();

    // 1 ─ Read & greyscale
    let srcV = cv.imread(canvasVis);
    let srcT = cv.imread(canvasThm);
    let grayV = new cv.Mat(), grayT = new cv.Mat();
    cv.cvtColor(srcV, grayV, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(srcT, grayT, cv.COLOR_RGBA2GRAY);
    srcV.delete(); srcT.delete();

    // 2 ─ Resize to FFT_SIZE² & float32
    let resV = new cv.Mat(), resT = new cv.Mat();
    const sz = new cv.Size(FFT_SIZE, FFT_SIZE);
    cv.resize(grayV, resV, sz, 0, 0, cv.INTER_AREA);
    cv.resize(grayT, resT, sz, 0, 0, cv.INTER_AREA);
    grayV.delete(); grayT.delete();

    let fV = new cv.Mat(), fT = new cv.Mat();
    resV.convertTo(fV, cv.CV_32FC1);
    resT.convertTo(fT, cv.CV_32FC1);
    resV.delete(); resT.delete();

    // 3 ─ Windowing
    if (windowAlpha > 0.01) {
      const winArr = generateWindow2D(FFT_SIZE, windowAlpha);
      const winMat = new cv.Mat(FFT_SIZE, FFT_SIZE, cv.CV_32FC1);
      winMat.data32F.set(winArr);
      cv.multiply(fV, winMat, fV);
      cv.multiply(fT, winMat, fT);
      winMat.delete();
    }

    // 4 ─ DFT
    let cV = new cv.Mat(), cT = new cv.Mat();
    cv.dft(fV, cV, cv.DFT_COMPLEX_OUTPUT);
    cv.dft(fT, cT, cv.DFT_COMPLEX_OUTPUT);
    fV.delete(); fT.delete();

    // 5 ─ Cross-Power Spectrum  F · G*  (manual — mulSpectrums unavailable in OpenCV.js)
    //     F·G* = (ReF·ReG + ImF·ImG) + j(ImF·ReG − ReF·ImG)
    let chV = new cv.MatVector(), chT = new cv.MatVector();
    cv.split(cV, chV); cv.split(cT, chT);
    let reV = chV.get(0), imV = chV.get(1);
    let reT = chT.get(0), imT = chT.get(1);
    chV.delete(); chT.delete(); cV.delete(); cT.delete();

    let tA = new cv.Mat(), tB = new cv.Mat();
    let crossRe = new cv.Mat(), crossIm = new cv.Mat();

    // crossRe = reV*reT + imV*imT
    cv.multiply(reV, reT, tA);
    cv.multiply(imV, imT, tB);
    cv.add(tA, tB, crossRe);

    // crossIm = imV*reT - reV*imT
    cv.multiply(imV, reT, tA);
    cv.multiply(reV, imT, tB);
    cv.subtract(tA, tB, crossIm);

    tA.delete(); tB.delete(); reV.delete(); imV.delete(); reT.delete(); imT.delete();

    // 6 ─ Phase-only normalisation: divide by magnitude
    //     mag = sqrt(re² + im²)
    let re2 = new cv.Mat(), im2 = new cv.Mat(), sum2 = new cv.Mat(), mag = new cv.Mat();
    cv.multiply(crossRe, crossRe, re2);
    cv.multiply(crossIm, crossIm, im2);
    cv.add(re2, im2, sum2);
    cv.sqrt(sum2, mag);
    re2.delete(); im2.delete(); sum2.delete();

    // Add epsilon to avoid /0
    let eps = new cv.Mat(FFT_SIZE, FFT_SIZE, cv.CV_32FC1, new cv.Scalar(1e-5));
    cv.add(mag, eps, mag); eps.delete();

    let nRe = new cv.Mat(), nIm = new cv.Mat();
    cv.divide(crossRe, mag, nRe);
    cv.divide(crossIm, mag, nIm);
    crossRe.delete(); crossIm.delete(); mag.delete();

    let norm = new cv.Mat();
    let nch = new cv.MatVector();
    nch.push_back(nRe); nch.push_back(nIm);
    cv.merge(nch, norm);
    nch.delete(); nRe.delete(); nIm.delete();

    // 7 ─ Band-limit (BLPOC)
    applyBandLimitMask(norm, bandwidthLimit);

    // 8 ─ Inverse DFT  (cv.idft unavailable — use cv.dft with DFT_INVERSE flag)
    let spatial = new cv.Mat();
    cv.dft(norm, spatial, cv.DFT_INVERSE | cv.DFT_REAL_OUTPUT | cv.DFT_SCALE);
    norm.delete();

    // 9 ─ FFT shift (centre the peak)
    fftShift2D_real(spatial);

    // 10 ─ Find peak
    let minV = 0, maxV = 0, minL = new cv.Point(), maxL = new cv.Point();
    cv.minMaxLoc(spatial, minV, maxV, minL, maxL);

    // 11 ─ Normalise & colourmap for display
    let disp = new cv.Mat();
    cv.normalize(spatial, disp, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1);
    let colour = new cv.Mat();
    cv.applyColorMap(disp, colour, cv.COLORMAP_JET);
    cv.imshow(canvasOut, colour);
    disp.delete(); colour.delete(); spatial.delete();

    return { peakValue: maxV, peakX: maxL.x, peakY: maxL.y, timeMs: performance.now() - t0 };
  }

  // ── Public API ──────────────────────────────────────────
  return {
    BASE_URL, MAX_ID, FFT_SIZE,
    padId, visualUrl, thermalUrl,
    loadImage, drawFit, process
  };
})();
