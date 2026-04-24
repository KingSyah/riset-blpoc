# How It Works — Technical Detail

Deep-dive into the POC/BLPOC system implementation. No file listing, just the mechanics.

---

## 1. Dataset & Image Loading

Images are fetched from GitHub Pages:
```
https://kingsyah.github.io/riset-blpoc/dataset/
├── visual/{id}.visual_gray.jpg    ← visible light face
└── thermal/{id}.termal_gray.jpg   ← thermal face
```

- IDs: `1` to `100` (no zero-padding)
- Loaded via `new Image()` with `crossOrigin="Anonymous"`
- Drawn onto 256×256 canvas (scaled to fit, black padding)
- Read back via `cv.imread()` → RGBA → grayscale (`cv.COLOR_RGBA2GRAY`) → `CV_32FC1`

---

## 2. Algorithm (matches SIMBLPOC_v2.m exactly)

### 2.1 FFT

Both images undergo 2D Discrete Fourier Transform:
```
F(u,v) = fft2(f(x,y))     ← visual image
G(u,v) = fft2(g(x,y))     ← thermal image
```

Implemented as `cv.dft(src, dst, DFT_COMPLEX_OUTPUT)` — produces 2-channel complex matrix (channel 0 = real, channel 1 = imaginary).

### 2.2 Cross-Power Spectrum

The core formula (from the MATLAB code):
```matlab
R(u,v) = F(u,v) · G*(u,v)  /  |F(u,v) · G*(u,v)|
```

Where `G*` is the complex conjugate of G. This **normalizes by magnitude**, keeping only **phase information** — hence "Phase-Only Correlation."

**Manual implementation** (no `cv.mulSpectrums` in OpenCV.js):
```javascript
// F · G* = (aRe + j·aIm)(bRe - j·bIm)
prodRe = aRe * bRe + aIm * bIm;    // real part
prodIm = aIm * bRe - aRe * bIm;    // imaginary part

// Normalize by magnitude (phase-only)
mag = sqrt(prodRe² + prodIm²) + ε;
R_re = prodRe / mag;
R_im = prodIm / mag;
```

### 2.3 POC vs BLPOC

| | POC | BLPOC |
|---|---|---|
| **FFT input** | Full `rows×cols` | Same |
| **Frequency crop** | None (use all) | Keep first `bw×bw` components |
| **Output size** | Same as input | `bw×bw` |
| **Peak sharpness** | Moderate | Sharper (less noise) |
| **Search range** | Full image | Reduced (lower resolution) |

The bandwidth crop is the key difference:
```javascript
// POC: use all frequencies
for (r = 0; r < rows; r++)
  for (c = 0; c < cols; c++)
    // compute R(r,c)

// BLPOC: crop to first bw frequencies
for (r = 0; r < bw; r++)
  for (c = 0; c < bw; c++)
    // compute R(r,c)
```

This matches MATLAB: `P2 = P1(1:20,1:20)` keeps only the lowest 20×20 frequency components.

### 2.4 Inverse FFT & Absolute Value

```javascript
correlation = abs(ifft2(R));
```

- `cv.dft(R, spatial, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE)`
- Take `Math.abs()` of every pixel — matches MATLAB `abs(ifft2(R))`
- The **peak** (maximum value) indicates the best match location

---

## 3. Peak Detection

Simple full-scan over the correlation map:
```javascript
for (r = 0; r < rows; r++)
  for (c = 0; c < cols; c++)
    if (data[r*cols+c] > maxVal) {
      maxVal = data[r*cols+c];
      peakX = c; peakY = r;
    }
```

- **High peak** → images are a good match (same subject)
- **Low/flat peak** → images don't match
- Peak location `(x,y)` indicates spatial offset between the two images

---

## 4. Optional: Laplacian of Gaussian (LoG) Pre-filter

Applied **before** FFT when enabled. Works in frequency domain:

```
LoG(u,v) = -(u²+v²) · exp(-2π²σ²(u²+v²))
```

- Highlights edges, suppresses low-frequency illumination differences
- σ (sigma) controls the filter bandwidth
- Applied by multiplying the DFT of each image with the LoG transfer function, then inverse DFT

---

## 5. Rendering

### 2D Heatmap
- Normalize correlation map to 0–255 (`cv.NORM_MINMAX`)
- Apply JET colormap (`cv.COLORMAP_JET`) — blue=low, red=high
- Display via `cv.imshow()`

### 3D Surface Mesh
- Three.js `PlaneGeometry` with vertex displacement (y = correlation value)
- JET colormap applied per-vertex
- Downsampled to max 128×128 for performance
- Mouse drag = rotate, scroll = zoom

---

## 6. Batch Processing

Sequential loop over all 100 image pairs:
```
for id = 1 to 100:
  load visual/{id}.visual_gray.jpg
  load thermal/{id}.termal_gray.jpg
  process → POC peak, BLPOC peak
  append to results table
```

Results exported as CSV:
```
Pair,POC Peak,BLPOC Peak,Peak X,Peak Y,Method,Time (ms)
001,0.031200,0.185400,128,128,BLPOC(20),45.2
...
```

---

## 7. OpenCV.js Functions Used

Only these 7 — all verified available in standard OpenCV.js builds:

| Function | Purpose |
|---|---|
| `cv.imread()` | Read canvas pixels into Mat |
| `cv.cvtColor()` | RGBA → Grayscale |
| `cv.resize()` | Scale image (unused in current version) |
| `cv.dft()` | Forward & inverse DFT |
| `cv.applyColorMap()` | Grayscale → JET colormap |
| `cv.imshow()` | Render Mat to canvas |
| `mat.convertTo()` | Type conversion (uint8 → float32) |

All other math (cross-power spectrum, magnitude, normalization, peak finding) is done with native `Float32Array` operations to avoid OpenCV.js binding issues.

---

## 8. Memory Management

Every `cv.Mat` created is `.delete()`d after use:
```javascript
let src = cv.imread(canvas);
let gray = new cv.Mat();
cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
src.delete();  // free intermediate

let f = new cv.Mat();
gray.convertTo(f, cv.CV_32FC1);
gray.delete(); // free intermediate

// ... use f ...
f.delete();    // free when done
```

OpenCV.js runs in WebAssembly — no garbage collection. Missing `.delete()` causes memory leaks that accumulate over batch runs.

---

## 9. Key Differences from MATLAB

| Aspect | MATLAB (SIMBLPOC_v2.m) | JavaScript (this app) |
|---|---|---|
| FFT input | Original image size | 256×256 canvas (padded) |
| Bandwidth | `P1(1:20,1:20)` literal | Same logic, configurable slider |
| Conjugate | `conj(Q1)` built-in | Manual: negate imaginary part |
| Normalization | `./  (abs(...))` | Same, with ε=1e-10 |
| Visualization | `mesh()` + `colormap jet` | 2D heatmap + Three.js 3D mesh |
| Output | `saveas()` to PNG | `cv.imshow()` to canvas |

---

## 10. License & Copyright

```
© 2026 KingSyah. All rights reserved.
```

This project and its associated **research data** (face image datasets, MATLAB reference outputs, algorithm implementations) are the **intellectual property** of the author.

**You may NOT:**
- Use, copy, modify, or distribute the research data without explicit written permission
- Use this work or its data for commercial purposes without authorization
- Claim authorship or redistribute the dataset under your own name

**You may:**
- View and study the source code for educational purposes
- Reference this work in academic publications with proper citation

**To request permission**, contact via:
- GitHub: [kingsyah](https://github.com/kingsyah)
- Trakteer: [trakteer.id/KingSyah](https://trakteer.id/KingSyah)

### Citation

If you use this work in academic research, please cite:
```
Based on: "Cross-Spectral Face Matching Based on Phase Correlation"
IEEE Access, 2019. https://ieeexplore.ieee.org/document/8875642/
```

### Third-Party Assets

| Asset | Source | License |
|---|---|---|
| OpenCV.js 4.9.0 | CDN | Apache 2.0 |
| Three.js r128 | CDN | MIT |
| Inter font | Google Fonts | Open Font License |

---

*KingSyah · kingsyah.github.io*
