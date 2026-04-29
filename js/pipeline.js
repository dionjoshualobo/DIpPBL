/* =============================================================
   MIRA · Digital Image Processing Pipeline
   Each stage is a discrete DIP operation, designed as a teaching
   moment. The full chain transforms a raw retinal fundus image
   into a vessel + lesion segmentation overlay, exposing every
   classical DIP technique along the way.
   ============================================================= */

/**
 * Stage shape:
 *  id            unique identifier
 *  num           display number ('01'..'15')
 *  name          short rail title
 *  sub           rail subtitle (operation summary)
 *  detailTitle   full title shown in the detail panel
 *  eyebrow       small label above detailTitle
 *  inputFrom     id of the upstream stage whose output is consumed
 *  defaultMethod id of the optimal method
 *  fixed         true when the stage has no method choices (e.g. acquisition)
 *  branch        true when the stage's output is purely informative
 *                (no downstream stage consumes it; safe to skip if errored)
 *  methods[]     selectable algorithms — each:
 *      id, name, optimal?, rationale, formula?, params{}, run(input, cv, params, pipeline) -> Mat
 *  description   shown in the detail panel for fixed stages
 */

// ─── Helpers (created lazily so cv is available) ───────────────
const make = {
  rect: (cv, w, h) => cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(w, h)),
  ellipse: (cv, w, h) => cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(w, h)),
  cross: (cv, w, h) => cv.getStructuringElement(cv.MORPH_CROSS, new cv.Size(w, h)),
  zerosLike: (cv, src, type) => cv.Mat.zeros(src.rows, src.cols, type ?? src.type()),
};

// Convert any 1-channel Mat to RGBA so it renders correctly via cv.imshow
function toRgba(cv, src) {
  const out = new cv.Mat();
  if (src.channels() === 1) {
    cv.cvtColor(src, out, cv.COLOR_GRAY2RGBA);
  } else if (src.channels() === 3) {
    cv.cvtColor(src, out, cv.COLOR_RGB2RGBA);
  } else {
    src.copyTo(out);
  }
  return out;
}

// Extract a single channel and return as a CV_8UC1 Mat
function extractChannel(cv, srcRgba, idx) {
  const channels = new cv.MatVector();
  cv.split(srcRgba, channels);
  const ch = channels.get(idx).clone();
  channels.delete();
  return ch;
}

// Extract one channel from a non-RGB color space
function extractColorSpaceChannel(cv, srcRgba, conversionCode, idx) {
  const rgb = new cv.Mat();
  cv.cvtColor(srcRgba, rgb, cv.COLOR_RGBA2RGB);
  const converted = new cv.Mat();
  cv.cvtColor(rgb, converted, conversionCode);
  rgb.delete();

  const channels = new cv.MatVector();
  cv.split(converted, channels);
  const ch = channels.get(idx).clone();
  channels.delete();
  converted.delete();
  return ch;
}

const PARAM_INFO = {
  illumination: {
    mean: {
      ksize: { input: 'number', description: 'Odd window size used by the mean filter that estimates the smooth illumination field.', integer: true, odd: true, min: 3, step: 2 },
    },
    median: {
      ksize: { input: 'number', description: 'Odd window size used by the median background estimator before subtraction.', integer: true, odd: true, min: 3, step: 2 },
    },
    morph: {
      kernel: { input: 'number', description: 'Diameter of the structuring element used to model the background surface.', integer: true, odd: true, min: 3, step: 2 },
    },
  },
  denoise: {
    gaussian: {
      ksize: { input: 'number', description: 'Odd Gaussian kernel size for spatial smoothing.', integer: true, odd: true, min: 3, step: 2 },
      sigma: { input: 'number', description: 'Standard deviation of the Gaussian blur.', min: 0, step: 0.1 },
    },
    median: {
      ksize: { input: 'number', description: 'Odd neighbourhood size used by the median filter.', integer: true, odd: true, min: 3, step: 2 },
    },
    bilateral: {
      d: { input: 'number', description: 'Pixel diameter of the bilateral neighbourhood.', integer: true, odd: true, min: 1, step: 2 },
      sigmaColor: { input: 'number', description: 'How strongly pixels with different intensities are mixed together.', min: 0, step: 1 },
      sigmaSpace: { input: 'number', description: 'How far the bilateral filter reaches spatially.', min: 0, step: 1 },
    },
  },
  contrast: {
    clahe: {
      clipLimit: { input: 'number', description: 'Contrast clip limit applied inside each CLAHE tile.', min: 0.1, step: 0.1 },
      tileGrid: { input: 'number', description: 'Number of tiles per axis used for local histogram equalisation.', integer: true, min: 1, step: 1 },
    },
    gamma: {
      gamma: { input: 'number', description: 'Power-law exponent used to brighten or darken midtones.', min: 0.05, step: 0.05 },
    },
  },
  frequency: {
    magnitude: {
      center: { input: 'text', description: 'Use "shifted" to move low frequencies to the center, or "raw" to keep the native FFT layout.', lowercase: true },
      scale: { input: 'text', description: 'Use "log" for log-magnitude compression, or "linear" for direct magnitude display.', lowercase: true },
    },
    phase: {
      center: { input: 'text', description: 'Use "shifted" to center the spectrum, or "raw" to leave quadrants in their original order.', lowercase: true },
    },
  },
  sharpening: {
    unsharp: {
      sigma: { input: 'number', description: 'Gaussian blur sigma used to build the unsharp mask.', min: 0.1, step: 0.1 },
      amount: { input: 'number', description: 'How strongly the recovered detail is added back into the image.', min: 0, step: 0.1 },
    },
    laplacian: {
      ksize: { input: 'number', description: 'Odd Laplacian kernel size used for the second derivative.', integer: true, odd: true, min: 1, step: 2 },
      alpha: { input: 'number', description: 'Weight applied to the Laplacian detail before subtraction from the source.', min: 0, step: 0.05 },
    },
    highboost: {
      A: { input: 'number', description: 'Boost factor applied to the original image before subtracting the blurred version.', min: 1, step: 0.1 },
      sigma: { input: 'number', description: 'Gaussian blur sigma used to form the high-boost mask.', min: 0.1, step: 0.1 },
    },
  },
  morphology: {
    blackhat: {
      kernel: { input: 'number', description: 'Diameter of the structuring element used for the Black-Hat operation.', integer: true, odd: true, min: 3, step: 2 },
      shape: { input: 'text', description: 'Kernel shape. Common values are "ellipse", "rect", or "cross".', lowercase: true },
    },
    tophat: {
      kernel: { input: 'number', description: 'Diameter of the structuring element used for the Top-Hat operation.', integer: true, odd: true, min: 3, step: 2 },
      shape: { input: 'text', description: 'Kernel shape. Common values are "ellipse", "rect", or "cross".', lowercase: true },
    },
    opening: {
      kernel: { input: 'number', description: 'Diameter of the structuring element used for opening.', integer: true, odd: true, min: 3, step: 2 },
      shape: { input: 'text', description: 'Kernel shape. Common values are "ellipse", "rect", or "cross".', lowercase: true },
    },
    closing: {
      kernel: { input: 'number', description: 'Diameter of the structuring element used for closing.', integer: true, odd: true, min: 3, step: 2 },
      shape: { input: 'text', description: 'Kernel shape. Common values are "ellipse", "rect", or "cross".', lowercase: true },
    },
  },
  edge: {
    canny: {
      lo: { input: 'number', description: 'Low hysteresis threshold used to keep weak edges that connect to strong ones.', min: 0, step: 1 },
      hi: { input: 'number', description: 'High hysteresis threshold used to mark strong edges in Canny detection.', min: 0, step: 1 },
    },
    sobel: {
      ksize: { input: 'number', description: 'Odd derivative kernel size used by the Sobel operator.', integer: true, odd: true, min: 1, step: 2 },
    },
    laplacian: {
      ksize: { input: 'number', description: 'Odd kernel size used by the Laplacian operator.', integer: true, odd: true, min: 1, step: 2 },
    },
    prewitt: {
      ksize: { input: 'number', description: 'Odd kernel size for the custom Prewitt derivative masks.', integer: true, odd: true, min: 3, step: 2 },
    },
  },
  segmentation: {
    adaptive: {
      block: { input: 'number', description: 'Odd local neighbourhood size used to compute the adaptive mean threshold.', integer: true, odd: true, min: 3, step: 2 },
      C: { input: 'number', description: 'Constant subtracted from each local threshold.', step: 1 },
    },
    'adaptive-gauss': {
      block: { input: 'number', description: 'Odd local neighbourhood size used to compute the Gaussian-weighted threshold.', integer: true, odd: true, min: 3, step: 2 },
      C: { input: 'number', description: 'Constant subtracted from the Gaussian-weighted local threshold.', step: 1 },
    },
    fixed: {
      t: { input: 'number', description: 'Global intensity cut-off used by the binary threshold.', min: 0, max: 255, step: 1 },
    },
  },
  cleanup: {
    area: {
      connectivity: { input: 'number', description: 'Pixel connectivity used when labeling components. Typical values are 4 or 8.', integer: true, min: 4, max: 8, step: 4, choices: [4, 8] },
      minArea: { input: 'number', description: 'Smallest connected-component area, in pixels, that will be kept.', integer: true, min: 1, step: 1 },
    },
    'close-area': {
      kernel: { input: 'number', description: 'Diameter of the closing kernel used before connected-component filtering.', integer: true, odd: true, min: 3, step: 2 },
      minArea: { input: 'number', description: 'Smallest connected-component area, in pixels, that will be kept after closing.', integer: true, min: 1, step: 1 },
    },
  },
  skeleton: {
    morph: {
      struct: { input: 'text', description: 'Structuring element specification such as "cross 3x3", "ellipse 5x5", or "rect 3x3".', lowercase: true },
      maxIter: { input: 'number', description: 'Maximum number of thinning iterations before the loop stops.', integer: true, min: 1, step: 1 },
    },
  },
  lesion: {
    bright: {
      kernel: { input: 'number', description: 'Diameter of the Top-Hat kernel used to isolate bright lesions.', integer: true, odd: true, min: 3, step: 2 },
      percentile: { input: 'number', description: 'Percentile of response intensities used as the lesion threshold.', min: 0, max: 100, step: 0.5 },
    },
    dark: {
      kernel: { input: 'number', description: 'Diameter of the Black-Hat kernel used to isolate dark lesions.', integer: true, odd: true, min: 3, step: 2 },
      percentile: { input: 'number', description: 'Percentile of response intensities used as the lesion threshold.', min: 0, max: 100, step: 0.5 },
    },
  },
  opticDisc: {
    hough: {
      dp: { input: 'number', description: 'Inverse accumulator resolution used by the Hough circle transform.', min: 0.1, step: 0.1 },
      minDist: { input: 'text', description: 'Minimum distance between detected circle centers. You can use rows, cols, min, and max in the expression.', lowercase: true },
      minR: { input: 'text', description: 'Minimum search radius. You can use rows, cols, min, and max in the expression.', lowercase: true },
      maxR: { input: 'text', description: 'Maximum search radius. You can use rows, cols, min, and max in the expression.', lowercase: true },
    },
    centroid: {
      percentile: { input: 'number', description: 'Intensity percentile used to keep only the brightest candidate optic-disc pixels.', min: 0, max: 100, step: 0.5 },
    },
  },
  composite: {
    overlay: {
      vesselColor: { input: 'color', description: 'Hex color used for the vessel overlay mask.' },
      lesionColor: { input: 'color', description: 'Hex color used for lesion candidates.' },
      odColor: { input: 'color', description: 'Hex color used for the optic-disc ring marker.' },
      alpha: { input: 'number', description: 'Base opacity used when blending mask overlays on top of the source image.', min: 0, max: 1, step: 0.05 },
    },
  },
};

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function oddInt(value, fallback, min = 1) {
  let out = Number.isFinite(value) ? Math.round(value) : fallback;
  out = Math.max(min, out);
  if (out % 2 === 0) out += 1;
  return out;
}

function intOr(value, fallback, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY) {
  if (!Number.isFinite(value)) return fallback;
  return clamp(Math.round(value), min, max);
}

function floatOr(value, fallback, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY) {
  if (!Number.isFinite(value)) return fallback;
  return clamp(value, min, max);
}

function normalizeParamValue(raw, defaultValue, meta = {}) {
  if (raw == null || raw === '') return defaultValue;

  if (meta.input === 'color') {
    const value = String(raw).trim();
    if (!/^#?[0-9a-f]{6}$/i.test(value)) {
      throw new Error(`Invalid color for ${meta.label || 'parameter'}`);
    }
    return value.startsWith('#') ? value.toUpperCase() : `#${value.toUpperCase()}`;
  }

  if (meta.input === 'number' || typeof defaultValue === 'number') {
    const numeric = Number(raw);
    if (!Number.isFinite(numeric)) {
      throw new Error(`Invalid number for ${meta.label || 'parameter'}`);
    }
    let normalized;
    if (meta.odd) normalized = oddInt(numeric, defaultValue, meta.min ?? 1);
    else if (meta.integer) normalized = intOr(numeric, defaultValue, meta.min, meta.max);
    else normalized = floatOr(numeric, defaultValue, meta.min, meta.max);
    if (meta.choices?.length) {
      return meta.choices.reduce((best, choice) =>
        Math.abs(choice - normalized) < Math.abs(best - normalized) ? choice : best,
      meta.choices[0]);
    }
    return normalized;
  }

  const value = String(raw).trim();
  if (!value) return defaultValue;
  return meta.lowercase ? value.toLowerCase() : value;
}

function makeKernel(cv, shape, width, height = width) {
  const text = String(shape || 'ellipse').trim().toLowerCase();
  if (text === 'rect' || text === 'rectangle') return make.rect(cv, width, height);
  if (text === 'cross') return make.cross(cv, width, height);
  return make.ellipse(cv, width, height);
}

function parseStructSpec(spec) {
  const text = String(spec || 'cross 3x3').trim().toLowerCase().replaceAll('×', 'x');
  const match = text.match(/(rect|rectangle|ellipse|cross)[^\d]*(\d+)(?:\s*x\s*(\d+))?/);
  if (!match) return { shape: 'cross', width: 3, height: 3 };
  return {
    shape: match[1],
    width: oddInt(Number(match[2]), 3, 1),
    height: oddInt(Number(match[3] || match[2]), 3, 1),
  };
}

function parseRelativeExpr(expr, input, fallback) {
  if (typeof expr === 'number') return expr;
  const text = String(expr ?? '').trim();
  if (!text) return fallback;

  const vars = {
    rows: input.rows,
    cols: input.cols,
    min: Math.min(input.rows, input.cols),
    max: Math.max(input.rows, input.cols),
  };

  const replaced = text
    .toLowerCase()
    .replaceAll('·', '*')
    .replaceAll('×', '*')
    .replace(/\b(rows|cols|min|max)\b/g, (_, key) => String(vars[key]));

  if (/[^0-9+\-*/().\s]/.test(replaced)) return fallback;

  try {
    const value = Number(Function(`"use strict"; return (${replaced});`)());
    return Number.isFinite(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

function percentileValue(data, percentile) {
  if (!data || !data.length) return 0;
  const p = clamp(Number(percentile) || 0, 0, 100);
  const targetIndex = Math.floor((data.length - 1) * (p / 100));

  if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) {
    const bins = new Uint32Array(256);
    for (let i = 0; i < data.length; i++) bins[data[i]]++;

    let cumulative = 0;
    for (let value = 0; value < bins.length; value++) {
      cumulative += bins[value];
      if (cumulative > targetIndex) return value;
    }
    return 255;
  }

  const sorted = Array.from(data).sort((a, b) => a - b);
  return sorted[targetIndex];
}

function hexToRgb(hex, fallback) {
  const value = String(hex || '').trim();
  const normalized = value.startsWith('#') ? value.slice(1) : value;
  if (!/^[0-9a-f]{6}$/i.test(normalized)) return fallback;
  return [
    parseInt(normalized.slice(0, 2), 16),
    parseInt(normalized.slice(2, 4), 16),
    parseInt(normalized.slice(4, 6), 16),
  ];
}

function makePrewittKernel(cv, axis, size) {
  const ksize = oddInt(size, 3, 3);
  const half = (ksize - 1) / 2;
  const values = [];
  for (let y = -half; y <= half; y++) {
    for (let x = -half; x <= half; x++) {
      values.push(axis === 'x' ? x : y);
    }
  }
  return cv.matFromArray(ksize, ksize, cv.CV_32F, values);
}

// ─── STAGE DEFINITIONS ─────────────────────────────────────────
export const STAGES = [
  // ═══════════════════════════════════════════════════════════
  {
    id: 'acquisition',
    num: '01',
    name: 'Image Acquisition',
    sub: 'Raw RGB capture',
    detailTitle: 'Image Acquisition',
    eyebrow: 'Stage 01 · Acquisition',
    inputFrom: null,
    fixed: true,
    description:
      'The raw retinal fundus image, captured at high resolution from a specialised camera. This is the spatial signal f(x, y) the entire pipeline operates on. Quality here directly bounds every downstream result.',
    methods: [],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'channel',
    num: '02',
    name: 'Channel Extraction',
    sub: 'Color space → 1ch',
    detailTitle: 'Channel Extraction',
    eyebrow: 'Stage 02 · Spectral Decomposition',
    inputFrom: 'acquisition',
    defaultMethod: 'green',
    methods: [
      {
        id: 'green',
        name: 'Green (RGB)',
        optimal: true,
        rationale:
          'The green channel offers the highest contrast between retinal vessels and background. Hemoglobin absorbs strongly in green, so vessels appear distinctly darker than the surrounding tissue.',
        formula: 'G(x,y) = I(x,y).g',
        params: {},
        run: (input, cv) => extractChannel(cv, input, 1),
      },
      {
        id: 'red',
        name: 'Red (RGB)',
        rationale:
          'The red channel is bright and saturated by the choroid, washing out vessel detail. Useful for highlighting the optic disc and lesions, but suboptimal for vasculature.',
        formula: 'R(x,y) = I(x,y).r',
        params: {},
        run: (input, cv) => extractChannel(cv, input, 0),
      },
      {
        id: 'blue',
        name: 'Blue (RGB)',
        rationale:
          'The blue channel is noisy and low-contrast in retinal photography due to weak reflectance. Rarely used standalone.',
        formula: 'B(x,y) = I(x,y).b',
        params: {},
        run: (input, cv) => extractChannel(cv, input, 2),
      },
      {
        id: 'gray',
        name: 'Luminance Y (BT.601)',
        rationale:
          'Standard luminance conversion using ITU-R BT.601 weights. Balanced but loses spectral specificity for vessel detection.',
        formula: 'Y = 0.299·R + 0.587·G + 0.114·B',
        params: {},
        run: (input, cv) => {
          const out = new cv.Mat();
          cv.cvtColor(input, out, cv.COLOR_RGBA2GRAY);
          return out;
        },
      },
      {
        id: 'hsv-v',
        name: 'HSV · Value',
        rationale:
          'The Value channel of HSV represents perceived brightness. Less vessel-specific than green but useful for illumination-invariant features.',
        formula: 'V = max(R, G, B)',
        params: {},
        run: (input, cv) => extractColorSpaceChannel(cv, input, cv.COLOR_RGB2HSV, 2),
      },
      {
        id: 'lab-l',
        name: 'CIE L*a*b* · L',
        rationale:
          'The L* channel of CIE Lab is a perceptually uniform lightness. Often used as a colour-space-independent input for medical preprocessing.',
        formula: 'L* = 116·f(Y/Yn) − 16',
        params: {},
        run: (input, cv) => extractColorSpaceChannel(cv, input, cv.COLOR_RGB2Lab, 0),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'illumination',
    num: '03',
    name: 'Illumination Correction',
    sub: 'Background subtraction',
    detailTitle: 'Illumination Correction',
    eyebrow: 'Stage 03 · Shading Compensation',
    inputFrom: 'channel',
    defaultMethod: 'mean',
    methods: [
      {
        id: 'mean',
        name: 'Mean Background Subtract',
        optimal: true,
        rationale:
          'Estimate the slowly-varying illumination by smoothing with a large mean kernel, then subtract from the input. Removes the central brightness gradient typical of fundus camera flash.',
        formula: 'I′(x,y) = I(x,y) − μ_K(x,y) + 128',
        params: { ksize: 41 },
        run: (input, cv, params) => {
          const ksize = oddInt(params.ksize, 41, 3);
          const bg = new cv.Mat();
          cv.boxFilter(input, bg, -1, new cv.Size(ksize, ksize));
          const out = new cv.Mat();
          cv.addWeighted(input, 1.0, bg, -1.0, 128, out);
          bg.delete();
          return out;
        },
      },
      {
        id: 'median',
        name: 'Median Background',
        rationale:
          'Median estimation of the background is more robust against bright punctate features (lesions) leaking into the background model.',
        formula: 'I′ = I − median_K(I) + 128',
        params: { ksize: 31 },
        run: (input, cv, params) => {
          const ksize = oddInt(params.ksize, 31, 3);
          const bg = new cv.Mat();
          cv.medianBlur(input, bg, ksize);
          const out = new cv.Mat();
          cv.addWeighted(input, 1.0, bg, -1.0, 128, out);
          bg.delete();
          return out;
        },
      },
      {
        id: 'morph',
        name: 'Morphological Background',
        rationale:
          'A wide morphological opening models the smooth illumination surface. Subtracting it isolates fine structures while preserving overall brightness.',
        formula: 'I′ = I − (I ∘ k_large) + 128',
        params: { kernel: 51 },
        run: (input, cv, params) => {
          const kernel = oddInt(params.kernel, 51, 3);
          const k = make.ellipse(cv, kernel, kernel);
          const bg = new cv.Mat();
          cv.morphologyEx(input, bg, cv.MORPH_OPEN, k);
          const out = new cv.Mat();
          cv.addWeighted(input, 1.0, bg, -1.0, 128, out);
          k.delete(); bg.delete();
          return out;
        },
      },
      {
        id: 'none',
        name: 'No Correction',
        rationale:
          'Skip illumination correction. The unmasked vignetting from the fundus camera will remain visible and bias downstream thresholds.',
        params: {},
        run: (input, cv) => input.clone(),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'denoise',
    num: '04',
    name: 'Noise Reduction',
    sub: 'Spatial smoothing',
    detailTitle: 'Noise Reduction',
    eyebrow: 'Stage 04 · Spatial Filtering',
    inputFrom: 'illumination',
    defaultMethod: 'gaussian',
    methods: [
      {
        id: 'gaussian',
        name: 'Gaussian Blur',
        optimal: true,
        rationale:
          'Convolution with a Gaussian kernel attenuates high-frequency sensor noise while preserving smooth gradients. Ideal preprocessing before any contrast or edge operation.',
        formula: 'G(x,y) = (1/2πσ²) · exp(-(x²+y²) / 2σ²)',
        params: { ksize: 5, sigma: 1.2 },
        run: (input, cv, params) => {
          const ksize = oddInt(params.ksize, 5, 3);
          const sigma = floatOr(params.sigma, 1.2, 0);
          const out = new cv.Mat();
          cv.GaussianBlur(input, out, new cv.Size(ksize, ksize), sigma, sigma, cv.BORDER_DEFAULT);
          return out;
        },
      },
      {
        id: 'median',
        name: 'Median Filter',
        rationale:
          'Replaces every pixel with the median of its neighbourhood. Excellent at removing salt-and-pepper noise while preserving edges, but slightly slower than Gaussian.',
        formula: 'M(x,y) = median{I(s,t) | (s,t) ∈ N(x,y)}',
        params: { ksize: 5 },
        run: (input, cv, params) => {
          const ksize = oddInt(params.ksize, 5, 3);
          const out = new cv.Mat();
          cv.medianBlur(input, out, ksize);
          return out;
        },
      },
      {
        id: 'bilateral',
        name: 'Bilateral Filter',
        rationale:
          'Edge-preserving smoothing using both spatial and intensity Gaussian weights. Slow but very effective when fine vessel boundaries must be retained.',
        formula: 'BF[I](x) = (1/Wp) · Σ Gσs(‖x−xi‖) · Gσr(|I(x)−I(xi)|) · I(xi)',
        params: { d: 9, sigmaColor: 75, sigmaSpace: 75 },
        run: (input, cv, params) => {
          const d = oddInt(params.d, 9, 1);
          const sigmaColor = floatOr(params.sigmaColor, 75, 0);
          const sigmaSpace = floatOr(params.sigmaSpace, 75, 0);
          const out = new cv.Mat();
          cv.bilateralFilter(input, out, d, sigmaColor, sigmaSpace, cv.BORDER_DEFAULT);
          return out;
        },
      },
      {
        id: 'none',
        name: 'No Filter',
        rationale:
          'Skipping denoising keeps every original frequency component, which means subsequent edge / threshold stages will pick up sensor noise as features.',
        params: {},
        run: (input, cv) => input.clone(),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'contrast',
    num: '05',
    name: 'Contrast Enhancement',
    sub: 'Histogram remapping',
    detailTitle: 'Contrast Enhancement',
    eyebrow: 'Stage 05 · Intensity Transform',
    inputFrom: 'denoise',
    defaultMethod: 'clahe',
    methods: [
      {
        id: 'clahe',
        name: 'CLAHE',
        optimal: true,
        rationale:
          'Contrast-Limited Adaptive Histogram Equalisation works on small tiles and clips extreme bins to suppress noise amplification. Reveals fine capillary structure without blowing out the optic disc.',
        formula: 'tiles = 8×8 · clipLimit = 2.0',
        params: { clipLimit: 2.0, tileGrid: 8 },
        run: (input, cv, params) => {
          const clipLimit = floatOr(params.clipLimit, 2.0, 0.1);
          const tileGrid = intOr(params.tileGrid, 8, 1);
          const out = new cv.Mat();
          const clahe = new cv.CLAHE(clipLimit, new cv.Size(tileGrid, tileGrid));
          clahe.apply(input, out);
          clahe.delete();
          return out;
        },
      },
      {
        id: 'histeq',
        name: 'Histogram Eq.',
        rationale:
          'Global histogram equalisation stretches intensities uniformly. Boosts overall contrast but tends to over-amplify already-bright regions like the optic disc.',
        formula: 's = T(r) = (L−1) · ∫₀ʳ pr(w) dw',
        params: {},
        run: (input, cv) => {
          const out = new cv.Mat();
          cv.equalizeHist(input, out);
          return out;
        },
      },
      {
        id: 'gamma',
        name: 'Gamma Correction',
        rationale:
          'Power-law transform that selectively brightens (γ < 1) or darkens (γ > 1) midtones. Useful for non-linear display correction.',
        formula: 's = c · r^γ      (γ = 0.6)',
        params: { gamma: 0.6 },
        run: (input, cv, params) => {
          const gamma = floatOr(params.gamma, 0.6, 0.05);
          const lut = new cv.Mat(1, 256, cv.CV_8UC1);
          for (let i = 0; i < 256; i++) {
            lut.data[i] = Math.min(255, Math.round(255 * Math.pow(i / 255, gamma)));
          }
          const out = new cv.Mat();
          cv.LUT(input, lut, out);
          lut.delete();
          return out;
        },
      },
      {
        id: 'none',
        name: 'No Enhancement',
        rationale:
          'Contrast remains as captured. Vessels and lesions stay close in intensity to the background, hampering segmentation.',
        params: {},
        run: (input, cv) => input.clone(),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'frequency',
    num: '06',
    name: 'Frequency Spectrum',
    sub: 'DFT magnitude',
    detailTitle: 'Frequency-Domain Analysis',
    eyebrow: 'Stage 06 · Fourier Transform',
    inputFrom: 'contrast',
    branch: true,
    defaultMethod: 'magnitude',
    methods: [
      {
        id: 'magnitude',
        name: 'Log Magnitude Spectrum',
        optimal: true,
        rationale:
          'The Discrete Fourier Transform decomposes the image into its spatial frequency components. The shifted log-magnitude reveals dominant orientations, periodic noise, and global texture.',
        formula:
          'F(u,v) = Σ Σ f(x,y)·e^(-j2π(ux/M+vy/N))\nS(u,v) = log(1 + |F(u,v)|)',
        params: { center: 'shifted', scale: 'log' },
        run: (input, cv, params) => computeDftSpectrum(cv, input, false, params),
      },
      {
        id: 'phase',
        name: 'Phase Spectrum',
        rationale:
          'The phase angle of F(u,v) carries most of the structural information. Visualising it shows how shapes are encoded in the frequency domain.',
        formula: 'φ(u,v) = atan2(Im(F), Re(F))',
        params: { center: 'shifted' },
        run: (input, cv, params) => computeDftSpectrum(cv, input, true, params),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'sharpening',
    num: '07',
    name: 'Image Sharpening',
    sub: 'High-pass enhancement',
    detailTitle: 'Image Sharpening',
    eyebrow: 'Stage 07 · Spatial High-Pass',
    inputFrom: 'contrast',
    defaultMethod: 'unsharp',
    methods: [
      {
        id: 'unsharp',
        name: 'Unsharp Mask',
        optimal: true,
        rationale:
          'Subtract a Gaussian-blurred copy from the original to recover high-frequency detail, then add it back to the input weighted by an "amount". Boosts vessel edges without ringing.',
        formula: 'I_s = I + α · (I − G_σ * I)',
        params: { sigma: 1.5, amount: 1.0 },
        run: (input, cv, params) => {
          const sigma = floatOr(params.sigma, 1.5, 0.1);
          const amount = floatOr(params.amount, 1.0, 0);
          const blurred = new cv.Mat();
          cv.GaussianBlur(input, blurred, new cv.Size(0, 0), sigma);
          const out = new cv.Mat();
          cv.addWeighted(input, 1.0 + amount, blurred, -amount, 0, out);
          blurred.delete();
          return out;
        },
      },
      {
        id: 'laplacian',
        name: 'Laplacian Sharpening',
        rationale:
          'Uses the second-derivative Laplacian operator as the high-pass kernel. Subtracting it from the original sharpens edges in all directions.',
        formula: 'I_s = I − α · ∇²I',
        params: { ksize: 3, alpha: 0.7 },
        run: (input, cv, params) => {
          const ksize = oddInt(params.ksize, 3, 1);
          const alpha = floatOr(params.alpha, 0.7, 0);
          const lap = new cv.Mat();
          cv.Laplacian(input, lap, cv.CV_16S, ksize);
          const lapAbs = new cv.Mat();
          cv.convertScaleAbs(lap, lapAbs);
          const out = new cv.Mat();
          cv.addWeighted(input, 1.0, lapAbs, -alpha, 0, out);
          lap.delete(); lapAbs.delete();
          return out;
        },
      },
      {
        id: 'highboost',
        name: 'High-Boost Filter',
        rationale:
          'Generalisation of unsharp masking with amplification factor A > 1. Brightens the result while still emphasising edges.',
        formula: 'I_s = A·I − G_σ * I    (A = 1.5)',
        params: { A: 1.5, sigma: 1.5 },
        run: (input, cv, params) => {
          const A = floatOr(params.A, 1.5, 1);
          const sigma = floatOr(params.sigma, 1.5, 0.1);
          const blurred = new cv.Mat();
          cv.GaussianBlur(input, blurred, new cv.Size(0, 0), sigma);
          const out = new cv.Mat();
          cv.addWeighted(input, A, blurred, -(A - 1), 0, out);
          blurred.delete();
          return out;
        },
      },
      {
        id: 'none',
        name: 'No Sharpening',
        rationale:
          'Pass-through. Soft edges may slip through the morphology and segmentation stages without being emphasised.',
        params: {},
        run: (input, cv) => input.clone(),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'morphology',
    num: '08',
    name: 'Morphological Op.',
    sub: 'Structural emphasis',
    detailTitle: 'Morphological Operation',
    eyebrow: 'Stage 08 · Mathematical Morphology',
    inputFrom: 'sharpening',
    defaultMethod: 'blackhat',
    methods: [
      {
        id: 'blackhat',
        name: 'Black-Hat',
        optimal: true,
        rationale:
          'Black-Hat = closing − input. It isolates DARK structures smaller than the structuring element — exactly the profile of retinal vessels against the brighter retinal background.',
        formula: 'B(I) = (I • k) − I        k = ellipse(15)',
        params: { kernel: 15, shape: 'ellipse' },
        run: (input, cv, params) => {
          const kernel = oddInt(params.kernel, 15, 3);
          const k = makeKernel(cv, params.shape, kernel, kernel);
          const out = new cv.Mat();
          cv.morphologyEx(input, out, cv.MORPH_BLACKHAT, k);
          k.delete();
          return out;
        },
      },
      {
        id: 'tophat',
        name: 'Top-Hat',
        rationale:
          'Top-Hat = input − opening. Highlights BRIGHT structures smaller than the kernel — useful for hard exudates and bright lesions but weak for vessels.',
        formula: 'T(I) = I − (I ∘ k)        k = ellipse(15)',
        params: { kernel: 15, shape: 'ellipse' },
        run: (input, cv, params) => {
          const kernel = oddInt(params.kernel, 15, 3);
          const k = makeKernel(cv, params.shape, kernel, kernel);
          const out = new cv.Mat();
          cv.morphologyEx(input, out, cv.MORPH_TOPHAT, k);
          k.delete();
          return out;
        },
      },
      {
        id: 'opening',
        name: 'Opening',
        rationale:
          'Erosion followed by dilation. Removes small bright noise islands while preserving the overall shape of larger structures.',
        formula: 'I ∘ k = (I ⊖ k) ⊕ k',
        params: { kernel: 5, shape: 'ellipse' },
        run: (input, cv, params) => {
          const kernel = oddInt(params.kernel, 5, 3);
          const k = makeKernel(cv, params.shape, kernel, kernel);
          const out = new cv.Mat();
          cv.morphologyEx(input, out, cv.MORPH_OPEN, k);
          k.delete();
          return out;
        },
      },
      {
        id: 'closing',
        name: 'Closing',
        rationale:
          'Dilation followed by erosion. Fills small dark holes — handy after thresholding to consolidate vessel regions.',
        formula: 'I • k = (I ⊕ k) ⊖ k',
        params: { kernel: 5, shape: 'ellipse' },
        run: (input, cv, params) => {
          const kernel = oddInt(params.kernel, 5, 3);
          const k = makeKernel(cv, params.shape, kernel, kernel);
          const out = new cv.Mat();
          cv.morphologyEx(input, out, cv.MORPH_CLOSE, k);
          k.delete();
          return out;
        },
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'edge',
    num: '09',
    name: 'Edge Detection',
    sub: 'First / second derivative',
    detailTitle: 'Edge Detection',
    eyebrow: 'Stage 09 · Gradient Operators',
    inputFrom: 'sharpening',
    branch: true,
    defaultMethod: 'canny',
    methods: [
      {
        id: 'canny',
        name: 'Canny',
        optimal: true,
        rationale:
          'Multi-stage edge detector: smoothing → gradient → non-maximum suppression → hysteresis. Yields thin, single-pixel edge contours.',
        formula: '∇I → NMS → hysteresis(40, 100)',
        params: { lo: 40, hi: 100 },
        run: (input, cv, params) => {
          const lo = floatOr(params.lo, 40, 0);
          const hi = floatOr(params.hi, 100, 0);
          const out = new cv.Mat();
          cv.Canny(input, out, lo, hi, 3, false);
          return out;
        },
      },
      {
        id: 'sobel',
        name: 'Sobel Magnitude',
        rationale:
          'Discrete first-order derivative using 3×3 kernels. The gradient magnitude image highlights all intensity transitions before thresholding.',
        formula: '|∇I| = √(Gx² + Gy²)',
        params: { ksize: 3 },
        run: (input, cv, params) => {
          const ksize = oddInt(params.ksize, 3, 1);
          const gx = new cv.Mat();
          const gy = new cv.Mat();
          cv.Sobel(input, gx, cv.CV_16S, 1, 0, ksize);
          cv.Sobel(input, gy, cv.CV_16S, 0, 1, ksize);
          const ax = new cv.Mat();
          const ay = new cv.Mat();
          cv.convertScaleAbs(gx, ax);
          cv.convertScaleAbs(gy, ay);
          const out = new cv.Mat();
          cv.addWeighted(ax, 0.5, ay, 0.5, 0, out);
          gx.delete(); gy.delete(); ax.delete(); ay.delete();
          return out;
        },
      },
      {
        id: 'laplacian',
        name: 'Laplacian',
        rationale:
          'Isotropic second-order derivative. Captures edges in all directions in a single pass; very sensitive to noise.',
        formula: '∇²I = ∂²I/∂x² + ∂²I/∂y²',
        params: { ksize: 3 },
        run: (input, cv, params) => {
          const ksize = oddInt(params.ksize, 3, 1);
          const lap = new cv.Mat();
          cv.Laplacian(input, lap, cv.CV_16S, ksize);
          const out = new cv.Mat();
          cv.convertScaleAbs(lap, out);
          lap.delete();
          return out;
        },
      },
      {
        id: 'prewitt',
        name: 'Prewitt',
        rationale:
          'Like Sobel but with uniform (un-weighted) neighbourhood — slightly more sensitive to noise but easier to derive theoretically.',
        formula: 'kx = [-1 0 1; -1 0 1; -1 0 1]',
        params: { ksize: 3 },
        run: (input, cv, params) => {
          const ksize = oddInt(params.ksize, 3, 3);
          const kx = makePrewittKernel(cv, 'x', ksize);
          const ky = makePrewittKernel(cv, 'y', ksize);
          const gx = new cv.Mat();
          const gy = new cv.Mat();
          cv.filter2D(input, gx, cv.CV_16S, kx);
          cv.filter2D(input, gy, cv.CV_16S, ky);
          const ax = new cv.Mat();
          const ay = new cv.Mat();
          cv.convertScaleAbs(gx, ax);
          cv.convertScaleAbs(gy, ay);
          const out = new cv.Mat();
          cv.addWeighted(ax, 0.5, ay, 0.5, 0, out);
          kx.delete(); ky.delete(); gx.delete(); gy.delete();
          ax.delete(); ay.delete();
          return out;
        },
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'segmentation',
    num: '10',
    name: 'Vessel Segmentation',
    sub: 'Binary thresholding',
    detailTitle: 'Vessel Segmentation',
    eyebrow: 'Stage 10 · Thresholding',
    inputFrom: 'morphology',
    defaultMethod: 'otsu',
    methods: [
      {
        id: 'otsu',
        name: "Otsu's Method",
        optimal: true,
        rationale:
          "Otsu's algorithm finds the threshold that maximises between-class variance. Fully data-driven — no parameter to tune — and works well on the bimodal histogram produced by Black-Hat.",
        formula: 't* = argmax_t  ω₀(t)·ω₁(t)·[μ₀(t)−μ₁(t)]²',
        params: {},
        run: (input, cv) => {
          const out = new cv.Mat();
          cv.threshold(input, out, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
          return out;
        },
      },
      {
        id: 'adaptive',
        name: 'Adaptive Mean',
        rationale:
          'Threshold computed locally over a sliding window. Robust to uneven illumination but tends to leak vessel boundaries.',
        formula: 't(x,y) = mean(N(x,y)) − C',
        params: { block: 31, C: 5 },
        run: (input, cv, params) => {
          const block = oddInt(params.block, 31, 3);
          const c = floatOr(params.C, 5);
          const out = new cv.Mat();
          cv.adaptiveThreshold(input, out, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block, c);
          return out;
        },
      },
      {
        id: 'adaptive-gauss',
        name: 'Adaptive Gaussian',
        rationale:
          'Same idea as adaptive mean but the local threshold uses a Gaussian-weighted neighbourhood — smoother and less blocky boundaries.',
        formula: 't(x,y) = G_σ * I(x,y) − C',
        params: { block: 31, C: 5 },
        run: (input, cv, params) => {
          const block = oddInt(params.block, 31, 3);
          const c = floatOr(params.C, 5);
          const out = new cv.Mat();
          cv.adaptiveThreshold(input, out, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block, c);
          return out;
        },
      },
      {
        id: 'fixed',
        name: 'Fixed Threshold',
        rationale:
          'Manually-tuned global cut-off. Simple and fast but brittle when illumination varies.',
        formula: 'B(x,y) = 255 if I(x,y) > 60 else 0',
        params: { t: 60 },
        run: (input, cv, params) => {
          const t = floatOr(params.t, 60, 0, 255);
          const out = new cv.Mat();
          cv.threshold(input, out, t, 255, cv.THRESH_BINARY);
          return out;
        },
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'cleanup',
    num: '11',
    name: 'Region Cleanup',
    sub: 'Connected components',
    detailTitle: 'Post-Processing & Region Filtering',
    eyebrow: 'Stage 11 · Region Analysis',
    inputFrom: 'segmentation',
    defaultMethod: 'area',
    methods: [
      {
        id: 'area',
        name: 'Area Filter',
        optimal: true,
        rationale:
          'Label connected components in the binary mask and discard any region whose pixel count falls below a minimum. Purges salt-and-pepper noise and tiny non-vessel artefacts.',
        formula: 'keep R if |R| ≥ A_min     (A_min = 40 px)',
        params: { connectivity: 8, minArea: 40 },
        run: (input, cv, params) =>
          connectedComponentAreaFilter(cv, input, intOr(params.minArea, 40, 1), intOr(params.connectivity, 8, 4, 8)),
      },
      {
        id: 'close-area',
        name: 'Closing + Area',
        rationale:
          'Apply a morphological closing first to bridge small gaps along the vessel skeleton, then filter by component size. Produces continuous vessels.',
        formula: '(I • k_close) → CC ≥ A_min',
        params: { kernel: 3, minArea: 40 },
        run: (input, cv, params) => {
          const kernel = oddInt(params.kernel, 3, 3);
          const minArea = intOr(params.minArea, 40, 1);
          const k = make.ellipse(cv, kernel, kernel);
          const closed = new cv.Mat();
          cv.morphologyEx(input, closed, cv.MORPH_CLOSE, k);
          const out = connectedComponentAreaFilter(cv, closed, minArea, 8);
          k.delete(); closed.delete();
          return out;
        },
      },
      {
        id: 'none',
        name: 'No Cleanup',
        rationale:
          'Use the raw segmentation mask. Tiny noise components and disconnected fragments will reach the composite.',
        params: {},
        run: (input, cv) => input.clone(),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'skeleton',
    num: '12',
    name: 'Skeletonization',
    sub: 'Vessel centerlines',
    detailTitle: 'Skeletonization',
    eyebrow: 'Stage 12 · Morphological Thinning',
    inputFrom: 'cleanup',
    defaultMethod: 'morph',
    methods: [
      {
        id: 'morph',
        name: 'Lantuéjoul Skeleton',
        optimal: true,
        rationale:
          'Iterative morphological skeleton: at every step, accumulate (I − opening(I)) and erode. The union of these residues collapses each vessel to its 1-pixel-thick centerline.',
        formula:
          'S(I) = ∪ₖ ( εᵏ(I) − [εᵏ(I) ∘ k_b] )',
        params: { struct: 'cross 3×3', maxIter: 100 },
        run: (input, cv, params) => morphologicalSkeleton(cv, input, params),
      },
      {
        id: 'distance',
        name: 'Distance Ridge',
        rationale:
          'Compute the distance transform of the binary mask, then keep local maxima as the medial axis. Faster but gives a thicker, gradient-style skeleton.',
        formula: 'ridge = max(D − D_smooth, 0)',
        params: {},
        run: (input, cv) => distanceRidge(cv, input),
      },
      {
        id: 'none',
        name: 'Disabled',
        rationale: 'Skip skeletonization — output is the cleaned vessel mask itself.',
        params: {},
        run: (input, cv) => input.clone(),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'lesion',
    num: '13',
    name: 'Lesion Detection',
    sub: 'Bright / dark anomalies',
    detailTitle: 'Lesion Detection',
    eyebrow: 'Stage 13 · Anomaly Mask',
    inputFrom: 'contrast',
    defaultMethod: 'bright',
    methods: [
      {
        id: 'bright',
        name: 'Bright Lesions (Top-Hat)',
        optimal: true,
        rationale:
          'Top-Hat on the enhanced channel exposes hard exudates and cotton-wool spots — bright clusters that the morphological opening cannot fit into. A high percentile threshold isolates them.',
        formula: 'T(I) ≥ P98(T(I))',
        params: { kernel: 21, percentile: 98 },
        run: (input, cv, params) => {
          const kernel = oddInt(params.kernel, 21, 3);
          const percentile = floatOr(params.percentile, 98, 0, 100);
          const k = make.ellipse(cv, kernel, kernel);
          const tophat = new cv.Mat();
          cv.morphologyEx(input, tophat, cv.MORPH_TOPHAT, k);
          const t = percentileValue(tophat.data, percentile);
          const out = new cv.Mat();
          cv.threshold(tophat, out, Math.min(254, t), 255, cv.THRESH_BINARY);
          k.delete(); tophat.delete();
          return out;
        },
      },
      {
        id: 'dark',
        name: 'Dark Lesions (Black-Hat)',
        rationale:
          'Black-Hat captures microaneurysms and haemorrhages — small dark blobs. Useful but tends to fire on vessel crossings too, so combine with the vessel mask in post.',
        formula: 'B(I) ≥ P97(B(I))',
        params: { kernel: 9, percentile: 97 },
        run: (input, cv, params) => {
          const kernel = oddInt(params.kernel, 9, 3);
          const percentile = floatOr(params.percentile, 97, 0, 100);
          const k = make.ellipse(cv, kernel, kernel);
          const blackhat = new cv.Mat();
          cv.morphologyEx(input, blackhat, cv.MORPH_BLACKHAT, k);
          const t = percentileValue(blackhat.data, percentile);
          const out = new cv.Mat();
          cv.threshold(blackhat, out, Math.min(254, t), 255, cv.THRESH_BINARY);
          k.delete(); blackhat.delete();
          return out;
        },
      },
      {
        id: 'none',
        name: 'Disabled',
        rationale: 'Lesion overlay turned off — output shows vessels only.',
        params: {},
        run: (input, cv) => make.zerosLike(cv, input, cv.CV_8UC1),
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'opticDisc',
    num: '14',
    name: 'Optic Disc',
    sub: 'Hough circle search',
    detailTitle: 'Optic Disc Localisation',
    eyebrow: 'Stage 14 · Hough Transform',
    inputFrom: 'contrast',
    branch: true,
    defaultMethod: 'hough',
    methods: [
      {
        id: 'hough',
        name: 'Hough Circle Transform',
        optimal: true,
        rationale:
          'A parameter-space voting scheme that searches over (cx, cy, r). Well-suited to the optic disc, which is one of the few near-circular bright structures in the fundus.',
        formula: '(x − a)² + (y − b)² = r²',
        params: { dp: 1, minDist: 'rows/4', minR: '0.05·min', maxR: '0.15·min' },
        run: (input, cv, params) => detectOpticDiscHough(cv, input, params),
      },
      {
        id: 'centroid',
        name: 'Brightest Centroid',
        rationale:
          'Heuristic baseline: blur heavily, threshold the brightest 5% of pixels, then return the centroid of the largest connected blob.',
        formula: 'c = (Σ x_i · I(x_i)) / Σ I(x_i)',
        params: { percentile: 95 },
        run: (input, cv, params) => detectOpticDiscCentroid(cv, input, params),
      },
      {
        id: 'none',
        name: 'Disabled',
        rationale: 'Skip optic disc localisation.',
        params: {},
        run: (input, cv) => {
          const out = new cv.Mat();
          cv.cvtColor(input, out, cv.COLOR_GRAY2RGBA);
          return out;
        },
      },
    ],
  },

  // ═══════════════════════════════════════════════════════════
  {
    id: 'composite',
    num: '15',
    name: 'Diagnostic Composite',
    sub: 'Vessels + Lesions + OD',
    detailTitle: 'Diagnostic Composite',
    eyebrow: 'Stage 15 · Visualisation',
    inputFrom: 'acquisition',
    fixed: true,
    defaultMethod: 'overlay',
    description:
      'Final visualisation: the original colour acquisition with the cleaned vessel mask in clinical teal, lesion candidates in warning amber, and the detected optic disc highlighted in sapphire. Toggle individual layers from the Output panel.',
    methods: [
      {
        id: 'overlay',
        name: 'Layered Overlay',
        optimal: true,
        rationale:
          'Output combines the original colour image with the cleaned vessel mask (clinical teal), the lesion mask (warning amber) and a sapphire ring on the detected optic disc.',
        params: { vesselColor: '#94D3C1', lesionColor: '#F1C27D', odColor: '#7DB7F1', alpha: 0.75 },
      },
    ],
  },
];

// ─── Frequency-domain helper ───────────────────────────────────
function computeDftSpectrum(cv, input, asPhase, params = {}) {
  const shouldShift = String(params.center ?? 'shifted').trim().toLowerCase() !== 'raw';
  const useLogScale = !asPhase && String(params.scale ?? 'log').trim().toLowerCase() !== 'linear';

  // 1. Pad to optimal DFT size
  const optW = cv.getOptimalDFTSize(input.cols);
  const optH = cv.getOptimalDFTSize(input.rows);
  const padded = new cv.Mat();
  cv.copyMakeBorder(input, padded, 0, optH - input.rows, 0, optW - input.cols, cv.BORDER_CONSTANT, new cv.Scalar(0));

  // 2. Build a 2-channel complex Mat
  const floatPadded = new cv.Mat();
  padded.convertTo(floatPadded, cv.CV_32F);
  padded.delete();

  const zeros = cv.Mat.zeros(floatPadded.rows, floatPadded.cols, cv.CV_32F);
  const planes = new cv.MatVector();
  planes.push_back(floatPadded);
  planes.push_back(zeros);

  const complex = new cv.Mat();
  cv.merge(planes, complex);
  planes.delete();
  floatPadded.delete();
  zeros.delete();

  // 3. Forward DFT
  cv.dft(complex, complex);

  // 4. Split into real/imag and compute magnitude or phase
  const split2 = new cv.MatVector();
  cv.split(complex, split2);
  const re = split2.get(0);
  const im = split2.get(1);

  const result = new cv.Mat();
  if (asPhase) {
    cv.phase(re, im, result, false);
  } else {
    cv.magnitude(re, im, result);
    if (useLogScale) {
      const ones = cv.Mat.ones(result.rows, result.cols, cv.CV_32F);
      cv.add(result, ones, result);
      ones.delete();
      cv.log(result, result);
    }
  }
  split2.delete();
  complex.delete();

  // 5. Crop to even size and optionally rearrange quadrants (FFT shift)
  const cx = (result.cols / 2) | 0;
  const cy = (result.rows / 2) | 0;
  const evenW = cx * 2;
  const evenH = cy * 2;
  const cropped = result.roi(new cv.Rect(0, 0, evenW, evenH)).clone();
  result.delete();

  if (shouldShift) {
    const q0 = cropped.roi(new cv.Rect(0, 0, cx, cy));
    const q1 = cropped.roi(new cv.Rect(cx, 0, cx, cy));
    const q2 = cropped.roi(new cv.Rect(0, cy, cx, cy));
    const q3 = cropped.roi(new cv.Rect(cx, cy, cx, cy));
    const tmp = new cv.Mat();
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    tmp.delete();
    q0.delete(); q1.delete(); q2.delete(); q3.delete();
  }

  // 6. Normalise to 0-255 and convert to uchar
  cv.normalize(cropped, cropped, 0, 255, cv.NORM_MINMAX);
  const out = new cv.Mat();
  cropped.convertTo(out, cv.CV_8U);
  cropped.delete();
  return out;
}

// ─── Connected-component area filter ───────────────────────────
function connectedComponentAreaFilter(cv, src, minArea, connectivity = 8) {
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();
  const cc = connectivity === 4 ? 4 : 8;
  const num = cv.connectedComponentsWithStats(src, labels, stats, centroids, cc, cv.CV_32S);

  // Decide which labels to keep
  const keep = new Uint8Array(num);
  for (let i = 1; i < num; i++) {
    if (stats.intAt(i, cv.CC_STAT_AREA) >= minArea) keep[i] = 1;
  }

  const out = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
  const labelData = labels.data32S;
  const outData = out.data;
  for (let i = 0; i < labelData.length; i++) {
    if (keep[labelData[i]]) outData[i] = 255;
  }

  labels.delete(); stats.delete(); centroids.delete();
  return out;
}

// ─── Morphological skeleton (Lantuéjoul) ───────────────────────
function morphologicalSkeleton(cv, src, params = {}) {
  const skel = cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
  let img = src.clone();
  const spec = parseStructSpec(params.struct);
  const k = makeKernel(cv, spec.shape, spec.width, spec.height);
  const maxIter = intOr(params.maxIter, 100, 1);

  let iter = 0;
  while (iter++ < maxIter) {
    const opened = new cv.Mat();
    cv.morphologyEx(img, opened, cv.MORPH_OPEN, k);

    const temp = new cv.Mat();
    cv.subtract(img, opened, temp);
    cv.bitwise_or(skel, temp, skel);

    const eroded = new cv.Mat();
    cv.erode(img, eroded, k);

    img.delete();
    img = eroded;

    const nz = cv.countNonZero(img);
    opened.delete();
    temp.delete();
    if (nz === 0) break;
  }
  img.delete();
  k.delete();
  return skel;
}

// ─── Distance-transform medial ridge ───────────────────────────
function distanceRidge(cv, src) {
  const dist = new cv.Mat();
  cv.distanceTransform(src, dist, cv.DIST_L2, 3);
  // Smooth, subtract, keep positive ridge
  const smooth = new cv.Mat();
  cv.GaussianBlur(dist, smooth, new cv.Size(0, 0), 1.5);
  const ridge = new cv.Mat();
  cv.subtract(dist, smooth, ridge);
  // Threshold above 0
  const ridge8 = new cv.Mat();
  cv.threshold(ridge, ridge8, 0.4, 255, cv.THRESH_BINARY);
  const out = new cv.Mat();
  ridge8.convertTo(out, cv.CV_8U);
  dist.delete(); smooth.delete(); ridge.delete(); ridge8.delete();
  return out;
}

// ─── Optic disc — Hough Circles ────────────────────────────────
function detectOpticDiscHough(cv, input, params = {}) {
  const blurred = new cv.Mat();
  cv.GaussianBlur(input, blurred, new cv.Size(9, 9), 2);

  const minDim = Math.min(input.rows, input.cols);
  const dp = floatOr(params.dp, 1, 0.1);
  const minDist = Math.max(1, Math.round(parseRelativeExpr(params.minDist, input, input.rows / 4)));
  const minR = Math.max(8, Math.round(parseRelativeExpr(params.minR, input, minDim * 0.05)));
  const maxR = Math.max(minR + 1, Math.round(parseRelativeExpr(params.maxR, input, minDim * 0.15)));

  const circles = new cv.Mat();
  cv.HoughCircles(blurred, circles, cv.HOUGH_GRADIENT, dp, minDist, 100, 28, minR, maxR);

  const out = new cv.Mat();
  cv.cvtColor(input, out, cv.COLOR_GRAY2RGBA);
  // Choose the brightest detected circle
  let best = null;
  let bestI = -1;
  for (let i = 0; i < circles.cols; i++) {
    const x = circles.data32F[i * 3];
    const y = circles.data32F[i * 3 + 1];
    const r = circles.data32F[i * 3 + 2];
    const cx = Math.round(x), cy = Math.round(y);
    if (cx >= 0 && cy >= 0 && cx < input.cols && cy < input.rows) {
      const v = input.ucharPtr(cy, cx)[0];
      if (v > bestI) { bestI = v; best = { x: cx, y: cy, r: Math.round(r) }; }
    }
  }
  if (best) {
    cv.circle(out, { x: best.x, y: best.y }, best.r, [125, 183, 241, 255], 3, cv.LINE_AA);
    cv.circle(out, { x: best.x, y: best.y }, 3, [241, 194, 125, 255], 3, cv.LINE_AA);
    out._opticDisc = best; // attach for composite use
  }
  blurred.delete(); circles.delete();
  return out;
}

// ─── Optic disc — brightest centroid ───────────────────────────
function detectOpticDiscCentroid(cv, input, params = {}) {
  const blurred = new cv.Mat();
  cv.GaussianBlur(input, blurred, new cv.Size(31, 31), 0);
  const mask = new cv.Mat();
  const percentile = floatOr(params.percentile, 95, 0, 100);
  const t = percentileValue(blurred.data, percentile);
  cv.threshold(blurred, mask, t, 255, cv.THRESH_BINARY);

  // Largest connected component
  const labels = new cv.Mat(); const stats = new cv.Mat(); const cents = new cv.Mat();
  const n = cv.connectedComponentsWithStats(mask, labels, stats, cents, 8, cv.CV_32S);
  let bestI = 0, bestArea = 0;
  for (let i = 1; i < n; i++) {
    const a = stats.intAt(i, cv.CC_STAT_AREA);
    if (a > bestArea) { bestArea = a; bestI = i; }
  }

  const out = new cv.Mat();
  cv.cvtColor(input, out, cv.COLOR_GRAY2RGBA);
  if (bestI > 0) {
    const cx = Math.round(cents.doubleAt(bestI, 0));
    const cy = Math.round(cents.doubleAt(bestI, 1));
    const w = stats.intAt(bestI, cv.CC_STAT_WIDTH);
    const h = stats.intAt(bestI, cv.CC_STAT_HEIGHT);
    const r = Math.round(Math.max(w, h) / 2);
    cv.circle(out, { x: cx, y: cy }, r, [125, 183, 241, 255], 3, cv.LINE_AA);
    cv.circle(out, { x: cx, y: cy }, 3, [241, 194, 125, 255], 3, cv.LINE_AA);
    out._opticDisc = { x: cx, y: cy, r };
  }
  blurred.delete(); mask.delete(); labels.delete(); stats.delete(); cents.delete();
  return out;
}

// ─── PIPELINE RUNNER ───────────────────────────────────────────
export class Pipeline {
  constructor(cv) {
    this.cv = cv;
    this.state = {};   // stageId -> Mat
    this.config = {};  // stageId -> methodId
    this.paramConfig = {}; // stageId -> methodId -> param overrides
    for (const stage of STAGES) {
      if (stage.defaultMethod) this.config[stage.id] = stage.defaultMethod;
    }
    // toggles
    this.showVessels = true;
    this.showLesions = true;
    this.showOpticDisc = true;
  }

  setMethod(stageId, methodId) {
    this.config[stageId] = methodId;
  }

  setMethodParams(stageId, values, methodId = this.config[stageId]) {
    const method = this.getMethod(stageId, methodId);
    if (!method) return {};

    const defaults = method.params || {};
    const schema = this.getParamSchema(stageId, methodId);
    const next = {};

    for (const [key, defaultValue] of Object.entries(defaults)) {
      next[key] = normalizeParamValue(values[key], defaultValue, schema[key] || {});
    }

    if (!this.paramConfig[stageId]) this.paramConfig[stageId] = {};
    this.paramConfig[stageId][methodId] = next;
    return { ...next };
  }

  hasSource() {
    return !!this.state.acquisition;
  }

  setSourceFromCanvas(canvas) {
    const cv = this.cv;
    this._free('acquisition');
    const mat = cv.imread(canvas);   // RGBA, CV_8UC4
    this.state.acquisition = mat;
  }

  getStage(id) { return STAGES.find(s => s.id === id); }
  getMethod(stageId, methodId = this.config[stageId]) {
    const stage = this.getStage(stageId);
    if (!stage || !stage.methods.length) return null;
    return stage.methods.find(m => m.id === methodId) || stage.methods[0];
  }
  getDefaultParams(stageId, methodId = this.config[stageId]) {
    const method = this.getMethod(stageId, methodId);
    return method ? { ...(method.params || {}) } : {};
  }
  getMethodParams(stageId, methodId = this.config[stageId]) {
    const defaults = this.getDefaultParams(stageId, methodId);
    return { ...defaults, ...(this.paramConfig[stageId]?.[methodId] || {}) };
  }
  getParamSchema(stageId, methodId = this.config[stageId]) {
    return PARAM_INFO[stageId]?.[methodId] || {};
  }
  isOptimal(stageId) {
    const m = this.getMethod(stageId);
    return !!(m && m.optimal);
  }
  statusFor(stageId) {
    const stage = this.getStage(stageId);
    if (stage.fixed) return 'optimal';
    if (this.isOptimal(stageId)) return 'optimal';
    return 'modified';
  }

  runStage(stageId) {
    const cv = this.cv;
    const stage = this.getStage(stageId);
    if (!stage) return;

    if (stage.id === 'composite') return this.runComposite();

    if (stage.id === 'acquisition') {
      if (!this.state.acquisition) throw new Error('No source image');
      return;
    }

    const input = this.state[stage.inputFrom];
    if (!input) throw new Error(`Stage ${stage.id} has no input from ${stage.inputFrom}`);

    const method = this.getMethod(stage.id);
    if (!method) throw new Error(`No method selected for ${stage.id}`);
    const params = this.getMethodParams(stage.id);

    this._free(stage.id);
    try {
      const out = method.run(input, cv, params, this);
      this.state[stage.id] = out;
    } catch (err) {
      // Branch failures shouldn't kill the whole pipeline
      console.warn(`Stage ${stage.id} failed:`, err);
      if (!stage.branch) throw err;
    }
  }

  runFrom(stageId) {
    let started = false;
    for (const s of STAGES) {
      if (!started && s.id !== stageId) continue;
      started = true;
      this.runStage(s.id);
    }
  }

  runAll() {
    for (const s of STAGES) this.runStage(s.id);
  }

  // ─── Composite ───────────────────────────────────────────────
  runComposite() {
    const cv = this.cv;
    const original = this.state.acquisition;
    if (!original) return;
    const overlayParams = this.getMethodParams('composite', 'overlay');
    const vesselColor = hexToRgb(overlayParams.vesselColor, [148, 211, 193]);
    const lesionColor = hexToRgb(overlayParams.lesionColor, [241, 194, 125]);
    const opticColor = hexToRgb(overlayParams.odColor, [125, 183, 241]);
    const alpha = floatOr(overlayParams.alpha, 0.75, 0, 1);

    const rgb = new cv.Mat();
    cv.cvtColor(original, rgb, cv.COLOR_RGBA2RGB);

    // Vessel layer — prefer cleanup, fall back to segmentation
    if (this.showVessels) {
      const vesselMask = this.state.cleanup || this.state.segmentation;
      if (vesselMask) {
        const mask = this._matchSize(vesselMask, rgb);
        this._tintWithMask(rgb, mask, vesselColor, alpha);
        mask.delete();
      }
    }
    // Lesion layer
    if (this.showLesions && this.state.lesion) {
      const mask = this._matchSize(this.state.lesion, rgb);
      this._tintWithMask(rgb, mask, lesionColor, alpha);
      mask.delete();
    }

    const finalRgba = new cv.Mat();
    cv.cvtColor(rgb, finalRgba, cv.COLOR_RGB2RGBA);
    rgb.delete();

    // Optic disc indicator
    if (this.showOpticDisc && this.state.opticDisc && this.state.opticDisc._opticDisc) {
      const od = this.state.opticDisc._opticDisc;
      // Coordinates were computed on the contrast-stage Mat; that has the same
      // dimensions as the channel-extracted image. Scale to original size.
      const refMat = this.state.contrast || this.state.channel;
      const sx = refMat ? finalRgba.cols / refMat.cols : 1;
      const sy = refMat ? finalRgba.rows / refMat.rows : 1;
      const x = Math.round(od.x * sx);
      const y = Math.round(od.y * sy);
      const r = Math.round(od.r * Math.max(sx, sy));
      cv.circle(finalRgba, { x, y }, r, [...opticColor, 255], 3, cv.LINE_AA);
      cv.circle(finalRgba, { x, y }, 3, [...opticColor, 255], 3, cv.LINE_AA);
    }

    this._free('composite');
    this.state.composite = finalRgba;
  }

  _matchSize(mask, dst) {
    const cv = this.cv;
    if (mask.rows === dst.rows && mask.cols === dst.cols) return mask.clone();
    const out = new cv.Mat();
    cv.resize(mask, out, new cv.Size(dst.cols, dst.rows), 0, 0, cv.INTER_NEAREST);
    return out;
  }

  _tintWithMask(rgb, mask, color, alpha) {
    const data = rgb.data;
    const m = mask.data;
    const len = m.length;
    const a = alpha;
    const ia = 1 - alpha;
    for (let i = 0, j = 0; i < len; i++, j += 3) {
      if (m[i]) {
        data[j]     = ia * data[j]     + a * color[0];
        data[j + 1] = ia * data[j + 1] + a * color[1];
        data[j + 2] = ia * data[j + 2] + a * color[2];
      }
    }
  }

  // ─── Display helpers ─────────────────────────────────────────
  renderStage(stageId, canvas) {
    const cv = this.cv;
    const mat = this.state[stageId];
    if (!mat) return false;
    const rgba = mat.channels() === 4 ? mat : toRgba(cv, mat);
    cv.imshow(canvas, rgba);
    if (rgba !== mat) rgba.delete();
    return true;
  }

  statsFor(stageId) {
    const cv = this.cv;
    const mat = this.state[stageId];
    if (!mat) return null;

    let gray;
    if (mat.channels() === 1) {
      gray = mat;
    } else if (mat.channels() === 4) {
      gray = new cv.Mat();
      cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    } else {
      gray = new cv.Mat();
      cv.cvtColor(mat, gray, cv.COLOR_RGB2GRAY);
    }

    const bins = new Uint32Array(256);
    const data = gray.data;
    let min = 255, max = 0, sum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      bins[v]++;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    const n = data.length;
    const mean = sum / n;
    let varSum = 0;
    for (let i = 0; i < data.length; i++) {
      const d = data[i] - mean;
      varSum += d * d;
    }
    const std = Math.sqrt(varSum / n);

    let entropy = 0;
    for (let i = 0; i < 256; i++) {
      if (bins[i]) {
        const p = bins[i] / n;
        entropy -= p * Math.log2(p);
      }
    }

    if (gray !== mat) gray.delete();
    return { bins, mean, std, min, max, entropy };
  }

  // ─── Memory ──────────────────────────────────────────────────
  _free(stageId) {
    const m = this.state[stageId];
    if (m) {
      try { m.delete(); } catch {}
      delete this.state[stageId];
    }
  }
  destroy() {
    for (const k of Object.keys(this.state)) this._free(k);
  }
}
