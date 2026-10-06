import { contours as d3Contours } from "https://cdn.jsdelivr.net/npm/d3-contour@4/+esm";

const ENCODER_MODEL_URL =
  "https://raw.githubusercontent.com/krngd2/SamOnWeb/main/fp16.sam2_hiera_tiny.encoder.ort";
const DECODER_MODEL_URL =
  "https://raw.githubusercontent.com/krngd2/SamOnWeb/main/fp16.sam2_hiera_tiny.decoder.ort";

const MODEL_SIZE = 1024;
const MASK_INPUT_SIZE = 256;

let encoderSession = null;
let decoderSession = null;
let encoderOutputs = null;
let encodedImageId = null;
let encodedWidth = 0;
let encodedHeight = 0;
let preprocess = null;
let encodedEdgeMap = null;

export function isModelLoaded() {
  return !!encoderSession && !!decoderSession;
}

export async function loadAI(onStatus = () => {}) {
  if (isModelLoaded()) return;
  if (!globalThis.ort) throw new Error("ONNX Runtime Webを読み込めませんでした。");

  // This fp16 export is known to run in-browser. Avoid forcing WebGPU here:
  // ORT will select the compatible backend for the model.
  onStatus("互換fp16版 SAM2 Hiera Tiny エンコーダを読み込んでいます…");
  encoderSession = await ort.InferenceSession.create(ENCODER_MODEL_URL);

  onStatus("SAM2デコーダを読み込んでいます…");
  decoderSession = await ort.InferenceSession.create(DECODER_MODEL_URL);

  onStatus("SAM2準備完了");
}

export async function encodeImage(imageId, url, onStatus = () => {}) {
  if (!isModelLoaded()) await loadAI(onStatus);
  if (encodedImageId === imageId && encoderOutputs) return;

  onStatus("SAM2で画像特徴を抽出しています…");
  const img = await loadImage(url);
  encodedWidth = img.naturalWidth || img.width;
  encodedHeight = img.naturalHeight || img.height;

  const prepared = preprocessImage(img);
  preprocess = prepared.transform;
  encodedEdgeMap = prepared.edgeMap;

  encoderOutputs = await encoderSession.run({
    image: prepared.tensor,
  });

  if (!encoderOutputs.image_embed) {
    throw new Error(
      "SAM2 encoder output image_embed が見つかりません: " +
      Object.keys(encoderOutputs).join(", ")
    );
  }

  encodedImageId = imageId;
  onStatus("画像解析完了。対象を指定してください。");
}

export function clearEncodedImage() {
  encoderOutputs = null;
  encodedImageId = null;
  encodedWidth = 0;
  encodedHeight = 0;
  preprocess = null;
  encodedEdgeMap = null;
}

export async function segment(points, box = null, options = {}) {
  if (!encoderOutputs || !preprocess) {
    throw new Error("画像のSAM2解析がまだ完了していません。");
  }

  const promptPoints = [];

  if (box) {
    const p1 = normalizedToModel(box.x1, box.y1);
    const p2 = normalizedToModel(box.x2, box.y2);
    promptPoints.push({ x: p1.x, y: p1.y, label: 2 });
    promptPoints.push({ x: p2.x, y: p2.y, label: 3 });
  }

  for (const p of points || []) {
    const q = normalizedToModel(p.x, p.y);
    promptPoints.push({
      x: q.x,
      y: q.y,
      label: p.label === 0 ? 0 : 1,
    });
  }

  if (!promptPoints.length) {
    throw new Error("SAM2への指定点がありません。");
  }

  const coordsF32 = new Float32Array(promptPoints.length * 2);
  const labelsF32 = new Float32Array(promptPoints.length);
  promptPoints.forEach((p, i) => {
    coordsF32[i * 2] = p.x;
    coordsF32[i * 2 + 1] = p.y;
    labelsF32[i] = p.label;
  });

  const pointCoords = new ort.Tensor(
    "float16",
    float32ArrayToFloat16(coordsF32),
    [1, promptPoints.length, 2]
  );
  const pointLabels = new ort.Tensor(
    "float16",
    float32ArrayToFloat16(labelsF32),
    [1, promptPoints.length]
  );

  const maskInput = new ort.Tensor(
    "float16",
    new Uint16Array(MASK_INPUT_SIZE * MASK_INPUT_SIZE),
    [1, 1, MASK_INPUT_SIZE, MASK_INPUT_SIZE]
  );

  const hasMaskInput = new ort.Tensor(
    "float16",
    float32ArrayToFloat16(new Float32Array([0])),
    [1]
  );

  const feeds = {
    image_embed: encoderOutputs.image_embed,
    point_coords: pointCoords,
    point_labels: pointLabels,
    mask_input: maskInput,
    has_mask_input: hasMaskInput,
  };

  if (encoderOutputs.high_res_feats_0) {
    feeds.high_res_feats_0 = encoderOutputs.high_res_feats_0;
  }
  if (encoderOutputs.high_res_feats_1) {
    feeds.high_res_feats_1 = encoderOutputs.high_res_feats_1;
  }

  const results = await decoderSession.run(feeds);
  const masks =
    results.masks ||
    results.pred_masks ||
    Object.values(results).find((v) => v?.dims?.length >= 3);

  if (!masks) {
    throw new Error(
      "SAM2 mask output が見つかりません: " +
      Object.keys(results).join(", ")
    );
  }

  const dims = masks.dims;
  const maskH = dims[dims.length - 2];
  const maskW = dims[dims.length - 1];
  const plane = maskW * maskH;
  const channels = Math.max(1, Math.floor(masks.data.length / plane));

  let bestChannel = 0;
  let bestScore = -Infinity;

  const iou =
    results.iou_predictions ||
    results.iou_scores ||
    results.predicted_iou;

  for (let c = 0; c < channels; c++) {
    let correct = 0;
    let area = 0;

    for (const p of points || []) {
      const q = normalizedToMask(p.x, p.y, maskW, maskH);
      const idx = c * plane + q.y * maskW + q.x;
      const inside = Number(masks.data[idx]) > 0;
      if ((p.label === 1 && inside) || (p.label === 0 && !inside)) correct++;
    }

    for (let i = 0; i < plane; i += 8) {
      if (Number(masks.data[c * plane + i]) > 0) area++;
    }

    const promptFit = points?.length ? correct / points.length : 0.5;
    const areaRatio = area / Math.ceil(plane / 8);
    const iouScore = iou?.data?.[c] != null ? Number(iou.data[c]) : 0;
    const sizePenalty = areaRatio > 0.92 ? (areaRatio - 0.92) * 4 : 0;

    const score = promptFit * 0.65 + iouScore * 0.35 - sizePenalty;
    if (score > bestScore) {
      bestScore = score;
      bestChannel = c;
    }
  }

  // Resample model mask back into the original encoded image coordinate system.
  let binary = new Uint8Array(encodedWidth * encodedHeight);

  for (let y = 0; y < encodedHeight; y++) {
    for (let x = 0; x < encodedWidth; x++) {
      const m = originalToMask(x, y, maskW, maskH);
      const idx = bestChannel * plane + m.y * maskW + m.x;
      binary[y * encodedWidth + x] = Number(masks.data[idx]) > 0 ? 1 : 0;
    }
  }

  const positive = (points || []).find((p) => p.label === 1);
  const anchor = positive
    ? {
        x: Math.round(clamp01(positive.x) * (encodedWidth - 1)),
        y: Math.round(clamp01(positive.y) * (encodedHeight - 1)),
      }
    : box
      ? {
          x: Math.round(((box.x1 + box.x2) / 2) * (encodedWidth - 1)),
          y: Math.round(((box.y1 + box.y2) / 2) * (encodedHeight - 1)),
        }
      : {
          x: Math.floor(encodedWidth / 2),
          y: Math.floor(encodedHeight / 2),
        };

  binary = selectPromptComponent(binary, encodedWidth, encodedHeight, anchor);

  let polygon = maskToPolygon(binary, encodedWidth, encodedHeight, anchor);
  if (polygon.length < 3) {
    throw new Error("SAM2で有効な輪郭を取得できませんでした。");
  }

  if (encodedEdgeMap && options.edgeSnap !== false) {
    polygon = snapPolygonToEdges(
      polygon,
      encodedWidth,
      encodedHeight,
      encodedEdgeMap
    );
  }

  return {
    width: encodedWidth,
    height: encodedHeight,
    score: bestScore,
    polygon,
  };
}

function normalizedToModel(nx, ny) {
  const x = clamp01(nx) * encodedWidth;
  const y = clamp01(ny) * encodedHeight;
  return {
    x: x * preprocess.scale + preprocess.offsetX,
    y: y * preprocess.scale + preprocess.offsetY,
  };
}

function normalizedToMask(nx, ny, maskW, maskH) {
  const p = normalizedToModel(nx, ny);
  return {
    x: Math.max(0, Math.min(maskW - 1, Math.round((p.x / MODEL_SIZE) * (maskW - 1)))),
    y: Math.max(0, Math.min(maskH - 1, Math.round((p.y / MODEL_SIZE) * (maskH - 1)))),
  };
}

function originalToMask(x, y, maskW, maskH) {
  const mx = x * preprocess.scale + preprocess.offsetX;
  const my = y * preprocess.scale + preprocess.offsetY;
  return {
    x: Math.max(0, Math.min(maskW - 1, Math.round((mx / MODEL_SIZE) * (maskW - 1)))),
    y: Math.max(0, Math.min(maskH - 1, Math.round((my / MODEL_SIZE) * (maskH - 1)))),
  };
}

async function loadImage(url) {
  return await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("画像を読み込めませんでした。"));
    img.src = url;
  });
}

function preprocessImage(image) {
  const canvas = document.createElement("canvas");
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  const scale = Math.min(MODEL_SIZE / width, MODEL_SIZE / height);
  const scaledWidth = width * scale;
  const scaledHeight = height * scale;
  const offsetX = (MODEL_SIZE - scaledWidth) / 2;
  const offsetY = (MODEL_SIZE - scaledHeight) / 2;

  ctx.clearRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  ctx.drawImage(image, offsetX, offsetY, scaledWidth, scaledHeight);

  const rgba = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const plane = MODEL_SIZE * MODEL_SIZE;
  const f16 = new Uint16Array(3 * plane);
  const gray = new Float32Array(width * height);

  for (let i = 0; i < plane; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];

    f16[i] = float32ToFloat16((r / 255 - 0.485) / 0.229);
    f16[plane + i] = float32ToFloat16((g / 255 - 0.456) / 0.224);
    f16[plane * 2 + i] = float32ToFloat16((b / 255 - 0.406) / 0.225);
  }

  // Build edge map in original encoded-image coordinates.
  const originalCanvas = document.createElement("canvas");
  originalCanvas.width = width;
  originalCanvas.height = height;
  const octx = originalCanvas.getContext("2d", { willReadFrequently: true });
  octx.drawImage(image, 0, 0, width, height);
  const org = octx.getImageData(0, 0, width, height).data;

  for (let i = 0; i < width * height; i++) {
    gray[i] =
      0.299 * org[i * 4] +
      0.587 * org[i * 4 + 1] +
      0.114 * org[i * 4 + 2];
  }

  return {
    tensor: new ort.Tensor("float16", f16, [1, 3, MODEL_SIZE, MODEL_SIZE]),
    transform: { scale, offsetX, offsetY, scaledWidth, scaledHeight },
    edgeMap: sobelMagnitude(gray, width, height),
  };
}

function sobelMagnitude(gray, width, height) {
  const out = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const tl = gray[i - width - 1], tc = gray[i - width], tr = gray[i - width + 1];
      const ml = gray[i - 1], mr = gray[i + 1];
      const bl = gray[i + width - 1], bc = gray[i + width], br = gray[i + width + 1];

      const gx = -tl - 2 * ml - bl + tr + 2 * mr + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      out[i] = Math.hypot(gx, gy);
    }
  }
  return out;
}

function selectPromptComponent(binary, width, height, anchor) {
  const visited = new Uint8Array(binary.length);
  const dirs = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]];
  const components = [];

  for (let start = 0; start < binary.length; start++) {
    if (!binary[start] || visited[start]) continue;

    const stack = [start];
    visited[start] = 1;
    const pixels = [];
    let minD2 = Infinity;

    while (stack.length) {
      const idx = stack.pop();
      pixels.push(idx);
      const x = idx % width;
      const y = (idx / width) | 0;
      const d2 = (x - anchor.x) ** 2 + (y - anchor.y) ** 2;
      if (d2 < minD2) minD2 = d2;

      for (const [dx, dy] of dirs) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const ni = ny * width + nx;
        if (binary[ni] && !visited[ni]) {
          visited[ni] = 1;
          stack.push(ni);
        }
      }
    }

    components.push({ pixels, minD2 });
  }

  if (!components.length) return binary;

  components.sort((a, b) => {
    if (Math.abs(a.minD2 - b.minD2) > 1) return a.minD2 - b.minD2;
    return b.pixels.length - a.pixels.length;
  });

  const out = new Uint8Array(binary.length);
  for (const i of components[0].pixels) out[i] = 1;
  return out;
}

function maskToPolygon(binary, width, height, anchor) {
  const values = Float32Array.from(binary);
  const contour = d3Contours()
    .size([width, height])
    .thresholds([0.5])(values)[0];

  if (!contour?.coordinates?.length) return [];

  const rings = [];
  for (const poly of contour.coordinates) {
    const ring = poly?.[0];
    if (!ring || ring.length < 4) continue;
    rings.push({
      ring,
      contains: pointInRing(anchor, ring),
      area: Math.abs(signedArea(ring)),
    });
  }

  if (!rings.length) return [];

  const candidates = rings.filter((r) => r.contains);
  const chosen = (candidates.length ? candidates : rings)
    .sort((a, b) => b.area - a.area)[0];

  let points = chosen.ring.slice(0, -1).map(([x, y]) => ({ x, y }));

  const epsilon = Math.max(0.2, Math.min(width, height) * 0.00025);
  points = simplifyClosed(points, epsilon);

  if (points.length > 1800) {
    const step = Math.ceil(points.length / 1800);
    points = points.filter((_, i) => i % step === 0);
  }

  return points;
}

function snapPolygonToEdges(points, width, height, edgeMap) {
  if (!points || points.length < 6) return points;

  const radius = Math.max(3, Math.min(10, Math.round(Math.min(width, height) * 0.012)));
  const snapped = new Array(points.length);

  for (let i = 0; i < points.length; i++) {
    const prev = points[(i - 1 + points.length) % points.length];
    const cur = points[i];
    const next = points[(i + 1) % points.length];

    const tx = next.x - prev.x;
    const ty = next.y - prev.y;
    const len = Math.hypot(tx, ty);

    if (len < 1e-6) {
      snapped[i] = { ...cur };
      continue;
    }

    const nx = -ty / len;
    const ny = tx / len;

    let best = { x: cur.x, y: cur.y };
    let bestScore = sampleEdge(edgeMap, width, height, cur.x, cur.y);

    for (let d = -radius; d <= radius; d += 0.5) {
      const x = cur.x + nx * d;
      const y = cur.y + ny * d;
      if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) continue;

      const edge = sampleEdge(edgeMap, width, height, x, y);
      const score = edge - 8.0 * Math.abs(d);

      if (score > bestScore) {
        bestScore = score;
        best = { x, y };
      }
    }

    snapped[i] = best;
  }

  return smoothClosedPolygon(snapped, 1);
}

function sampleEdge(edgeMap, width, height, x, y) {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;

  const a = edgeMap[y0 * width + x0];
  const b = edgeMap[y0 * width + x1];
  const c = edgeMap[y1 * width + x0];
  const d = edgeMap[y1 * width + x1];

  return (a * (1 - fx) + b * fx) * (1 - fy) +
         (c * (1 - fx) + d * fx) * fy;
}

function smoothClosedPolygon(points, passes = 1) {
  let out = points.map((p) => ({ ...p }));

  for (let pass = 0; pass < passes; pass++) {
    const next = new Array(out.length);
    for (let i = 0; i < out.length; i++) {
      const a = out[(i - 1 + out.length) % out.length];
      const b = out[i];
      const c = out[(i + 1) % out.length];
      next[i] = {
        x: a.x * 0.15 + b.x * 0.70 + c.x * 0.15,
        y: a.y * 0.15 + b.y * 0.70 + c.y * 0.15,
      };
    }
    out = next;
  }

  return out;
}

function pointInRing(p, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    const hit =
      ((yi > p.y) !== (yj > p.y)) &&
      (p.x < ((xj - xi) * (p.y - yi)) / ((yj - yi) || 1e-12) + xi);
    if (hit) inside = !inside;
  }
  return inside;
}

function signedArea(points) {
  let area = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    area += points[j][0] * points[i][1] - points[i][0] * points[j][1];
  }
  return area / 2;
}

function simplifyClosed(points, epsilon) {
  if (points.length < 4) return points;
  const closed = [...points, points[0]];
  const simplified = rdp(closed, epsilon);
  if (simplified.length > 1) simplified.pop();
  return simplified;
}

function rdp(points, epsilon) {
  if (points.length < 3) return points.slice();

  const first = points[0];
  const last = points[points.length - 1];
  let maxDist = 0;
  let index = 0;

  for (let i = 1; i < points.length - 1; i++) {
    const d = pointLineDistance(points[i], first, last);
    if (d > maxDist) {
      maxDist = d;
      index = i;
    }
  }

  if (maxDist > epsilon) {
    const left = rdp(points.slice(0, index + 1), epsilon);
    const right = rdp(points.slice(index), epsilon);
    return left.slice(0, -1).concat(right);
  }

  return [first, last];
}

function pointLineDistance(p, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;

  if (dx === 0 && dy === 0) {
    return Math.hypot(p.x - a.x, p.y - a.y);
  }

  const t = Math.max(
    0,
    Math.min(
      1,
      ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy)
    )
  );

  return Math.hypot(
    p.x - (a.x + t * dx),
    p.y - (a.y + t * dy)
  );
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function float32ArrayToFloat16(arr) {
  const out = new Uint16Array(arr.length);
  for (let i = 0; i < arr.length; i++) out[i] = float32ToFloat16(arr[i]);
  return out;
}

const float32ToFloat16 = (() => {
  const f = new Float32Array(1);
  const i = new Int32Array(f.buffer);

  return (val) => {
    f[0] = val;
    const x = i[0];

    const sign = (x >> 16) & 0x8000;
    let exp = ((x >> 23) & 0xff) - 127 + 15;
    let mant = x & 0x7fffff;

    if (exp <= 0) {
      if (exp < -10) return sign;
      mant = (mant | 0x800000) >> (1 - exp);
      return sign | ((mant + 0x1000) >> 13);
    }

    if (exp >= 31) {
      return sign | 0x7c00;
    }

    return sign | (exp << 10) | ((mant + 0x1000) >> 13);
  };
})();
