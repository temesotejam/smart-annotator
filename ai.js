import { contours as d3Contours } from "https://cdn.jsdelivr.net/npm/d3-contour@4/+esm";

const ENCODER_MODEL_URL = "https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_tiny.encoder.ort";
const DECODER_MODEL_URL = "https://storage.googleapis.com/lb-artifacts-testing-public/sam2/sam2_hiera_tiny.decoder.onnx";
const INPUT_SIZE = 1024;
const MASK_INPUT_SIZE = 256;

let encoderSession = null;
let decoderSession = null;
let imageEmbedding = null;
let encodedImageId = null;

export function isModelLoaded() {
  return !!encoderSession && !!decoderSession;
}

export async function loadAI(onStatus = () => {}) {
  if (isModelLoaded()) return;
  if (!globalThis.ort) throw new Error("ONNX Runtime Webを読み込めませんでした。");

  onStatus("SAM2 Hiera Tinyエンコーダを読み込んでいます…");
  const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];

  encoderSession = await ort.InferenceSession.create(ENCODER_MODEL_URL, {
    executionProviders: providers,
  });

  onStatus("SAM2デコーダを読み込んでいます…");
  decoderSession = await ort.InferenceSession.create(DECODER_MODEL_URL, {
    executionProviders: providers,
  });

  onStatus("SAM2準備完了");
}

export async function encodeImage(imageId, url, onStatus = () => {}) {
  if (!isModelLoaded()) await loadAI(onStatus);
  if (encodedImageId === imageId && imageEmbedding) return;

  onStatus("SAM2で画像特徴を抽出しています…");
  const img = await loadImage(url);
  const tensor = imageToTensor(img);
  const results = await encoderSession.run({ image: tensor });

  imageEmbedding =
    results.image_embed ||
    results.image_embeddings ||
    Object.values(results)[0];

  if (!imageEmbedding) throw new Error("SAM2エンコーダ出力を取得できませんでした。");

  encodedImageId = imageId;
  onStatus("画像解析完了。対象を指定してください。");
}

export function clearEncodedImage() {
  imageEmbedding = null;
  encodedImageId = null;
}

export async function segment(points, box = null, options = {}) {
  if (!imageEmbedding) throw new Error("画像のSAM2解析がまだ完了していません。");

  const promptPoints = [];

  // SAM2 represents box prompts as two points with labels 2 and 3.
  if (box) {
    promptPoints.push({
      x: clamp01(box.x1) * INPUT_SIZE,
      y: clamp01(box.y1) * INPUT_SIZE,
      label: 2,
    });
    promptPoints.push({
      x: clamp01(box.x2) * INPUT_SIZE,
      y: clamp01(box.y2) * INPUT_SIZE,
      label: 3,
    });
  }

  for (const p of points || []) {
    promptPoints.push({
      x: clamp01(p.x) * INPUT_SIZE,
      y: clamp01(p.y) * INPUT_SIZE,
      label: p.label === 0 ? 0 : 1,
    });
  }

  if (!promptPoints.length) throw new Error("SAM2への指定点がありません。");

  const coords = new Float32Array(promptPoints.length * 2);
  const labels = new Float32Array(promptPoints.length);
  promptPoints.forEach((p, i) => {
    coords[i * 2] = p.x;
    coords[i * 2 + 1] = p.y;
    labels[i] = p.label;
  });

  const inputs = {
    image_embed: imageEmbedding,
    point_coords: new ort.Tensor("float32", coords, [1, promptPoints.length, 2]),
    point_labels: new ort.Tensor("float32", labels, [1, promptPoints.length]),
    mask_input: new ort.Tensor(
      "float32",
      new Float32Array(MASK_INPUT_SIZE * MASK_INPUT_SIZE),
      [1, 1, MASK_INPUT_SIZE, MASK_INPUT_SIZE],
    ),
    has_mask_input: new ort.Tensor("float32", new Float32Array([0]), [1]),
    high_res_feats_0: new ort.Tensor(
      "float32",
      new Float32Array(1 * 32 * 256 * 256),
      [1, 32, 256, 256],
    ),
    high_res_feats_1: new ort.Tensor(
      "float32",
      new Float32Array(1 * 64 * 128 * 128),
      [1, 64, 128, 128],
    ),
  };

  const results = await decoderSession.run(inputs);
  const maskTensor =
    results.masks ||
    results.pred_masks ||
    Object.values(results).find((v) => v?.dims?.length === 4);

  if (!maskTensor) throw new Error("SAM2マスク出力を取得できませんでした。");

  const height = maskTensor.dims[maskTensor.dims.length - 2];
  const width = maskTensor.dims[maskTensor.dims.length - 1];
  const plane = width * height;
  const channels = Math.max(1, Math.floor(maskTensor.data.length / plane));

  // Select the mask that best respects prompts and has a sensible size.
  let bestChannel = 0;
  let bestScore = -Infinity;
  for (let c = 0; c < channels; c++) {
    let correct = 0;
    let positives = 0;
    let area = 0;

    for (const p of points || []) {
      const px = Math.max(0, Math.min(width - 1, Math.round(clamp01(p.x) * (width - 1))));
      const py = Math.max(0, Math.min(height - 1, Math.round(clamp01(p.y) * (height - 1))));
      const inside = maskTensor.data[c * plane + py * width + px] > 0;
      if (p.label === 1) positives++;
      if ((p.label === 1 && inside) || (p.label === 0 && !inside)) correct++;
    }
    for (let i = 0; i < plane; i += 4) {
      if (maskTensor.data[c * plane + i] > 0) area++;
    }

    const fit = points?.length ? correct / points.length : 0.5;
    const areaRatio = area / Math.ceil(plane / 4);
    const sizePenalty = areaRatio > 0.90 ? (areaRatio - 0.90) * 3 : 0;
    const score = fit - sizePenalty + (positives ? 0.1 : 0);
    if (score > bestScore) {
      bestScore = score;
      bestChannel = c;
    }
  }

  let binary = new Uint8Array(plane);
  for (let i = 0; i < plane; i++) {
    binary[i] = maskTensor.data[bestChannel * plane + i] > 0 ? 1 : 0;
  }

  const positive = (points || []).find((p) => p.label === 1);
  const anchor = positive
    ? {
        x: Math.round(clamp01(positive.x) * (width - 1)),
        y: Math.round(clamp01(positive.y) * (height - 1)),
      }
    : box
      ? {
          x: Math.round(((box.x1 + box.x2) / 2) * (width - 1)),
          y: Math.round(((box.y1 + box.y2) / 2) * (height - 1)),
        }
      : { x: Math.floor(width / 2), y: Math.floor(height / 2) };

  binary = selectPromptComponent(binary, width, height, anchor);
  const polygon = maskToPolygon(binary, width, height, anchor);
  if (polygon.length < 3) throw new Error("SAM2で有効な輪郭を取得できませんでした。");

  return {
    width,
    height,
    score: bestScore,
    polygon,
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

function imageToTensor(image) {
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const rgba = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const plane = INPUT_SIZE * INPUT_SIZE;
  const data = new Float32Array(3 * plane);

  for (let i = 0; i < plane; i++) {
    data[i] = (rgba[i * 4] / 255) * 2 - 1;
    data[plane + i] = (rgba[i * 4 + 1] / 255) * 2 - 1;
    data[plane * 2 + i] = (rgba[i * 4 + 2] / 255) * 2 - 1;
  }
  return new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
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
  const first = points[0], last = points[points.length - 1];
  let maxDist = 0, index = 0;

  for (let i = 1; i < points.length - 1; i++) {
    const d = pointLineDistance(points[i], first, last);
    if (d > maxDist) { maxDist = d; index = i; }
  }
  if (maxDist > epsilon) {
    const left = rdp(points.slice(0, index + 1), epsilon);
    const right = rdp(points.slice(index), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [first, last];
}

function pointLineDistance(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  if (dx === 0 && dy === 0) return Math.hypot(p.x - a.x, p.y - a.y);
  const t = Math.max(0, Math.min(1,
    ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy)
  ));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}
