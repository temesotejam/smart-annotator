import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0";
import { contours as d3Contours } from "https://cdn.jsdelivr.net/npm/d3-contour@4/+esm";

const MODEL_ID = "Xenova/sam-vit-base";
let model = null;
let processor = null;
let imageProcessed = null;
let imageEmbeddings = null;
let encodedImageId = null;

export function isModelLoaded() {
  return !!model && !!processor;
}

export async function loadAI(onStatus = () => {}) {
  if (isModelLoaded()) return;
  if (!navigator.gpu) {
    throw new Error("WebGPUが利用できません。最新版のChromeまたはEdgeで開いてください。");
  }

  onStatus("高精度SAM ViT-Bモデルを読み込んでいます…（初回は大きめのダウンロードがあります）");
  model = await SamModel.from_pretrained(MODEL_ID, {
    dtype: "fp16",
    device: "webgpu",
  });
  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  onStatus("AIモデル準備完了");
}

export async function encodeImage(imageId, url, onStatus = () => {}) {
  if (!isModelLoaded()) await loadAI(onStatus);
  if (encodedImageId === imageId && imageEmbeddings) return;

  onStatus("この画像をAI用に解析しています…");
  const input = await RawImage.fromURL(url);
  imageProcessed = await processor(input);
  imageEmbeddings = await model.get_image_embeddings(imageProcessed);
  encodedImageId = imageId;
  onStatus("画像解析完了。対象をクリックしてください。");
}

export function clearEncodedImage() {
  imageProcessed = null;
  imageEmbeddings = null;
  encodedImageId = null;
}

export async function segment(points, box = null) {
  if (!imageEmbeddings || !imageProcessed) {
    throw new Error("画像のAI解析がまだ完了していません。");
  }
  if (!points.length) throw new Error("対象点がありません。");

  const reshaped = imageProcessed.reshaped_input_sizes[0];
  const pointData = points
    .map((p) => [p.x * reshaped[1], p.y * reshaped[0]])
    .flat();
  const labels = points.map((p) => BigInt(p.label));

  const input_points = new Tensor("float32", pointData, [1, 1, points.length, 2]);
  const input_labels = new Tensor("int64", labels, [1, 1, points.length]);

  const modelInputs = {
    ...imageEmbeddings,
    input_points,
    input_labels,
  };

  if (box) {
    const x1 = Math.max(0, Math.min(1, box.x1)) * reshaped[1];
    const y1 = Math.max(0, Math.min(1, box.y1)) * reshaped[0];
    const x2 = Math.max(0, Math.min(1, box.x2)) * reshaped[1];
    const y2 = Math.max(0, Math.min(1, box.y2)) * reshaped[0];
    modelInputs.input_boxes = new Tensor("float32", [x1, y1, x2, y2], [1, 1, 4]);
  }

  const { pred_masks, iou_scores } = await model(modelInputs);

  const masks = await processor.post_process_masks(
    pred_masks,
    imageProcessed.original_sizes,
    imageProcessed.reshaped_input_sizes,
  );

  const rawMask = RawImage.fromTensor(masks[0][0]);
  const scores = Array.from(iou_scores.data);
  const nMasks = scores.length;

  // Choose the candidate that best obeys the user's prompts, not only SAM's IoU score.
  let best = 0;
  let bestQuality = -Infinity;
  for (let m = 0; m < nMasks; m++) {
    let correct = 0;
    for (const p of points) {
      const px = Math.max(0, Math.min(rawMask.width - 1, Math.round(p.x * (rawMask.width - 1))));
      const py = Math.max(0, Math.min(rawMask.height - 1, Math.round(p.y * (rawMask.height - 1))));
      const inside = rawMask.data[nMasks * (py * rawMask.width + px) + m] === 1;
      if ((p.label === 1 && inside) || (p.label === 0 && !inside)) correct++;
    }
    const promptFit = points.length ? correct / points.length : 0;

    let outsideRatio = 0;
    if (box) {
      const bx1 = Math.floor(Math.max(0, Math.min(1, box.x1)) * rawMask.width);
      const by1 = Math.floor(Math.max(0, Math.min(1, box.y1)) * rawMask.height);
      const bx2 = Math.ceil(Math.max(0, Math.min(1, box.x2)) * rawMask.width);
      const by2 = Math.ceil(Math.max(0, Math.min(1, box.y2)) * rawMask.height);
      let total = 0, outside = 0;
      for (let y = 0; y < rawMask.height; y += 2) {
        for (let x = 0; x < rawMask.width; x += 2) {
          if (rawMask.data[nMasks * (y * rawMask.width + x) + m] !== 1) continue;
          total++;
          if (x < bx1 || x > bx2 || y < by1 || y > by2) outside++;
        }
      }
      outsideRatio = total ? outside / total : 1;
    }

    const quality = scores[m] + 0.30 * promptFit - 0.45 * outsideRatio;
    if (quality > bestQuality) { bestQuality = quality; best = m; }
  }

  const binary = new Uint8Array(rawMask.width * rawMask.height);
  for (let i = 0; i < binary.length; i++) {
    binary[i] = rawMask.data[nMasks * i + best] === 1 ? 1 : 0;
  }

  const positive = points.find((p) => p.label === 1) || points[0];
  const anchor = {
    x: positive.x * rawMask.width,
    y: positive.y * rawMask.height,
  };
  const polygon = maskToPolygon(binary, rawMask.width, rawMask.height, anchor);
  if (polygon.length < 3) throw new Error("輪郭を取り出せませんでした。別の点をクリックしてください。");

  return {
    width: rawMask.width,
    height: rawMask.height,
    score: scores[best],
    polygon,
  };
}

function maskToPolygon(binary, width, height, anchor = null) {
  const values = Float32Array.from(binary);
  const geo = d3Contours()
    .size([width, height])
    .thresholds([0.5])(values)[0];

  if (!geo?.coordinates?.length) return [];

  const rings = [];
  for (const polygon of geo.coordinates) {
    const ring = polygon?.[0];
    if (!ring || ring.length < 4) continue;
    rings.push({
      ring,
      area: Math.abs(signedArea(ring)),
      containsAnchor: anchor ? pointInRing(anchor, ring) : false,
    });
  }
  if (!rings.length) return [];

  // Most important: use the contour that contains the user's positive prompt.
  // Fall back to the largest component only if no component contains it.
  const candidates = anchor ? rings.filter((r) => r.containsAnchor) : [];
  const chosen = (candidates.length ? candidates : rings)
    .sort((a, b) => b.area - a.area)[0];

  let points = chosen.ring.slice(0, -1).map(([x, y]) => ({ x, y }));

  // Keep substantially more contour detail than before.
  const epsilon = Math.max(0.7, Math.min(width, height) * 0.001);
  points = simplifyClosed(points, epsilon);

  // High-detail cap: enough for accurate boundaries while keeping JSON manageable.
  if (points.length > 700) {
    const step = Math.ceil(points.length / 700);
    points = points.filter((_, i) => i % step === 0);
  }
  return points;
}

function pointInRing(p, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    const intersect =
      ((yi > p.y) !== (yj > p.y)) &&
      (p.x < ((xj - xi) * (p.y - yi)) / ((yj - yi) || 1e-12) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function signedArea(points) {
  let a = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    a += points[j][0] * points[i][1] - points[i][0] * points[j][1];
  }
  return a / 2;
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
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}
