import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.0";
import { contours as d3Contours } from "https://cdn.jsdelivr.net/npm/d3-contour@4/+esm";

const MODEL_ID = "Xenova/slimsam-77-uniform";
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

  onStatus("SlimSAMモデルを読み込んでいます…");
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

export async function segment(points) {
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

  const { pred_masks, iou_scores } = await model({
    ...imageEmbeddings,
    input_points,
    input_labels,
  });

  const masks = await processor.post_process_masks(
    pred_masks,
    imageProcessed.original_sizes,
    imageProcessed.reshaped_input_sizes,
  );

  const rawMask = RawImage.fromTensor(masks[0][0]);
  const scores = Array.from(iou_scores.data);
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;

  const nMasks = scores.length;
  const binary = new Uint8Array(rawMask.width * rawMask.height);
  for (let i = 0; i < binary.length; i++) {
    binary[i] = rawMask.data[nMasks * i + best] === 1 ? 1 : 0;
  }

  const polygon = maskToPolygon(binary, rawMask.width, rawMask.height);
  if (polygon.length < 3) throw new Error("輪郭を取り出せませんでした。別の点をクリックしてください。");

  return {
    width: rawMask.width,
    height: rawMask.height,
    score: scores[best],
    polygon,
  };
}

function maskToPolygon(binary, width, height) {
  const values = Float32Array.from(binary);
  const geo = d3Contours()
    .size([width, height])
    .thresholds([0.5])(values)[0];

  if (!geo?.coordinates?.length) return [];

  let bestRing = null;
  let bestArea = -1;
  for (const polygon of geo.coordinates) {
    const ring = polygon?.[0];
    if (!ring || ring.length < 4) continue;
    const area = Math.abs(signedArea(ring));
    if (area > bestArea) {
      bestArea = area;
      bestRing = ring;
    }
  }
  if (!bestRing) return [];

  // D3 contour coordinates are on pixel boundaries. Remove duplicate closing point.
  let points = bestRing.slice(0, -1).map(([x, y]) => ({ x, y }));
  const epsilon = Math.max(1.5, Math.min(width, height) * 0.0025);
  points = simplifyClosed(points, epsilon);

  // Keep exports manageable while preserving the outline.
  if (points.length > 300) {
    const step = Math.ceil(points.length / 300);
    points = points.filter((_, i) => i % step === 0);
  }
  return points;
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
