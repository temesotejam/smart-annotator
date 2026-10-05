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
let encodedRawImage = null;

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
  encodedRawImage = input;
  imageProcessed = await processor(input);
  imageEmbeddings = await model.get_image_embeddings(imageProcessed);
  encodedImageId = imageId;
  onStatus("画像解析完了。対象をクリックしてください。");
}

export function clearEncodedImage() {
  imageProcessed = null;
  imageEmbeddings = null;
  encodedImageId = null;
  encodedRawImage = null;
}

export async function segment(points, box = null, options = {}) {
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

  let binary = new Uint8Array(rawMask.width * rawMask.height);
  for (let i = 0; i < binary.length; i++) {
    binary[i] = rawMask.data[nMasks * i + best] === 1 ? 1 : 0;
  }

  const positive = points.find((p) => p.label === 1) || points[0];
  const anchor = {
    x: Math.max(0, Math.min(rawMask.width - 1, Math.round(positive.x * (rawMask.width - 1)))),
    y: Math.max(0, Math.min(rawMask.height - 1, Math.round(positive.y * (rawMask.height - 1)))),
  };

  // Conservative cleanup only: keep the component linked to the positive prompt.
  binary = selectPromptComponent(binary, rawMask.width, rawMask.height, anchor);
  binary = removeTinyIslands(binary, rawMask.width, rawMask.height, Math.max(8, Math.floor(rawMask.width * rawMask.height * 0.000005)));

  // Optional legacy refinement. ROI-crop mode intentionally skips this because
  // it was too aggressive on small targets.
  if (options.refine !== false && box && encodedRawImage) {
    try {
      binary = await refineWithGrabCut(
        binary,
        rawMask.width,
        rawMask.height,
        points,
        box,
        encodedRawImage,
      );
      binary = selectPromptComponent(binary, rawMask.width, rawMask.height, anchor);

      // Remove thin bridges/appendages (e.g. chair legs or supports) while
      // preserving the main object boundary as much as possible.
      binary = pruneThinAttachments(binary, rawMask.width, rawMask.height, anchor, box);
    } catch (err) {
      console.warn("GrabCut refinement skipped:", err);
    }
  }

  if (options.prune !== false && box) {
    binary = pruneThinAttachments(binary, rawMask.width, rawMask.height, anchor, box);
  }

  const polygon = maskToPolygon(binary, rawMask.width, rawMask.height, anchor);
  if (polygon.length < 3) throw new Error("輪郭を取り出せませんでした。別の点をクリックしてください。");

  return {
    width: rawMask.width,
    height: rawMask.height,
    score: scores[best],
    polygon,
  };
}




function pruneThinAttachments(binary, width, height, anchor, box) {
  // Estimate a conservative radius from the prompted ROI size.
  const roiW = box ? Math.max(1, (box.x2 - box.x1) * width) : width * 0.15;
  const roiH = box ? Math.max(1, (box.y2 - box.y1) * height) : height * 0.15;
  const radius = Math.max(1, Math.min(4, Math.round(Math.min(roiW, roiH) * 0.018)));

  if (radius <= 0) return binary;

  const original = binary;
  let core = binary;

  // Erode just enough to break narrow bridges.
  for (let r = 0; r < radius; r++) core = erode8(core, width, height);

  // Keep the eroded core nearest to the positive prompt.
  core = selectPromptComponent(core, width, height, anchor);

  // Restore roughly the original thickness.
  let restored = core;
  for (let r = 0; r < radius; r++) restored = dilate8(restored, width, height);

  // Geodesic-like reconstruction: never grow outside the original SAM/GrabCut mask.
  const out = new Uint8Array(original.length);
  for (let i = 0; i < out.length; i++) out[i] = restored[i] && original[i] ? 1 : 0;

  // If pruning was too aggressive, safely fall back.
  let before = 0, after = 0;
  for (let i = 0; i < out.length; i++) {
    if (original[i]) before++;
    if (out[i]) after++;
  }
  if (!after || (before && after / before < 0.42)) return original;

  return out;
}

function erode8(src, width, height) {
  const out = new Uint8Array(src.length);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let keep = 1;
      for (let dy = -1; dy <= 1 && keep; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!src[(y + dy) * width + (x + dx)]) { keep = 0; break; }
        }
      }
      out[y * width + x] = keep;
    }
  }
  return out;
}

function dilate8(src, width, height) {
  const out = new Uint8Array(src.length);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let on = 0;
      for (let dy = -1; dy <= 1 && !on; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (src[(y + dy) * width + (x + dx)]) { on = 1; break; }
        }
      }
      out[y * width + x] = on;
    }
  }
  return out;
}

async function waitForOpenCV(timeoutMs = 12000) {
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    const cv = globalThis.cv;
    if (cv && cv.Mat && cv.grabCut) return cv;
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("OpenCV.jsの初期化が間に合いませんでした");
}

async function refineWithGrabCut(binary, width, height, points, box, rawImage) {
  const cv = await waitForOpenCV();

  // RawImage is RGB. Build an OpenCV CV_8UC3 image directly.
  const src = new cv.Mat(height, width, cv.CV_8UC3);
  const srcData = src.data;
  const raw = rawImage.data;
  if (rawImage.width !== width || rawImage.height !== height) {
    src.delete();
    throw new Error("GrabCut input size mismatch");
  }
  srcData.set(raw);

  const mask = new cv.Mat(height, width, cv.CV_8UC1);
  const md = mask.data;

  const bx1 = Math.max(0, Math.min(width - 1, Math.floor(box.x1 * width)));
  const by1 = Math.max(0, Math.min(height - 1, Math.floor(box.y1 * height)));
  const bx2 = Math.max(bx1 + 1, Math.min(width, Math.ceil(box.x2 * width)));
  const by2 = Math.max(by1 + 1, Math.min(height, Math.ceil(box.y2 * height)));

  // 0 = sure BG, 1 = sure FG, 2 = probable BG, 3 = probable FG.
  md.fill(0);
  for (let y = by1; y < by2; y++) {
    for (let x = bx1; x < bx2; x++) {
      const i = y * width + x;
      md[i] = binary[i] ? 3 : 2;
    }
  }

  const paintDisk = (nx, ny, radius, label) => {
    const cx = Math.round(nx * (width - 1));
    const cy = Math.round(ny * (height - 1));
    const r2 = radius * radius;
    for (let y = Math.max(0, cy - radius); y <= Math.min(height - 1, cy + radius); y++) {
      for (let x = Math.max(0, cx - radius); x <= Math.min(width - 1, cx + radius); x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r2) md[y * width + x] = label;
      }
    }
  };

  const radius = Math.max(2, Math.round(Math.min(width, height) * 0.004));
  for (const p of points) {
    paintDisk(p.x, p.y, radius, p.label === 1 ? 1 : 0);
  }

  // Strong background band near the bottom of the user box. This specifically
  // suppresses thin chair/table legs accidentally connected to the object.
  const bottomBand = Math.max(2, Math.round((by2 - by1) * 0.04));
  for (let y = Math.max(by1, by2 - bottomBand); y < by2; y++) {
    for (let x = bx1; x < bx2; x++) {
      if (md[y * width + x] !== 1) md[y * width + x] = 0;
    }
  }

  const bgdModel = new cv.Mat();
  const fgdModel = new cv.Mat();
  const rect = new cv.Rect(bx1, by1, Math.max(1, bx2 - bx1), Math.max(1, by2 - by1));

  try {
    cv.grabCut(src, mask, rect, bgdModel, fgdModel, 4, cv.GC_INIT_WITH_MASK);
    const out = new Uint8Array(width * height);
    for (let i = 0; i < out.length; i++) {
      const v = md[i];
      out[i] = (v === 1 || v === 3) ? 1 : 0;
    }
    return out;
  } finally {
    src.delete();
    mask.delete();
    bgdModel.delete();
    fgdModel.delete();
  }
}

function selectPromptComponent(binary, width, height, anchor) {
  const total = width * height;
  const visited = new Uint8Array(total);
  const components = [];
  const dirs = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]];

  for (let idx = 0; idx < total; idx++) {
    if (!binary[idx] || visited[idx]) continue;
    const stack = [idx];
    visited[idx] = 1;
    const pixels = [];
    let containsAnchor = false;
    let minAnchorD2 = Infinity;

    while (stack.length) {
      const cur = stack.pop();
      pixels.push(cur);
      const x = cur % width, y = (cur / width) | 0;
      const d2 = (x - anchor.x) ** 2 + (y - anchor.y) ** 2;
      if (d2 < minAnchorD2) minAnchorD2 = d2;
      if (x === anchor.x && y === anchor.y) containsAnchor = true;

      for (const [dx,dy] of dirs) {
        const nx=x+dx, ny=y+dy;
        if (nx<0||ny<0||nx>=width||ny>=height) continue;
        const ni=ny*width+nx;
        if (binary[ni]&&!visited[ni]) { visited[ni]=1; stack.push(ni); }
      }
    }
    components.push({pixels,containsAnchor,minAnchorD2});
  }

  if (!components.length) return binary;
  let chosen = components.find(c => c.containsAnchor);
  if (!chosen) {
    chosen = components.sort((a,b) => {
      if (Math.abs(a.minAnchorD2-b.minAnchorD2)>1) return a.minAnchorD2-b.minAnchorD2;
      return b.pixels.length-a.pixels.length;
    })[0];
  }

  const out = new Uint8Array(total);
  for (const i of chosen.pixels) out[i]=1;
  return out;
}

function removeTinyIslands(binary, width, height, minSize) {
  // selectPromptComponent usually leaves one component; this also cleans
  // artifacts that may reappear after later morphology.
  const total=width*height, visited=new Uint8Array(total), out=new Uint8Array(total);
  const dirs=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1],[1,-1],[-1,1]];
  for(let idx=0;idx<total;idx++){
    if(!binary[idx]||visited[idx])continue;
    const stack=[idx], pixels=[];visited[idx]=1;
    while(stack.length){
      const cur=stack.pop();pixels.push(cur);
      const x=cur%width,y=(cur/width)|0;
      for(const [dx,dy] of dirs){
        const nx=x+dx,ny=y+dy;
        if(nx<0||ny<0||nx>=width||ny>=height)continue;
        const ni=ny*width+nx;
        if(binary[ni]&&!visited[ni]){visited[ni]=1;stack.push(ni)}
      }
    }
    if(pixels.length>=minSize)for(const p of pixels)out[p]=1;
  }
  return out;
}

function morphClose3(src, width, height) {
  const dilated=new Uint8Array(src.length);
  for(let y=1;y<height-1;y++){
    for(let x=1;x<width-1;x++){
      let on=0;
      for(let dy=-1;dy<=1&&!on;dy++)for(let dx=-1;dx<=1;dx++){
        if(src[(y+dy)*width+(x+dx)]){on=1;break}
      }
      dilated[y*width+x]=on;
    }
  }
  const eroded=new Uint8Array(src.length);
  for(let y=1;y<height-1;y++){
    for(let x=1;x<width-1;x++){
      let on=1;
      for(let dy=-1;dy<=1&&on;dy++)for(let dx=-1;dx<=1;dx++){
        if(!dilated[(y+dy)*width+(x+dx)]){on=0;break}
      }
      eroded[y*width+x]=on;
    }
  }
  return eroded;
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
  // Preserve substantially more boundary detail. This is intentionally much
  // less aggressive than the previous simplification.
  const epsilon = Math.max(0.25, Math.min(width, height) * 0.00035);
  points = simplifyClosed(points, epsilon);

  // Allow dense contours for research/annotation use.
  if (points.length > 1600) {
    const step = Math.ceil(points.length / 1600);
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
