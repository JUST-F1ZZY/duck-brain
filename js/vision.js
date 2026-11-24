// Vision: the "eyes" part of the brain. Three models, all running in the phone browser.
//
// 1. MediaPipe ObjectDetector (EfficientDet-Lite0, COCO's 80 classes)
//    -> boxes + labels ("cup", "dog", "laptop"...). Fast, int8, ~4.6 MB.
// 2. MediaPipe ImageEmbedder (MobileNetV3-small) -> 1024-number "fingerprint"
//    of any image region. Two crops of the SAME mug give similar vectors, so
//    "have I seen this before?" becomes "is there a stored vector close to this one?".
// 3. face-api (TinyFaceDetector + 68-point landmarks + FaceRecognitionNet + expressions)
//    -> face boxes, a 128-number face fingerprint built for identity
//    (same person ≈ distance < 0.5), and a "happy" score for smile counting.
//
// Why two libraries? MobileNet fingerprints are about how a thing LOOKS (colour,
// shape, background), which is fine for objects but bad for telling two people
// apart. FaceRecognitionNet was trained specifically so lighting/angle change the
// vector less than identity does. That's what owner recognition needs.

import { FilesetResolver, ObjectDetector, ImageEmbedder } from '../vendor/mediapipe/vision_bundle.esm.js';
import * as faceapi from '../vendor/face-api/face-api.esm.js';

const base = new URL('..', import.meta.url).href;

// "person" boxes become kind "body" (owner re-identification by clothing).
// Big furniture is skipped too: it never moves, so it's just noise for curiosity.
const IGNORE_LABELS = new Set(['dining table', 'bed', 'couch', 'chair', 'tv']);

export class Vision {
  constructor() {
    this.detector = null;
    this.embedder = null;
    this.delegate = '?';
    this.faceBackend = '?';
    this._crop = document.createElement('canvas');
    this._cropCtx = this._crop.getContext('2d', { willReadFrequently: true });
  }

  async init(status = () => {}) {
    status('Loading MediaPipe runtime…');
    const fileset = await FilesetResolver.forVisionTasks(base + 'vendor/mediapipe/wasm');

    // Try the GPU (WebGL) delegate first, fall back to CPU (WASM SIMD).
    // ?cpu in the URL forces the CPU path (debugging / phones with flaky WebGL).
    const forceCpu = new URLSearchParams(location.search).has('cpu');
    for (const delegate of forceCpu ? ['CPU'] : ['GPU', 'CPU']) {
      try {
        status(`Loading object detector (${delegate})…`);
        this.detector = await ObjectDetector.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: base + 'models/efficientdet_lite0.tflite', delegate },
          runningMode: 'IMAGE',
          scoreThreshold: 0.45,
          maxResults: 6,
        });
        status(`Loading image embedder (${delegate})…`);
        this.embedder = await ImageEmbedder.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: base + 'models/mobilenet_v3_small.tflite', delegate },
          runningMode: 'IMAGE',
          l2Normalize: true,
          quantize: false,
        });
        this.delegate = delegate;
        break;
      } catch (e) {
        console.warn('MediaPipe delegate failed', delegate, e);
        this.detector = this.embedder = null;
      }
    }
    if (!this.detector) throw new Error('MediaPipe failed to load on both GPU and CPU');

    status('Loading face models…');
    for (const be of ['webgl', 'wasm', 'cpu']) {
      try {
        if (await faceapi.tf.setBackend(be)) { await faceapi.tf.ready(); this.faceBackend = be; break; }
      } catch (e) { /* try next */ }
    }
    const m = base + 'models/face-api';
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(m),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(m),
      faceapi.nets.faceRecognitionNet.loadFromUri(m),
      faceapi.nets.faceExpressionNet.loadFromUri(m),
    ]);
    this.faceOpts = new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 });
    this.faceCropOpts = new faceapi.TinyFaceDetectorOptions({ inputSize: 160, scoreThreshold: 0.3 });
    status('Ready');
  }

  // -> [{kind:'object', label, score, box:{x,y,w,h}}] in canvas pixels
  detectObjects(canvas) {
    const res = this.detector.detect(canvas);
    const out = [];
    for (const d of res.detections) {
      const c = d.categories[0];
      if (!c || IGNORE_LABELS.has(c.categoryName)) continue;
      const b = d.boundingBox;
      if (c.categoryName === 'person') { // bodies: used to recognise the owner from behind
        out.push({ kind: 'body', label: 'person', score: c.score, box: { x: b.originX, y: b.originY, w: b.width, h: b.height } });
        continue;
      }
      out.push({ kind: 'object', label: c.categoryName, score: c.score, box: { x: b.originX, y: b.originY, w: b.width, h: b.height } });
    }
    return out;
  }

  // -> [{kind:'face', label:'face', score, box}]
  async detectFaces(canvas) {
    const res = await faceapi.detectAllFaces(canvas, this.faceOpts);
    return res.map((d) => ({ kind: 'face', label: 'face', score: d.score, box: { x: d.box.x, y: d.box.y, w: d.box.width, h: d.box.height } }));
  }

  // Object fingerprint: embed only the box region (MediaPipe supports a region of interest).
  embedRegion(canvas, box) {
    const W = canvas.width, H = canvas.height;
    const roi = {
      left: clamp01(box.x / W), top: clamp01(box.y / H),
      right: clamp01((box.x + box.w) / W), bottom: clamp01((box.y + box.h) / H),
    };
    if (roi.right - roi.left < 0.02 || roi.bottom - roi.top < 0.02) return null;
    const r = this.embedder.embed(canvas, { regionOfInterest: roi });
    return Float32Array.from(r.embeddings[0].floatEmbedding);
  }

  // Whole-frame fingerprint, used for "did the scene change?" (contingency later).
  embedScene(canvas) {
    const r = this.embedder.embed(canvas);
    return Float32Array.from(r.embeddings[0].floatEmbedding);
  }

  // Face fingerprint: crop the face with padding, re-detect inside the crop, then
  // landmarks -> aligned face -> 128-d descriptor + expressions.
  async describeFace(canvas, box) {
    const pad = 0.35;
    const x = Math.max(0, box.x - box.w * pad), y = Math.max(0, box.y - box.h * pad);
    const w = Math.min(canvas.width - x, box.w * (1 + 2 * pad)), h = Math.min(canvas.height - y, box.h * (1 + 2 * pad));
    if (w < 24 || h < 24) return null;
    const s = Math.min(1, 200 / Math.max(w, h)) * Math.max(1, 120 / Math.max(w, h));
    this._crop.width = Math.round(w * s); this._crop.height = Math.round(h * s);
    this._cropCtx.drawImage(canvas, x, y, w, h, 0, 0, this._crop.width, this._crop.height);
    const r = await faceapi.detectSingleFace(this._crop, this.faceCropOpts)
      .withFaceLandmarks(true).withFaceExpressions().withFaceDescriptor();
    if (!r) return null;
    return { emb: Float32Array.from(r.descriptor), happy: r.expressions.happy || 0 };
  }
}

function clamp01(v) { return Math.min(1, Math.max(0, v)); }
