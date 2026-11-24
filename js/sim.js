// Sim world: a fake room so the brain can be tested with no camera at all.
//
// It plugs in where the real camera + models go:
//   sim.source : looks like a VideoSource (draws the duck's view into the canvas)
//   sim.vision : looks like Vision (returns detections + fingerprints)
// Everything after that (tracker, memory, novelty, boredom, owner + outfit
// matching, self-motion) is the REAL brain code, unchanged. So what you see here
// is how the brain will behave; only the eyes are fake.
//
// Closed loop: the brain's head yaw / turn commands rotate the view, and
// "walk forward" brings things in front closer (they get bigger).
//
// Fake fingerprints are random vectors fixed per identity, plus a bit of noise
// each time they're "seen", so the same cup always looks like the same cup and a
// different cup doesn't. Face vectors are scaled like real face-api descriptors
// (same person ≈ 0.3-0.4 apart, different people ≈ 1.4).

const HFOV = 66;           // degrees, matches DEFAULTS.hfovDeg
const OBJECTS = [
  ['cup', '☕'], ['teddy bear', '🧸'], ['banana', '🍌'], ['sports ball', '⚽'], ['book', '📕'],
  ['cell phone', '📱'], ['bottle', '🍾'], ['apple', '🍎'], ['scissors', '✂️'], ['potted plant', '🪴'],
  ['clock', '⏰'], ['umbrella', '☂️'],
];
const GADGETS = ['🔧', '🔌', '🛠️', '🧲', '🔋']; // not in COCO: drawn, but no detection box
const SHIRTS = ['#d62828', '#1d4ed8', '#15803d', '#f59e0b', '#7c3aed', '#111827', '#e5e7eb', '#ea580c'];

// --- tiny seeded random + vector helpers ---
let seedCounter = 1;
function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r()); }
function randVec(n, seed = seedCounter++) { const r = rng(seed * 9973); const v = new Float32Array(n); for (let i = 0; i < n; i++) v[i] = gauss(r); return norm(v); }
function norm(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); }
function mix(a, wa, b, wb) { return norm(a.map((x, i) => x * wa + b[i] * wb)); }
function noisy(v, sd) { return v.map((x) => x + gauss(Math.random) * sd); }
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const wrap = (a) => ((a + 540) % 360) - 180;

export class SimWorld {
  constructor() {
    this.items = [];
    this.removed = [];
    this.bodyYaw = 0;   // where the duck's body faces (deg, + = left)
    this.headYaw = 0;   // head yaw relative to body (deg)
    this.W = 480; this.H = 360;
    this.labelVecs = {};
    this.bg = Array.from({ length: 25 }, () => randVec(1024)); // background look every 15°
    this.hunter = null;
    this.stranger = null;
    this.hunterWander = false;
    this.t = 0;
    this.lastBoxes = [];
    this.kind = 'sim';
    const self = this;

    // ---------- VideoSource-shaped ----------
    this.source = {
      kind: 'sim', width: 640, height: 480, ready: true,
      async start() {}, stop() {},
      drawTo(ctx, w, h) { self.render(ctx, w, h); },
    };

    // ---------- Vision-shaped ----------
    this.vision = {
      delegate: 'sim', faceBackend: 'sim',
      detectObjects: () => this.lastBoxes.filter((b) => b.kind !== 'face').map(strip),
      detectFaces: async () => this.lastBoxes.filter((b) => b.kind === 'face').map(strip),
      embedRegion: (canvas, box) => this.embedRegion(box),
      embedScene: () => this.embedScene(),
      describeFace: async (canvas, box) => this.describeFace(box),
    };
    function strip(b) { return { kind: b.kind, label: b.label, score: 0.9, box: jitter(b.box) }; }
    function jitter(b) { const j = () => (Math.random() - 0.5) * 3; return { x: b.x + j(), y: b.y + j(), w: b.w + j(), h: b.h + j() }; }

    // Starting room: a few things to look at.
    for (let i = 0; i < 3; i++) this.addObject((i - 1) * 25);
  }

  // ---------- world changes (buttons) ----------
  addObject(angle) {
    const [label, emoji] = pick(OBJECTS);
    this.labelVecs[label] ||= randVec(1024);
    const it = {
      id: seedCounter++, kind: 'object', label, emoji,
      angle: angle ?? this.viewCenter() + (Math.random() * 2 - 1) * 50,
      dist: 1.5 + Math.random() * 2, y: 0.55 + Math.random() * 0.15,
    };
    it.vec = mix(this.labelVecs[label], 0.6, randVec(1024), 0.8); // two cups: related but different
    this.items.push(it);
    return it;
  }
  removeObject() {
    const objs = this.items.filter((i) => i.kind === 'object' || i.kind === 'unknown');
    if (!objs.length) return;
    const it = pick(objs);
    this.items = this.items.filter((i) => i !== it);
    this.removed.push(it);
  }
  bringBack() {
    const it = this.removed.shift();
    if (!it) return false;
    it.angle = this.viewCenter() + (Math.random() * 2 - 1) * 40;
    this.items.push(it);
    return true;
  }
  addGadget() {
    this.items.push({
      id: seedCounter++, kind: 'unknown', label: 'gadget', emoji: pick(GADGETS),
      angle: this.viewCenter() + (Math.random() * 2 - 1) * 25, dist: 1.5 + Math.random(), y: 0.62, vec: randVec(1024),
    });
  }
  shuffle() {
    for (const it of this.items) if (it.kind !== 'person') it.angle = this.viewCenter() + (Math.random() * 2 - 1) * 60;
  }
  makePerson(name, isHunter) {
    return {
      id: seedCounter++, kind: 'person', name, isHunter,
      angle: this.viewCenter() + (Math.random() * 2 - 1) * 20, dist: 2.2, away: false, smile: false,
      faceVec: randVec(128), shirt: pick(SHIRTS), outfitVec: randVec(1024), frontVec: randVec(1024), backVec: randVec(1024),
      skin: isHunter ? '#e0ac69' : pick(['#8d5524', '#f1c27d', '#c68642']), hair: isHunter ? '#3b2314' : pick(['#111', '#b45309', '#d4d4d4']),
    };
  }
  toggleHunter() {
    if (this.hunter) { this.items = this.items.filter((i) => i !== this.hunter); this.hunterOut = this.hunter; this.hunter = null; return; }
    this.hunter = this.hunterOut || this.makePerson('Hunter', true); // same face every time
    this.hunter.angle = this.viewCenter() + (Math.random() * 2 - 1) * 20;
    this.items.push(this.hunter);
  }
  toggleStranger() {
    if (this.stranger) { this.items = this.items.filter((i) => i !== this.stranger); this.stranger = null; return; }
    this.stranger = this.makePerson('stranger', false); // a new random person each time
    this.items.push(this.stranger);
  }
  hunterTurn() { if (this.hunter) this.hunter.away = !this.hunter.away; }
  hunterSmile() { if (this.hunter) this.hunter.smile = !this.hunter.smile; }
  hunterShirt() {
    if (!this.hunter) return;
    let c; do { c = pick(SHIRTS); } while (c === this.hunter.shirt);
    this.hunter.shirt = c;
    this.hunter.outfitVec = randVec(1024);
  }

  viewCenter() { return this.bodyYaw + this.headYaw; }

  // ---------- closed loop: brain command -> duck moves ----------
  step(cmd, dt) {
    this.t += dt;
    this.headYaw = (cmd.head.yaw * 180) / Math.PI;
    this.bodyYaw = wrap(this.bodyYaw + ((cmd.walk.wz * 180) / Math.PI) * dt);
    if (cmd.walk.vx > 0) {
      for (const it of this.items) {
        if (Math.abs(wrap(it.angle - this.bodyYaw)) < 25) it.dist = Math.max(it.kind === 'person' ? 1.3 : 0.9, it.dist - cmd.walk.vx * dt * 1.5);
      }
    }
    if (this.hunter && this.hunterWander) {
      this.hunter.angle += Math.sin(this.t * 0.7) * 12 * dt;
      this.hunter.dist = Math.min(4, Math.max(1.2, this.hunter.dist + Math.sin(this.t * 0.4) * 0.3 * dt));
    }
  }

  // ---------- rendering + ground-truth boxes ----------
  screenX(angle, W) { return W / 2 + (wrap(this.viewCenter() - angle) / (HFOV / 2)) * (W / 2); }

  render(ctx, W, H) {
    this.W = W; this.H = H;
    // Wall with coloured panels fixed in the world, so head turns visibly pan.
    ctx.fillStyle = '#cfc6b8'; ctx.fillRect(0, 0, W, H * 0.7);
    ctx.fillStyle = '#8b6b4a'; ctx.fillRect(0, H * 0.7, W, H * 0.3);
    for (let a = -180; a < 180; a += 30) {
      const x = this.screenX(a, W);
      if (x < -60 || x > W + 60) continue;
      ctx.fillStyle = `hsl(${(a + 180) * 1.5}, 25%, 70%)`;
      ctx.fillRect(x - 25, H * 0.12, 50, H * 0.3);
    }
    const boxes = [];
    const visible = this.items
      .map((it) => ({ it, x: this.screenX(it.angle, W) }))
      .filter(({ x }) => x > -W * 0.3 && x < W * 1.3)
      .sort((a, b) => b.it.dist - a.it.dist); // far things first
    for (const { it, x } of visible) {
      const s = 1 / it.dist;
      if (it.kind === 'person') {
        const bw = 90 * s * (W / 480), bh = 300 * s * (H / 360), r = 26 * s * (W / 480);
        const feet = H * 0.92, top = feet - bh;
        ctx.fillStyle = '#374151'; ctx.fillRect(x - bw * 0.35, top + bh * 0.62, bw * 0.7, bh * 0.38); // legs
        ctx.fillStyle = it.shirt; ctx.fillRect(x - bw / 2, top + bh * 0.2, bw, bh * 0.45);             // shirt
        ctx.fillStyle = it.away ? it.hair : it.skin;
        ctx.beginPath(); ctx.arc(x, top + r, r, 0, Math.PI * 2); ctx.fill();
        if (!it.away) {
          ctx.fillStyle = it.hair; ctx.fillRect(x - r, top - 2, r * 2, r * 0.5);
          ctx.fillStyle = '#111';
          ctx.beginPath(); ctx.arc(x - r * 0.35, top + r * 0.9, r * 0.12, 0, 7); ctx.arc(x + r * 0.35, top + r * 0.9, r * 0.12, 0, 7); ctx.fill();
          ctx.strokeStyle = '#111'; ctx.lineWidth = Math.max(1, r * 0.08); ctx.beginPath();
          if (it.smile) ctx.arc(x, top + r * 1.2, r * 0.4, 0.15 * Math.PI, 0.85 * Math.PI); else { ctx.moveTo(x - r * 0.3, top + r * 1.45); ctx.lineTo(x + r * 0.3, top + r * 1.45); }
          ctx.stroke();
        }
        const body = clip({ x: x - bw / 2, y: top, w: bw, h: bh }, W, H);
        if (body) boxes.push({ kind: 'body', label: 'person', box: body, item: it });
        const face = clip({ x: x - r, y: top, w: r * 2, h: r * 2 }, W, H);
        if (face && !it.away) boxes.push({ kind: 'face', label: 'face', box: face, item: it });
      } else {
        const size = 70 * s * (W / 480);
        ctx.font = `${size}px system-ui, "Noto Color Emoji", sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(it.emoji, x, H * it.y);
        if (it.kind === 'object') {
          const b = clip({ x: x - size / 2, y: H * it.y - size / 2, w: size, h: size }, W, H);
          if (b) boxes.push({ kind: 'object', label: it.label, box: b, item: it });
        }
      }
    }
    ctx.textAlign = 'start';
    // Real detectors miss now and then: drop ~3% of detections.
    this.lastBoxes = boxes.filter(() => Math.random() > 0.03);
  }

  // Which world thing does this region belong to? (most of the query box inside it)
  findItem(box, kinds) {
    let best = null, bestF = 0.3;
    for (const b of this.lastBoxes) {
      if (!kinds.includes(b.kind)) continue;
      const x1 = Math.max(box.x, b.box.x), y1 = Math.max(box.y, b.box.y);
      const x2 = Math.min(box.x + box.w, b.box.x + b.box.w), y2 = Math.min(box.y + box.h, b.box.y + b.box.h);
      const f = (Math.max(0, x2 - x1) * Math.max(0, y2 - y1)) / (box.w * box.h + 1e-6);
      if (f > bestF) { bestF = f; best = b; }
    }
    return best;
  }

  embedRegion(box) {
    const b = this.findItem(box, ['object', 'body']);
    if (!b) return randVec(1024, Math.floor(Math.random() * 1e9));
    const it = b.item;
    if (it.kind === 'person') {
      // Clothing "shape" fingerprint: same shirt, but front and back differ a bit.
      return norm(noisy(mix(it.outfitVec, 0.9, it.away ? it.backVec : it.frontVec, 0.45), 0.01));
    }
    return norm(noisy(it.vec, 0.006)); // ≈0.98 cosine between two looks at the same thing
  }

  describeFace(box) {
    const b = this.findItem(box, ['face']);
    if (!b) return null;
    const it = b.item;
    return { emb: noisy(it.faceVec, 0.02), happy: it.smile ? 0.95 : 0.05 };
  }

  embedScene() {
    const c = this.viewCenter();
    const k = (wrap(c) + 180) / 15, i = Math.floor(k) % 24, f = k - Math.floor(k);
    let v = mix(this.bg[i], 1 - f, this.bg[i + 1], f).map((x) => x * 1.5);
    for (const it of this.items) {
      if (Math.abs(wrap(it.angle - c)) > HFOV / 2) continue;
      const iv = it.kind === 'person' ? it.outfitVec : it.vec;
      v = v.map((x, j) => x + iv[j] * (0.8 / it.dist));
    }
    return norm(noisy(v, 0.002));
  }

  // Top-down-ish panorama: all 360°, with the duck's field of view marked.
  drawStrip(canvas) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const ax = (a) => ((wrap(a) + 180) / 360) * W;
    ctx.fillStyle = '#1d2027'; ctx.fillRect(0, 0, W, H);
    // Mirrored so that left in the room = left on the strip (angles are + = left).
    const sx = (a) => W - ax(a);
    const c = this.viewCenter(), fw = (HFOV / 360) * W;
    ctx.fillStyle = 'rgba(255,93,143,.22)';
    for (const off of [-W, 0, W]) ctx.fillRect(sx(c) - fw / 2 + off, 0, fw, H);
    ctx.strokeStyle = '#ffb703'; ctx.beginPath(); ctx.moveTo(sx(this.bodyYaw), 0); ctx.lineTo(sx(this.bodyYaw), H); ctx.stroke();
    ctx.font = `${H * 0.45}px system-ui, "Noto Color Emoji", sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const it of this.items) {
      const x = sx(it.angle);
      ctx.fillStyle = '#e8eaf0';
      ctx.fillText(it.kind === 'person' ? (it.isHunter ? (it.away ? '🔙' : '🧑') : '👤') : it.emoji, x, H / 2);
    }
    ctx.fillStyle = '#9aa1ad'; ctx.font = '11px system-ui'; ctx.textAlign = 'start';
    ctx.fillText('360° room · pink = what the duck sees · yellow line = body heading', 4, H - 6);
  }
}

function clip(b, W, H) {
  const x1 = Math.max(0, b.x), y1 = Math.max(0, b.y), x2 = Math.min(W, b.x + b.w), y2 = Math.min(H, b.y + b.h);
  if (x2 - x1 < b.w * 0.4 || y2 - y1 < b.h * 0.4 || x2 - x1 < 6) return null; // mostly off-screen: not detected
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}
