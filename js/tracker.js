// Tracker: links detections across frames so "the cup" stays the same thing
// for several seconds instead of being a brand-new detection every frame.
// Without this, every frame would look novel and the duck would never get bored.
//
// Method: greedy IoU matching (intersection-over-union of boxes). Cheap and
// good enough at 8-12 fps when things don't teleport.

let nextId = 1;

export function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter + 1e-6);
}

export class Tracker {
  constructor({ maxAgeMs = 900, minIou = 0.2 } = {}) {
    this.tracks = [];
    this.maxAgeMs = maxAgeMs;
    this.minIou = minIou;
    this.lastUpdate = 0;
  }

  // dets: [{kind,label,score,box}] -> { visible: Track[], lost: Track[] }
  update(dets, now) {
    // If the phone is slow (long gaps between frames), keep tracks alive longer
    // instead of forgetting everything between two frames.
    const frameGap = this.lastUpdate ? now - this.lastUpdate : 0;
    this.lastUpdate = now;
    const maxAge = Math.max(this.maxAgeMs, frameGap * 2.5);
    const pairs = [];
    this.tracks.forEach((t, ti) => {
      dets.forEach((d, di) => {
        if (t.kind !== d.kind) return;
        const o = iou(t.box, d.box);
        // Objects: same label, or big overlap (detector flickers "cup"/"vase").
        if (o >= this.minIou && (d.kind === 'face' || d.label === t.label || o > 0.6)) pairs.push([o, ti, di]);
      });
    });
    pairs.sort((a, b) => b[0] - a[0]);
    const usedT = new Set(), usedD = new Set();
    for (const [, ti, di] of pairs) {
      if (usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti); usedD.add(di);
      const t = this.tracks[ti], d = dets[di];
      t.box = lerpBox(t.box, d.box, 0.6);
      t.score = d.score;
      t.lastSeen = now;
      t.hits++;
    }
    dets.forEach((d, di) => {
      if (usedD.has(di)) return;
      this.tracks.push({
        id: nextId++, kind: d.kind, label: d.label, score: d.score, box: { ...d.box },
        born: now, lastSeen: now, hits: 1,
        emb: null, embN: 0, lastEmbT: 0,       // fingerprint (running mean)
        committed: false, protoId: null,        // link to long-term memory
        baseNovelty: 0.9, isOwner: false, name: '',
        happy: 0, smileCounted: false,
      });
    });
    const lost = [];
    this.tracks = this.tracks.filter((t) => {
      if (now - t.lastSeen > maxAge) { lost.push(t); return false; }
      return true;
    });
    const visible = this.tracks.filter((t) => t.lastSeen === now); // seen this frame
    return { visible, lost };
  }
}

function lerpBox(a, b, k) {
  return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, w: a.w + (b.w - a.w) * k, h: a.h + (b.h - a.h) * k };
}
