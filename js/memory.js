// Memory: the duck's long-term memory of things and people, stored on the phone
// in IndexedDB (survives reloads; per-origin, so use Export/Import to move it
// between GitHub Pages and the Pi-served page later).
//
// A "proto" (prototype) = one remembered thing:
//   { id, kind:'face'|'object', label, name, isOwner,
//     samples: Float32Array[]  // a few fingerprints from different angles
//     count,                   // separate sightings (visits), drives familiarity
//     firstSeen, lastSeen, smiles, thumb }
//
// Matching:
//   faces   -> smallest Euclidean distance to any sample (face-api: < ~0.5 = same person)
//   objects -> largest cosine similarity to any sample (vectors are L2-normalised, so dot = cosine)

const DB_NAME = 'duck-brain';
const MAX_SAMPLES = { face: 10, object: 6 };
const OWNER_SAMPLES = 24;
const MAX_PROTOS = 400;

export function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
export function euclid(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
export function normalize(v) { let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) || 1; return v.map((x) => x / n); }

// Score of an embedding against a proto. Lower=closer for faces, higher=closer for objects.
export function protoScore(p, emb) {
  if (p.kind === 'face') { let best = Infinity; for (const s of p.samples) best = Math.min(best, euclid(s, emb)); return best; }
  let best = -Infinity; for (const s of p.samples) best = Math.max(best, dot(s, emb)); return best;
}

function newId() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 7); }

function openDB() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB_NAME, 1);
    r.onupgradeneeded = () => {
      const db = r.result;
      db.createObjectStore('protos', { keyPath: 'id' });
      db.createObjectStore('events', { keyPath: 'id', autoIncrement: true });
    };
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}

function reqP(req) { return new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); }); }

export class Memory {
  constructor() {
    this.protos = new Map();
    this.events = [];
    this.dirty = new Set();
    this.deleted = new Set();
    this.db = null;
    this.onChange = () => {};
  }

  async load() {
    try {
      this.db = await openDB();
      const t = this.db.transaction(['protos', 'events']);
      const protos = await reqP(t.objectStore('protos').getAll());
      const events = await reqP(t.objectStore('events').getAll());
      for (const p of protos) this.protos.set(p.id, p);
      this.events = events.slice(-200).reverse();
    } catch (e) {
      console.warn('IndexedDB unavailable; memory will not persist', e);
    }
    setInterval(() => this.flush(), 2000);
  }

  async flush() {
    if (!this.db || (!this.dirty.size && !this.deleted.size)) return;
    const t = this.db.transaction('protos', 'readwrite');
    const s = t.objectStore('protos');
    for (const id of this.dirty) { const p = this.protos.get(id); if (p) s.put(p); }
    for (const id of this.deleted) s.delete(id);
    this.dirty.clear(); this.deleted.clear();
  }

  mark(p) { this.dirty.add(p.id); this.onChange(); }

  // Best match of a given kind -> { proto, score } (proto null if memory empty)
  nearest(kind, emb, filter = () => true) {
    let proto = null, score = kind === 'face' ? Infinity : -Infinity;
    for (const p of this.protos.values()) {
      if (p.kind !== kind || !filter(p)) continue;
      const s = protoScore(p, emb);
      if (kind === 'face' ? s < score : s > score) { score = s; proto = p; }
    }
    return { proto, score };
  }

  create(kind, label, emb, thumb) {
    const sameLabel = [...this.protos.values()].filter((p) => p.label === label).length;
    const p = {
      id: newId(), kind, label,
      name: kind === 'face' ? `person #${sameLabel + 1}` : `${label} #${sameLabel + 1}`,
      isOwner: false, samples: [emb], count: 1,
      firstSeen: Date.now(), lastSeen: Date.now(), smiles: 0, thumb: thumb || '',
    };
    this.protos.set(p.id, p);
    this.evict();
    this.mark(p);
    return p;
  }

  // Add a fingerprint sample only if it's different enough from what we have,
  // so the proto covers several angles instead of 10 copies of one view.
  addSample(p, emb) {
    const max = p.isOwner ? OWNER_SAMPLES : MAX_SAMPLES[p.kind];
    const s = protoScore(p, emb);
    const different = p.kind === 'face' ? s > 0.2 : s < 0.93;
    if (!different) return;
    if (p.samples.length >= max) p.samples.splice(p.isOwner ? 0 : 1, 1); // keep the first sample, drop an old one
    p.samples.push(emb);
    this.mark(p);
  }

  evict() {
    if (this.protos.size <= MAX_PROTOS) return;
    // Forget the least-visited, longest-unseen non-owner thing.
    let worst = null, worstScore = Infinity;
    for (const p of this.protos.values()) {
      if (p.isOwner) continue;
      const s = p.count * 10 - (Date.now() - p.lastSeen) / 3.6e6; // hours since seen
      if (s < worstScore) { worstScore = s; worst = p; }
    }
    if (worst) this.remove(worst.id);
  }

  remove(id) { this.protos.delete(id); this.deleted.add(id); this.dirty.delete(id); this.onChange(); }

  async clear() {
    this.protos.clear(); this.events = [];
    if (this.db) {
      const t = this.db.transaction(['protos', 'events'], 'readwrite');
      t.objectStore('protos').clear(); t.objectStore('events').clear();
    }
    this.onChange();
  }

  // Event log: the raw material for "event memory" (step 3) and affinity (step 2).
  log(type, p, extra = {}) {
    const e = { t: Date.now(), type, protoId: p?.id || null, name: p?.name || '', kind: p?.kind || '', ...extra };
    this.events.unshift(e);
    if (this.events.length > 200) this.events.length = 200;
    if (this.db) this.db.transaction('events', 'readwrite').objectStore('events').add(e);
    this.onChange();
  }

  owner() { for (const p of this.protos.values()) if (p.isOwner) return p; return null; }

  exportJSON() {
    const protos = [...this.protos.values()].map((p) => ({ ...p, samples: p.samples.map((s) => Array.from(s, (x) => +x.toFixed(5))),
      bodySamples: (p.bodySamples || []).map((b) => ({ t: b.t, emb: { shape: Array.from(b.emb.shape, (x) => +x.toFixed(5)), color: Array.from(b.emb.color, (x) => +x.toFixed(4)) } })) }));
    return JSON.stringify({ version: 1, exported: new Date().toISOString(), protos, events: this.events });
  }

  importJSON(text) {
    const data = JSON.parse(text);
    for (const p of data.protos || []) {
      p.samples = p.samples.map((s) => Float32Array.from(s));
      p.bodySamples = (p.bodySamples || []).map((b) => ({ t: b.t, emb: { shape: Float32Array.from(b.emb.shape), color: Float32Array.from(b.emb.color) } }));
      this.protos.set(p.id, p);
      this.dirty.add(p.id);
    }
    if (Array.isArray(data.events)) this.events = data.events.concat(this.events).slice(0, 200);
    this.onChange();
    return (data.protos || []).length;
  }
}
