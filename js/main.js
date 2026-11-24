// Main loop: grab frame -> detect -> track -> fingerprint (on a budget) -> brain -> draw.
// Budget matters on a phone: detection runs every frame, but fingerprints only for
// new things and then once every ~1-1.5 s per thing. That's what keeps 8-12 fps.

import { VideoSource } from './video-source.js';
import { Vision } from './vision.js';
import { SimWorld } from './sim.js';
import { Tracker } from './tracker.js';
import { Memory, dot } from './memory.js';
import { Brain, DEFAULTS } from './curiosity.js';
import { CommandSink, affinity, EventWatcher } from './hooks.js';

const $ = (id) => document.getElementById(id);
const PROC_W = 480; // processing width in pixels; lower = faster, less range

// ---------- settings (persisted per browser) ----------
const settings = { ...DEFAULTS };
try { Object.assign(settings, JSON.parse(localStorage.getItem('duck-settings') || '{}')); } catch (e) { /* private mode etc. */ }
const saveSettings = () => { try { localStorage.setItem('duck-settings', JSON.stringify(settings)); } catch (e) { /* ignore */ } };

const mem = new Memory();
// Real eyes, or the sim world's fake eyes. Everything else is shared.
const realVision = new Vision();
const camSrc = new VideoSource();
const sim = new SimWorld();
let vision = realVision;
let src = camSrc;
let simLast = 0;
const tracker = new Tracker();
const brain = new Brain(mem, settings);
const sink = new CommandSink();
const events = new EventWatcher();

const proc = document.createElement('canvas');
const procCtx = proc.getContext('2d');
const view = $('view');
const vctx = view.getContext('2d');
const thumbC = document.createElement('canvas');
thumbC.width = thumbC.height = 64;
const thumbCtx = thumbC.getContext('2d');

let running = false;
let enrolling = null;
let lastScene = 0;
let wakeLock = null;
const stats = { fps: 0, det: 0, face: 0, emb: 0, last: performance.now() };
let lastState = null;

window.duck = { mem, brain, vision, tracker, settings, get state() { return lastState; }, stats };

// ---------- boot ----------
function setStatus(msg, isErr = false) {
  $('status').textContent = msg;
  $('status').className = isErr ? 'err' : '';
}

async function boot() {
  buildSettingsUI();
  mem.onChange = throttle(renderMemory, 400);
  await mem.load();
  renderMemory();
  try {
    await realVision.init(setStatus);
    setStatus(`Ready. Vision: MediaPipe ${realVision.delegate}, faces ${realVision.faceBackend}. Pick a source and press Start.`);
  } catch (e) {
    console.error(e);
    setStatus('Model load failed: ' + e.message + ' (Sim world still works.)', true);
  }
  $('start').disabled = false;
}

// ---------- start / stop ----------
async function start() {
  try {
    setStatus('Starting video…');
    const kind = $('source').value;
    if (kind === 'sim') { src = sim.source; vision = sim.vision; }
    else {
      if (!realVision.detector) throw new Error('Vision models did not load, only Sim world is available.');
      src = camSrc; vision = realVision;
    }
    await src.start(kind, $('url').value.trim(), $('file').files[0]);
    proc.width = view.width = PROC_W;
    proc.height = view.height = Math.round((PROC_W * src.height) / src.width);
    running = true;
    $('start').disabled = true; $('stop').disabled = false;
    setStatus('Watching…');
    try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (e) { /* not critical */ }
    loop();
  } catch (e) {
    console.error(e);
    setStatus(e.message, true);
  }
}

function stop() {
  running = false;
  src.stop();
  $('start').disabled = false; $('stop').disabled = true;
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
  setStatus('Stopped.');
}

async function loop() {
  while (running) {
    try { await frame(); } catch (e) {
      console.error(e);
      if (String(e).includes('tainted') || String(e).includes('SecurityError')) {
        setStatus('Cross-origin stream: the browser blocks reading its pixels. Serve the page from the same host as the stream, or add CORS headers (see README).', true);
        stop(); return;
      }
      setStatus('Frame error: ' + e.message, true);
    }
    await new Promise((r) => requestAnimationFrame(r));
  }
}

// ---------- one frame ----------
async function frame() {
  if (!src.ready) return;
  const W = proc.width, H = proc.height;
  src.drawTo(procCtx, W, H);

  let t = performance.now();
  const objs = vision.detectObjects(proc);
  stats.det = ema(stats.det, performance.now() - t);
  t = performance.now();
  const faces = await vision.detectFaces(proc);
  stats.face = ema(stats.face, performance.now() - t);

  const now = performance.now();
  const { visible } = tracker.update([...objs, ...faces], now);

  t = performance.now();
  if (enrolling) await enrollStep(visible, now);
  else await fingerprintFaces(visible, now);
  fingerprintObjects(visible, now);
  fingerprintBodies(visible, now);
  if (now - lastScene > 1000) { brain.observeScene(vision.embedScene(proc)); lastScene = now; }
  stats.emb = ema(stats.emb, performance.now() - t);

  events.update(visible, now);
  const state = brain.tick(visible, W, H, now);
  lastState = state;
  if (src === sim.source) {
    sim.step(state.command, simLast ? Math.min(0.5, (now - simLast) / 1000) : 0);
    simLast = now;
    sim.drawStrip($('simStrip'));
  }
  sink.send(state.command);

  draw(state, visible);
  const dtf = now - stats.last; stats.last = now;
  stats.fps = ema(stats.fps, 1000 / Math.max(1, dtf), 0.1);
  renderHud(state, visible);
}

async function fingerprintFaces(visible, now) {
  // At most one face descriptor per frame: new faces first, then the stalest.
  const todo = visible.filter((t) => t.kind === 'face' && (!t.committed || now - t.lastEmbT > 1200) && !(t.fails > 3 && now - t.lastTry < 800));
  todo.sort((a, b) => (a.committed - b.committed) || (a.lastEmbT - b.lastEmbT));
  const tr = todo[0];
  if (!tr) return;
  tr.lastTry = now;
  const r = await vision.describeFace(proc, tr.box);
  if (!r) { tr.fails = (tr.fails || 0) + 1; return; }
  tr.fails = 0;
  brain.observe(tr, r.emb, { happy: r.happy, thumb: thumbOf(tr.box) }, now);
}

// ---------- body re-identification (recognise Hunter from behind) ----------
// Faces are the long-term identity. Bodies are recognised by CLOTHING appearance
// (MobileNet fingerprint of the person box), which only holds while you wear the
// same outfit. So the duck learns it automatically: whenever your face is
// recognised inside a person box, that box's fingerprint is saved as "today's
// outfit" (expires after bodyMemoryHours). With your back turned, a person box
// whose fingerprint matches today's outfit = you.
// Also: once a person track was linked to you by face, it stays you while it is
// tracked, even if you turn around (tracking continuity is the strongest cue).
function faceInBody(f, b) {
  const cx = f.box.x + f.box.w / 2, cy = f.box.y + f.box.h / 2;
  return cx > b.box.x && cx < b.box.x + b.box.w && cy > b.box.y && cy < b.box.y + b.box.h * 0.6;
}

function fingerprintBodies(visible, now) {
  const owner = mem.owner();
  const faces = visible.filter((t) => t.kind === 'face');
  let outfit = [];
  if (owner) {
    const maxAge = settings.bodyMemoryHours * 3.6e6;
    outfit = (owner.bodySamples || []).filter((b) => Date.now() - b.t < maxAge);
    if (owner.bodySamples && outfit.length !== owner.bodySamples.length) { owner.bodySamples = outfit; mem.mark(owner); }
  }
  let budget = 1; // one body fingerprint per frame
  const bodies = visible.filter((t) => t.kind === 'body').sort((a, b) => a.lastEmbT - b.lastEmbT);
  for (const b of bodies) {
    const face = faces.find((f) => faceInBody(f, b));
    b.hasFace = !!face;
    b.committed = true; // bodies are not stored as separate memories
    if (face?.isOwner) { b.isOwner = true; b.name = owner?.name || face.name; b.ownerVia = 'face'; }
    else if (face?.committed) { b.isOwner = false; b.ownerVia = null; b.name = face.name; } // someone else's face

    if (!owner || budget <= 0 || now - b.lastEmbT < 1000) continue;
    // Torso only: the head looks totally different from behind (hair vs face),
    // but shirt/jacket colour and texture are similar front and back.
    const torso = { x: b.box.x + b.box.w * 0.1, y: b.box.y + b.box.h * 0.22, w: b.box.w * 0.8, h: b.box.h * 0.45 };
    const shape = vision.embedRegion(proc, torso);
    b.lastEmbT = now; budget--;
    if (!shape) continue;
    const emb = { shape, color: colorHist(torso) };
    if (face?.isOwner) { addOutfitSample(owner, emb); continue; }
    if (face?.committed || b.ownerVia === 'face') continue;
    let sim = -1;
    for (const o of outfit) sim = Math.max(sim, outfitSim(o.emb, emb));
    b.bodySim = sim;
    const thr = b.isOwner ? settings.bodyMatchSim - 0.06 : settings.bodyMatchSim; // easier to stay than to become
    const was = b.isOwner;
    b.isOwner = sim > thr;
    b.name = b.isOwner ? owner.name : '';
    if (b.isOwner && !was) { b.ownerVia = 'body'; mem.log('owner-body', owner, { sim: +sim.toFixed(2) }); }
  }
}

// Outfit fingerprint = MobileNet vector (texture/shape) + colour histogram.
// MobileNet alone barely cares about colour, but colour is the most stable
// clothing cue front-vs-back. Histogram: 8 hues x 2 saturation x 3 brightness,
// compared with histogram intersection (1 = identical colour mix).
const histC = document.createElement('canvas');
histC.width = histC.height = 32;
const histCtx = histC.getContext('2d', { willReadFrequently: true });
function colorHist(r) {
  histCtx.drawImage(proc, r.x, r.y, Math.max(1, r.w), Math.max(1, r.h), 0, 0, 32, 32);
  const d = histCtx.getImageData(0, 0, 32, 32).data;
  const h = new Float32Array(8 * 2 * 3 + 3); // + 3 grey bins (black / grey / white)
  for (let i = 0; i < d.length; i += 4) {
    const R = d[i] / 255, G = d[i + 1] / 255, B = d[i + 2] / 255;
    const mx = Math.max(R, G, B), mn = Math.min(R, G, B), v = mx, sat = mx ? (mx - mn) / mx : 0;
    const vb = v < 0.33 ? 0 : v < 0.66 ? 1 : 2;
    if (sat < 0.2 || v < 0.12) { h[48 + vb]++; continue; }
    let hue = mx === R ? ((G - B) / (mx - mn)) % 6 : mx === G ? (B - R) / (mx - mn) + 2 : (R - G) / (mx - mn) + 4;
    hue = (hue + 6) % 6;
    h[(Math.floor(hue / 0.75) % 8) * 6 + (sat < 0.55 ? 0 : 3) + vb]++;
  }
  for (let i = 0; i < h.length; i++) h[i] /= 1024;
  return h;
}
function outfitSim(a, b) {
  let inter = 0;
  for (let i = 0; i < a.color.length; i++) inter += Math.min(a.color[i], b.color[i]);
  return 0.4 * dot(a.shape, b.shape) + 0.6 * inter;
}

function addOutfitSample(owner, emb) {
  owner.bodySamples ||= [];
  let best = -1;
  for (const o of owner.bodySamples) best = Math.max(best, outfitSim(o.emb, emb));
  if (best > 0.95) return; // nearly identical view already stored
  owner.bodySamples.push({ t: Date.now(), emb });
  if (owner.bodySamples.length > 12) owner.bodySamples.shift();
  mem.mark(owner);
}

function fingerprintObjects(visible, now) {
  const todo = visible.filter((t) => t.kind === 'object' && (!t.committed || now - t.lastEmbT > 1500));
  todo.sort((a, b) => (a.committed - b.committed) || (a.lastEmbT - b.lastEmbT));
  for (const tr of todo.slice(0, 2)) {
    const emb = vision.embedRegion(proc, tr.box);
    if (emb) brain.observe(tr, emb, { thumb: thumbOf(tr.box) }, now);
  }
}

function thumbOf(b) {
  try {
    const s = Math.max(b.w, b.h);
    thumbCtx.fillStyle = '#000'; thumbCtx.fillRect(0, 0, 64, 64);
    thumbCtx.drawImage(proc, b.x + b.w / 2 - s / 2, b.y + b.h / 2 - s / 2, s, s, 0, 0, 64, 64);
    return thumbC.toDataURL('image/jpeg', 0.7);
  } catch (e) { return ''; }
}

// ---------- owner enrollment ----------
function startEnroll() {
  if (!running) { setStatus('Start the camera first, then enroll.', true); return; }
  enrolling = { name: $('ownerName').value.trim() || 'Hunter', samples: [], until: performance.now() + 12000, lastT: 0 };
  setStatus('Enrolling: look at the camera, slowly turn your head left/right and up/down…');
}

async function enrollStep(visible, now) {
  // Prefer faces that already gave a good fingerprint (real faces), then the biggest.
  const faces = visible.filter((t) => t.kind === 'face' && !(t.fails > 2 && !t.emb))
    .sort((a, b) => (!!b.emb - !!a.emb) || (b.box.w * b.box.h - a.box.w * a.box.h));
  const f = faces[0];
  if (f && now - enrolling.lastT > 250) {
    enrolling.lastT = now;
    const r = await vision.describeFace(proc, f.box);
    if (r) { enrolling.samples.push(r.emb); enrolling.thumb ||= thumbOf(f.box); if (!f.emb) f.emb = r.emb; }
    else f.fails = (f.fails || 0) + 1;
  }
  setStatus(`Enrolling ${enrolling.name}: ${enrolling.samples.length}/16 samples. Turn your head slowly.`);
  if (enrolling.samples.length >= 16 || now > enrolling.until) finishEnroll();
}

function finishEnroll() {
  const e = enrolling; enrolling = null;
  if (e.samples.length < 5) { setStatus('Enrollment failed: not enough clear face views. Try again with more light.', true); return; }
  const mean = e.samples[0].map((_, i) => e.samples.reduce((s, v) => s + v[i], 0) / e.samples.length);
  // If the duck already remembers this face as a stranger, promote that memory to owner.
  let p = mem.owner();
  if (!p) {
    const { proto, score } = mem.nearest('face', mean);
    if (proto && score < settings.faceMatchDist) p = proto;
  }
  if (!p) p = mem.create('face', 'face', mean, e.thumb);
  p.isOwner = true;
  p.name = e.name;
  p.samples = p.samples.concat(e.samples).slice(-24);
  if (e.thumb) p.thumb = e.thumb;
  mem.mark(p);
  mem.log('enrolled', p);
  // Re-label any face tracks that match now.
  for (const t of tracker.tracks) {
    if (t.kind === 'face' && t.emb && Math.min(...p.samples.map((s) => dist(s, t.emb))) < settings.faceMatchDist) {
      t.protoId = p.id; t.isOwner = true; t.name = p.name; t.committed = true; t.baseNovelty = 0.1;
    }
  }
  setStatus(`Enrolled ${p.name} with ${e.samples.length} samples. The duck now knows your face. Now step back so it sees your whole body for a few seconds: it learns today's outfit and can then follow you from behind.`);
}
const dist = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); };

// ---------- drawing ----------
function draw(state, visible) {
  const W = view.width, H = view.height;
  vctx.drawImage(proc, 0, 0);
  vctx.font = '13px system-ui, sans-serif';
  vctx.textBaseline = 'bottom';
  for (const t of visible) {
    const isTarget = state.target === t;
    const col = t.isOwner ? '#ffd23f' : t.kind === 'face' ? '#4cc9f0' : t.kind === 'body' ? '#b8b8ff' : '#80ed99';
    vctx.strokeStyle = col;
    vctx.lineWidth = isTarget ? 4 : t.kind === 'body' ? 1 : 1.5;
    vctx.setLineDash(t.kind === 'body' ? [6, 4] : []);
    vctx.strokeRect(t.box.x, t.box.y, t.box.w, t.box.h);
    if (t.kind === 'body' && t.hasFace && !t.isOwner) continue; // the face box already says it
    const name = (t.isOwner ? `★ ${t.name}` : t.name || t.label) + (t.kind === 'body' ? ' (body)' : '');
    const tag = t.kind === 'body' ? '' : !t.committed ? '…' : t.baseNovelty > 0.7 ? 'NEW' : '';
    const txt = `${name} ${tag} ${Math.round((t.interest || 0) * 100)}%`;
    const tw = vctx.measureText(txt).width + 8;
    vctx.fillStyle = 'rgba(0,0,0,.6)';
    vctx.fillRect(t.box.x, Math.max(0, t.box.y - 18), tw, 18);
    vctx.fillStyle = col;
    vctx.fillText(txt, t.box.x + 4, Math.max(18, t.box.y));
  }
  vctx.setLineDash([]);
  // Gaze reticle: where the duck's head is pointing.
  const gx = ((state.gaze.x + 1) / 2) * W, gy = ((state.gaze.y + 1) / 2) * H;
  vctx.strokeStyle = '#ff5d8f'; vctx.lineWidth = 2;
  vctx.beginPath(); vctx.arc(gx, gy, 14, 0, Math.PI * 2); vctx.stroke();
  vctx.beginPath(); vctx.moveTo(gx - 22, gy); vctx.lineTo(gx - 8, gy); vctx.moveTo(gx + 8, gy); vctx.lineTo(gx + 22, gy);
  vctx.moveTo(gx, gy - 22); vctx.lineTo(gx, gy - 8); vctx.moveTo(gx, gy + 8); vctx.lineTo(gx, gy + 22); vctx.stroke();
}

const MOODS = {
  wander: '🦆 looking around',
  bored: '😴 bored',
  glance: '👀 mildly curious',
  inspect: '🤩 very curious, going to inspect',
  follow_owner: '🧡 following',
  glance_follow: '🧡👀 following, glancing at something new',
  search_owner: '🔎 lost Hunter, looking for him',
};

let hudT = 0;
function renderHud(state, visible) {
  const now = performance.now();
  if (now - hudT < 150) return;
  hudT = now;
  const c = Math.round(state.curiosity * 100);
  $('meterFill').style.width = c + '%';
  $('meterVal').textContent = c + '%';
  let mood = MOODS[state.mode] || state.mode;
  if (state.mode === 'follow_owner' && state.target) mood += ' ' + state.target.name;
  if (state.mode === 'inspect' && state.target) mood += ': ' + (state.target.name || state.target.label);
  $('mood').textContent = mood;
  // Duck eye pupil follows gaze.
  $('pupil').setAttribute('cx', 50 + state.gaze.x * 18);
  $('pupil').setAttribute('cy', 50 + state.gaze.y * 18);
  $('pupil').setAttribute('r', 10 + state.curiosity * 8); // pupils dilate when curious
  $('cmd').textContent = JSON.stringify(state.command);
  $('stats').textContent =
    `${stats.fps.toFixed(1)} fps · objects ${stats.det.toFixed(0)} ms · faces ${stats.face.toFixed(0)} ms · fingerprints ${stats.emb.toFixed(0)} ms\n` +
    `MediaPipe ${vision.delegate} · face-api ${vision.faceBackend} · ${visible.length} tracked · ${mem.protos.size} remembered · scene change ${(brain.scene.change * 100).toFixed(0)}% (${state.selfMoving ? 'ignored, I am moving' : 'watching'}) · link ${sink.status}`;
}

// ---------- memory panel ----------
function ago(ms) {
  const s = (Date.now() - ms) / 1000;
  if (s < 60) return Math.round(s) + 's ago';
  if (s < 3600) return Math.round(s / 60) + 'm ago';
  if (s < 86400) return Math.round(s / 3600) + 'h ago';
  return Math.round(s / 86400) + 'd ago';
}

function renderMemory() {
  const list = $('memList');
  const protos = [...mem.protos.values()].sort((a, b) => b.isOwner - a.isOwner || b.lastSeen - a.lastSeen);
  $('memCount').textContent = `${protos.length} things · ${protos.filter((p) => p.kind === 'face').length} people`;
  list.replaceChildren(...protos.slice(0, 60).map((p) => {
    const row = document.createElement('div');
    row.className = 'mem' + (p.isOwner ? ' owner' : '');
    const img = document.createElement('img');
    img.src = p.thumb || 'data:,'; img.alt = '';
    const info = document.createElement('div');
    info.className = 'info';
    const aff = (p.kind === 'face' ? ` · smiles ${p.smiles || 0} · affinity ${affinity(p)}` : '') + (p.isOwner ? ` · outfit views ${(p.bodySamples || []).length}` : '');
    info.innerHTML = `<b></b><small>${p.kind === 'face' ? 'person' : p.label} · seen ${p.count}× · ${ago(p.lastSeen)}${aff} · ${p.samples.length} views</small>`;
    info.querySelector('b').textContent = (p.isOwner ? '★ ' : '') + p.name;
    const ren = button('Rename', () => { const n = prompt('Name', p.name); if (n) { p.name = n; mem.mark(p); syncTrackNames(p); } });
    const del = button('Forget', () => { mem.remove(p.id); for (const t of tracker.tracks) if (t.protoId === p.id) { t.committed = false; t.protoId = null; t.isOwner = false; t.name = ''; } });
    row.append(img, info, ren, del);
    return row;
  }));
  const ev = $('events');
  ev.textContent = mem.events.slice(0, 15).map((e) => `${new Date(e.t).toLocaleTimeString()}  ${e.type.padEnd(8)} ${e.name}`).join('\n') || '(nothing yet)';
}

function syncTrackNames(p) { for (const t of tracker.tracks) if (t.protoId === p.id) t.name = p.name; }

function button(label, fn) { const b = document.createElement('button'); b.textContent = label; b.onclick = fn; return b; }

// ---------- settings UI ----------
const SETTING_INFO = {
  objMatchSim: ['Object match similarity', 0.5, 0.99, 0.01],
  faceMatchDist: ['Face match distance', 0.3, 0.7, 0.01],
  boredomTau: ['Boredom time (s)', 1, 30, 0.5],
  revisitGapSec: ['New visit after (s)', 5, 600, 5],
  ownerFollowBias: ['Owner interest floor', 0, 1, 0.05],
  glanceThreshold: ['Glance-away novelty (while following)', 0, 1, 0.05],
  glanceSec: ['Max glance while following (s)', 0, 10, 0.5],
  searchSec: ['Search for lost owner (s)', 0, 30, 1],
  switchMargin: ['Gaze switch margin', 0, 0.5, 0.01],
  bodyMatchSim: ['Body (outfit) match similarity', 0.5, 0.99, 0.01],
  bodyMemoryHours: ['Forget outfit after (h)', 1, 72, 1],
  hfovDeg: ['Camera FOV (°)', 40, 120, 1],
};
function buildSettingsUI() {
  const box = $('settings');
  for (const [k, [label, min, max, step]] of Object.entries(SETTING_INFO)) {
    const row = document.createElement('label');
    row.innerHTML = `<span>${label}</span><input type="number" min="${min}" max="${max}" step="${step}">`;
    const inp = row.querySelector('input');
    inp.value = settings[k];
    inp.onchange = () => { const v = parseFloat(inp.value); if (!isNaN(v)) { settings[k] = v; saveSettings(); } };
    box.append(row);
  }
  box.append(button('Reset to defaults', () => { Object.assign(settings, DEFAULTS); saveSettings(); box.replaceChildren(); buildSettingsUI(); }));
}

// ---------- utils ----------
function ema(prev, v, k = 0.2) { return prev ? prev + (v - prev) * k : v; }
function throttle(fn, ms) {
  let t = 0, pending = null;
  return () => {
    const now = Date.now();
    if (now - t > ms) { t = now; fn(); } else if (!pending) pending = setTimeout(() => { pending = null; t = Date.now(); fn(); }, ms);
  };
}

// ---------- wire up ----------
$('start').onclick = start;
$('stop').onclick = stop;
$('enroll').onclick = startEnroll;
$('source').onchange = () => {
  const v = $('source').value;
  $('urlRow').hidden = !['mjpeg', 'snapshot', 'video'].includes(v);
  $('fileRow').hidden = v !== 'file';
  $('simPanel').hidden = v !== 'sim';
};
$('source').onchange();
$('export').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([mem.exportJSON()], { type: 'application/json' }));
  a.download = `duck-memory-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
$('importFile').onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try { const n = mem.importJSON(await f.text()); setStatus(`Imported ${n} memories.`); } catch (err) { setStatus('Import failed: ' + err.message, true); }
};
$('forgetAll').onclick = async () => {
  if (!confirm('Forget everything, including the owner?')) return;
  await mem.clear();
  for (const t of tracker.tracks) { t.committed = false; t.protoId = null; t.isOwner = false; t.name = ''; }
};
// Phone gyro: while the Pixel's own camera is the eye, moving the phone = the duck moving.
window.addEventListener('devicemotion', (e) => {
  const r = e.rotationRate;
  brain.deviceRate = r && src.kind?.startsWith('camera') ? Math.hypot(r.alpha || 0, r.beta || 0, r.gamma || 0) : 0;
});
// Sim world buttons
const simButtons = {
  simAdd: () => sim.addObject(),
  simRemove: () => sim.removeObject(),
  simBack: () => { if (!sim.bringBack()) setStatus('Nothing removed yet: press "Remove an object" first.'); },
  simGadget: () => sim.addGadget(),
  simShuffle: () => sim.shuffle(),
  simHunter: () => sim.toggleHunter(),
  simTurn: () => sim.hunterTurn(),
  simSmile: () => sim.hunterSmile(),
  simShirt: () => sim.hunterShirt(),
  simWander: () => { sim.hunterWander = !sim.hunterWander; },
  simStranger: () => sim.toggleStranger(),
};
for (const [id, fn] of Object.entries(simButtons)) $(id).onclick = fn;
$('wsConnect').onclick = () => sink.connect($('wsUrl').value.trim());

boot();
