// Brain: turns tracked things + memory into "what do I look at, and how curious am I?"
//
// Interest of a thing = novelty × (1 − boredom)
//   novelty  (long-term): how unfamiliar it is, from memory.
//            - no close match in memory       -> high (0.75..1, more if very far)
//            - matched, seen `count` times     -> 0.8·e^(−count/3) + 0.2·distance-term
//            - not seen for > 1 day            -> +0.15 ("oh, you again!")
//   boredom  (short-term): grows while the duck stares at it, e.g. 1 − e^(−seconds/τ),
//            and slowly recovers when it looks elsewhere. This is habituation, the
//            same trick real animals use, and it stops the duck staring at one mug forever.
//
// The owner gets a floor on interest (ownerFollowBias) so when nothing new is
// around, the duck drifts back to following Hunter.
//
// Gaze switching uses hysteresis (new target must beat the current one by a margin)
// so the head doesn't twitch between two similar things every frame.

import { normalize } from './memory.js';

export const DEFAULTS = {
  objMatchSim: 0.80,     // cosine similarity above this = same object
  faceMatchDist: 0.50,   // face distance below this = same person
  boredomTau: 5,         // seconds of staring to get ~63% bored
  revisitGapSec: 20,     // a sighting counts as a new "visit" after this gap
  ownerFollowBias: 0.35, // minimum interest in the owner
  switchMargin: 0.15,    // hysteresis for changing gaze target
  bodyMatchSim: 0.82,    // clothing-fingerprint similarity to call a body "owner"
  bodyMemoryHours: 12,   // outfits change, so body samples expire
  hfovDeg: 66,           // camera horizontal field of view (Pi cam v2 ≈ 62, Pixel main ≈ 70-80)
};

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));

export class Brain {
  constructor(memory, settings) {
    this.mem = memory;
    this.s = settings;
    this.exposure = new Map(); // key -> seconds stared at (decays)
    this.gaze = { x: 0, y: 0 };
    this.targetKey = null;
    this.curiosity = 0;
    this.lastT = performance.now();
    this.scene = { emb: null, change: 0 };
    // Self-motion ("efference copy"): the brain knows when it is moving its own
    // head/body, so it doesn't mistake its own motion for "the world changed".
    this.deviceRate = 0;      // phone gyro, deg/s (set by main.js while the phone camera is the eye)
    this.lastHead = null;
    this.stillSince = performance.now();
    this.selfMoving = false;
    this.idle = { x: 0, y: 0, until: 0 };
  }

  key(t) { return t.protoId || 't' + t.id; }
  boredom(key) { return 1 - Math.exp(-(this.exposure.get(key) || 0) / this.s.boredomTau); }

  // Called whenever a track gets a fresh fingerprint (+ optional extras like happy).
  observe(track, emb, { happy = 0, thumb = '' } = {}, now) {
    // Running mean of the track's fingerprint (re-normalised for objects).
    if (!track.emb) track.emb = emb;
    else {
      const k = 1 / Math.min(track.embN + 1, 5);
      track.emb = track.emb.map((v, i) => v * (1 - k) + emb[i] * k);
      if (track.kind === 'object') track.emb = normalize(track.emb);
    }
    track.embN++;
    track.lastEmbT = now;
    track.happy = Math.max(track.happy, happy);

    if (!track.committed) {
      if (now - track.born < 300) return; // wait a beat so one-frame junk detections don't get remembered
      this.commit(track, thumb);
    } else {
      const p = this.mem.protos.get(track.protoId);
      if (p) this.mem.addSample(p, emb);
    }
    this.countSmile(track);
  }

  commit(track, thumb) {
    const isFace = track.kind === 'face';
    const { proto, score } = this.mem.nearest(track.kind, track.emb);
    const matched = proto && (isFace ? score < this.s.faceMatchDist : score > this.s.objMatchSim);
    // distance term: 0 = identical, 1 = very different
    const dTerm = proto ? (isFace ? clamp((score - 0.3) / 0.3) : clamp((0.95 - score) / 0.35)) : 1;
    track.matchScore = proto ? score : null;

    let p;
    if (matched) {
      p = proto;
      const gapSec = (Date.now() - p.lastSeen) / 1000;
      const fam = 1 - Math.exp(-p.count / 3);
      let nov = 0.8 * (1 - fam) + 0.2 * dTerm;
      if (gapSec > 86400) nov += 0.15;
      if (p.isOwner) nov *= 0.6; // the owner is familiar by definition
      track.baseNovelty = clamp(nov);
      if (gapSec > this.s.revisitGapSec) {
        p.count++;
        this.mem.log(p.isOwner ? 'owner' : 'seen', p);
      }
      p.lastSeen = Date.now();
      if (!p.thumb && thumb) p.thumb = thumb;
      this.mem.addSample(p, track.emb);
      this.mem.mark(p);
    } else {
      p = this.mem.create(track.kind, track.label, track.emb, thumb);
      track.baseNovelty = clamp(0.75 + 0.25 * dTerm);
      this.mem.log('new', p);
    }
    track.committed = true;
    track.protoId = p.id;
    track.isOwner = !!p.isOwner;
    track.name = p.name;
  }

  // Smiles feed "affinity" later: people who smile at the duck become people of interest.
  countSmile(track) {
    if (track.kind !== 'face' || track.smileCounted || track.happy < 0.7 || !track.protoId) return;
    const p = this.mem.protos.get(track.protoId);
    if (!p) return;
    p.smiles = (p.smiles || 0) + 1;
    track.smileCounted = true;
    this.mem.log('smile', p);
    this.mem.mark(p);
  }

  // Scene fingerprint every ~second: big jumps = "something changed" (lamp on, thing moved).
  // While the duck is moving (or just stopped), the view changes because of US,
  // so just re-baseline: "this is what the world looks like from here now".
  // Real "something changed" is only judged while the head has been still.
  observeScene(emb) {
    if (this.selfMoving) { this.scene.emb = emb; this.scene.change = 0; return; }
    if (this.scene.emb) {
      let d = 0; for (let i = 0; i < emb.length; i++) d += emb[i] * this.scene.emb[i];
      this.scene.change = clamp(1 - d);
      this.scene.emb = normalize(this.scene.emb.map((v, i) => v * 0.8 + emb[i] * 0.2));
    } else this.scene.emb = emb;
  }

  // Once per frame. Returns the brain state + a suggested robot command.
  tick(visible, W, H, now = performance.now()) {
    const dt = clamp((now - this.lastT) / 1000, 0, 0.5);
    this.lastT = now;

    // Boredom slowly recovers for everything (≈1 minute).
    for (const [k, v] of this.exposure) {
      const nv = v * Math.exp(-dt / 60);
      // Tiny threshold on purpose: at 60 fps one frame only adds 0.016 s of staring.
      if (nv < 0.001) this.exposure.delete(k); else this.exposure.set(k, nv);
    }

    let best = null, bestScore = 0, current = null, currentScore = 0;
    for (const t of visible) {
      // Not fingerprinted yet = intriguing. But a "face" that keeps failing the
      // fingerprint step is probably a false detection (a pattern, a poster): ignore it.
      let nov = t.committed ? t.baseNovelty : t.fails > 2 ? 0 : 0.9;
      // Bodies: if a face is visible inside the body box, the face is the target
      // (better identity, better place to look). A body alone (person from behind)
      // is mildly interesting, or "owner" if its clothes match Hunter's today.
      if (t.kind === 'body') {
        if (t.hasFace) { t.interest = 0; continue; }
        nov = t.isOwner ? 0.1 : 0.3;
      }
      let interest = nov * (1 - this.boredom(this.key(t)));
      if (t.isOwner) interest = Math.max(interest, this.s.ownerFollowBias);
      const area = (t.box.w * t.box.h) / (W * H);
      const salience = 0.75 + 0.25 * clamp(Math.sqrt(area) * 3);
      const score = interest * salience * (t.kind === 'face' ? 1.1 : 1); // social creature
      t.interest = interest;
      t.gazeScore = score;
      if (score > bestScore) { bestScore = score; best = t; }
      if (this.key(t) === this.targetKey) { current = t; currentScore = score; }
    }
    // Hysteresis.
    let target = best;
    if (current && best !== current && bestScore < currentScore + this.s.switchMargin) target = current;
    this.targetKey = target ? this.key(target) : null;

    // Stare = get bored.
    if (target) this.exposure.set(this.targetKey, (this.exposure.get(this.targetKey) || 0) + dt);

    // Smooth gaze toward target centre (normalised −1..1), idle scan when nothing is there.
    let gx, gy;
    if (target) {
      gx = ((target.box.x + target.box.w / 2) / W) * 2 - 1;
      const cy = target.kind === 'body' ? target.box.y + target.box.h * 0.15 : target.box.y + target.box.h / 2; // look at the head
      gy = (cy / H) * 2 - 1;
    } else {
      // Idle scan in hops (look, hold ~2.5 s, look elsewhere) instead of a constant
      // sweep: the holds are when the duck can notice that something changed.
      if (now > this.idle.until) this.idle = { x: (Math.random() * 2 - 1) * 0.6, y: (Math.random() * 2 - 1) * 0.15, until: now + 2500 };
      gx = this.idle.x; gy = this.idle.y;
    }
    const k = clamp(dt * 5);
    this.gaze.x += (gx - this.gaze.x) * k;
    this.gaze.y += (gy - this.gaze.y) * k;

    const want = target ? target.interest : this.scene.change * 2;
    this.curiosity += (clamp(want) - this.curiosity) * clamp(dt * 1.5);

    let mode = 'wander';
    if (target) {
      if (target.isOwner && target.interest <= this.s.ownerFollowBias + 0.05) mode = 'follow_owner';
      else if (this.curiosity > 0.55) mode = 'inspect';
      else if (this.curiosity > 0.2) mode = 'glance';
      else mode = 'bored';
    }

    const command = this.command(target, mode, W, H);
    this.updateSelfMotion(command, dt, now);
    return { target, mode, curiosity: this.curiosity, gaze: { ...this.gaze }, command, selfMoving: this.selfMoving };
  }

  // Moving = head turning faster than ~9°/s, walking, or (phone prototype) the
  // phone itself rotating > 15°/s. Stay "moving" for 0.7 s after stopping so the
  // camera image can settle (motion blur, auto-exposure).
  updateSelfMotion(cmd, dt, now) {
    let headRate = 0;
    if (this.lastHead && dt > 0) headRate = (Math.abs(cmd.head.yaw - this.lastHead.yaw) + Math.abs(cmd.head.pitch - this.lastHead.pitch)) / dt;
    this.lastHead = cmd.head;
    // Smooth over ~0.3 s so tiny frame-to-frame jitter at high fps doesn't count as moving.
    this.headRateS = (this.headRateS || 0) + (headRate - (this.headRateS || 0)) * clamp(dt / 0.3);
    const moving = this.headRateS > 0.15 || cmd.walk.vx !== 0 || cmd.walk.wz !== 0 || this.deviceRate > 15;
    if (moving) this.stillSince = now;
    this.selfMoving = now - this.stillSince < 700;
  }

  // Suggested command for the robot. The Pi side maps this to the policy inputs
  // (head yaw/pitch + walk velocity). Sign convention: yaw + = look LEFT (ROS style),
  // pitch + = look DOWN. Flip on the Pi if the duck looks the wrong way.
  command(target, mode, W, H) {
    const half = (this.s.hfovDeg * Math.PI) / 360;
    const vhalf = half * (H / W);
    const cmd = {
      t: Date.now(), mode,
      head: { yaw: +(-this.gaze.x * half).toFixed(3), pitch: +(this.gaze.y * vhalf).toFixed(3) },
      walk: { vx: 0, wz: 0 },
      curiosity: +this.curiosity.toFixed(2),
    };
    if (target && (mode === 'inspect' || mode === 'follow_owner')) {
      const area = (target.box.w * target.box.h) / (W * H);
      // How big the target should look before the duck stops walking toward it.
      const closeEnough = target.kind === 'body' ? 0.2 : mode === 'inspect' ? 0.08 : 0.03;
      if (Math.abs(this.gaze.x) > 0.35) cmd.walk.wz = +(-this.gaze.x * 0.6).toFixed(2); // turn body toward it
      else if (area < closeEnough) cmd.walk.vx = 0.15;                                    // walk up to it
    }
    return cmd;
  }
}
