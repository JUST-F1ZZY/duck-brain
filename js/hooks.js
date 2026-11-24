// Hooks for later steps. Kept small on purpose: they define the shape of the
// pieces so step 2-4 slot in without rewriting the loop.

// --- CommandSink: where the brain's decisions go -------------------------------
// Now: just shown on screen. Later: a WebSocket to the Pi on the phone hotspot,
// e.g. ws://duck.local:8765. The Pi side validates and maps to policy inputs.
// Failsafe belongs on the Pi: no message for 500 ms -> stand still, head only.
// (An HTTPS page can't open ws:// to the Pi; serve this page FROM the Pi later.)
export class CommandSink {
  constructor() { this.ws = null; this.status = 'off'; this.lastSend = 0; }
  connect(url) {
    this.close();
    if (!url) return;
    try {
      this.ws = new WebSocket(url);
      this.status = 'connecting';
      this.ws.onopen = () => (this.status = 'connected');
      this.ws.onclose = () => (this.status = 'closed');
      this.ws.onerror = () => (this.status = 'error');
    } catch (e) { this.status = 'error: ' + e.message; }
  }
  send(cmd, hz = 10) {
    const now = performance.now();
    if (!this.ws || this.ws.readyState !== 1 || now - this.lastSend < 1000 / hz) return;
    this.lastSend = now;
    this.ws.send(JSON.stringify(cmd));
  }
  close() { if (this.ws) this.ws.close(); this.ws = null; this.status = 'off'; }
}

// --- Affinity (step 2) ------------------------------------------------------------
// People of interest = seen often + smile at the duck. Data is already logged:
// proto.count (visits) and proto.smiles. This is the scoring rule to start from.
export function affinity(p) {
  if (p.kind !== 'face') return 0;
  if (p.isOwner) return 1;
  const visits = 1 - Math.exp(-(p.count || 0) / 5);
  const smiles = 1 - Math.exp(-(p.smiles || 0) / 3);
  return +(0.5 * visits + 0.5 * smiles).toFixed(2);
}

// --- Event memory (step 3) ---------------------------------------------------------
// Idea: when a face track and an object track overlap for a while AND the object's
// fingerprint changes afterwards, log "person X did something to object Y".
// The tracker already gives both tracks per frame; this is the place for it.
export class EventWatcher {
  update(/* visibleTracks, now */) { /* TODO step 3 */ }
}

// --- Contingency learning (step 4) -------------------------------------------------
// Loop: duck acts (e.g. toggles a Shelly smart plug via the Pi) -> watch the scene
// fingerprint (Brain.scene.change) for ~2 s -> if change >> baseline, "I did that!"
// -> raise excitement, trigger the dance (head bobs, stomp, beak). Repeat to learn
// which actions reliably cause which changes.
export class Contingency {
  constructor(sink) { this.sink = sink; this.trials = []; }
  async act(/* name */) { /* TODO step 4: sink.send({ action: 'plug_toggle' }) */ }
  observe(/* sceneChange, now */) { /* TODO step 4 */ }
}
