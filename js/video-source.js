// VideoSource: one interface for every way the duck can "see".
// Everything downstream only calls drawTo(ctx, w, h) and reads width/height,
// so swapping the Pixel camera for the Pi's CSI stream later is a dropdown change.
//
// Kinds:
//   camera-back / camera-front : getUserMedia (needs HTTPS or localhost)
//   mjpeg   : <img src="http://pi.local:8080/stream.mjpg">  (multipart MJPEG)
//   snapshot: polls a JPEG URL (e.g. /snapshot.jpg) as fast as it can
//   video   : any URL a <video> can play (mp4, webm, HLS on Android Chrome)
//   file    : a local video file, handy for testing without a camera
//
// Cross-origin note: the models read pixels from a canvas. If the image comes
// from another origin without CORS headers, the canvas gets "tainted" and
// reading it throws. Easiest fix later: let the Pi serve this page AND the
// stream (same origin). Or add "Access-Control-Allow-Origin: *" on the Pi.

export class VideoSource {
  constructor() {
    this.kind = null;
    this.el = null;        // HTMLVideoElement | HTMLImageElement | ImageBitmap holder
    this.stream = null;
    this._poll = null;
    this._bitmap = null;
  }

  get width() {
    if (this._bitmap) return this._bitmap.width;
    if (!this.el) return 0;
    return this.el.videoWidth || this.el.naturalWidth || 0;
  }
  get height() {
    if (this._bitmap) return this._bitmap.height;
    if (!this.el) return 0;
    return this.el.videoHeight || this.el.naturalHeight || 0;
  }
  get ready() { return this.width > 0 && this.height > 0; }

  async start(kind, url = '', file = null) {
    this.stop();
    this.kind = kind;
    if (kind === 'camera-back' || kind === 'camera-front') {
      if (!window.isSecureContext) {
        throw new Error('Camera needs HTTPS (or localhost). Open this page from GitHub Pages, or see README for the Chrome flag.');
      }
      const facingMode = kind === 'camera-back' ? 'environment' : 'user';
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      });
      this.el = makeVideo();
      this.el.srcObject = this.stream;
      await this.el.play();
    } else if (kind === 'mjpeg') {
      this.el = new Image();
      this.el.crossOrigin = 'anonymous';
      this.el.src = url;
    } else if (kind === 'snapshot') {
      this._startSnapshotLoop(url);
    } else if (kind === 'video' || kind === 'file') {
      this.el = makeVideo();
      this.el.crossOrigin = 'anonymous';
      this.el.loop = true;
      this.el.src = kind === 'file' ? URL.createObjectURL(file) : url;
      await this.el.play();
    } else {
      throw new Error('Unknown source ' + kind);
    }
    await waitFor(() => this.ready, 8000, 'Video source never produced a frame. Check the URL / permissions.');
  }

  _startSnapshotLoop(url) {
    let alive = true;
    this._poll = () => { alive = false; };
    const loop = async () => {
      while (alive) {
        try {
          const sep = url.includes('?') ? '&' : '?';
          const r = await fetch(url + sep + 't=' + Date.now(), { cache: 'no-store' });
          const bmp = await createImageBitmap(await r.blob());
          if (this._bitmap) this._bitmap.close();
          this._bitmap = bmp;
        } catch (e) {
          await new Promise((r) => setTimeout(r, 500));
        }
      }
    };
    loop();
  }

  drawTo(ctx, w, h) {
    const src = this._bitmap || this.el;
    if (src) ctx.drawImage(src, 0, 0, w, h);
  }

  stop() {
    if (this._poll) this._poll();
    this._poll = null;
    if (this._bitmap) { this._bitmap.close(); this._bitmap = null; }
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.el && this.el.src && this.el.src.startsWith('blob:')) URL.revokeObjectURL(this.el.src);
    if (this.el) { this.el.src = ''; this.el.srcObject = null; }
    this.el = null;
  }
}

function makeVideo() {
  const v = document.createElement('video');
  v.playsInline = true;
  v.muted = true;
  v.autoplay = true;
  return v;
}

async function waitFor(cond, ms, msg) {
  const t0 = performance.now();
  while (!cond()) {
    if (performance.now() - t0 > ms) throw new Error(msg);
    await new Promise((r) => setTimeout(r, 50));
  }
}
