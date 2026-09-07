// Detection is now YOLOv8n (yolo-detector.js) run via onnxruntime-web,
// replacing the old Hough-based detection.js circle finder — real turf
// texture flooded Hough with 80-190 spurious "circles" per photo (see
// test/fixtures/real/), while YOLO reliably finds real bowls even at low
// confidence, filtered to sports-ball/bowl classes. computeScore is still
// shared from detection.js (pure ranking-list logic, detector-agnostic);
// fusion.js needed zero changes at all, since it only ever depended on the
// {jack, bowls, ranking, usable} shape, not on how it was produced.

// Surface any uncaught error on-screen instead of failing silently — this is
// the only way to see what went wrong on a phone with no console attached.
// Queries the DOM directly rather than the statusEl variable below, since
// this must work even if the crash happened before that variable was set.
window.addEventListener('error', e => {
  const el = document.getElementById('status');
  if (el) el.textContent = 'Script error: ' + e.message;
});

const YOLO_MODEL_URL = './models/yolov8n.onnx';
const EMBEDDING_MODEL_URL = './models/mobilenetv2-embedding.onnx';
const REGISTRY_STORAGE_KEY = 'lawnBowlsRegistry';
const REGISTRATION_DURATION_MS = 6000;
const REGISTRATION_VIEW_LIMIT = 12;

let cameraReady = false;
let modelReady = false;
let domReady = false;
let scanning = false;
let registering = false;
let rafId = null;
let yoloSession = null;
let embeddingSession = null;

let fusion = LawnBowlsFusion.createFusion(); // accumulated map, fed every usable frame while scanning
let frozen = null; // { detections, jack, ranking } laid out from the fused map on Stop, for tap-to-assign
let assignments = []; // parallel to frozen.ranking: 'mine' | 'theirs' | null
let registry = LawnBowlsRegistry.createRegistry(); // player roster + appearance galleries, persisted in localStorage

let video, overlay, overlayCtx, statusEl, rankingEl, scanBtn, registerBtn, registryEl;
let letterboxCanvas, letterboxCtx; // offscreen 640x640: YOLO's fixed input size
let frameCanvas, frameCtx; // offscreen, native video resolution: source for bowl crops
let cropCanvas, cropCtx; // offscreen 224x224: embedding model's fixed input size

document.addEventListener('DOMContentLoaded', () => {
  video = document.getElementById('video');
  overlay = document.getElementById('overlay');
  overlayCtx = overlay.getContext('2d');
  statusEl = document.getElementById('status');
  rankingEl = document.getElementById('ranking');
  scanBtn = document.getElementById('scanBtn');
  registerBtn = document.getElementById('registerBtn');
  registryEl = document.getElementById('registry');

  letterboxCanvas = document.createElement('canvas');
  letterboxCanvas.width = LawnBowlsYolo.INPUT_SIZE;
  letterboxCanvas.height = LawnBowlsYolo.INPUT_SIZE;
  letterboxCtx = letterboxCanvas.getContext('2d', { willReadFrequently: true });

  frameCanvas = document.createElement('canvas');
  frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true });

  cropCanvas = document.createElement('canvas');
  cropCanvas.width = LawnBowlsEmbedding.INPUT_SIZE;
  cropCanvas.height = LawnBowlsEmbedding.INPUT_SIZE;
  cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

  loadRegistry();
  renderRegistryList();

  scanBtn.addEventListener('click', toggleScan);
  registerBtn.addEventListener('click', startRegistration);
  overlay.addEventListener('click', handleCanvasTap);
  window.addEventListener('resize', sizeOverlay);

  domReady = true;
  startCamera();
  loadModels();
});

function maybeEnableScan() {
  if (cameraReady && modelReady) {
    setStatus('Hold the phone over the rink and tap Start Scan.');
    scanBtn.disabled = false;
    registerBtn.disabled = false;
  }
}

async function loadModels() {
  try {
    setStatus('Loading detection model…');
    yoloSession = await ort.InferenceSession.create(YOLO_MODEL_URL, { executionProviders: ['wasm'] });
    setStatus('Loading player-matching model…');
    embeddingSession = await ort.InferenceSession.create(EMBEDDING_MODEL_URL, { executionProviders: ['wasm'] });
    modelReady = true;
    maybeEnableScan();
  } catch (err) {
    setStatus('Model load error: ' + err.message);
  }
}

function loadRegistry() {
  try {
    const stored = localStorage.getItem(REGISTRY_STORAGE_KEY);
    if (stored) registry = LawnBowlsRegistry.deserialize(stored);
  } catch (err) {
    // Corrupt/unavailable storage — start fresh rather than block the app.
    registry = LawnBowlsRegistry.createRegistry();
  }
}

function saveRegistry() {
  try {
    localStorage.setItem(REGISTRY_STORAGE_KEY, LawnBowlsRegistry.serialize(registry));
  } catch (err) {
    // Storage full/unavailable (private browsing, etc.) — registry still
    // works for the rest of this session, just won't persist.
  }
}

async function startCamera() {
  try {
    setStatus('Requesting camera…');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    sizeOverlay();
    cameraReady = true;
    maybeEnableScan();
  } catch (err) {
    setStatus('Camera error: ' + err.message);
  }
}

function sizeOverlay() {
  if (!video.videoWidth) return;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  frameCanvas.width = video.videoWidth;
  frameCanvas.height = video.videoHeight;
}

// Crops a square region around a detection (a bit larger than its radius,
// for context) from a native-resolution source canvas, scaled to the
// embedding model's fixed input size. Returns null if the detection is too
// close to the frame edge to crop meaningfully.
function cropBowlImageData(sourceCanvas, bowl) {
  const side = bowl.r * 2.4;
  let sx = bowl.x - side / 2;
  let sy = bowl.y - side / 2;
  let sw = side;
  let sh = side;
  if (sx < 0) { sw += sx; sx = 0; }
  if (sy < 0) { sh += sy; sy = 0; }
  if (sx + sw > sourceCanvas.width) sw = sourceCanvas.width - sx;
  if (sy + sh > sourceCanvas.height) sh = sourceCanvas.height - sy;
  if (sw <= 0 || sh <= 0) return null;

  const size = LawnBowlsEmbedding.INPUT_SIZE;
  cropCtx.clearRect(0, 0, size, size);
  cropCtx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, size, size);
  return cropCtx.getImageData(0, 0, size, size);
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function toggleScan() {
  if (scanning) {
    scanning = false;
    if (rafId) cancelAnimationFrame(rafId);
    scanBtn.textContent = 'Start Scan';
    registerBtn.disabled = false;
    video.pause();

    // Lay out the fused map (not just the last frame) so bowls seen anywhere
    // during the pan — even if out of frame now — have stable positions to tap.
    // confirmedOnly: a stray object detected once or twice (a shoe, a
    // hand) shouldn't get permanently scored as a bowl just because it was
    // in frame briefly — only landmarks seen consistently make the cut.
    const snapshot = LawnBowlsFusion.getSnapshot(fusion, { confirmedOnly: true });
    if (snapshot.bowls.length > 0) {
      frozen = LawnBowlsFusion.layoutForCanvas(snapshot, overlay.width, overlay.height);
      // Pre-fill from any confident registry match made live during the scan;
      // anything unmatched stays null for the existing tap-to-assign fallback.
      assignments = frozen.ranking.map(entry => (entry.bowl.identity ? entry.bowl.identity.team : null));
      const matchedCount = assignments.filter(a => a !== null).length;
      const matchedNote = matchedCount > 0 ? ` ${matchedCount} auto-matched from the registry.` : '';
      setStatus(`Map built from ${fusion.frameCount} frame(s).${matchedNote} Tap any flag to set or correct whose it is.`);
      renderFrozen();
    } else {
      frozen = null;
      assignments = [];
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      rankingEl.innerHTML = '';
      setStatus('No bowl was seen consistently enough to trust — rescan and hold steadier over each one.');
    }
  } else {
    scanning = true;
    frozen = null;
    assignments = [];
    fusion = LawnBowlsFusion.createFusion();
    rankingEl.innerHTML = '';
    scanBtn.textContent = 'Stop Scan';
    registerBtn.disabled = true;
    setStatus('Scanning…');
    video.play();
    rafId = requestAnimationFrame(processFrame);
  }
}

// Self-paced rather than a fixed rAF rate: inference is much heavier than
// Hough was, so the next frame is only requested once this one's detection
// (including the await) has actually finished, instead of queuing overlapping
// inference calls every screen refresh.
async function processFrame() {
  if (!scanning) return;

  try {
    const letterbox = LawnBowlsYolo.computeLetterbox(video.videoWidth, video.videoHeight, LawnBowlsYolo.INPUT_SIZE);
    letterboxCtx.fillStyle = 'rgb(114,114,114)';
    letterboxCtx.fillRect(0, 0, LawnBowlsYolo.INPUT_SIZE, LawnBowlsYolo.INPUT_SIZE);
    letterboxCtx.drawImage(video, letterbox.padX, letterbox.padY, letterbox.newWidth, letterbox.newHeight);
    const imageData640 = letterboxCtx.getImageData(0, 0, LawnBowlsYolo.INPUT_SIZE, LawnBowlsYolo.INPUT_SIZE);

    const result = await LawnBowlsYolo.detectAndRank(ort, yoloSession, imageData640, letterbox);

    if (result.usable) {
      // Native-resolution frame to crop bowls from — the 640x640 letterboxed
      // one is too downscaled for a clean embedding crop.
      frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
      for (const entry of result.ranking) {
        const crop = cropBowlImageData(frameCanvas, entry.bowl);
        if (!crop) continue;
        const embedding = await LawnBowlsEmbedding.embedCrop(ort, embeddingSession, crop);
        entry.identity = LawnBowlsRegistry.matchBowl(registry, embedding);
      }
      LawnBowlsFusion.addFrame(fusion, result);
    }

    drawOverlay(result.detections, result.jack, result.usable ? result.ranking : []);
    renderRanking(result.ranking, result.usable, result.reason, result.detections.length);

    const mapSnapshot = LawnBowlsFusion.getSnapshot(fusion);
    const confirmedCount = mapSnapshot.ranking.filter(r => r.confirmed).length;
    setStatus(`Scanning… ${mapSnapshot.bowls.length} bowl(s) tracked, ${confirmedCount} confirmed. Stop when ready.`);
  } catch (err) {
    scanning = false;
    scanBtn.textContent = 'Start Scan';
    registerBtn.disabled = false;
    setStatus('Scan error: ' + err.message);
    return;
  }

  rafId = requestAnimationFrame(processFrame);
}

// --- Player registration: capture multiple views per player's bowls -------

async function startRegistration() {
  if (!modelReady || !cameraReady || scanning || registering) return;

  const name = (prompt('Player name?') || '').trim();
  if (!name) return;
  const isMine = confirm("Is this player on your team?\nOK = mine, Cancel = the opponent's.");
  const team = isMine ? 'mine' : 'theirs';

  const playerId = LawnBowlsRegistry.addPlayer(registry, name, team);

  registering = true;
  scanBtn.disabled = true;
  registerBtn.disabled = true;
  video.play();

  let viewCount = 0;
  const stopAt = Date.now() + REGISTRATION_DURATION_MS;

  try {
    while (Date.now() < stopAt && viewCount < REGISTRATION_VIEW_LIMIT) {
      const letterbox = LawnBowlsYolo.computeLetterbox(video.videoWidth, video.videoHeight, LawnBowlsYolo.INPUT_SIZE);
      letterboxCtx.fillStyle = 'rgb(114,114,114)';
      letterboxCtx.fillRect(0, 0, LawnBowlsYolo.INPUT_SIZE, LawnBowlsYolo.INPUT_SIZE);
      letterboxCtx.drawImage(video, letterbox.padX, letterbox.padY, letterbox.newWidth, letterbox.newHeight);
      const imageData640 = letterboxCtx.getImageData(0, 0, LawnBowlsYolo.INPUT_SIZE, LawnBowlsYolo.INPUT_SIZE);

      // Registration doesn't need jack/bowl classification or ranking — any
      // round object in view during this window is assumed to be this
      // player's own bowl, shown deliberately.
      const result = await LawnBowlsYolo.detectAndRank(ort, yoloSession, imageData640, letterbox);
      frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
      drawOverlay(result.detections, null, []);

      for (const d of result.detections) {
        if (viewCount >= REGISTRATION_VIEW_LIMIT) break;
        const crop = cropBowlImageData(frameCanvas, d);
        if (!crop) continue;
        const embedding = await LawnBowlsEmbedding.embedCrop(ort, embeddingSession, crop);
        LawnBowlsRegistry.addGalleryView(registry, playerId, embedding);
        viewCount++;
      }

      setStatus(`Registering ${name}… show their bowls, rotating a bit. ${viewCount} view(s) captured.`);
      await new Promise(resolve => setTimeout(resolve, 400));
    }
  } catch (err) {
    setStatus('Registration error: ' + err.message);
  }

  registering = false;
  scanBtn.disabled = false;
  registerBtn.disabled = false;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  saveRegistry();
  renderRegistryList();

  if (frozen) {
    renderFrozen();
  } else {
    setStatus(`Registered ${name} (${team === 'mine' ? 'yours' : "opponent's"}) with ${viewCount} view(s). Hold the phone over the rink and tap Start Scan.`);
  }
}

function renderRegistryList() {
  registryEl.innerHTML = '';
  if (registry.players.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No players registered yet.';
    registryEl.appendChild(li);
    return;
  }

  registry.players.forEach(player => {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = `${player.name} (${player.team === 'mine' ? 'yours' : "opponent's"}) — ${player.gallery.length} view(s)`;
    const removeBtn = document.createElement('button');
    removeBtn.textContent = '×';
    removeBtn.title = 'Remove player';
    removeBtn.addEventListener('click', () => {
      LawnBowlsRegistry.removePlayer(registry, player.id);
      saveRegistry();
      renderRegistryList();
    });
    li.appendChild(label);
    li.appendChild(removeBtn);
    registryEl.appendChild(li);
  });
}

// Closest-to-farthest color scale for ranked bowls; unranked detections (jack
// aside) fall back to blue.
const RANK_COLORS = ['#66bb6a', '#9ccc65', '#ffee58', '#ffb74d', '#ef5350'];
const JACK_COLOR = '#ffd54f';
const UNRANKED_COLOR = '#42a5f5';

function drawOverlay(detections, jack, ranking) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  for (const d of detections) {
    const isJack = d === jack;
    const rankIndex = ranking.findIndex(entry => entry.bowl === d);
    const color = isJack ? JACK_COLOR : rankIndex >= 0 ? RANK_COLORS[Math.min(rankIndex, RANK_COLORS.length - 1)] : UNRANKED_COLOR;
    drawAura(d, color, isJack);
  }

  ranking.forEach((entry, i) => drawFlag(entry.bowl, i + 1, i === 0 ? '#2e7d32' : '#1565c0'));
}

function drawAura(d, color, isJack) {
  const glowR = d.r * 2.2;
  const gradient = overlayCtx.createRadialGradient(d.x, d.y, d.r * 0.6, d.x, d.y, glowR);
  gradient.addColorStop(0, hexToRgba(color, 0.5));
  gradient.addColorStop(1, hexToRgba(color, 0));
  overlayCtx.fillStyle = gradient;
  overlayCtx.beginPath();
  overlayCtx.arc(d.x, d.y, glowR, 0, 2 * Math.PI);
  overlayCtx.fill();

  overlayCtx.beginPath();
  overlayCtx.arc(d.x, d.y, d.r, 0, 2 * Math.PI);
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = isJack ? 3 : 2.5;
  overlayCtx.stroke();
}

function hexToRgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

// Small pennant on a pole above a bowl, labeled with its closeness rank (1 = closest to jack).
function drawFlag(d, rank, fillColor) {
  const poleX = d.x;
  const baseY = d.y - d.r;
  const topY = baseY - d.r * 1.8;

  overlayCtx.beginPath();
  overlayCtx.moveTo(poleX, baseY);
  overlayCtx.lineTo(poleX, topY);
  overlayCtx.strokeStyle = '#ffffff';
  overlayCtx.lineWidth = 2;
  overlayCtx.stroke();

  const flagW = Math.max(18, d.r * 1.1);
  const flagH = flagW * 0.65;
  overlayCtx.beginPath();
  overlayCtx.moveTo(poleX, topY);
  overlayCtx.lineTo(poleX + flagW, topY + flagH * 0.3);
  overlayCtx.lineTo(poleX, topY + flagH);
  overlayCtx.closePath();
  overlayCtx.fillStyle = fillColor;
  overlayCtx.fill();
  overlayCtx.strokeStyle = 'rgba(0,0,0,0.4)';
  overlayCtx.lineWidth = 1;
  overlayCtx.stroke();

  overlayCtx.fillStyle = '#ffffff';
  overlayCtx.font = `bold ${Math.max(11, flagH * 0.55)}px system-ui, sans-serif`;
  overlayCtx.textAlign = 'center';
  overlayCtx.textBaseline = 'middle';
  overlayCtx.fillText(String(rank), poleX + flagW * 0.4, topY + flagH * 0.32);
}

function renderRanking(ranking, usable, reason, detectionCount) {
  rankingEl.innerHTML = '';

  if (!usable) {
    const li = document.createElement('li');
    li.textContent = `${reason} (${detectionCount} circle(s) detected)`;
    rankingEl.appendChild(li);
    return;
  }

  ranking.forEach((entry, i) => {
    const li = document.createElement('li');
    li.textContent = `#${i + 1} bowl — ${entry.dist.toFixed(2)} bowl-diameters from jack`;
    rankingEl.appendChild(li);
  });
}

// --- Frozen end: tap-to-assign team ownership, closest bowl first --------

const TEAM_COLORS = { mine: '#2e7d32', theirs: '#c62828' };
const UNASSIGNED_FLAG_COLOR = '#616161';
const ASSIGNMENT_CYCLE = [null, 'mine', 'theirs'];

function renderFrozen() {
  // Fused positions aren't tied to whatever the camera currently sees (some
  // bowls may be out of frame by now), so this is an abstract top-down map,
  // not an overlay on the live picture — paint over the paused video feed.
  overlayCtx.fillStyle = '#1b5e20';
  overlayCtx.fillRect(0, 0, overlay.width, overlay.height);

  for (const d of frozen.detections) {
    const isJack = d === frozen.jack;
    const rankIndex = frozen.ranking.findIndex(entry => entry.bowl === d);
    const color = isJack ? JACK_COLOR : rankIndex >= 0 ? RANK_COLORS[Math.min(rankIndex, RANK_COLORS.length - 1)] : UNRANKED_COLOR;
    drawAura(d, color, isJack);
  }

  frozen.ranking.forEach((entry, i) => {
    const flagColor = assignments[i] ? TEAM_COLORS[assignments[i]] : UNASSIGNED_FLAG_COLOR;
    drawFlag(entry.bowl, i + 1, flagColor);
  });

  renderScoreUI();
}

function renderScoreUI() {
  rankingEl.innerHTML = '';

  const summary = document.createElement('li');
  summary.style.fontWeight = 'bold';

  if (frozen.ranking.length === 0) {
    summary.textContent = 'No bowls to score (only the jack was detected).';
  } else {
    const score = LawnBowlsDetection.computeScore(frozen.ranking, assignments);
    if (score.team === null) {
      summary.textContent = "Tap the closest bowl's flag to say whose it is.";
    } else {
      const label = score.team === 'mine' ? 'You' : 'Opponent';
      if (score.pending) {
        summary.textContent = `${label}: at least ${score.count} — keep tagging to confirm`;
      } else if (score.tooClose) {
        summary.textContent = `${label}: ${score.count} — but the deciding bowls are too close to call from the scan, measure by hand`;
      } else {
        summary.textContent = `${label} score this end: ${score.count}`;
      }
    }
  }
  rankingEl.appendChild(summary);

  frozen.ranking.forEach((entry, i) => {
    const li = document.createElement('li');
    const team = assignments[i];
    const label = team === 'mine' ? 'Mine' : team === 'theirs' ? "Theirs" : 'Unassigned — tap to set';
    li.textContent = `#${i + 1} bowl — ${entry.dist.toFixed(2)} bowl-diameters from jack — ${label}`;
    li.style.cursor = 'pointer';
    li.addEventListener('click', () => cycleAssignment(i));
    rankingEl.appendChild(li);
  });
}

function cycleAssignment(i) {
  const current = ASSIGNMENT_CYCLE.indexOf(assignments[i]);
  assignments[i] = ASSIGNMENT_CYCLE[(current + 1) % ASSIGNMENT_CYCLE.length];
  renderFrozen();
}

function handleCanvasTap(evt) {
  if (!frozen) return;

  const pt = canvasPointFromEvent(evt);
  let bestIndex = -1;
  let bestDist = Infinity;

  frozen.ranking.forEach((entry, i) => {
    const d = entry.bowl;
    // Bias the hit region up toward the flag, since that's the visible tap target.
    const hitCx = d.x;
    const hitCy = d.y - d.r * 1.4;
    const hitR = Math.max(d.r * 1.8, 26);
    const dist = Math.hypot(pt.x - hitCx, pt.y - hitCy);
    if (dist < hitR && dist < bestDist) {
      bestDist = dist;
      bestIndex = i;
    }
  });

  if (bestIndex >= 0) cycleAssignment(bestIndex);
}

// Maps a click's CSS-pixel position to the canvas's internal pixel space,
// accounting for object-fit:cover scaling (which crops rather than stretches).
function canvasPointFromEvent(evt) {
  const rect = overlay.getBoundingClientRect();
  const cssX = evt.clientX - rect.left;
  const cssY = evt.clientY - rect.top;

  const scale = Math.max(rect.width / overlay.width, rect.height / overlay.height);
  const renderedW = overlay.width * scale;
  const renderedH = overlay.height * scale;
  const offsetX = (rect.width - renderedW) / 2;
  const offsetY = (rect.height - renderedH) / 2;

  return {
    x: (cssX - offsetX) / scale,
    y: (cssY - offsetY) / scale,
  };
}
