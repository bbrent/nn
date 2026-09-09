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
const MAX_RECENT_OBSERVATIONS = 40; // how many recent bowl thumbnails the picker keeps
const MAX_DETECTIONS_EMBEDDED_PER_FRAME = 15; // bound embedding cost on a noisy frame
const THUMBNAIL_SIZE = 64;

let cameraReady = false;
let modelReady = false;
let domReady = false;
let scanning = false;
let picking = false; // registration picker's live capture loop is running
let rafId = null;
let yoloSession = null;
let embeddingSession = null;

let fusion = LawnBowlsFusion.createFusion(); // accumulated map, fed every usable frame while scanning
let frozen = null; // { detections, jack, ranking } laid out from the fused map on Stop, for tap-to-assign
let assignments = []; // parallel to frozen.ranking: 'mine' | 'theirs' | null
let registry = LawnBowlsRegistry.createRegistry(); // player roster + appearance galleries, persisted in localStorage
let recentObservations = []; // newest first: { id, thumbnail (data URL), embedding } — feeds the registration picker
let pickerSelection = new Set(); // observation ids currently selected in the open picker

let video, overlay, overlayCtx, statusEl, rankingEl, scanBtn, registerBtn, registryEl;
let pickerModal, pickerGrid, pickerConfirmBtn, pickerCancelBtn;
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
  pickerModal = document.getElementById('pickerModal');
  pickerGrid = document.getElementById('pickerGrid');
  pickerConfirmBtn = document.getElementById('pickerConfirm');
  pickerCancelBtn = document.getElementById('pickerCancel');

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
  registerBtn.addEventListener('click', openRegistrationPicker);
  pickerConfirmBtn.addEventListener('click', confirmPicker);
  pickerCancelBtn.addEventListener('click', cancelPicker);
  overlay.addEventListener('click', handleCanvasTap);
  window.addEventListener('resize', sizeOverlay);

  domReady = true;
  startCamera();
  loadModels();
});

function maybeEnableScan() {
  if (cameraReady && modelReady) {
    setStatus('Scanning… point the camera at the bowls.');
    registerBtn.disabled = false;
    // Auto-start scanning instead of waiting for user to click
    if (!scanning && !frozen) {
      toggleScan();
    }
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

// cropCanvas still holds whatever cropBowlImageData last drew onto it — reuse
// it directly as the thumbnail source rather than re-decoding the ImageData.
function makeThumbnailFromCropCanvas() {
  const thumbCanvas = document.createElement('canvas');
  thumbCanvas.width = THUMBNAIL_SIZE;
  thumbCanvas.height = THUMBNAIL_SIZE;
  thumbCanvas.getContext('2d').drawImage(cropCanvas, 0, 0, THUMBNAIL_SIZE, THUMBNAIL_SIZE);
  return thumbCanvas.toDataURL('image/jpeg', 0.7);
}

// Records a just-embedded crop as a recent observation for the registration
// picker (newest first, capped). Call right after cropBowlImageData so
// cropCanvas still holds the matching crop.
function recordObservation(embedding) {
  recentObservations.unshift({
    id: 'obs_' + Math.random().toString(36).slice(2, 10),
    thumbnail: makeThumbnailFromCropCanvas(),
    embedding,
  });
  if (recentObservations.length > MAX_RECENT_OBSERVATIONS) {
    recentObservations.length = MAX_RECENT_OBSERVATIONS;
  }
  if (picking) renderPickerGrid();
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function toggleScan() {
  if (scanning) {
    scanning = false;
    if (rafId) cancelAnimationFrame(rafId);
    scanBtn.hidden = true;
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
      setStatus(`Map built from ${fusion.frameCount} frame(s).${matchedNote} Tap any dashed outline to assign a team.`);
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
    scanBtn.hidden = false;
    scanBtn.textContent = 'Stop Scan';
    registerBtn.disabled = true;
    setStatus('Scanning… point the camera at the bowls.');
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

    // Native-resolution frame to crop bowls from — the 640x640 letterboxed
    // one is too downscaled for a clean embedding crop. Embed every raw
    // detection (capped) once, regardless of usable/ranked status — this
    // also feeds the registration picker's "recently seen" thumbnails, not
    // just registry matching for fusion.
    frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    const embeddingByDetection = new Map();
    for (const d of result.detections.slice(0, MAX_DETECTIONS_EMBEDDED_PER_FRAME)) {
      const crop = cropBowlImageData(frameCanvas, d);
      if (!crop) continue;
      const embedding = await LawnBowlsEmbedding.embedCrop(ort, embeddingSession, crop);
      embeddingByDetection.set(d, embedding);
      recordObservation(embedding);
    }

    if (result.usable) {
      for (const entry of result.ranking) {
        const embedding = embeddingByDetection.get(entry.bowl);
        entry.identity = embedding ? LawnBowlsRegistry.matchBowl(registry, embedding) : null;
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

// --- Player registration: pick from recently-seen bowls, then name them ---

function openRegistrationPicker() {
  if (!modelReady || !cameraReady || scanning || picking) return;

  pickerSelection.clear();
  picking = true;
  scanBtn.disabled = true;
  registerBtn.disabled = true;
  pickerModal.hidden = false;
  renderPickerGrid();
  video.play();

  pickerLoop();
}

// Runs until the picker is closed (Confirm/Cancel) — keeps detecting and
// adding new thumbnails to the top of the grid (via recordObservation) so the
// user can point the camera around and watch candidates appear, live, before
// selecting any — rather than a blind timed capture with no visual check.
async function pickerLoop() {
  while (picking) {
    try {
      const letterbox = LawnBowlsYolo.computeLetterbox(video.videoWidth, video.videoHeight, LawnBowlsYolo.INPUT_SIZE);
      letterboxCtx.fillStyle = 'rgb(114,114,114)';
      letterboxCtx.fillRect(0, 0, LawnBowlsYolo.INPUT_SIZE, LawnBowlsYolo.INPUT_SIZE);
      letterboxCtx.drawImage(video, letterbox.padX, letterbox.padY, letterbox.newWidth, letterbox.newHeight);
      const imageData640 = letterboxCtx.getImageData(0, 0, LawnBowlsYolo.INPUT_SIZE, LawnBowlsYolo.INPUT_SIZE);

      // No jack/bowl classification needed here — any round object seen
      // while the picker is open is just a candidate to review and select.
      const result = await LawnBowlsYolo.detectAndRank(ort, yoloSession, imageData640, letterbox);
      frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);

      for (const d of result.detections.slice(0, MAX_DETECTIONS_EMBEDDED_PER_FRAME)) {
        const crop = cropBowlImageData(frameCanvas, d);
        if (!crop) continue;
        const embedding = await LawnBowlsEmbedding.embedCrop(ort, embeddingSession, crop);
        recordObservation(embedding); // re-renders the grid itself while picking
      }
    } catch (err) {
      setStatus('Registration error: ' + err.message);
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 400));
  }
}

function renderPickerGrid() {
  pickerGrid.innerHTML = '';
  recentObservations.forEach(obs => {
    const cell = document.createElement('div');
    cell.className = 'pickerThumb' + (pickerSelection.has(obs.id) ? ' selected' : '');
    const img = document.createElement('img');
    img.src = obs.thumbnail;
    cell.appendChild(img);
    cell.addEventListener('click', () => {
      if (pickerSelection.has(obs.id)) pickerSelection.delete(obs.id);
      else pickerSelection.add(obs.id);
      renderPickerGrid();
    });
    pickerGrid.appendChild(cell);
  });
}

function closePicker() {
  picking = false;
  pickerModal.hidden = true;
  scanBtn.disabled = false;
  registerBtn.disabled = false;
}

function cancelPicker() {
  closePicker();
  if (frozen) renderFrozen();
}

function confirmPicker() {
  const selected = recentObservations.filter(obs => pickerSelection.has(obs.id));
  if (selected.length === 0) {
    alert('Select at least one bowl first.');
    return;
  }

  const name = (prompt('Player name?') || '').trim();
  if (!name) return;
  const isMine = confirm("Is this player on your team?\nOK = mine, Cancel = the opponent's.");
  const team = isMine ? 'mine' : 'theirs';

  const playerId = LawnBowlsRegistry.addPlayer(registry, name, team);
  selected.forEach(obs => LawnBowlsRegistry.addGalleryView(registry, playerId, obs.embedding));
  saveRegistry();
  renderRegistryList();

  closePicker();
  if (frozen) {
    renderFrozen();
  } else {
    setStatus(`Registered ${name} (${team === 'mine' ? 'yours' : "opponent's"}) with ${selected.length} view(s). Hold the phone over the rink and tap Start Scan.`);
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
    // Use confidence for opacity fade — low confidence bowls appear ghosted
    const opacity = isJack ? 1.0 : (d.conf || 0.5);
    drawAura(d, color, isJack, opacity);
  }

  ranking.forEach((entry, i) => drawFlag(entry.bowl, i + 1, i === 0 ? '#2e7d32' : '#1565c0'));
}

function drawAura(d, color, isJack, opacity, isDashed) {
  opacity = opacity !== undefined ? opacity : 1.0;

  const glowR = d.r * 2.2;
  const gradient = overlayCtx.createRadialGradient(d.x, d.y, d.r * 0.6, d.x, d.y, glowR);
  gradient.addColorStop(0, hexToRgba(color, 0.5 * opacity));
  gradient.addColorStop(1, hexToRgba(color, 0));
  overlayCtx.fillStyle = gradient;
  overlayCtx.beginPath();
  overlayCtx.arc(d.x, d.y, glowR, 0, 2 * Math.PI);
  overlayCtx.fill();

  overlayCtx.beginPath();
  overlayCtx.arc(d.x, d.y, d.r, 0, 2 * Math.PI);
  overlayCtx.strokeStyle = color;
  overlayCtx.lineWidth = isJack ? 3 : 2.5;
  if (isDashed) {
    overlayCtx.setLineDash([6, 4]);
  }
  overlayCtx.globalAlpha = opacity;
  overlayCtx.stroke();
  overlayCtx.setLineDash([]);
  overlayCtx.globalAlpha = 1.0;
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
    // Dashed outline for unassigned bowls, solid for assigned
    const isUnassigned = rankIndex >= 0 && assignments[rankIndex] === null;
    drawAura(d, color, isJack, 1.0, isUnassigned);
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

  if (bestIndex >= 0) {
    // Unassigned bowl: prompt for team; assigned: cycle to change
    if (assignments[bestIndex] === null) {
      promptTeamAssignment(bestIndex);
    } else {
      cycleAssignment(bestIndex);
    }
  }
}

function promptTeamAssignment(i) {
  const isYours = confirm('Is this bowl yours (red)? OK=yours, Cancel=opponent\'s (blue).');
  assignments[i] = isYours ? 'mine' : 'theirs';
  renderFrozen();
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
