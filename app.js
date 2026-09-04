// Detection logic lives in detection.js, cross-frame fusion in fusion.js
// (both shared with the Node test harness).

// Surface any uncaught error on-screen instead of failing silently — this is
// the only way to see what went wrong on a phone with no console attached.
// Queries the DOM directly rather than the statusEl variable below, since
// this must work even if the crash happened before that variable was set.
window.addEventListener('error', e => {
  const el = document.getElementById('status');
  if (el) el.textContent = 'Script error: ' + e.message;
});

let cvReady = false;
let domReady = false;
let scanning = false;
let rafId = null;

let fusion = LawnBowlsFusion.createFusion(); // accumulated map, fed every usable frame while scanning
let frozen = null; // { detections, jack, ranking } laid out from the fused map on Stop, for tap-to-assign
let assignments = []; // parallel to frozen.ranking: 'mine' | 'theirs' | null

let video, overlay, overlayCtx, statusEl, rankingEl, scanBtn;
let captureCanvas, captureCtx; // offscreen: cv.imread() needs a canvas/img, not a <video>, as its source

function onOpenCvReady() {
  cv['onRuntimeInitialized'] = () => {
    cvReady = true;
    maybeStart();
  };
}

document.addEventListener('DOMContentLoaded', () => {
  video = document.getElementById('video');
  overlay = document.getElementById('overlay');
  overlayCtx = overlay.getContext('2d');
  statusEl = document.getElementById('status');
  rankingEl = document.getElementById('ranking');
  scanBtn = document.getElementById('scanBtn');
  captureCanvas = document.createElement('canvas');
  captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });

  scanBtn.addEventListener('click', toggleScan);
  overlay.addEventListener('click', handleCanvasTap);
  window.addEventListener('resize', sizeOverlay);

  domReady = true;
  maybeStart();
});

function maybeStart() {
  if (cvReady && domReady) startCamera();
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
    setStatus('Hold the phone over the rink and tap Start Scan.');
    scanBtn.disabled = false;
  } catch (err) {
    setStatus('Camera error: ' + err.message);
  }
}

function sizeOverlay() {
  if (!video.videoWidth) return;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  captureCanvas.width = video.videoWidth;
  captureCanvas.height = video.videoHeight;
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function toggleScan() {
  if (scanning) {
    scanning = false;
    if (rafId) cancelAnimationFrame(rafId);
    scanBtn.textContent = 'Start Scan';
    video.pause();

    // Lay out the fused map (not just the last frame) so bowls seen anywhere
    // during the pan — even if out of frame now — have stable positions to tap.
    const snapshot = LawnBowlsFusion.getSnapshot(fusion);
    if (snapshot.bowls.length > 0) {
      frozen = LawnBowlsFusion.layoutForCanvas(snapshot, overlay.width, overlay.height);
      assignments = new Array(frozen.ranking.length).fill(null);
      setStatus(`Map built from ${fusion.frameCount} frame(s). Tap each flag, closest first, to mark it yours or theirs.`);
      renderFrozen();
    } else {
      frozen = null;
      assignments = [];
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      rankingEl.innerHTML = '';
      setStatus('No usable frames were scanned — rescan and hold steady over the bowls.');
    }
  } else {
    scanning = true;
    frozen = null;
    assignments = [];
    fusion = LawnBowlsFusion.createFusion();
    rankingEl.innerHTML = '';
    scanBtn.textContent = 'Stop Scan';
    setStatus('Scanning…');
    video.play();
    rafId = requestAnimationFrame(processFrame);
  }
}

function processFrame() {
  if (!scanning) return;

  try {
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const src = cv.imread(captureCanvas);
    const result = LawnBowlsDetection.detectAndRank(cv, src);
    src.delete();

    if (result.usable) LawnBowlsFusion.addFrame(fusion, result);

    drawOverlay(result.detections, result.jack, result.usable ? result.ranking : []);
    renderRanking(result.ranking, result.usable, result.reason, result.detections.length);

    const mapSnapshot = LawnBowlsFusion.getSnapshot(fusion);
    const confirmedCount = mapSnapshot.ranking.filter(r => r.confirmed).length;
    setStatus(`Scanning… ${mapSnapshot.bowls.length} bowl(s) tracked, ${confirmedCount} confirmed. Stop when ready.`);
  } catch (err) {
    scanning = false;
    scanBtn.textContent = 'Start Scan';
    setStatus('Scan error: ' + err.message);
    return;
  }

  rafId = requestAnimationFrame(processFrame);
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
