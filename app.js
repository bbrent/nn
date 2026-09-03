// Milestone 1: single-frame circle detection + relative distance to jack.
// Detection logic lives in detection.js (shared with the Node test harness).
// Cross-frame fusion (panning the phone over the whole rink) comes next.

let cvReady = false;
let domReady = false;
let scanning = false;
let rafId = null;

let lastResult = null; // most recent live detectAndRank() result, refreshed every frame
let frozen = null; // { detections, jack, ranking } captured on Stop, for tap-to-assign
let assignments = []; // parallel to frozen.ranking: 'mine' | 'theirs' | null

let video, overlay, overlayCtx, statusEl, rankingEl, scanBtn;

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
}

function setStatus(msg) {
  statusEl.textContent = msg;
}

function toggleScan() {
  if (scanning) {
    scanning = false;
    if (rafId) cancelAnimationFrame(rafId);
    scanBtn.textContent = 'Start Scan';

    // Freeze the last usable frame so bowls have stable positions to tap.
    if (lastResult && lastResult.usable) {
      frozen = lastResult;
      assignments = new Array(frozen.ranking.length).fill(null);
      setStatus('End frozen. Tap each flag, closest first, to mark it yours or theirs.');
      renderFrozen();
    } else {
      frozen = null;
      assignments = [];
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      rankingEl.innerHTML = '';
      setStatus("Last frame wasn't reliable enough to score — rescan and stop on a clear view.");
    }
  } else {
    scanning = true;
    frozen = null;
    assignments = [];
    rankingEl.innerHTML = '';
    scanBtn.textContent = 'Stop Scan';
    setStatus('Scanning…');
    rafId = requestAnimationFrame(processFrame);
  }
}

function processFrame() {
  if (!scanning) return;

  const src = cv.imread(video);
  const result = LawnBowlsDetection.detectAndRank(cv, src);
  src.delete();
  lastResult = result;

  drawOverlay(result.detections, result.jack, result.usable ? result.ranking : []);
  renderRanking(result.ranking, result.usable, result.reason, result.detections.length);

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
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

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
    const score = LawnBowlsDetection.computeScore(assignments);
    if (score.team === null) {
      summary.textContent = "Tap the closest bowl's flag to say whose it is.";
    } else {
      const label = score.team === 'mine' ? 'You' : 'Opponent';
      summary.textContent = score.pending
        ? `${label}: at least ${score.count} — keep tagging to confirm`
        : `${label} score this end: ${score.count}`;
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
