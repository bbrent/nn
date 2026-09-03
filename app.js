// Milestone 1: single-frame circle detection + relative distance to jack.
// Detection logic lives in detection.js (shared with the Node test harness).
// Cross-frame fusion (panning the phone over the whole rink) comes next.

let cvReady = false;
let domReady = false;
let scanning = false;
let rafId = null;

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
  scanning = !scanning;
  scanBtn.textContent = scanning ? 'Stop Scan' : 'Start Scan';
  if (scanning) {
    rafId = requestAnimationFrame(processFrame);
  } else if (rafId) {
    cancelAnimationFrame(rafId);
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }
}

function processFrame() {
  if (!scanning) return;

  const src = cv.imread(video);
  const { detections, jack, ranking, usable, reason } = LawnBowlsDetection.detectAndRank(cv, src);
  src.delete();

  drawOverlay(detections, jack, usable ? ranking : []);
  renderRanking(ranking, usable, reason, detections.length);

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

  ranking.forEach((entry, i) => drawFlag(entry.bowl, i + 1));
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
function drawFlag(d, rank) {
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
  overlayCtx.fillStyle = rank === 1 ? '#2e7d32' : '#1565c0';
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
