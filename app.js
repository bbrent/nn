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
  const { detections, jack, ranking } = LawnBowlsDetection.detectAndRank(cv, src);
  src.delete();

  drawOverlay(detections, jack);
  renderRanking(ranking, jack !== null, detections.length);

  rafId = requestAnimationFrame(processFrame);
}

function drawOverlay(detections, jack) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  for (const d of detections) {
    overlayCtx.beginPath();
    overlayCtx.arc(d.x, d.y, d.r, 0, 2 * Math.PI);
    overlayCtx.strokeStyle = d === jack ? '#ffd54f' : '#42a5f5';
    overlayCtx.lineWidth = 3;
    overlayCtx.stroke();
  }
}

function renderRanking(ranking, haveJack, detectionCount) {
  rankingEl.innerHTML = '';

  if (!haveJack) {
    const li = document.createElement('li');
    li.textContent = detectionCount
      ? `${detectionCount} circle(s) detected, no jack identified yet`
      : 'No circles detected';
    rankingEl.appendChild(li);
    return;
  }

  ranking.forEach((entry, i) => {
    const li = document.createElement('li');
    li.textContent = `#${i + 1} bowl — ${entry.dist.toFixed(2)} bowl-diameters from jack`;
    rankingEl.appendChild(li);
  });
}
