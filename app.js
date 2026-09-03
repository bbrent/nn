// Milestone 1: single-frame circle detection + relative distance to jack.
// Cross-frame fusion (panning the phone over the whole rink) comes next.

const WORK_WIDTH = 480; // downscaled width used for cv processing

let cvReady = false;
let domReady = false;
let scanning = false;
let rafId = null;

let video, overlay, overlayCtx, workCanvas, workCtx, statusEl, rankingEl, scanBtn;

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
  workCanvas = document.createElement('canvas');
  workCtx = workCanvas.getContext('2d', { willReadFrequently: true });
  statusEl = document.getElementById('status');
  rankingEl = document.getElementById('ranking');
  scanBtn = document.getElementById('scanBtn');

  scanBtn.addEventListener('click', toggleScan);
  window.addEventListener('resize', sizeCanvases);

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
    sizeCanvases();
    setStatus('Hold the phone over the rink and tap Start Scan.');
    scanBtn.disabled = false;
  } catch (err) {
    setStatus('Camera error: ' + err.message);
  }
}

function sizeCanvases() {
  if (!video.videoWidth) return;
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  const scale = WORK_WIDTH / video.videoWidth;
  workCanvas.width = WORK_WIDTH;
  workCanvas.height = Math.round(video.videoHeight * scale);
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

  workCtx.drawImage(video, 0, 0, workCanvas.width, workCanvas.height);
  const src = cv.imread(workCanvas);
  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.medianBlur(gray, gray, 5);

  const circles = new cv.Mat();
  cv.HoughCircles(
    gray,
    circles,
    cv.HOUGH_GRADIENT,
    1,
    gray.rows / 12, // min distance between circle centers
    100,            // Canny high threshold
    30,             // accumulator threshold (lower = more false positives)
    8,              // min radius, px at work resolution
    60              // max radius, px at work resolution
  );

  const detections = [];
  for (let i = 0; i < circles.cols; i++) {
    detections.push({
      x: circles.data32F[i * 3],
      y: circles.data32F[i * 3 + 1],
      r: circles.data32F[i * 3 + 2],
    });
  }

  drawOverlay(detections);

  src.delete();
  gray.delete();
  circles.delete();

  rafId = requestAnimationFrame(processFrame);
}

function drawOverlay(detections) {
  const scaleX = overlay.width / workCanvas.width;
  const scaleY = overlay.height / workCanvas.height;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  if (detections.length === 0) {
    renderRanking([], false);
    return;
  }

  // Jack is the ~2.5"-vs-4.6" outlier: markedly smaller than the median circle here.
  const sortedR = [...detections].map(d => d.r).sort((a, b) => a - b);
  const medianR = sortedR[Math.floor(sortedR.length / 2)];
  const jackThreshold = medianR * 0.65;

  let jack = null;
  const bowls = [];
  for (const d of detections) {
    if (d.r < jackThreshold) {
      if (!jack || d.r < jack.r) {
        if (jack) bowls.push(jack);
        jack = d;
      } else {
        bowls.push(d);
      }
    } else {
      bowls.push(d);
    }
  }

  const avgBowlDiameter = bowls.length
    ? bowls.reduce((sum, b) => sum + b.r * 2, 0) / bowls.length
    : medianR * 2;

  for (const d of detections) {
    const isJack = d === jack;
    const cx = d.x * scaleX;
    const cy = d.y * scaleY;
    const r = d.r * scaleX;
    overlayCtx.beginPath();
    overlayCtx.arc(cx, cy, r, 0, 2 * Math.PI);
    overlayCtx.strokeStyle = isJack ? '#ffd54f' : '#42a5f5';
    overlayCtx.lineWidth = 3;
    overlayCtx.stroke();
  }

  const ranking = bowls.map(b => ({
    bowl: b,
    dist: jack ? Math.hypot(b.x - jack.x, b.y - jack.y) / avgBowlDiameter : null,
  }));

  if (jack) ranking.sort((a, b) => a.dist - b.dist);

  renderRanking(ranking, jack !== null);
}

function renderRanking(ranking, haveJack) {
  rankingEl.innerHTML = '';

  if (!haveJack) {
    const li = document.createElement('li');
    li.textContent = ranking.length
      ? `${ranking.length} circle(s) detected, no jack identified yet`
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
