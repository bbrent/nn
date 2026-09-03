// Generates synthetic "video frame" fixtures: a turf-green background with
// non-overlapping bowl + jack circles at known positions, plus a JSON
// sidecar with ground truth. These validate the detection/classification/
// ranking *logic*; they are not a substitute for real phone photos (see
// test/fixtures/real/).

const fs = require('fs');
const path = require('path');
const { createCanvas } = require('canvas');

const OUT_DIR = path.join(__dirname, 'fixtures', 'synthetic');

// Deterministic PRNG so fixtures are reproducible across runs/machines.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const BOWL_COLORS = ['#1b1b1b', '#c62828', '#1565c0', '#f9a825', '#4a148c', '#00695c'];
const JACK_COLOR = '#fdf6e3';

function drawTurf(ctx, w, h, rng) {
  ctx.fillStyle = '#2e7d32';
  ctx.fillRect(0, 0, w, h);
  // Sparse speckle noise so edges aren't detected against a perfectly flat field.
  for (let i = 0; i < (w * h) / 40; i++) {
    const x = rng() * w;
    const y = rng() * h;
    const shade = rng() > 0.5 ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.05)';
    ctx.fillStyle = shade;
    ctx.fillRect(x, y, 2, 2);
  }
}

function drawCircle(ctx, x, y, r, color) {
  const gradient = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r);
  gradient.addColorStop(0, lighten(color, 0.35));
  gradient.addColorStop(1, color);
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.lineWidth = Math.max(1, r * 0.06);
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.stroke();
}

function lighten(hex, amount) {
  const n = parseInt(hex.slice(1), 16);
  const r = Math.min(255, ((n >> 16) & 255) + 255 * amount);
  const g = Math.min(255, ((n >> 8) & 255) + 255 * amount);
  const b = Math.min(255, (n & 255) + 255 * amount);
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

function placeCircles(rng, w, h, count, bowlRadius, jackRadius, margin) {
  const circles = [];
  const jackIndex = Math.floor(rng() * count);

  for (let i = 0; i < count; i++) {
    const r = i === jackIndex ? jackRadius : bowlRadius * (0.9 + rng() * 0.2);
    let placed = false;
    for (let attempt = 0; attempt < 300 && !placed; attempt++) {
      const x = margin + r + rng() * (w - 2 * (margin + r));
      const y = margin + r + rng() * (h - 2 * (margin + r));
      const clash = circles.some(c => Math.hypot(c.x - x, c.y - y) < c.r + r + 14);
      if (!clash) {
        circles.push({ x, y, r, isJack: i === jackIndex });
        placed = true;
      }
    }
    if (!placed) break; // ran out of room; scene will just have fewer circles
  }
  return circles;
}

function generateScene(name, { seed, width, height, bowlCount, bowlRadius, jackRadius }) {
  const rng = mulberry32(seed);
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  drawTurf(ctx, width, height, rng);
  const margin = bowlRadius + 20;
  const circles = placeCircles(rng, width, height, bowlCount, bowlRadius, jackRadius, margin);

  // Draw bowls first, jack last so it renders on top if anything is close.
  const bowls = circles.filter(c => !c.isJack);
  const jack = circles.find(c => c.isJack);
  bowls.forEach((c, i) => drawCircle(ctx, c.x, c.y, c.r, BOWL_COLORS[i % BOWL_COLORS.length]));
  if (jack) drawCircle(ctx, jack.x, jack.y, jack.r, JACK_COLOR);

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const pngPath = path.join(OUT_DIR, `${name}.png`);
  fs.writeFileSync(pngPath, canvas.toBuffer('image/png'));

  const truth = { width, height, circles };
  fs.writeFileSync(path.join(OUT_DIR, `${name}.json`), JSON.stringify(truth, null, 2));
  console.log(`wrote ${pngPath} (${circles.length} circles, jack=${jack ? 'yes' : 'no'})`);
}

const scenes = [
  { name: 'scene-3bowls', seed: 1, width: 960, height: 540, bowlCount: 4, bowlRadius: 32, jackRadius: 17 },
  { name: 'scene-6bowls', seed: 2, width: 960, height: 540, bowlCount: 7, bowlRadius: 30, jackRadius: 16 },
  { name: 'scene-tight-cluster', seed: 3, width: 960, height: 540, bowlCount: 9, bowlRadius: 26, jackRadius: 14 },
  { name: 'scene-wide-shot', seed: 4, width: 1280, height: 720, bowlCount: 6, bowlRadius: 24, jackRadius: 13 },
];

for (const scene of scenes) {
  const { name, ...opts } = scene;
  generateScene(name, opts);
}
