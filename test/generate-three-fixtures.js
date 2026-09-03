// Renders synthetic "video frame" fixtures with a real WebGL 3D scene
// (three.js, driven headlessly via Playwright/Chromium) instead of flat 2D
// circles. This gives perspective foreshortening, shadows, and specular
// highlights as camera angle steepens — things a 2D canvas drawing can't
// exercise — plus post-render sensor noise. Ground truth (screen-space
// position/radius per sphere) comes from three.js's own camera projection,
// computed inside the page after rendering.

const fs = require('fs');
const path = require('path');
const http = require('http');
const { chromium } = require('playwright');
const { createCanvas, loadImage } = require('canvas');

const ROOT = path.join(__dirname, '..');
const OUT_DIR = path.join(__dirname, 'fixtures', 'three-js');
const CHROMIUM_PATH = '/opt/pw-browsers/chromium';

const BOWL_COLORS = [0x1b1b1b, 0xc62828, 0x1565c0, 0xf9a825, 0x4a148c, 0x00695c];
const JACK_COLOR = 0xfdf6e3;
const BOWL_RADIUS_M = 0.058; // regulation ~116mm diameter
const JACK_RADIUS_M = 0.0315; // regulation ~63mm diameter

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function placeCircles(rng, count, halfExtent) {
  const circles = [];
  const jackIndex = Math.floor(rng() * count);
  for (let i = 0; i < count; i++) {
    const isJack = i === jackIndex;
    const radius = isJack ? JACK_RADIUS_M : BOWL_RADIUS_M * (0.9 + rng() * 0.2);
    for (let attempt = 0; attempt < 300; attempt++) {
      const x = (rng() * 2 - 1) * (halfExtent - radius);
      const z = (rng() * 2 - 1) * (halfExtent - radius);
      const clash = circles.some(c => Math.hypot(c.x - x, c.z - z) < c.radius + radius + 0.015);
      if (!clash) {
        circles.push({ x, z, radius, isJack });
        break;
      }
    }
  }
  return circles;
}

// Uniform per-pixel sensor-style noise, applied after the WebGL render.
function addNoise(imageData, amount, rng) {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (rng() - 0.5) * amount;
    d[i] = clamp(d[i] + n);
    d[i + 1] = clamp(d[i + 1] + n);
    d[i + 2] = clamp(d[i + 2] + n);
  }
  return imageData;
}

function clamp(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function startServer() {
  const server = http.createServer((req, res) => {
    const filePath = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
    if (!filePath.startsWith(ROOT)) {
      res.writeHead(403);
      res.end();
      return;
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404);
        res.end();
        return;
      }
      const ext = path.extname(filePath);
      const type = { '.html': 'text/html', '.js': 'text/javascript' }[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': type });
      res.end(data);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

const scenes = [
  {
    name: 'three-top-down',
    seed: 11,
    width: 960,
    height: 540,
    groundSize: 3,
    bowlCount: 6,
    halfExtent: 0.9,
    noise: 6,
    camera: { fov: 60, position: [0, 1.5, 0.2], lookAt: [0, 0, 0] },
  },
  {
    name: 'three-angled',
    seed: 12,
    width: 960,
    height: 540,
    groundSize: 4,
    bowlCount: 7,
    halfExtent: 0.9,
    noise: 8,
    camera: { fov: 65, position: [0, 1.3, 1.1], lookAt: [0, 0, -0.1] },
  },
  {
    name: 'three-low-angle',
    seed: 13,
    width: 960,
    height: 540,
    groundSize: 5,
    bowlCount: 6,
    halfExtent: 0.85,
    noise: 10,
    camera: { fov: 70, position: [0, 0.9, 1.8], lookAt: [0, 0, -0.2] },
  },
];

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const server = await startServer();
  const port = server.address().port;

  const browser = await chromium.launch({
    executablePath: CHROMIUM_PATH,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
  });

  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}/test/three-scene.html`);
    await page.waitForFunction('window.__sceneReady === true');

    for (const scene of scenes) {
      const rng = mulberry32(scene.seed);
      const placed = placeCircles(rng, scene.bowlCount, scene.halfExtent);
      const circles = placed.map((c, i) => ({
        ...c,
        color: c.isJack ? JACK_COLOR : BOWL_COLORS[i % BOWL_COLORS.length],
      }));

      const config = {
        width: scene.width,
        height: scene.height,
        groundSize: scene.groundSize,
        seed: scene.seed,
        camera: scene.camera,
        circles,
      };

      const groundTruthPx = await page.evaluate(cfg => window.renderScene(cfg), config);
      // Let the compositor pick up the WebGL draw before screenshotting the canvas.
      await page.waitForTimeout(100);

      const canvasHandle = await page.$('#c');
      const pngBuffer = await canvasHandle.screenshot();

      const image = await loadImage(pngBuffer);
      const canvas = createCanvas(scene.width, scene.height);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      const imageData = ctx.getImageData(0, 0, scene.width, scene.height);
      addNoise(imageData, scene.noise, mulberry32(scene.seed + 1));
      ctx.putImageData(imageData, 0, 0);

      const pngPath = path.join(OUT_DIR, `${scene.name}.png`);
      fs.writeFileSync(pngPath, canvas.toBuffer('image/png'));
      fs.writeFileSync(
        path.join(OUT_DIR, `${scene.name}.json`),
        JSON.stringify({ width: scene.width, height: scene.height, circles: groundTruthPx }, null, 2)
      );
      console.log(`wrote ${pngPath} (${groundTruthPx.length} circles)`);
    }
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
