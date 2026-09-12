// End-to-end scan harness: renders a real 3D green, walks a camera over it,
// and pushes every rendered frame through the actual detection ->
// rectification -> mapping chain, then checks the resulting map against
// ground truth.
//
// This is the one test where nothing is handed to the code. The unit tests
// give ground.js and slam.js perfect detections from a geometric simulator,
// which is the right way to check the maths but says nothing about what
// happens when the detector misses half the bowls, invents one in the turf,
// or nudges a centre a few pixels. Here the input is pixels and the detector
// has to find the bowls itself, exactly as on a phone.
//
// Two things it can establish that nothing else can:
//   - whether the whole chain reproduces distances a score can be trusted to,
//     starting from an image rather than from coordinates
//   - whether the uncertainty the map reports is honest. That number is
//     currently self-reported, and the "too close to call" warning rests
//     entirely on it, so it is checked here against the true error.
//
// Slow (a WebGL render plus a YOLO inference per frame), so it is run on
// demand with `npm run scan` rather than as part of `npm test`.

const fs = require('fs');
const path = require('path');
const http = require('http');
const { chromium } = require('playwright');

const ROOT = path.join(__dirname, '..');
const CHROMIUM_PATH = '/opt/pw-browsers/chromium';

const BOWL_RADIUS_M = 0.058; // regulation, ~116mm across
const JACK_RADIUS_M = 0.0315; // regulation, ~63mm across
const BOWL_COLORS = [0x1b1b1b, 0x2a2a2a, 0x8b1a1a, 0x16367a, 0x3d2b1f, 0x1b1b1b, 0x24486b, 0x7a1f1f];

const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
};

function startServer() {
  const server = http.createServer((req, res) => {
    const filePath = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
    if (!filePath.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); return res.end(); }
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      res.end(data);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

// A head of bowls around the jack, at distinct distances so ordering means
// something, and spread wider than one frame can hold.
function buildHead() {
  const objects = [{ name: 'jack', x: 0, z: 0, radius: JACK_RADIUS_M, color: 0xf2efe4, isJack: true }];
  const spec = [
    [0.19, 0.4], [0.31, 1.9], [0.46, 3.3], [0.63, 5.0],
    [0.82, 2.5], [1.02, 4.3], [1.25, 0.8], [1.5, 2.9],
  ];
  spec.forEach(([radius, angle], i) => {
    objects.push({
      name: 'b' + i,
      x: radius * Math.cos(angle),
      z: radius * Math.sin(angle),
      radius: BOWL_RADIUS_M,
      color: BOWL_COLORS[i % BOWL_COLORS.length],
    });
  });
  return objects;
}

// A scan the way someone would actually do it: sweep across the head from one
// side to the other and back, at a natural holding height and angle, never
// containing the whole head at once.
function buildPath() {
  const poses = [];
  const height = 1.15;
  const back = 1.0;
  for (const t of [-1, -0.65, -0.3, 0, 0.3, 0.65, 1, 0.5, 0, -0.5, -0.9]) {
    poses.push({
      x: t * 0.85,
      y: height,
      z: back + Math.abs(t) * 0.12,
      tx: t * 0.4,
      ty: 0,
      tz: t * 0.15,
    });
  }
  return poses;
}

function summarise(label, values) {
  if (!values.length) return `${label}: none`;
  const sorted = values.slice().sort((a, b) => a - b);
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return `${label}: mean ${mean.toFixed(3)}, median ${sorted[Math.floor(sorted.length / 2)].toFixed(3)}, worst ${sorted[sorted.length - 1].toFixed(3)}`;
}

async function main() {
  const server = await startServer();
  const port = server.address().port;
  const browser = await chromium.launch({
    executablePath: CHROMIUM_PATH,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
  });

  try {
    const page = await browser.newPage();
    const pageErrors = [];
    page.on('pageerror', e => pageErrors.push(e.message));
    page.on('console', m => { if (m.type() === 'error') pageErrors.push('console: ' + m.text()); });

    await page.goto(`http://127.0.0.1:${port}/test/scan-scene.html`);
    await page.waitForFunction('window.__scanReady === true', null, { timeout: 30000 });

    // Vertical field of view. 41 degrees is about 67 across the frame, which
    // is what a phone's main camera does and what ground.js assumes when it
    // has to guess a focal length. Passing something else is how the harness
    // measures what a wrong guess actually costs end to end.
    const fov = Number(process.env.SCAN_FOV || 41.1);
    const objects = buildHead();
    console.log(`Lens: ${fov.toFixed(1)} degrees vertical (assumed focal ${(0.75 * 1280).toFixed(0)}px, ` +
      `true ${((720 / 2) / Math.tan(fov * Math.PI / 360)).toFixed(0)}px)\n`);
    await page.evaluate(cfg => window.buildScene(cfg), {
      width: 1280,
      height: 720,
      fov,
      groundSize: 8,
      seed: 21,
      objects,
    });

    const scanPath = buildPath();
    console.log(`Rendering and processing ${scanPath.length} frames through the real pipeline…\n`);
    const result = await page.evaluate(p => window.runScan(p), scanPath);

    if (pageErrors.length) {
      console.log('Page errors:\n  ' + pageErrors.slice(0, 5).join('\n  ') + '\n');
    }

    // --- what the detector managed, frame by frame ---
    console.log('Per frame (detector on rendered pixels):');
    console.log('  #   visible  found  spurious  jack  usable  merged');
    result.frames.forEach((f, i) => {
      console.log(
        `  ${String(i).padStart(2)}   ${String(f.visible).padStart(5)}   ${String(f.found).padStart(4)}` +
        `   ${String(f.spurious).padStart(6)}   ${f.jackFound ? ' y' : ' n'}    ${f.classifierUsable ? ' y' : ' n'}` +
        `      ${f.merged ? 'y' : 'NO — ' + f.mergeReason}` +
        (f.merged ? `  scale=${f.poseScale} matched=${f.matched} new=${f.newLandmarks}${f.carriedPlane ? ' CARRIED' : ''}` : '')
      );
    });

    // How faithfully the detector reports each bowl, separated into the part
    // that averages out and the part that does not.
    const allRatios = result.frames.flatMap(f => f.radiusRatios || []);
    const allCentres = result.frames.flatMap(f => f.centreErrors || []);
    if (allRatios.length) {
      const meanRatio = allRatios.reduce((s, v) => s + v, 0) / allRatios.length;
      const spread = Math.sqrt(allRatios.reduce((s, v) => s + (v - meanRatio) ** 2, 0) / allRatios.length);
      console.log(`\nDetector fidelity against what was drawn (${allRatios.length} matched detections):`);
      console.log(`  reported radius / true radius: mean ${meanRatio.toFixed(3)} (+/- ${spread.toFixed(3)})`);
      console.log(`    -> a ${((meanRatio - 1) * 100).toFixed(1)}% systematic radius bias becomes a` +
        ` ${((1 / meanRatio - 1) * 100).toFixed(1)}% depth bias, the same in every frame`);
      console.log('  ' + summarise('centre error (bowl radii)', allCentres));
    }

    // Which shape does the detector's radius error take? If its box runs large
    // by a fixed number of pixels, the excess is constant and hurts small far
    // bowls most; if it runs large by a percentage, the ratio is constant and
    // it is a harmless uniform scale. The two call for completely different
    // corrections, so it is worth knowing which.
    const pairs = result.frames.flatMap(f => f.radiusPairs || []);
    if (pairs.length > 4) {
      const buckets = [[0, 8], [8, 14], [14, 22], [22, 100]];
      console.log('\nShape of the detector\'s radius error:');
      console.log('  true radius    n    excess px   excess %');
      for (const [lo, hi] of buckets) {
        const inBucket = pairs.filter(p => p[0] >= lo && p[0] < hi);
        if (!inBucket.length) continue;
        const excessPx = inBucket.reduce((s, p) => s + (p[1] - p[0]), 0) / inBucket.length;
        const excessPct = inBucket.reduce((s, p) => s + (p[1] / p[0] - 1), 0) / inBucket.length * 100;
        console.log('  ' + (lo + '-' + hi + 'px').padEnd(13) + String(inBucket.length).padStart(3) +
          '     ' + excessPx.toFixed(2).padStart(6) + '     ' + excessPct.toFixed(1).padStart(6) + '%');
      }
    }

    const tiltErrors = result.frames.map(f => f.tiltErrorDeg).filter(v => v !== null && isFinite(v));
    if (tiltErrors.length) {
      const sorted = tiltErrors.slice().sort((a, b) => a - b);
      console.log(`\nGreen tilt: fitted from the bowls vs what gravity would report (${tiltErrors.length} frames)`);
      console.log(`  median ${sorted[Math.floor(sorted.length / 2)].toFixed(2)} degrees off, ` +
        `worst ${sorted[sorted.length - 1].toFixed(2)}`);
    }

    const totalVisible = result.frames.reduce((s, f) => s + f.visible, 0);
    const totalFound = result.frames.reduce((s, f) => s + f.found, 0);
    const totalSpurious = result.frames.reduce((s, f) => s + f.spurious, 0);
    const mergedCount = result.frames.filter(f => f.merged).length;
    console.log(`\n  detector recall ${totalFound}/${totalVisible} (${(100 * totalFound / totalVisible).toFixed(0)}%),` +
      ` ${totalSpurious} spurious, ${mergedCount}/${result.frames.length} frames merged`);

    // --- what the map made of it ---
    console.log(`\nMap: ${result.landmarks.length} landmarks, ${result.trackedCount} tracked, ` +
      `${result.confirmed.ranking.length} confirmed bowls (${objects.length - 1} real bowls + jack)`);
    console.log(`  scorable: ${result.confirmed.usable}${result.confirmed.reason ? ' — ' + result.confirmed.reason : ''}`);

    if (!result.confirmed.usable) {
      console.log('\nNo scorable map — cannot check distances.');
      return;
    }

    // Ground truth distances from the jack, in bowl-diameters.
    const diameter = BOWL_RADIUS_M * 2;
    const trueDistances = objects
      .filter(o => !o.isJack)
      .map(o => ({ name: o.name, dist: Math.hypot(o.x, o.z) / diameter }))
      .sort((a, b) => a.dist - b.dist);

    const measured = result.confirmed.ranking.map(r => r.dist).sort((a, b) => a - b);

    // Match each measurement to its nearest unclaimed truth: the map need not
    // have found every bowl, so a positional comparison would misalign.
    const remaining = trueDistances.slice();
    const errors = [];
    const unmatched = [];
    for (const value of measured) {
      if (!remaining.length) {
        // More bowls on the map than exist — whatever these are, they are not
        // real, and there is no truth to compare them against.
        unmatched.push(value);
        continue;
      }
      let bestIndex = 0;
      let bestError = Infinity;
      remaining.forEach((t, i) => {
        const e = Math.abs(t.dist - value);
        if (e < bestError) { bestError = e; bestIndex = i; }
      });
      const matched = remaining.splice(bestIndex, 1)[0];
      errors.push({ name: matched.name, measured: value, truth: matched.dist, error: bestError });
    }
    if (unmatched.length) {
      console.log(`\n  ${unmatched.length} bowl(s) on the map that do not exist, at ` +
        unmatched.map(v => v.toFixed(2)).join(', ') + ' from the jack');
    }

    console.log('\nDistance from jack (bowl-diameters):');
    console.log('  bowl   measured   ±sigma    true     error');
    result.confirmed.ranking
      .slice()
      .sort((a, b) => a.dist - b.dist)
      .forEach((r, i) => {
        const e = errors[i];
        if (!e) return;
        console.log(`  ${e.name.padEnd(5)}  ${e.measured.toFixed(3).padStart(7)}   ${r.sigma.toFixed(3)}` +
          `   ${e.truth.toFixed(3).padStart(6)}  ${e.error.toFixed(3).padStart(7)}`);
      });

    console.log('\n' + summarise('  distance error', errors.map(e => e.error)));

    // --- is the reported uncertainty honest? ---
    // The whole "too close to call" warning rests on sigma meaning something.
    // If the true error routinely exceeds it, the app is confidently wrong; if
    // it is always far inside, the app cries close-call when it needn't.
    const sigmas = result.confirmed.ranking.slice().sort((a, b) => a.dist - b.dist).map(r => r.sigma);
    const withinOne = errors.filter((e, i) => e.error <= sigmas[i]).length;
    const withinTwo = errors.filter((e, i) => e.error <= 2 * sigmas[i]).length;
    console.log(`\nUncertainty calibration (is the reported error bar honest?):`);
    console.log(`  within 1 sigma: ${withinOne}/${errors.length} (expect about 68%)`);
    console.log(`  within 2 sigma: ${withinTwo}/${errors.length} (expect about 95%)`);
    console.log('  ' + summarise('reported sigma', sigmas));

    const ratios = errors.map((e, i) => (sigmas[i] > 0 ? e.error / sigmas[i] : Infinity));
    console.log('  ' + summarise('error / sigma', ratios.filter(r => isFinite(r))));

    // --- would the score come out right? ---
    const measuredOrder = errors.map(e => e.name);
    const truthOrder = trueDistances
      .filter(t => measuredOrder.includes(t.name))
      .sort((a, b) => a.dist - b.dist)
      .map(t => t.name);
    const orderOk = measuredOrder.every((n, i) => n === truthOrder[i]);
    console.log(`\nRanking: ${orderOk ? 'CORRECT' : 'WRONG'}`);
    console.log(`  measured: ${measuredOrder.join(' ')}`);
    console.log(`  truth:    ${truthOrder.join(' ')}`);
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
