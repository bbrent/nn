// Runs real photographs of real ends through the real pipeline.
//
// test/fixtures/real holds one folder per end; every image in a folder is the
// same head of bowls seen from a different position, which is what the app
// gets while someone scans. This pushes them through detection, rectification
// and mapping exactly as the live app does, and reports what came out.
//
// Where the simulated harness (scan-harness.js) can check answers against
// ground truth it generated itself, this one mostly cannot — nobody measured
// the green with a tape. What it can do is show how the pipeline behaves on
// real turf, real shadows and a real lens, and there is a lot to learn from
// that without any ground truth at all: whether views of one end tie together
// into a single map, whether the bowl count comes out right, how far the map
// disagrees with itself between views, and whether the uncertainty it reports
// matches that disagreement. An end.json with tape measurements turns those
// consistency checks into accuracy ones (see the README there).
//
//   npm run real-scan            every end
//   npm run real-scan end-03     just that one

const fs = require('fs');
const path = require('path');
const http = require('http');
const { chromium } = require('playwright');
const exif = require('./exif.js');

const ROOT = path.join(__dirname, '..');
const REAL_DIR = path.join(__dirname, 'fixtures', 'real');
const CHROMIUM_PATH = '/opt/pw-browsers/chromium';

// Photos arrive at full camera resolution; the app works from a video frame.
// Processing at the same size keeps the detector facing the same amount of
// detail it would on a phone, so what happens here transfers.
// Overridable so the harness can measure what capture resolution actually
// costs the detector — which is a real decision for the app, not a test knob.
const PROCESS_WIDTH = Number(process.env.SCAN_WIDTH || 1280);
const PROCESS_HEIGHT = Math.round(PROCESS_WIDTH * 9 / 16);

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.json': 'application/json',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
  '.onnx': 'application/octet-stream',
};

function startServer() {
  const server = http.createServer((req, res) => {
    const filePath = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
    if (!filePath.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); return res.end(); }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream' });
      res.end(data);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

function listEnds(filter) {
  if (!fs.existsSync(REAL_DIR)) return [];
  return fs.readdirSync(REAL_DIR)
    .filter(name => fs.statSync(path.join(REAL_DIR, name)).isDirectory())
    .filter(name => !filter || name === filter)
    .map(name => {
      const dir = path.join(REAL_DIR, name);
      const images = fs.readdirSync(dir).filter(f => /\.(jpe?g|png)$/i.test(f)).sort();
      let meta = null;
      const metaPath = path.join(dir, 'end.json');
      if (fs.existsSync(metaPath)) {
        try { meta = JSON.parse(fs.readFileSync(metaPath, 'utf8')); }
        catch (err) { console.log(`  (${name}/end.json is not valid JSON — ignoring it)`); }
      }
      return { name, dir, images, meta };
    })
    .filter(end => end.images.length > 0);
}

// The lens each photo was taken with, from its own EXIF. The live app never
// gets this — a camera stream carries no lens data — so it is here purely to
// check the guess the app is forced to make.
function lensFor(imagePath, meta) {
  const tags = exif.read(imagePath);
  const fromExif = exif.focalLengthPx(tags, PROCESS_WIDTH);
  if (fromExif) {
    return { px: fromExif, source: `EXIF ${tags.focalLength35mm}mm`, device: [tags.make, tags.model].filter(Boolean).join(' ') };
  }
  if (meta && meta.focalLength35mm) {
    return { px: (meta.focalLength35mm / exif.FILM_WIDTH_MM) * PROCESS_WIDTH, source: `end.json ${meta.focalLength35mm}mm`, device: null };
  }
  return { px: null, source: 'unknown — using the app default', device: null };
}

function stats(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  return {
    mean: values.reduce((s, v) => s + v, 0) / values.length,
    median: sorted[Math.floor(sorted.length / 2)],
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
}

async function runEnd(page, end) {
  console.log(`\n${'='.repeat(72)}\n${end.name} — ${end.images.length} view(s)`);
  if (end.meta && end.meta.note) console.log(`  "${end.meta.note}"`);

  const views = end.images.map(file => {
    const lens = lensFor(path.join(end.dir, file), end.meta);
    return {
      name: file,
      url: `/test/fixtures/real/${end.name}/${encodeURIComponent(file)}`,
      focalLengthPx: lens.px,
      lens,
    };
  });

  const device = views.map(v => v.lens.device).find(Boolean);
  if (device) console.log(`  camera: ${device}`);
  const lensNote = views[0].lens;
  const assumed = Math.max(PROCESS_WIDTH, PROCESS_HEIGHT) * 0.70;
  console.log(`  lens: ${lensNote.source}` +
    (lensNote.px ? ` -> ${lensNote.px.toFixed(0)}px (the app would have guessed ${assumed.toFixed(0)}px, ` +
      `${((assumed / lensNote.px - 1) * 100).toFixed(0)}% out)` : ''));

  const result = await page.evaluate(job => window.runRealScan(job), {
    views: views.map(v => ({ name: v.name, url: v.url, focalLengthPx: v.focalLengthPx })),
    width: PROCESS_WIDTH,
    height: PROCESS_HEIGHT,
  });

  console.log('\n  Per view:');
  console.log('    view            found  jack  usable  kept  tilt   residual  merged');
  for (const v of result.views) {
    console.log(
      `    ${v.name.padEnd(14)}  ${String(v.detections).padStart(4)}   ${v.jackFound ? 'y' : 'n'}` +
      `     ${v.classifierUsable ? 'y' : 'n'}    ${String(v.keptAfterOutliers).padStart(3)}` +
      `  ${v.tilt === null ? '  -  ' : String(v.tilt).padStart(5)}` +
      `  ${v.planeResidual === null ? '   -  ' : String(v.planeResidual).padStart(6)}` +
      `   ${v.merged ? (v.relocalised ? 'y (relocalised)' : 'y') : 'NO — ' + v.mergeReason}`
    );
    if (!v.classifierUsable && v.classifierReason) console.log(`        classifier: ${v.classifierReason}`);
    if (!v.rectified && v.rectifyReason) console.log(`        rectify: ${v.rectifyReason}`);
    if (v.jackRejected) console.log(`        the labelled jack didn't sit on the green — discarded`);
  }

  const mergedCount = result.views.filter(v => v.merged).length;
  console.log(`\n  ${mergedCount}/${result.views.length} views merged into one map`);
  console.log(`  ${result.landmarks.length} landmarks, ${result.tracked} tracked, ${result.confirmed.bowls} confirmed`);

  if (end.meta && typeof end.meta.bowls === 'number') {
    const expected = end.meta.bowls;
    const got = result.confirmed.bowls;
    console.log(`  expected ${expected} bowls — ${got === expected ? 'MATCHES' :
      got > expected ? `${got - expected} TOO MANY` : `${expected - got} MISSING`}`);
  } else {
    console.log('  (add "bowls" to end.json to check the count automatically)');
  }

  if (!result.confirmed.usable) {
    console.log(`  not scorable: ${result.confirmed.reason}`);
    return;
  }

  console.log('\n  Map (distance from jack, bowl-diameters):');
  result.confirmed.ranking.forEach((r, i) => {
    console.log(`    #${i + 1}  ${r.dist.toFixed(3).padStart(7)}  +/- ${r.sigma.toFixed(3)}  (${r.observations} look(s))`);
  });

  if (result.confirmed.uncertainPairs.length) {
    console.log('\n  Too close to call:');
    for (const p of result.confirmed.uncertainPairs) {
      console.log(`    #${p.near + 1} and #${p.far + 1}: ${p.gap.toFixed(3)} apart, ` +
        `good only to ${p.noise.toFixed(3)} — would ask for a closer look`);
    }
  } else {
    console.log('\n  Every pair is separable at the reported confidence.');
  }

  // With several views, how far the map disagrees with itself is a real
  // measurement of its own repeatability — no tape measure needed.
  const spreads = result.landmarks.filter(l => l.observations > 1).map(l => l.sigma);
  const summary = stats(spreads);
  if (summary && result.views.length > 1) {
    console.log(`\n  Repeatability across views: sigma ${summary.min.toFixed(3)}-${summary.max.toFixed(3)}` +
      ` (median ${summary.median.toFixed(3)} bowl-diameters, about ${(summary.median * 125).toFixed(0)}mm)`);
  }

  const measured = end.meta && end.meta.measured;
  if (measured && measured.length) {
    console.log('\n  Against the tape:');
    for (const m of measured) {
      if (typeof m.metres === 'number' && m.fromJack === 'nearest' && result.confirmed.ranking.length) {
        const got = result.confirmed.ranking[0].dist * 0.125;
        console.log(`    nearest bowl to jack: measured ${m.metres.toFixed(3)}m, app says ${got.toFixed(3)}m ` +
          `(${((got - m.metres) * 1000).toFixed(0)}mm out)`);
      }
    }
  } else {
    console.log('\n  (no tape measurements in end.json — consistency only, not accuracy)');
  }
}

async function main() {
  const filter = process.argv[2];
  const ends = listEnds(filter);
  if (!ends.length) {
    console.log(filter ? `No end folder called "${filter}" in test/fixtures/real.` :
      'No end folders in test/fixtures/real — see the README there.');
    return;
  }

  const server = await startServer();
  const port = server.address().port;
  const browser = await chromium.launch({ executablePath: CHROMIUM_PATH });

  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(`http://127.0.0.1:${port}/test/real-scan.html`);
    await page.waitForFunction('window.__realReady === true', null, { timeout: 30000 });

    for (const end of ends) {
      await runEnd(page, end);
    }

    if (errors.length) console.log('\nPage errors:\n  ' + errors.slice(0, 5).join('\n  '));

    const singleView = ends.filter(e => e.images.length === 1);
    if (singleView.length) {
      console.log(`\n${'='.repeat(72)}`);
      console.log(`${singleView.length} end(s) have only one view: ${singleView.map(e => e.name).join(', ')}`);
      console.log('A single photo exercises detection and rectification, but nothing that');
      console.log('combines views — no repeatability, no relocalisation, no consensus. Several');
      console.log('shots of the same head from different positions is what tests the rest.');
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
