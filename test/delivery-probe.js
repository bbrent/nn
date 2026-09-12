// Is tracking a delivery feasible? A measurement, not an opinion.
//
// Renders a bowl rolling a full length of green, from where somebody would
// actually stand to watch it, and puts every frame through two ways of finding
// it: the detector the app already uses, and plain frame differencing.
//
// The distinction matters because the two fail for opposite reasons. The
// detector needs something that looks like an object, and a bowl at the far
// end of a rink is about two pixels across. Frame differencing needs only
// something that moved, and does not care how small it is — but it needs the
// camera held still and nothing else in shot moving.
//
// Run with `npm run delivery-probe`. Nothing in the app depends on this; it
// exists to answer whether a trajectory feature is worth attempting.

const fs = require('fs');
const path = require('path');
const http = require('http');
const { chromium } = require('playwright');

const ROOT = path.join(__dirname, '..');
const CHROMIUM_PATH = '/opt/pw-browsers/chromium';

const WIDTH = 1280;
const HEIGHT = 720;
const FOV = 41.1; // vertical; about 67 across, a phone main camera
const CAMERA_HEIGHT = 1.4; // chest height, standing on the mat

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.json': 'application/json',
  '.onnx': 'application/octet-stream',
};

function startServer() {
  const server = http.createServer((req, res) => {
    const filePath = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
    if (!filePath.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
    fs.readFile(filePath, (err, data) => {
      if (err) { res.writeHead(404); return res.end(); }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' });
      res.end(data);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

// A delivery: away down the green, slowing as it goes, and drawing across as
// the bias takes hold — most of the curve happening in the last third, which
// is what makes the shot interesting and what a trajectory would be for.
function buildDelivery(frameCount, endDistance, drawMetres) {
  const steps = [];
  for (let i = 0; i < frameCount; i++) {
    const t = i / (frameCount - 1);
    // Distance eases out, as a bowl losing speed does.
    const along = endDistance * (1 - Math.pow(1 - t, 2));
    // The draw builds with the cube of progress: barely anything early, most
    // of it at the end.
    const across = drawMetres * Math.pow(t, 3);
    steps.push({ x: across, z: -along });
  }
  return steps;
}

function summarise(values) {
  const clean = values.filter(v => v !== null && isFinite(v));
  if (!clean.length) return null;
  const sorted = clean.slice().sort((a, b) => a - b);
  return {
    n: clean.length,
    mean: clean.reduce((s, v) => s + v, 0) / clean.length,
    median: sorted[Math.floor(sorted.length / 2)],
    worst: sorted[sorted.length - 1],
  };
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
    page.on('pageerror', e => console.log('PAGEERROR', e.message));
    await page.goto(`http://127.0.0.1:${port}/test/delivery-scene.html`);
    await page.waitForFunction('window.__deliveryReady === true', null, { timeout: 30000 });

    const endDistance = 25;
    const draw = 1.8;

    await page.evaluate(cfg => window.buildGreen(cfg), {
      width: WIDTH, height: HEIGHT, fov: FOV,
      // Standing on the mat, looking up the green and slightly down.
      camera: { x: 0, y: CAMERA_HEIGHT, z: 1.2, tx: 0, ty: 0, tz: -12 },
      jack: { x: draw, z: -endDistance },
    });

    // 40 steps over the roll — about what 3 seconds at 13fps would give, or a
    // sampled-down version of a 30fps capture.
    const delivery = buildDelivery(40, endDistance, draw);
    console.log(`A ${endDistance}m delivery drawing ${draw}m, seen from the mat at ${CAMERA_HEIGHT}m.`);
    console.log(`${delivery.length} frames through the detector and through frame differencing.\n`);

    const { frames } = await page.evaluate(
      args => window.runDelivery(args[0], args[1]),
      [delivery, { diffThreshold: 18 }]
    );

    console.log('  dist   bowl    detector            frame differencing');
    console.log('          size   found  off    found  off     changed px   ground error');
    for (const f of frames) {
      if (frames.indexOf(f) % 2 !== 0) continue; // every other row, to keep it readable
      console.log(
        `  ${f.distance.toFixed(1).padStart(4)}m  ${f.trueRadiusPx.toFixed(1).padStart(4)}px` +
        `   ${f.detectorFound ? ' y ' : ' . '}  ${f.detectorOffPx === null ? '  -  ' : f.detectorOffPx.toFixed(0).padStart(4) + 'px'}` +
        `    ${f.changeFound ? ' y ' : ' . '}  ${f.changeOffPx === null ? '  -  ' : f.changeOffPx.toFixed(1).padStart(4) + 'px'}` +
        `    ${String(f.changePixels).padStart(6)}` +
        `     ${f.groundErrorChange === null ? '   -' : (f.groundErrorChange * 100).toFixed(0).padStart(4) + 'cm'}`
      );
    }

    const detectorHits = frames.filter(f => f.detectorFound);
    const changeHits = frames.filter(f => f.changeFound);
    console.log(`\n  Detector found the bowl in ${detectorHits.length}/${frames.length} frames` +
      (detectorHits.length ? `, out to ${Math.max(...detectorHits.map(f => f.distance)).toFixed(1)}m` : ''));
    console.log(`  Frame differencing found it in ${changeHits.length}/${frames.length} frames` +
      (changeHits.length ? `, out to ${Math.max(...changeHits.map(f => f.distance)).toFixed(1)}m` : ''));

    const ground = summarise(frames.map(f => f.groundErrorChange));
    const lateral = summarise(frames.map(f => f.lateralErrorChange));
    const along = summarise(frames.map(f => f.alongErrorChange));
    if (ground) {
      console.log(`\n  Position on the green, from frame differencing and a known ground plane:`);
      console.log(`    overall   median ${(ground.median * 100).toFixed(0)}cm, worst ${(ground.worst * 100).toFixed(0)}cm`);
      console.log(`    across    median ${(lateral.median * 100).toFixed(1)}cm, worst ${(lateral.worst * 100).toFixed(0)}cm   <- the draw`);
      console.log(`    up-green  median ${(along.median * 100).toFixed(0)}cm, worst ${(along.worst * 100).toFixed(0)}cm   <- the length`);
    }

    // The far third is where a trajectory would earn its keep, and where both
    // methods are most stretched.
    const far = frames.filter(f => f.distance > endDistance * 0.66);
    const farChange = summarise(far.map(f => f.lateralErrorChange));
    if (farChange) {
      console.log(`\n  Over the last third (${(endDistance * 0.66).toFixed(0)}m+), where the bowl draws:`);
      console.log(`    found in ${far.filter(f => f.changeFound).length}/${far.length} frames,` +
        ` across-green error median ${(farChange.median * 100).toFixed(1)}cm`);
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
