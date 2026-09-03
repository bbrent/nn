// Runs the shared detection pipeline (detection.js) against static fixture
// images and checks results against ground truth. Run: npm test
//
// Three kinds of fixtures:
//   fixtures/synthetic/*.png + .json  — flat 2D canvas circles, ground truth, asserted
//   fixtures/three-js/*.png + .json   — real WebGL 3D render (perspective, shadows,
//                                        noise), ground truth, asserted
//   fixtures/real/*.jpg               — real phone photos, no ground truth;
//                                        printed for manual eyeballing only

const fs = require('fs');
const path = require('path');
const { loadImage, createCanvas } = require('canvas');
const LawnBowlsDetection = require('../detection.js');

const GROUND_TRUTH_DIRS = [
  { label: '2D synthetic', dir: path.join(__dirname, 'fixtures', 'synthetic') },
  { label: '3D (three.js)', dir: path.join(__dirname, 'fixtures', 'three-js') },
];
const REAL_DIR = path.join(__dirname, 'fixtures', 'real');

// Ground-truth circles use image-pixel radius directly; detector position
// tolerance is generous since Hough centers can drift a few px from the
// true center, especially for the smaller jack circle.
const POSITION_TOLERANCE_PX = 12;

async function matFromImage(cv, imagePath) {
  const image = await loadImage(imagePath);
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  const imageData = ctx.getImageData(0, 0, image.width, image.height);
  return cv.matFromImageData(imageData);
}

function nearest(point, candidates) {
  let best = null;
  let bestDist = Infinity;
  for (const c of candidates) {
    const d = Math.hypot(c.x - point.x, c.y - point.y);
    if (d < bestDist) {
      bestDist = d;
      best = c;
    }
  }
  return { match: best, dist: bestDist };
}

async function runSynthetic(cv, dir, name) {
  const pngPath = path.join(dir, `${name}.png`);
  const truth = JSON.parse(fs.readFileSync(path.join(dir, `${name}.json`), 'utf8'));

  const src = await matFromImage(cv, pngPath);
  const { detections, jack, ranking, usable, reason } = LawnBowlsDetection.detectAndRank(cv, src);
  src.delete();

  const failures = [];
  const truthBowls = truth.circles.filter(c => !c.isJack);
  const truthJack = truth.circles.find(c => c.isJack);

  // A frame is only expected to be fully solvable if the true jack clears the
  // same confidence floor the pipeline itself uses. Below that, the correct
  // behavior is to say so (usable:false) and get skipped by the caller — not
  // to force a full detection, which is what real footage will actually do
  // too (bad frames get discarded once multi-frame fusion lands).
  const truthJackWorkR = truthJack ? (truthJack.r * LawnBowlsDetection.WORK_WIDTH) / truth.width : 0;
  const expectSolvable = truthJack && truthJackWorkR >= LawnBowlsDetection.CONFIDENT_JACK_WORK_RADIUS_PX;

  if (!expectSolvable) {
    if (usable) {
      failures.push(`jack is only ${truthJackWorkR.toFixed(1)}px in the work image (too small to trust) but pipeline reported usable`);
    }
    return { name, failures, skipped: !failures.length };
  }

  if (!usable) {
    failures.push(`expected a usable frame but got usable:false (${reason})`);
    return { name, failures };
  }

  if (detections.length !== truth.circles.length) {
    failures.push(`expected ${truth.circles.length} circles, detected ${detections.length}`);
  }

  {
    const { dist } = nearest(jack, [truthJack]);
    if (dist > POSITION_TOLERANCE_PX) {
      failures.push(`jack detected ${dist.toFixed(1)}px from ground truth (tolerance ${POSITION_TOLERANCE_PX}px)`);
    }
  }

  // Expected ranking order: sort ground-truth bowls by true distance to the true jack.
  const expectedOrder = truthBowls
    .map((b, i) => ({ i, dist: Math.hypot(b.x - truthJack.x, b.y - truthJack.y) }))
    .sort((a, b) => a.dist - b.dist)
    .map(e => e.i);

  // Map each ranked detection back to its nearest ground-truth bowl, in ranked order.
  const detectedOrder = ranking.map(entry => {
    const { match } = nearest(entry.bowl, truthBowls);
    return truthBowls.indexOf(match);
  });

  if (detectedOrder.length === expectedOrder.length) {
    const orderMatches = detectedOrder.every((idx, i) => idx === expectedOrder[i]);
    if (!orderMatches) {
      failures.push(`ranking order mismatch: expected [${expectedOrder}], got [${detectedOrder}]`);
    }
  } else {
    failures.push(`expected ${expectedOrder.length} ranked bowls, got ${detectedOrder.length}`);
  }

  return { name, failures };
}

async function runReal(cv, filename) {
  const src = await matFromImage(cv, path.join(REAL_DIR, filename));
  const { detections, jack, ranking } = LawnBowlsDetection.detectAndRank(cv, src);
  src.delete();
  console.log(
    `  ${filename}: ${detections.length} circle(s), jack=${jack ? 'found' : 'NOT FOUND'}, ` +
    `ranked bowls=${ranking.length}`
  );
}

async function main() {
  const cv = await require('@techstark/opencv-js');

  let totalCount = 0;
  let failCount = 0;

  for (const mod of ['./score.test.js', './fusion.test.js']) {
    const result = require(mod).run();
    totalCount += result.total;
    if (result.failures.length === 0) {
      console.log(`PASS  ${result.name} (${result.total} cases)\n`);
    } else {
      failCount += result.failures.length;
      console.log(`FAIL  ${result.name}`);
      result.failures.forEach(f => console.log(`        - ${f}`));
      console.log('');
    }
  }

  for (const { label, dir } of GROUND_TRUTH_DIRS) {
    if (!fs.existsSync(dir)) continue;
    const sceneNames = fs
      .readdirSync(dir)
      .filter(f => f.endsWith('.png'))
      .map(f => f.replace(/\.png$/, ''));

    console.log(`${label} (${sceneNames.length} fixture(s)):`);
    for (const name of sceneNames) {
      totalCount++;
      const result = await runSynthetic(cv, dir, name);
      if (result.failures.length === 0 && result.skipped) {
        console.log(`  SKIP  ${name} (jack too small to trust — correctly flagged unusable)`);
      } else if (result.failures.length === 0) {
        console.log(`  PASS  ${name}`);
      } else {
        failCount++;
        console.log(`  FAIL  ${name}`);
        result.failures.forEach(f => console.log(`          - ${f}`));
      }
    }
    console.log('');
  }

  const realFiles = fs.existsSync(REAL_DIR)
    ? fs.readdirSync(REAL_DIR).filter(f => /\.(jpe?g|png)$/i.test(f))
    : [];

  if (realFiles.length > 0) {
    console.log(`Real photo fixtures (no ground truth, informational only):`);
    for (const f of realFiles) {
      await runReal(cv, f);
    }
  } else {
    console.log(`No real-photo fixtures in test/fixtures/real/ yet — drop phone photos there to sanity-check detection on actual bowls.`);
  }

  console.log(`\n${totalCount - failCount}/${totalCount} fixtures passed.`);
  process.exit(failCount > 0 ? 1 : 0);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
