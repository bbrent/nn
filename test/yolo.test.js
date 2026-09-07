// Tests for yolo-detector.js's pure pre/post-processing. No ONNX runtime
// involved — decodeOutput is validated against a real raw model output
// tensor (test/fixtures/yolo/gameplay-2-raw-output.bin), captured by running
// the actual yolov8n.onnx model (via Python onnxruntime) against a real
// gameplay photo, with the expected boxes cross-checked against
// ultralytics' own reference NMS on that exact tensor. This is the part
// most likely to have a subtle bug (tensor layout, letterbox math, NMS) and
// least amenable to reasoning by eye, so it's tested against ground truth
// rather than hand-derived expectations.

const fs = require('fs');
const path = require('path');
const LawnBowlsYolo = require('../yolo-detector.js');

function approxEqual(a, b, tol) {
  return Math.abs(a - b) < tol;
}

function run() {
  const failures = [];

  // Case 1: letterbox math — a wide image fit into a square target should
  // scale to fill width, center-pad top/bottom.
  {
    const lb = LawnBowlsYolo.computeLetterbox(4032, 2268, 640);
    if (!approxEqual(lb.scale, 640 / 4032, 1e-6)) failures.push(`case1: expected scale ${640 / 4032}, got ${lb.scale}`);
    if (lb.newWidth !== 640) failures.push(`case1: expected newWidth 640, got ${lb.newWidth}`);
    if (!approxEqual(lb.padX, 0, 1e-6)) failures.push(`case1: expected padX 0, got ${lb.padX}`);
    if (lb.padY <= 0) failures.push(`case1: expected positive padY for a wide image, got ${lb.padY}`);
  }

  // Case 2: imageDataToTensor — NCHW layout and /255 normalization, checked
  // on a tiny 2x1 synthetic image so the expected values can be hand-verified.
  {
    const imageData = { width: 2, height: 1, data: new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255]) };
    const tensor = LawnBowlsYolo.imageDataToTensor(imageData);
    const expected = [1, 0, 0, 1, 0, 0]; // R plane: [1,0], G plane: [0,1], B plane: [0,0]
    const matches = expected.every((v, i) => approxEqual(tensor.data[i], v, 1e-6));
    if (!matches) failures.push(`case2: expected ${expected}, got ${Array.from(tensor.data)}`);
    if (tensor.dims.join(',') !== '1,3,1,2') failures.push(`case2: expected dims [1,3,1,2], got ${tensor.dims}`);
  }

  // Case 3: decodeOutput against real captured model output, cross-validated
  // against ultralytics' own NMS on the identical raw tensor (see comment
  // above). This is the real integration proof — same tensor layout
  // indexing, same letterbox un-warp, same NMS behavior as the actual model.
  {
    const fixtureDir = path.join(__dirname, 'fixtures', 'yolo');
    const raw = fs.readFileSync(path.join(fixtureDir, 'gameplay-2-raw-output.bin'));
    const outputData = new Float32Array(raw.buffer, raw.byteOffset, raw.length / 4);
    const letterbox = JSON.parse(fs.readFileSync(path.join(fixtureDir, 'gameplay-2-letterbox.json'), 'utf8'));

    const detections = LawnBowlsYolo.decodeOutput(outputData, letterbox, { confThreshold: 0.1 });

    // Expected: Python/ultralytics NMS on this exact tensor found 3 raw boxes
    // (2 sports-ball, 1 scissors); the scissors one must be dropped by the
    // class allowlist, leaving exactly these 2 real detections.
    const expected = [
      { x: 1913.9, y: 613.9, r: 153.7, score: 0.694 },
      { x: 2072.3, y: 1173.0, r: 86.9, score: 0.130 },
    ];

    if (detections.length !== expected.length) {
      failures.push(`case3: expected ${expected.length} detections, got ${detections.length}`);
    } else {
      detections.forEach((d, i) => {
        const e = expected[i];
        if (!approxEqual(d.x, e.x, 0.5) || !approxEqual(d.y, e.y, 0.5) || !approxEqual(d.r, e.r, 0.5) || !approxEqual(d.score, e.score, 0.01)) {
          failures.push(`case3: detection ${i} expected ${JSON.stringify(e)}, got ${JSON.stringify({ x: d.x, y: d.y, r: d.r, score: d.score })}`);
        }
      });
    }
  }

  // Case 4: classifyAndRank — same jack/bowl split logic as detection.js,
  // but gated on score instead of an absolute pixel-size floor.
  {
    const detections = [
      { x: 100, y: 100, r: 40, score: 0.6 }, // bowl
      { x: 200, y: 100, r: 38, score: 0.5 }, // bowl
      { x: 150, y: 150, r: 20, score: 0.3 }, // jack (clearly smaller)
    ];
    const result = LawnBowlsYolo.classifyAndRank(detections);
    if (!result.jack || result.jack.r !== 20) failures.push(`case4: expected jack r=20, got ${JSON.stringify(result.jack)}`);
    if (!result.usable) failures.push(`case4: expected usable, got not usable (${result.reason})`);
    if (result.bowls.length !== 2) failures.push(`case4: expected 2 bowls, got ${result.bowls.length}`);
  }

  // Case 5: classifyAndRank — a low-confidence jack candidate should be
  // flagged unusable rather than trusted.
  {
    const detections = [
      { x: 100, y: 100, r: 40, score: 0.6 },
      { x: 200, y: 100, r: 38, score: 0.5 },
      { x: 150, y: 150, r: 20, score: 0.02 }, // jack-sized but too low confidence
    ];
    const result = LawnBowlsYolo.classifyAndRank(detections);
    if (result.usable) failures.push('case5: expected unusable due to low jack confidence, got usable');
  }

  return { name: 'yolo-detector', total: 5, failures };
}

module.exports = { run };

if (require.main === module) {
  const result = run();
  if (result.failures.length === 0) {
    console.log(`PASS  yolo-detector (${result.total} cases)`);
    process.exit(0);
  } else {
    console.log(`FAIL  yolo-detector`);
    result.failures.forEach(f => console.log(`        - ${f}`));
    process.exit(1);
  }
}
