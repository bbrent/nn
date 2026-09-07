// Pure-math tests for embedding.js. No ONNX runtime involved — embedCrop's
// actual inference needs a real session (browser-only, onnxruntime-web); what
// gets tested here is the preprocessing math (ImageNet normalization, NCHW
// layout) and the similarity functions, on small hand-computed cases.

const LawnBowlsEmbedding = require('../embedding.js');

function approxEqual(a, b, tol) {
  return Math.abs(a - b) < tol;
}

function run() {
  const failures = [];

  // Case 1: imageDataToTensor — NCHW layout + ImageNet mean/std normalization,
  // hand-verified on a 2x1 synthetic image (pure red, then pure black).
  {
    const imageData = { width: 2, height: 1, data: new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 0, 255]) };
    const tensor = LawnBowlsEmbedding.imageDataToTensor(imageData);
    const [meanR, meanG, meanB] = LawnBowlsEmbedding.MEAN;
    const [stdR, stdG, stdB] = LawnBowlsEmbedding.STD;
    const expected = [
      (1 - meanR) / stdR, (0 - meanR) / stdR, // R plane: pixel0=red(1), pixel1=black(0)
      (0 - meanG) / stdG, (0 - meanG) / stdG, // G plane: both 0
      (0 - meanB) / stdB, (0 - meanB) / stdB, // B plane: both 0
    ];
    const matches = expected.every((v, i) => approxEqual(tensor.data[i], v, 1e-5));
    if (!matches) failures.push(`case1: expected ${expected}, got ${Array.from(tensor.data)}`);
    if (tensor.dims.join(',') !== '1,3,1,2') failures.push(`case1: expected dims [1,3,1,2], got ${tensor.dims}`);
  }

  // Case 2: normalize produces a unit vector.
  {
    const v = LawnBowlsEmbedding.normalize(new Float32Array([3, 4]));
    if (!approxEqual(v[0], 0.6, 1e-5) || !approxEqual(v[1], 0.8, 1e-5)) {
      failures.push(`case2: expected [0.6, 0.8], got ${Array.from(v)}`);
    }
  }

  // Case 3: cosineSimilarity on unit vectors — identical, orthogonal, opposite.
  {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    const c = new Float32Array([-1, 0]);
    if (!approxEqual(LawnBowlsEmbedding.cosineSimilarity(a, a), 1, 1e-6)) failures.push('case3: identical vectors should have similarity 1');
    if (!approxEqual(LawnBowlsEmbedding.cosineSimilarity(a, b), 0, 1e-6)) failures.push('case3: orthogonal vectors should have similarity 0');
    if (!approxEqual(LawnBowlsEmbedding.cosineSimilarity(a, c), -1, 1e-6)) failures.push('case3: opposite vectors should have similarity -1');
  }

  return { name: 'embedding', total: 3, failures };
}

module.exports = { run };

if (require.main === module) {
  const result = run();
  if (result.failures.length === 0) {
    console.log(`PASS  embedding (${result.total} cases)`);
    process.exit(0);
  } else {
    console.log(`FAIL  embedding`);
    result.failures.forEach(f => console.log(`        - ${f}`));
    process.exit(1);
  }
}
