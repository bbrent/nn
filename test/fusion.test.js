// Pure-logic tests for fusion.js. No images/opencv — addFrame only consumes
// the jack-centered localX/localY that detection.js already computes, so
// frames can be synthesized directly by rotating a known ground-truth layout
// and revealing a subset of bowls, simulating a phone panning over the rink.

const LawnBowlsFusion = require('../fusion.js');

// Ground truth: jack at origin, 5 bowls at distinct distances so matching
// in the test's own verification isn't itself ambiguous.
const WORLD_BOWLS = [
  { x: 1.0, y: 0.0 },
  { x: -0.5, y: 0.8 },
  { x: 0.3, y: -1.2 },
  { x: 2.0, y: 1.0 },
  { x: -1.5, y: -0.5 },
];

function rotate(p, theta) {
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  return { x: p.x * cos - p.y * sin, y: p.x * sin + p.y * cos };
}

// Builds a synthetic usable detectAndRank()-shaped result: `visibleIndices`
// of WORLD_BOWLS, as seen by a camera at orientation `theta` relative to world.
function makeFrame(visibleIndices, theta) {
  const ranking = visibleIndices.map(i => {
    const p = rotate(WORLD_BOWLS[i], theta);
    return { bowl: { worldIndex: i }, dist: Math.hypot(p.x, p.y), localX: p.x, localY: p.y };
  });
  return { usable: true, jack: { x: 0, y: 0 }, ranking };
}

function approxEqual(a, b, tol) {
  return Math.abs(a - b) < tol;
}

// Matches fused landmarks back to WORLD_BOWLS by nearest distance-to-jack
// (unambiguous since WORLD_BOWLS distances are all distinct), then checks
// every pairwise distance (including to the jack) is preserved — this
// verifies 2D placement, not just radius.
function checkPairwiseDistances(fusedBowls, worldIndices, tol) {
  const failures = [];
  const worldDists = WORLD_BOWLS.map(p => Math.hypot(p.x, p.y));

  const matched = fusedBowls.map(fb => {
    const dist = Math.hypot(fb.x, fb.y);
    let best = -1;
    let bestDiff = Infinity;
    worldIndices.forEach(i => {
      const diff = Math.abs(worldDists[i] - dist);
      if (diff < bestDiff) {
        bestDiff = diff;
        best = i;
      }
    });
    return { fb, worldIndex: best };
  });

  const usedWorldIndices = new Set(matched.map(m => m.worldIndex));
  if (usedWorldIndices.size !== worldIndices.length) {
    failures.push(`expected ${worldIndices.length} distinct matched landmarks, got ${usedWorldIndices.size}`);
  }

  for (let a = 0; a < matched.length; a++) {
    for (let b = a + 1; b < matched.length; b++) {
      const gotDist = Math.hypot(matched[a].fb.x - matched[b].fb.x, matched[a].fb.y - matched[b].fb.y);
      const trueDist = Math.hypot(
        WORLD_BOWLS[matched[a].worldIndex].x - WORLD_BOWLS[matched[b].worldIndex].x,
        WORLD_BOWLS[matched[a].worldIndex].y - WORLD_BOWLS[matched[b].worldIndex].y
      );
      if (!approxEqual(gotDist, trueDist, tol)) {
        failures.push(`pairwise distance mismatch (world ${matched[a].worldIndex}<->${matched[b].worldIndex}): expected ${trueDist.toFixed(3)}, got ${gotDist.toFixed(3)}`);
      }
    }
  }
  return failures;
}

function run() {
  const failures = [];

  // Case 1: full overlap across two differently-rotated frames should merge
  // into exactly 5 landmarks, not 10.
  {
    const fusion = LawnBowlsFusion.createFusion();
    LawnBowlsFusion.addFrame(fusion, makeFrame([0, 1, 2, 3, 4], 0));
    const r2 = LawnBowlsFusion.addFrame(fusion, makeFrame([0, 1, 2, 3, 4], 0.7));
    if (!r2.merged) failures.push('case1: second frame should have merged (full overlap)');
    if (fusion.bowls.length !== 5) failures.push(`case1: expected 5 landmarks, got ${fusion.bowls.length}`);
    failures.push(...checkPairwiseDistances(fusion.bowls, [0, 1, 2, 3, 4], 0.02).map(f => `case1: ${f}`));
  }

  // Case 2: panning — each frame only sees a subset, consecutive frames
  // overlap by 2 bowls. Final map should be the union (5), correctly placed.
  {
    const fusion = LawnBowlsFusion.createFusion();
    LawnBowlsFusion.addFrame(fusion, makeFrame([0, 1, 2], 0));
    const r2 = LawnBowlsFusion.addFrame(fusion, makeFrame([1, 2, 3], 1.1));
    const r3 = LawnBowlsFusion.addFrame(fusion, makeFrame([3, 4], -0.9));
    if (!r2.merged || !r3.merged) failures.push('case2: overlapping pan frames should all merge');
    if (fusion.bowls.length !== 5) failures.push(`case2: expected 5 landmarks (union), got ${fusion.bowls.length}`);
    failures.push(...checkPairwiseDistances(fusion.bowls, [0, 1, 2, 3, 4], 0.02).map(f => `case2: ${f}`));
  }

  // Case 3: a frame with no bowls in common with the existing map (all-new,
  // distinct distances) can't be confidently aligned and should be dropped.
  {
    const fusion = LawnBowlsFusion.createFusion();
    LawnBowlsFusion.addFrame(fusion, makeFrame([0, 1], 0));
    const disjointFrame = {
      usable: true,
      jack: { x: 0, y: 0 },
      ranking: [{ bowl: {}, dist: 10, localX: 10, localY: 0 }],
    };
    const r2 = LawnBowlsFusion.addFrame(fusion, disjointFrame);
    if (r2.merged) failures.push('case3: disjoint frame should not have merged');
    if (fusion.bowls.length !== 2) failures.push(`case3: landmark count should stay 2, got ${fusion.bowls.length}`);
  }

  // Case 4: repeated observations of the same bowls should raise their
  // observation count toward CONFIRM_OBSERVATIONS without creating duplicates.
  {
    const fusion = LawnBowlsFusion.createFusion();
    for (let i = 0; i < LawnBowlsFusion.CONFIRM_OBSERVATIONS; i++) {
      LawnBowlsFusion.addFrame(fusion, makeFrame([0, 1, 2], i * 0.3));
    }
    if (fusion.bowls.length !== 3) failures.push(`case4: expected 3 landmarks, got ${fusion.bowls.length}`);
    const snapshot = LawnBowlsFusion.getSnapshot(fusion);
    const allConfirmed = snapshot.ranking.every(r => r.confirmed);
    if (!allConfirmed) failures.push('case4: all landmarks should be confirmed after enough repeated views');
  }

  return { name: 'fusion', total: 4, failures };
}

module.exports = { run };

if (require.main === module) {
  const result = run();
  if (result.failures.length === 0) {
    console.log(`PASS  fusion (${result.total} cases)`);
    process.exit(0);
  } else {
    console.log(`FAIL  fusion`);
    result.failures.forEach(f => console.log(`        - ${f}`));
    process.exit(1);
  }
}
