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

// Like makeFrame, but for an arbitrary point set — used by the pruning cases
// below, which need geometry closer together than WORLD_BOWLS provides.
function makeFrameFrom(points, visibleIndices, theta) {
  const ranking = visibleIndices.map(i => {
    const p = rotate(points[i], theta);
    return { bowl: { i }, dist: Math.hypot(p.x, p.y), localX: p.x, localY: p.y };
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

  // Case 5: a stray object seen only once or twice (a shoe, a hand) — should
  // count toward the live "tracked" total but be dropped from confirmedOnly,
  // since that's what feeds the frozen/scored map.
  {
    const fusion = LawnBowlsFusion.createFusion();
    // Bowls 0,1,2 seen enough times to confirm...
    for (let i = 0; i < LawnBowlsFusion.CONFIRM_OBSERVATIONS; i++) {
      LawnBowlsFusion.addFrame(fusion, makeFrame([0, 1, 2], i * 0.3));
    }
    // ...plus one frame with a stray extra point (distance 5, far from any
    // real bowl) seen only this once. Uses bowls 0 and 2 (well-separated
    // distances) rather than 0 and 1, which are close enough to each other
    // to be mutually ambiguous under MATCH_MARGIN — correctly so, that's a
    // real limitation, not what this case is testing.
    const strayFrame = makeFrame([0, 2], 0.9);
    strayFrame.ranking.push({ bowl: { stray: true }, dist: 5, localX: 5, localY: 0 });
    LawnBowlsFusion.addFrame(fusion, strayFrame);

    const unfiltered = LawnBowlsFusion.getSnapshot(fusion);
    const filtered = LawnBowlsFusion.getSnapshot(fusion, { confirmedOnly: true });

    if (unfiltered.bowls.length !== 4) {
      failures.push(`case5: expected 4 tracked landmarks (3 real + 1 stray), got ${unfiltered.bowls.length}`);
    }
    if (filtered.bowls.length !== 3) {
      failures.push(`case5: expected 3 confirmed landmarks (stray excluded), got ${filtered.bowls.length}`);
    }
    if (filtered.bowls.some(b => Math.hypot(b.x, b.y) > 4)) {
      failures.push('case5: stray landmark leaked into confirmedOnly snapshot');
    }
  }

  // Case 6: a landmark that's genuinely gone (a false positive that isn't
  // there on re-look, or literally removed from the rink) should be pruned
  // once frames that clearly re-check its neighborhood keep coming up empty.
  {
    const CLOSE_PTS = [
      { x: 1.0, y: 0.0 }, // A
      { x: 1.3, y: 0.3 }, // B
      { x: 0.7, y: -0.3 }, // C — "goes missing" after seeding
    ];
    const fusion = LawnBowlsFusion.createFusion();
    for (let i = 0; i < 3; i++) LawnBowlsFusion.addFrame(fusion, makeFrameFrom(CLOSE_PTS, [0, 1, 2], i * 0.15));
    if (fusion.bowls.length !== 3) failures.push(`case6: expected 3 seeded landmarks, got ${fusion.bowls.length}`);

    let sawRemoval = false;
    for (let i = 0; i < LawnBowlsFusion.MISS_THRESHOLD; i++) {
      const r = LawnBowlsFusion.addFrame(fusion, makeFrameFrom(CLOSE_PTS, [0, 1], 0.4 + i * 0.1));
      if (r.removedLandmarks > 0) sawRemoval = true;
    }
    if (!sawRemoval) failures.push('case6: landmark C should have been pruned after repeated clean misses');
    if (fusion.bowls.length !== 2) failures.push(`case6: expected 2 landmarks left after pruning, got ${fusion.bowls.length}`);
    if (fusion.bowls.some(b => Math.hypot(b.x - 0.7, b.y - (-0.3)) < 0.3)) {
      failures.push('case6: pruned landmark C is still present');
    }
  }

  // Case 7: a landmark that's just genuinely out of frame (far from anything
  // this frame re-confirmed) must NOT be pruned — only a spot the camera
  // clearly just re-checked counts as evidence of absence.
  {
    const fusion = LawnBowlsFusion.createFusion();
    for (let i = 0; i < 3; i++) LawnBowlsFusion.addFrame(fusion, makeFrame([0, 2, 3], i * 0.2));
    for (let i = 0; i < 3; i++) LawnBowlsFusion.addFrame(fusion, makeFrame([0, 3], 0.5 + i * 0.15));
    if (fusion.bowls.length !== 3) {
      failures.push(`case7: bowl 2 (out of frame, not absent) should not have been pruned, got ${fusion.bowls.length} landmarks`);
    }
  }

  return { name: 'fusion', total: 7, failures };
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
