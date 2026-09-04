// Cross-frame fusion: merges bowl positions from many overlapping camera
// frames into one consistent map, so the phone can be panned over a rink
// wider than a single shot and bowls currently out of frame still end up
// correctly placed. No camera pose/trajectory tracking (no SLAM, no
// calibration) — the bowls themselves are the landmarks.
//
// Every usable frame from detection.js already reports each bowl's position
// jack-centered, in bowl-diameter units (ranking[i].localX/localY). Since
// there is always exactly one jack, using it as the origin in every frame
// means alignment between two frames reduces to solving a single rotation
// (translation is already zero by construction) — no full SLAM/pose-graph
// machinery needed, just point-set registration between frames that share
// enough bowls.
//
// Shared with the browser app (window.LawnBowlsFusion) and the Node test
// harness (synthetic multi-frame sequences, no images/opencv involved).

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.LawnBowlsFusion = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  // Two local bowls are candidate matches to the same world landmark if their
  // jack-distances differ by less than this (bowl-diameter units).
  const MATCH_DIST_TOLERANCE = 0.35;
  // A candidate match must beat the next-best candidate by this margin to be
  // trusted for solving rotation — otherwise two bowls at similar distances
  // from the jack could get swapped.
  const MATCH_MARGIN = 0.15;
  // After rotating a frame's points into world space, how close counts as
  // "the same landmark" rather than a newly-revealed bowl.
  const MERGE_RADIUS = 0.4;
  // Observations needed before a landmark is considered stable ("consensus").
  const CONFIRM_OBSERVATIONS = 3;

  function createFusion() {
    return { jackSeen: 0, bowls: [], frameCount: 0 };
  }

  // Finds each local bowl's best-matching world landmark by comparing
  // distance-to-jack (rotation-invariant, so this works before rotation is
  // known). Returns only unambiguous matches — good enough to solve the
  // frame's rotation from, even if not every bowl matches confidently.
  function findConfidentMatches(localPoints, worldBowls) {
    const matches = [];
    for (let i = 0; i < localPoints.length; i++) {
      const localDist = Math.hypot(localPoints[i].x, localPoints[i].y);
      const candidates = worldBowls
        .map((w, j) => ({ j, diff: Math.abs(Math.hypot(w.x, w.y) - localDist) }))
        .filter(c => c.diff < MATCH_DIST_TOLERANCE)
        .sort((a, b) => a.diff - b.diff);

      if (candidates.length === 0) continue;
      if (candidates.length > 1 && candidates[1].diff - candidates[0].diff < MATCH_MARGIN) continue; // ambiguous

      matches.push({ localIndex: i, worldIndex: candidates[0].j, diff: candidates[0].diff });
    }
    return matches;
  }

  // Best-fit rotation (no scale, no translation — both point sets are already
  // jack-centered) minimizing weighted sum of squared distances between
  // matched pairs. Weighting by match tightness matters: a "confident" match
  // just means no *competing* candidate, not that it's actually correct — two
  // real bowls can sit at similar distances from the jack, so a near-exact
  // match (diff ~0) needs to dominate a merely-passable one (diff ~tolerance)
  // rather than being averaged with it into a compromise rotation that fits
  // neither well.
  function solveRotation(localPoints, worldBowls, matches) {
    let sinSum = 0;
    let cosSum = 0;
    for (const { localIndex, worldIndex, diff } of matches) {
      const l = localPoints[localIndex];
      const w = worldBowls[worldIndex];
      const weight = 1 / Math.pow((diff || 0) + 0.02, 2);
      sinSum += weight * (l.x * w.y - l.y * w.x);
      cosSum += weight * (l.x * w.x + l.y * w.y);
    }
    return Math.atan2(sinSum, cosSum);
  }

  function rotate(p, theta) {
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    return { x: p.x * cos - p.y * sin, y: p.x * sin + p.y * cos };
  }

  // A "confident" distance match can still be wrong — two real bowls can
  // legitimately sit at similar distances from the jack, close enough to
  // both pass the tolerance check. Solve rotation, then check whether any
  // matched pair disagrees badly with the rest; if one clearly does, drop it
  // and re-solve once rather than let it corrupt the whole frame's alignment.
  function solveRotationRobust(localPoints, worldBowls, matches) {
    let theta = solveRotation(localPoints, worldBowls, matches);
    if (matches.length < 2) return theta;

    const residuals = matches.map(m => {
      const rotated = rotate(localPoints[m.localIndex], theta);
      const w = worldBowls[m.worldIndex];
      return Math.hypot(rotated.x - w.x, rotated.y - w.y);
    });

    const worstIndex = residuals.indexOf(Math.max(...residuals));
    const otherResiduals = residuals.filter((_, i) => i !== worstIndex).sort((a, b) => a - b);
    const medianOther = otherResiduals[Math.floor(otherResiduals.length / 2)];

    const isOutlier = residuals[worstIndex] > MERGE_RADIUS && residuals[worstIndex] > medianOther * 2;
    if (isOutlier) {
      const filtered = matches.filter((_, i) => i !== worstIndex);
      theta = solveRotation(localPoints, worldBowls, filtered);
    }
    return theta;
  }

  // Merges one usable detectAndRank() result into the running map. Frames
  // that don't share enough bowls with the existing map to align confidently
  // are dropped — same principle as the single-frame usable gate.
  function addFrame(fusion, frameResult) {
    if (!frameResult.usable || !frameResult.jack) {
      return { merged: false, reason: 'frame not usable' };
    }

    fusion.frameCount++;
    fusion.jackSeen++;

    const localPoints = frameResult.ranking.map(r => ({ x: r.localX, y: r.localY }));

    if (fusion.bowls.length === 0) {
      // First frame: its own orientation becomes the world frame.
      for (const p of localPoints) fusion.bowls.push({ x: p.x, y: p.y, observations: 1 });
      return { merged: true, newLandmarks: localPoints.length, reason: null };
    }

    const matches = findConfidentMatches(localPoints, fusion.bowls);
    if (matches.length === 0) {
      return { merged: false, reason: 'no confident overlap with existing map' };
    }

    const theta = solveRotationRobust(localPoints, fusion.bowls, matches);

    let newLandmarks = 0;
    for (const p of localPoints) {
      const rotated = rotate(p, theta);

      let bestIndex = -1;
      let bestDist = Infinity;
      fusion.bowls.forEach((w, j) => {
        const d = Math.hypot(rotated.x - w.x, rotated.y - w.y);
        if (d < MERGE_RADIUS && d < bestDist) {
          bestDist = d;
          bestIndex = j;
        }
      });

      if (bestIndex >= 0) {
        const landmark = fusion.bowls[bestIndex];
        const n = landmark.observations;
        landmark.x = (landmark.x * n + rotated.x) / (n + 1);
        landmark.y = (landmark.y * n + rotated.y) / (n + 1);
        landmark.observations = n + 1;
      } else {
        fusion.bowls.push({ x: rotated.x, y: rotated.y, observations: 1 });
        newLandmarks++;
      }
    }

    return { merged: true, newLandmarks, reason: null };
  }

  // Bowls ranked by distance to the (always jack-centered-at-origin) jack,
  // shaped like detection.js's ranking so the same UI code can consume either.
  // confirmedOnly drops landmarks seen fewer than CONFIRM_OBSERVATIONS times —
  // a stray object (a shoe, a hand) that got Hough-detected once or twice
  // should not end up permanently scored as a bowl. Live status display wants
  // the unfiltered view (to show "N tracked, M confirmed"); freezing the map
  // for scoring wants confirmedOnly so noise doesn't get locked in.
  function getSnapshot(fusion, opts) {
    const confirmedOnly = opts && opts.confirmedOnly;
    const bowls = confirmedOnly
      ? fusion.bowls.filter(b => b.observations >= CONFIRM_OBSERVATIONS)
      : fusion.bowls;
    const ranking = bowls
      .map(b => ({ bowl: b, dist: Math.hypot(b.x, b.y), confirmed: b.observations >= CONFIRM_OBSERVATIONS }))
      .sort((a, b) => a.dist - b.dist);
    return { jack: { x: 0, y: 0 }, bowls, ranking, frameCount: fusion.frameCount };
  }

  // Maps a snapshot's bowl-diameter-unit coordinates into canvas pixel space
  // (uniform fit-to-canvas, with a fixed display radius since apparent size
  // has no meaning in this abstract map). Returns detections/jack/ranking
  // shaped exactly like a live detectAndRank() result.
  function layoutForCanvas(snapshot, width, height, opts) {
    const bowlR = (opts && opts.bowlRadiusPx) || Math.min(width, height) * 0.035;
    const jackR = bowlR * 0.55;
    const padding = (opts && opts.paddingPx) || bowlR * 3;

    const points = [{ x: 0, y: 0 }, ...snapshot.bowls.map(b => ({ x: b.x, y: b.y }))];
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = Math.max(maxX - minX, 0.001);
    const spanY = Math.max(maxY - minY, 0.001);
    const scale = Math.min((width - 2 * padding) / spanX, (height - 2 * padding) / spanY);
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    function project(p, r) {
      return {
        x: width / 2 + (p.x - centerX) * scale,
        y: height / 2 + (p.y - centerY) * scale,
        r,
      };
    }

    const jack = project({ x: 0, y: 0 }, jackR);
    const bowlPoints = snapshot.bowls.map(b => project(b, bowlR));
    const detections = [jack, ...bowlPoints];

    const ranking = snapshot.ranking.map(entry => {
      const idx = snapshot.bowls.indexOf(entry.bowl);
      return { bowl: bowlPoints[idx], dist: entry.dist, confirmed: entry.confirmed };
    });

    return { detections, jack, ranking };
  }

  return {
    MATCH_DIST_TOLERANCE,
    MATCH_MARGIN,
    MERGE_RADIUS,
    CONFIRM_OBSERVATIONS,
    createFusion,
    addFrame,
    getSnapshot,
    layoutForCanvas,
  };
});
