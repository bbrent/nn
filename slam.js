// SLAM-style mapping: estimates where the camera is for every frame and
// builds one shared map of bowl positions from all of them, so the whole end
// can be scanned freely — pan, walk around, let bowls leave the frame — with
// no requirement to keep everything (or anything in particular) in shot.
//
// Why this replaces fusion.js's approach: that one pinned every frame to the
// jack, which made alignment a single-rotation solve, but it also meant the
// jack had to be visible in EVERY frame or the frame was discarded outright.
// In practice that threw away most of a real scan. Here the jack is just
// another landmark: it needs to be seen once, anywhere in the scan, and only
// because scoring is measured from it.
//
// The map is 2D and metric in bowl-diameters. Frames arrive already rectified
// onto the green by ground.js, which uses each bowl's apparent size as a depth
// cue, so what this module aligns are true top-down positions rather than raw
// image coordinates. That matters for more than tidiness: measuring distances
// straight off the image assumes the phone points vertically down, and at any
// realistic angle it foreshortens the far side of the head badly enough to
// invert which bowl is closest — the one thing the app exists to decide.
//
// Because rectified coordinates are metric, the same bowl measures the same in
// every frame no matter where the camera stood, so alignment between frames is
// a plain rigid transform with no scale left to guess at.
//
// Every landmark also carries the spread of the positions it has been seen at,
// which is what lets the app say how sure it is: two bowls whose gap is
// smaller than that spread are reported as too close to call rather than
// silently ranked.
//
// Shared with the browser app (window.LawnBowlsSlam) and the Node test
// harness (synthetic frame sequences, no images or camera involved).

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('./ground.js'));
  } else {
    root.LawnBowlsSlam = factory(root.LawnBowlsGround);
  }
})(typeof self !== 'undefined' ? self : this, function (LawnBowlsGround) {
  // How close a projected landmark and a detection must be (bowl-diameters)
  // to be treated as the same thing when estimating a pose.
  const ASSOC_GATE = 0.55;
  // After the pose is solved, how close counts as "already mapped" rather
  // than a newly-revealed bowl. Slightly looser than the gate so a landmark
  // that only just failed association doesn't immediately get duplicated.
  const MERGE_RADIUS = 0.7;
  // Correspondences needed to solve a pose at all. Two exactly determine a
  // similarity transform; fewer is unsolvable.
  const MIN_INLIERS = 2;
  // A two-point hypothesis with a short baseline gives a wildly unstable
  // rotation (tiny position noise swings the angle), so require some spread.
  const MIN_BASELINE = 0.8;
  // Rectified coordinates are metric in bowl-diameters, set by the bowls'
  // real size, so the same bowl measures the same in every frame and the
  // frame-to-frame scale should sit at 1. The tolerance left here covers
  // detection noise and a guessed focal length, nothing more; anything beyond
  // it is a mis-association explaining itself away by resizing the world.
  const MIN_SCALE = 0.85;
  const MAX_SCALE = 1.18;
  // Observations before a landmark is trusted as a real bowl rather than a
  // one-off false positive (a shoe, a hand, a bag).
  const CONFIRM_OBSERVATIONS = 3;
  // Clean looks that came back empty before a landmark is dropped.
  const MISS_THRESHOLD = 3;
  // Inset from the frame edge (bowl-diameters) when deciding whether a
  // landmark was really inside this frame's view — detections right at the
  // border are unreliable, so don't count a miss against them.
  const VIEW_MARGIN = 0.75;
  // Cap on hypotheses tried during relocalisation, so a frame can never
  // stall the capture loop.
  const RANSAC_BUDGET = 400;
  // A pose is only rejected outright when it claims to be staring at a lot of
  // mapped bowls and lines up with almost none of them. Deliberately lenient:
  // the detector routinely finds only a few of the bowls actually in shot, and
  // such a frame is still perfectly well positioned by the ones it did find.
  const MIN_EXPLAINED_FOR_POSE = 0.25;
  // Concluding a mapped bowl is NOT there is a much stronger claim than
  // placing the camera, so it takes a frame that found most of what it
  // expected. Otherwise one bad detection frame would start deleting real
  // bowls that were sitting in plain view the whole time.
  const MIN_EXPLAINED_FOR_PRUNE = 0.8;
  // Relocalisation has no prior to lean on, so a two-point fit there is a
  // coincidence waiting to happen; while tracking, the previous pose already
  // rules out everything except a near-continuation of it.
  const MIN_INLIERS_TRACKING = 2;
  const MIN_INLIERS_RELOCALISING = 3;
  // Floor on how precisely a position is ever claimed to be known, in
  // bowl-diameters. Repeated looks that agree shrink the measured spread
  // toward zero, but that spread only ever captures the part of the error
  // that varies between frames. The dominant part does not vary.
  //
  // Measured end to end (test/scan-harness.js renders a green, runs the real
  // detector over it and compares the resulting map against what was drawn):
  // the detector reports bowl radii about 3.5% larger than they truly are,
  // consistently, and depth is read from radius — so that becomes the same
  // depth error in every frame, invisible to any amount of averaging. Worse,
  // box precision is roughly constant in pixels, so the relative radius error
  // grows for smaller, further bowls; the recovered green is then slightly
  // tilted and distances come out wrong by differing amounts rather than by a
  // uniform factor. Across a full simulated scan that left distance errors
  // averaging 0.34 bowl-diameters and reaching 0.78, against a spread-derived
  // sigma of about 0.10 — confidently wrong by roughly threefold, and wrong in
  // the direction that matters, since it would declare bowls separable that
  // this pipeline cannot actually separate.
  //
  // This floor is set so two sigma covers the worst error seen there. It makes
  // the app decline more close calls, which is the honest outcome: those are
  // exactly the ends a player would put the measure on anyway. Calibrated on
  // simulation and due a revisit against real footage.
  const MIN_POSITION_ERROR = 0.28;
  // A landmark seen once has no measurable spread at all. That is not
  // precision, it is ignorance, so it is treated as very uncertain until a
  // second look either confirms or contradicts it.
  const UNMEASURED_POSITION_ERROR = 0.5;
  // How many standard errors two bowls must be apart before their order is
  // called rather than flagged. Two is roughly 95% confidence.
  const CONFIDENCE_SIGMAS = 2;
  // How badly a frame's detections may fail to lie on one plane before the
  // frame is thrown out. Bowls on a green are coplanar by definition, so a
  // large residual means something in the frame isn't what it was taken for —
  // most importantly a bowl mislabelled as the jack, which gets scaled to the
  // wrong depth and drags the whole fit with it.
  //
  // Calibrated by simulation rather than guessed: legitimate frames sit at
  // 0.02-0.06 with realistic detector noise (p95 ~0.10, rising to ~0.16 when
  // noise is heavy), while a bowl mislabelled as the jack sits around 0.26.
  // The gap is real but not wide enough for a tight gate without field data,
  // so this is deliberately lenient and only catches egregious cases. Milder
  // distortion needs no gate of its own: a frame with skewed geometry
  // contributes positions that disagree with the others, which widens the
  // measured spread and shows up as lower confidence — which is the honest
  // outcome anyway.
  const MAX_PLANE_RESIDUAL = 0.3;

  function createSlam() {
    return {
      landmarks: [], // { x, y, observations, misses, jackVotes, identity }
      poses: [], // one { x, y, theta, scale } per merged frame, in order
      lastPose: null,
      frameCount: 0,
      mergedCount: 0,
    };
  }

  // --- pose algebra -------------------------------------------------------
  // A pose maps this frame's local coordinates into the shared world frame:
  //   world = scale * R(theta) * local + (x, y)

  const IDENTITY_POSE = { x: 0, y: 0, theta: 0, scale: 1 };

  function applyPose(pose, p) {
    const cos = Math.cos(pose.theta);
    const sin = Math.sin(pose.theta);
    return {
      x: pose.scale * (p.x * cos - p.y * sin) + pose.x,
      y: pose.scale * (p.x * sin + p.y * cos) + pose.y,
    };
  }

  function applyInversePose(pose, p) {
    const cos = Math.cos(pose.theta);
    const sin = Math.sin(pose.theta);
    const dx = (p.x - pose.x) / pose.scale;
    const dy = (p.y - pose.y) / pose.scale;
    return { x: dx * cos + dy * sin, y: -dx * sin + dy * cos };
  }

  // Closed-form similarity transform (Umeyama) minimising the weighted sum of
  // squared distances between corresponding points. Weighted rather than
  // plain least-squares because a tight correspondence is far better evidence
  // than one that merely scraped inside the gate, and averaging the two
  // equally produces a pose that fits neither.
  function solveSimilarity(pairs) {
    let wSum = 0;
    let lcx = 0, lcy = 0, wcx = 0, wcy = 0;
    for (const pair of pairs) {
      const w = pair.weight === undefined ? 1 : pair.weight;
      wSum += w;
      lcx += w * pair.local.x;
      lcy += w * pair.local.y;
      wcx += w * pair.world.x;
      wcy += w * pair.world.y;
    }
    if (wSum === 0) return null;
    lcx /= wSum; lcy /= wSum; wcx /= wSum; wcy /= wSum;

    let a = 0; // sum of dot products    -> cos component
    let b = 0; // sum of cross products  -> sin component
    let localVar = 0;
    for (const pair of pairs) {
      const w = pair.weight === undefined ? 1 : pair.weight;
      const lx = pair.local.x - lcx;
      const ly = pair.local.y - lcy;
      const wx = pair.world.x - wcx;
      const wy = pair.world.y - wcy;
      a += w * (lx * wx + ly * wy);
      b += w * (lx * wy - ly * wx);
      localVar += w * (lx * lx + ly * ly);
    }
    if (localVar < 1e-9) return null; // all correspondences on one spot

    const theta = Math.atan2(b, a);
    const scale = Math.sqrt(a * a + b * b) / localVar;
    if (!isFinite(scale) || scale < MIN_SCALE || scale > MAX_SCALE) return null;

    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    return {
      theta,
      scale,
      x: wcx - scale * (lcx * cos - lcy * sin),
      y: wcy - scale * (lcx * sin + lcy * cos),
    };
  }

  // Exact similarity transform from two correspondences. Used to seed RANSAC
  // hypotheses: two points are the minimum that determines one.
  function solveFromPair(l1, w1, l2, w2) {
    const dlx = l2.x - l1.x;
    const dly = l2.y - l1.y;
    const dwx = w2.x - w1.x;
    const dwy = w2.y - w1.y;
    const dl = Math.hypot(dlx, dly);
    const dw = Math.hypot(dwx, dwy);
    if (dl < MIN_BASELINE || dw < MIN_BASELINE) return null;

    const scale = dw / dl;
    if (scale < MIN_SCALE || scale > MAX_SCALE) return null;

    const theta = Math.atan2(dwy, dwx) - Math.atan2(dly, dlx);
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    return {
      theta,
      scale,
      x: w1.x - scale * (l1.x * cos - l1.y * sin),
      y: w1.y - scale * (l1.x * sin + l1.y * cos),
    };
  }

  // --- data association ---------------------------------------------------

  // Greedy nearest-neighbour matching under a candidate pose, closest pairs
  // claimed first so a landmark can't be stolen by a worse match that merely
  // got iterated over earlier. One-to-one in both directions.
  function associate(localPoints, landmarks, pose, gate) {
    const candidates = [];
    for (let i = 0; i < localPoints.length; i++) {
      const projected = applyPose(pose, localPoints[i]);
      for (let j = 0; j < landmarks.length; j++) {
        const dist = Math.hypot(projected.x - landmarks[j].x, projected.y - landmarks[j].y);
        if (dist < gate) candidates.push({ localIndex: i, landmarkIndex: j, dist });
      }
    }
    candidates.sort((p, q) => p.dist - q.dist);

    const usedLocal = new Set();
    const usedLandmark = new Set();
    const pairs = [];
    for (const c of candidates) {
      if (usedLocal.has(c.localIndex) || usedLandmark.has(c.landmarkIndex)) continue;
      usedLocal.add(c.localIndex);
      usedLandmark.add(c.landmarkIndex);
      pairs.push(c);
    }
    return pairs;
  }

  function pairsToCorrespondences(pairs, localPoints, landmarks) {
    return pairs.map(p => ({
      local: localPoints[p.localIndex],
      world: landmarks[p.landmarkIndex],
      // Tight matches dominate marginal ones; the epsilon keeps a perfect
      // match from becoming an infinite weight.
      weight: 1 / Math.pow(p.dist + 0.05, 2),
    }));
  }

  // Refines a candidate pose by alternating association and solving, which
  // pulls in correspondences that were just outside the gate under the
  // initial guess. Converges in a couple of rounds at this scale.
  function refinePose(localPoints, landmarks, pose, rounds) {
    let current = pose;
    let pairs = associate(localPoints, landmarks, current, ASSOC_GATE);
    for (let i = 0; i < rounds; i++) {
      if (pairs.length < MIN_INLIERS) break;
      const solved = solveSimilarity(pairsToCorrespondences(pairs, localPoints, landmarks));
      if (!solved) break;
      const nextPairs = associate(localPoints, landmarks, solved, ASSOC_GATE);
      current = solved;
      if (nextPairs.length <= pairs.length) { pairs = nextPairs; break; }
      pairs = nextPairs;
    }
    return { pose: current, pairs };
  }

  // Landmarks that this frame's field of view says should have been visible.
  // Having a real pose is what makes this possible at all — the old jack-
  // anchored fusion had to guess with a fixed radius around matched points.
  // Pruning exists to clear out things that were never bowls — a shoe, a bag,
  // a passing hand caught once or twice. It is not a mechanism for removing
  // bowls that are genuinely there, so two kinds of landmark are exempt.
  //
  // A landmark confirmed by several independent looks is not a false positive
  // by definition, and deleting it because the detector had a poor run costs
  // a real bowl for no gain. If a bowl truly is moved mid-end, a fresh scan is
  // the right remedy, not silent deletion during this one.
  //
  // The jack is exempt outright. It is half a bowl across and correspondingly
  // easy to miss, so its misses carry far less meaning than a bowl's — and
  // losing it costs the whole score rather than one position, since every
  // distance is measured from it. That asymmetry makes protecting it the only
  // sensible default, and it is what makes "look at the jack once" a promise
  // the app can actually keep.
  function isPrunable(landmark) {
    if (landmark.jackVotes > 0) return false;
    return landmark.observations < CONFIRM_OBSERVATIONS;
  }

  // Landmarks lying inside the patch of green this frame actually covered.
  // An empty footprint means the view was too oblique to bound (the horizon
  // was in shot), and the honest answer there is that nothing can be
  // concluded about what was or wasn't visible — so nothing is.
  function landmarksInView(landmarks, pose, footprint) {
    if (!footprint) return [];
    const inView = [];
    for (let j = 0; j < landmarks.length; j++) {
      const local = applyInversePose(pose, landmarks[j]);
      if (LawnBowlsGround.containsPoint(footprint, local)) inView.push(j);
    }
    return inView;
  }

  // Fraction of the landmarks this pose claims to be looking at that it
  // actually matched. A pose that says "twelve mapped bowls are in shot" but
  // only lines up with two of them has almost certainly locked onto a
  // coincidence, and merging it would corrupt the map.
  function explainedFraction(pairs, localPoints, landmarks, pose, footprint) {
    const expected = landmarksInView(landmarks, pose, footprint);
    if (expected.length === 0) return 1;
    const matched = new Set(pairs.map(p => p.landmarkIndex));
    let hits = 0;
    for (const j of expected) if (matched.has(j)) hits++;
    return hits / expected.length;
  }

  function poseQuality(pairs, localPoints, landmarks, pose, footprint, minInliers) {
    if (pairs.length < minInliers) return null;
    // Two-point poses ride entirely on their baseline; a short one is noise.
    if (pairs.length === 2) {
      const a = localPoints[pairs[0].localIndex];
      const b = localPoints[pairs[1].localIndex];
      if (Math.hypot(a.x - b.x, a.y - b.y) < MIN_BASELINE) return null;
    }
    const explained = explainedFraction(pairs, localPoints, landmarks, pose, footprint);
    if (explained < MIN_EXPLAINED_FOR_POSE) return null;
    const residual = pairs.reduce((s, p) => s + p.dist, 0) / pairs.length;
    // Prefer more correspondences first, then tighter ones.
    return pairs.length + (1 - Math.min(residual / ASSOC_GATE, 1));
  }

  // Estimates this frame's pose. Tries to track from the previous pose first
  // (the overwhelmingly common case while panning), and falls back to a
  // bounded RANSAC over pairwise correspondences when tracking fails —
  // which is what lets the camera be swung away and brought back without
  // having to restart the scan.
  function estimatePose(localPoints, landmarks, priorPose, footprint) {
    let best = null;
    let bestQuality = -Infinity;

    function consider(candidate, minInliers) {
      if (!candidate) return;
      const refined = refinePose(localPoints, landmarks, candidate, 3);
      const quality = poseQuality(refined.pairs, localPoints, landmarks, refined.pose, footprint, minInliers);
      if (quality !== null && quality > bestQuality) {
        bestQuality = quality;
        best = refined;
      }
    }

    if (priorPose) {
      consider(priorPose, MIN_INLIERS_TRACKING);
      // Good enough to skip the search: it agrees with where the camera was a
      // moment ago and has more correspondences than a fluke would produce.
      if (best && best.pairs.length >= MIN_INLIERS_RELOCALISING) {
        return { ok: true, pose: best.pose, pairs: best.pairs, relocalised: false };
      }
    }
    const trackedBest = best;

    // Relocalisation. Long baselines give the best-conditioned rotations, so
    // try those first and let the budget cut off the rest.
    const localPairs = [];
    for (let i = 0; i < localPoints.length; i++) {
      for (let j = i + 1; j < localPoints.length; j++) {
        const sep = Math.hypot(localPoints[i].x - localPoints[j].x, localPoints[i].y - localPoints[j].y);
        if (sep >= MIN_BASELINE) localPairs.push({ i, j, sep });
      }
    }
    localPairs.sort((p, q) => q.sep - p.sep);

    let budget = RANSAC_BUDGET;
    for (const lp of localPairs) {
      for (let a = 0; a < landmarks.length && budget > 0; a++) {
        for (let b = 0; b < landmarks.length && budget > 0; b++) {
          if (a === b) continue;
          const worldSep = Math.hypot(landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y);
          const ratio = worldSep / lp.sep;
          if (ratio < MIN_SCALE || ratio > MAX_SCALE) continue; // implausible scale, skip before solving
          budget--;
          consider(solveFromPair(localPoints[lp.i], landmarks[a], localPoints[lp.j], landmarks[b]), MIN_INLIERS_RELOCALISING);
        }
      }
      if (budget <= 0) break;
    }

    if (!best) return { ok: false, reason: 'could not place this view in the map' };
    return { ok: true, pose: best.pose, pairs: best.pairs, relocalised: best !== trackedBest };
  }

  // --- frame ingestion ----------------------------------------------------

  // Converts image-pixel detections into this frame's local coordinates, by
  // rectifying them onto the green (see ground.js). The result is metric in
  // bowl-diameters and genuinely top-down, so distances mean the same thing
  // wherever the camera was standing.
  //
  // A frame the rectifier can't place is rejected outright rather than falling
  // back to the old "divide everything by one median radius" scaling. Those
  // two conventions describe different spaces — one is the ground, the other
  // is the image plane — and quietly mixing them in a single map would put
  // bowls in positions that belong to neither.
  function toLocalFrame(detections, jack, viewWidth, viewHeight, focalLength) {
    const rectified = LawnBowlsGround.rectify(detections, jack, {
      width: viewWidth,
      height: viewHeight,
      focalLength,
      insetDiameters: VIEW_MARGIN,
    });
    if (!rectified.ok) return { ok: false, reason: rectified.reason };
    if (rectified.residual > MAX_PLANE_RESIDUAL) {
      return { ok: false, reason: "this view doesn't sit flat on the green — something in it isn't a bowl" };
    }

    const bowlRadii = detections.filter(d => d !== jack && d.r > 0).map(d => d.r).sort((p, q) => p - q);
    const medianRadius = bowlRadii.length ? bowlRadii[Math.floor(bowlRadii.length / 2)] : 0;

    return {
      ok: true,
      points: rectified.points.map(p => ({
        x: p.x,
        y: p.y,
        isJack: p.detection === jack,
        identity: p.detection.identity || null,
      })),
      footprint: rectified.footprint,
      tilt: rectified.tilt,
      residual: rectified.residual,
      // Kept only so the live overlay can draw mapped bowls back at the right
      // apparent size; the map itself no longer depends on it.
      diameter: medianRadius * 2,
    };
  }

  // Merges one frame of raw detections into the map.
  //
  // frame: {
  //   detections: [{ x, y, r, identity? }]  image pixels, jack included
  //   jack:       one of detections, or null if it isn't in this shot
  //   width, height: the frame's pixel dimensions
  // }
  //
  // Unlike the old jack-anchored fusion, a frame with no jack in it is
  // perfectly usable — that is the whole point.
  function addFrame(slam, frame) {
    slam.frameCount++;

    const detections = frame.detections || [];
    if (detections.length === 0) {
      return { merged: false, reason: 'nothing detected in this frame' };
    }

    const local = toLocalFrame(detections, frame.jack, frame.width, frame.height, frame.focalLength);
    if (!local.ok) {
      return { merged: false, reason: local.reason };
    }

    if (slam.landmarks.length === 0) {
      // First frame defines the world: its own coordinates become the map's.
      for (const p of local.points) {
        slam.landmarks.push({
          x: p.x,
          y: p.y,
          observations: 1,
          misses: 0,
          m2: 0,
          jackVotes: p.isJack ? 1 : 0,
          identity: p.identity,
        });
      }
      slam.lastPose = IDENTITY_POSE;
      slam.poses.push(IDENTITY_POSE);
      slam.mergedCount++;
      return {
        merged: true,
        pose: IDENTITY_POSE,
        bowlDiameterPx: local.diameter,
        newLandmarks: local.points.length,
        removedLandmarks: 0,
        relocalised: false,
        reason: null,
      };
    }

    const estimate = estimatePose(local.points, slam.landmarks, slam.lastPose, local.footprint);
    if (!estimate.ok) {
      return { merged: false, reason: estimate.reason };
    }

    const pose = estimate.pose;
    const worldPoints = local.points.map(p => applyPose(pose, p));

    let newLandmarks = 0;
    const seen = new Set();

    for (let i = 0; i < worldPoints.length; i++) {
      const wp = worldPoints[i];
      const p = local.points[i];

      let bestIndex = -1;
      let bestDist = Infinity;
      for (let j = 0; j < slam.landmarks.length; j++) {
        if (seen.has(j)) continue;
        const d = Math.hypot(wp.x - slam.landmarks[j].x, wp.y - slam.landmarks[j].y);
        if (d < MERGE_RADIUS && d < bestDist) {
          bestDist = d;
          bestIndex = j;
        }
      }

      if (bestIndex >= 0) {
        const landmark = slam.landmarks[bestIndex];
        // Welford: keeps the running mean and the spread around it in one
        // pass, numerically stable. The spread is what lets the app say how
        // sure it is — repeated looks that agree tighten the estimate, ones
        // that disagree widen it, and both are real information about whether
        // a close call can be trusted.
        const n = landmark.observations + 1;
        const dx = wp.x - landmark.x;
        const dy = wp.y - landmark.y;
        landmark.x += dx / n;
        landmark.y += dy / n;
        landmark.m2 += dx * (wp.x - landmark.x) + dy * (wp.y - landmark.y);
        landmark.observations = n;
        landmark.misses = 0;
        if (p.isJack) landmark.jackVotes++;
        // Identity is sticky and only ever upgraded: one frame's crop
        // failing to match must not erase a confident earlier match.
        if (p.identity && (!landmark.identity || p.identity.similarity > landmark.identity.similarity)) {
          landmark.identity = p.identity;
        }
        seen.add(bestIndex);
      } else {
        slam.landmarks.push({
          x: wp.x,
          y: wp.y,
          observations: 1,
          misses: 0,
          m2: 0,
          jackVotes: p.isJack ? 1 : 0,
          identity: p.identity,
        });
        seen.add(slam.landmarks.length - 1);
        newLandmarks++;
      }
    }

    // Anything the pose says was squarely in shot but that matched nothing
    // gets a miss against it. Repeated clean looks that come back empty mean
    // the bowl isn't there (it was a false positive, or it has been moved).
    //
    // But only a frame that found most of what it expected is allowed to make
    // that claim. When the detector has an off frame and returns a handful of
    // the bowls actually in view, the honest reading is "this frame saw
    // poorly", not "those bowls are gone" — charging misses there would
    // quietly delete real bowls sitting in plain sight.
    const expected = landmarksInView(slam.landmarks, pose, local.footprint);
    const expectedHits = expected.filter(j => seen.has(j)).length;
    const informative = expected.length === 0 || expectedHits / expected.length >= MIN_EXPLAINED_FOR_PRUNE;

    let removedLandmarks = 0;
    if (informative) {
      const doomed = new Set();
      for (const j of expected) {
        if (seen.has(j)) continue;
        const landmark = slam.landmarks[j];
        landmark.misses++;
        if (landmark.misses >= MISS_THRESHOLD && isPrunable(landmark)) {
          doomed.add(j);
          removedLandmarks++;
        }
      }
      if (doomed.size > 0) {
        slam.landmarks = slam.landmarks.filter((_, j) => !doomed.has(j));
      }
    }

    slam.lastPose = pose;
    slam.poses.push(pose);
    slam.mergedCount++;

    return {
      merged: true,
      pose,
      bowlDiameterPx: local.diameter,
      newLandmarks,
      removedLandmarks,
      relocalised: estimate.relocalised,
      matched: estimate.pairs.length,
      reason: null,
    };
  }

  // --- reading the map ----------------------------------------------------

  // The jack is whichever landmark the detector has voted for most often
  // across the whole scan — a single frame mistaking a distant bowl for the
  // jack can't hijack the scoring reference.
  function findJack(landmarks) {
    let best = null;
    for (const landmark of landmarks) {
      if (landmark.jackVotes > 0 && (!best || landmark.jackVotes > best.jackVotes)) best = landmark;
    }
    return best;
  }

  // How far a landmark's averaged position might still be off, in
  // bowl-diameters — the standard error of the mean of everywhere it has been
  // seen. Looks that agree pull it down; looks that disagree push it up; more
  // looks shrink it as 1/sqrt(n), which is why moving in closer and gathering
  // more of them is genuinely worth telling someone to do.
  function positionError(landmark) {
    if (!landmark || landmark.observations < 2) return UNMEASURED_POSITION_ERROR;
    const variance = Math.max(landmark.m2, 0) / (landmark.observations - 1);
    return Math.max(Math.sqrt(variance / landmark.observations), MIN_POSITION_ERROR);
  }

  // Adjacent bowls in the ranking whose order the measurements cannot actually
  // separate. This is what turns "the app says you're up by one" into "these
  // two are too close to call from here" — the honest answer, and the same one
  // a player would reach for the measure over.
  function uncertainPairs(snapshot, opts) {
    const sigmas = (opts && opts.sigmas) || CONFIDENCE_SIGMAS;
    const pairs = [];
    for (let i = 0; i + 1 < snapshot.ranking.length; i++) {
      const near = snapshot.ranking[i];
      const far = snapshot.ranking[i + 1];
      const gap = far.dist - near.dist;
      const noise = sigmas * Math.hypot(near.sigma, far.sigma);
      if (gap < noise) {
        pairs.push({ nearIndex: i, farIndex: i + 1, gap, noise, near, far });
      }
    }
    return pairs;
  }

  // Bowls ranked by distance from the jack, shaped like a detectAndRank()
  // result so the same UI and scoring code consumes either.
  // confirmedOnly drops landmarks seen fewer than CONFIRM_OBSERVATIONS times,
  // so a one-frame false positive never gets locked into a score.
  function getSnapshot(slam, opts) {
    const confirmedOnly = opts && opts.confirmedOnly;
    const jack = findJack(slam.landmarks);

    const visible = confirmedOnly
      ? slam.landmarks.filter(b => b.observations >= CONFIRM_OBSERVATIONS)
      : slam.landmarks.slice();
    const bowls = visible.filter(b => b !== jack);

    if (!jack) {
      return {
        jack: null,
        bowls,
        ranking: [],
        usable: false,
        reason: 'jack not seen yet — point the camera at it once',
        frameCount: slam.frameCount,
        mergedCount: slam.mergedCount,
      };
    }

    // Distance uncertainty carries the jack's own error as well as the bowl's:
    // everything is measured from the jack, so if it is loosely placed then
    // every distance taken from it is loose too.
    const jackError = positionError(jack);
    const ranking = bowls
      .map(b => ({
        bowl: b,
        dist: Math.hypot(b.x - jack.x, b.y - jack.y),
        sigma: Math.hypot(positionError(b), jackError),
        confirmed: b.observations >= CONFIRM_OBSERVATIONS,
      }))
      .sort((a, b) => a.dist - b.dist);

    return {
      jack,
      bowls,
      ranking,
      usable: true,
      reason: null,
      frameCount: slam.frameCount,
      mergedCount: slam.mergedCount,
    };
  }

  // Maps a snapshot's bowl-diameter coordinates into canvas pixels (uniform
  // fit-to-canvas, fixed display radius since apparent size means nothing in
  // an abstract top-down map). Returns detections/jack/ranking shaped exactly
  // like a live detection result.
  function layoutForCanvas(snapshot, width, height, opts) {
    const bowlR = (opts && opts.bowlRadiusPx) || Math.min(width, height) * 0.035;
    const jackR = bowlR * 0.55;
    const padding = (opts && opts.paddingPx) || bowlR * 3;

    const origin = snapshot.jack || { x: 0, y: 0 };
    const points = [origin, ...snapshot.bowls];
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

    function project(p, r, identity) {
      return {
        x: width / 2 + (p.x - centerX) * scale,
        y: height / 2 + (p.y - centerY) * scale,
        r,
        identity: identity || null,
      };
    }

    const jack = project(origin, jackR);
    const bowlPoints = snapshot.bowls.map(b => project(b, bowlR, b.identity));
    const detections = [jack, ...bowlPoints];

    const ranking = snapshot.ranking.map(entry => {
      const idx = snapshot.bowls.indexOf(entry.bowl);
      return { bowl: bowlPoints[idx], dist: entry.dist, sigma: entry.sigma, confirmed: entry.confirmed };
    });

    return { detections, jack, ranking };
  }

  // Where the camera is now, in map coordinates — for drawing the operator's
  // own position on the map and for projecting mapped landmarks back into the
  // live view so bowls that have drifted out of shot can still be drawn.
  function currentPose(slam) {
    return slam.lastPose;
  }

  // Projects mapped landmarks back into the current frame's pixel space, so
  // the live overlay can show a bowl it knows about even on a frame where the
  // detector missed it. bowlDiameterPx is this frame's own scale (median
  // detected bowl diameter in pixels).
  function projectToFrame(slam, pose, bowlDiameterPx) {
    if (!pose) return [];
    return slam.landmarks.map(landmark => {
      const local = applyInversePose(pose, landmark);
      return {
        x: local.x * bowlDiameterPx,
        y: local.y * bowlDiameterPx,
        r: (bowlDiameterPx / 2) * pose.scale,
        landmark,
        confirmed: landmark.observations >= CONFIRM_OBSERVATIONS,
      };
    });
  }

  return {
    ASSOC_GATE,
    MERGE_RADIUS,
    MIN_INLIERS,
    MIN_BASELINE,
    MIN_SCALE,
    MAX_SCALE,
    CONFIRM_OBSERVATIONS,
    MISS_THRESHOLD,
    VIEW_MARGIN,
    MIN_EXPLAINED_FOR_POSE,
    MIN_EXPLAINED_FOR_PRUNE,
    MIN_INLIERS_TRACKING,
    MIN_INLIERS_RELOCALISING,
    MIN_POSITION_ERROR,
    UNMEASURED_POSITION_ERROR,
    CONFIDENCE_SIGMAS,
    MAX_PLANE_RESIDUAL,
    IDENTITY_POSE,
    createSlam,
    applyPose,
    applyInversePose,
    solveSimilarity,
    solveFromPair,
    associate,
    estimatePose,
    addFrame,
    getSnapshot,
    positionError,
    uncertainPairs,
    layoutForCanvas,
    currentPose,
    projectToFrame,
  };
});
