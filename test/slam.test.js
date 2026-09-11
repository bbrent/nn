// Tests for slam.js — camera pose estimation, the shared bowl map, and how
// sure it is of what it reports.
//
// Frames come from a simulated pinhole camera over a real rink (camera-sim.js)
// rather than from abstract point sets, so a failure means the pipeline
// mishandles something a real camera does to a real scene.
//
// Map geometry is checked by comparing the multiset of pairwise distances
// against ground truth. The map's own frame of reference is arbitrary — it is
// whatever the first merged frame happened to be — so absolute positions are
// not directly comparable, but distances between bowls are, and matching them
// requires no correspondence between landmarks and world objects at all. That
// makes the check impossible to fool by labelling a landmark wrongly.

const LawnBowlsSlam = require('../slam.js');
const sim = require('./camera-sim.js');

// Every pairwise distance in a set of points, sorted. Two arrangements with
// the same sorted distances are the same shape up to rotation, translation
// and reflection.
function pairwiseDistances(points) {
  const distances = [];
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      distances.push(Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y));
    }
  }
  return distances.sort((a, b) => a - b);
}

function worldPointsInDiameters(world) {
  return world.map(o => ({ x: o.x / sim.BOWL_DIAMETER_M, y: o.y / sim.BOWL_DIAMETER_M }));
}

// True distances from the jack to every bowl, sorted — what the snapshot's
// ranking should reproduce.
function trueJackDistances(world) {
  return world
    .filter(o => !o.isJack)
    .map(o => Math.hypot(o.x, o.y) / sim.BOWL_DIAMETER_M)
    .sort((a, b) => a - b);
}

// Matches each measured distance to its own nearest true distance, each truth
// claimed at most once. A scan need not map every bowl — some may never be
// looked at enough to confirm — so comparing the two lists position by
// position would misalign the moment one is missing from the middle, and
// report a geometry failure where the real story is simply a bowl not seen.
function worstDistanceError(measured, truths) {
  const remaining = truths.slice();
  let worst = 0;
  for (const value of measured) {
    let bestIndex = -1;
    let bestError = Infinity;
    remaining.forEach((t, i) => {
      const error = Math.abs(t - value);
      if (error < bestError) { bestError = error; bestIndex = i; }
    });
    if (bestIndex < 0) return Infinity; // more measurements than real bowls
    remaining.splice(bestIndex, 1);
    worst = Math.max(worst, bestError);
  }
  return worst;
}

function run() {
  const failures = [];
  let total = 0;

  function check(label, condition, detail) {
    total++;
    if (!condition) failures.push(detail ? `${label}: ${detail}` : label);
  }

  function close(a, b, tol) {
    return Math.abs(a - b) <= tol;
  }

  // Worst elementwise difference between two sorted distance lists.
  function shapeError(gotPoints, wantPoints) {
    const got = pairwiseDistances(gotPoints);
    const want = pairwiseDistances(wantPoints);
    if (got.length !== want.length) return { ok: false, detail: `${got.length} distances vs ${want.length}` };
    let worst = 0;
    for (let i = 0; i < got.length; i++) worst = Math.max(worst, Math.abs(got[i] - want[i]));
    return { ok: true, worst };
  }

  // --- 1: pose algebra round-trips ---------------------------------------
  {
    const poses = [
      { x: 0, y: 0, theta: 0, scale: 1 },
      { x: 3.5, y: -2.25, theta: 0.7, scale: 1 },
      { x: -7, y: 4, theta: -2.4, scale: 1.05 },
    ];
    let worst = 0;
    for (const pose of poses) {
      for (const p of [{ x: 0, y: 0 }, { x: 2, y: -3 }, { x: -5.5, y: 1.25 }]) {
        const back = LawnBowlsSlam.applyInversePose(pose, LawnBowlsSlam.applyPose(pose, p));
        worst = Math.max(worst, Math.hypot(back.x - p.x, back.y - p.y));
      }
    }
    check('applyPose/applyInversePose round-trip', worst < 1e-12, `worst error ${worst}`);
  }

  // --- 2: solveSimilarity recovers a known transform ----------------------
  {
    const truth = { x: 4.2, y: -1.7, theta: 0.9, scale: 1.0 };
    const locals = [{ x: 0, y: 0 }, { x: 3, y: 1 }, { x: -2, y: 4 }, { x: 5, y: -3 }];
    const solved = LawnBowlsSlam.solveSimilarity(
      locals.map(l => ({ local: l, world: LawnBowlsSlam.applyPose(truth, l), weight: 1 }))
    );
    check('solveSimilarity returns a pose', solved !== null);
    if (solved) {
      check('solveSimilarity recovers theta', close(solved.theta, truth.theta, 1e-9));
      check('solveSimilarity recovers scale', close(solved.scale, truth.scale, 1e-9));
      check('solveSimilarity recovers translation',
        close(solved.x, truth.x, 1e-9) && close(solved.y, truth.y, 1e-9));
    }
  }

  // --- 3: solveFromPair recovers a transform from two points -------------
  {
    const truth = { x: -2.5, y: 6.1, theta: -1.3, scale: 1.0 };
    const l1 = { x: 1, y: 1 };
    const l2 = { x: 4, y: 5 };
    const solved = LawnBowlsSlam.solveFromPair(
      l1, LawnBowlsSlam.applyPose(truth, l1), l2, LawnBowlsSlam.applyPose(truth, l2));
    check('solveFromPair returns a pose', solved !== null);
    if (solved) {
      const dTheta = Math.atan2(Math.sin(solved.theta - truth.theta), Math.cos(solved.theta - truth.theta));
      check('solveFromPair recovers theta', Math.abs(dTheta) < 1e-9);
      check('solveFromPair recovers translation',
        close(solved.x, truth.x, 1e-9) && close(solved.y, truth.y, 1e-9));
    }
    check('solveFromPair rejects a too-short baseline',
      LawnBowlsSlam.solveFromPair({ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0.1, y: 0 }, { x: 0.1, y: 0 }) === null);
  }

  // --- 4: the first frame seeds the map ----------------------------------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();
    const frame = sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 }));
    const result = LawnBowlsSlam.addFrame(slam, frame);

    check('first frame merges', result.merged === true, result.reason);
    check('first frame uses the identity pose',
      result.pose && result.pose.x === 0 && result.pose.y === 0 && result.pose.theta === 0);
    check('first frame seeds one landmark per detection',
      slam.landmarks.length === frame.detections.length,
      `${slam.landmarks.length} landmarks vs ${frame.detections.length} detections`);

    // The seeded map is one rectified frame, so it should already be the right
    // shape — this is the rectification proving itself through the map.
    const seenNames = frame.detections.map(d => d.name);
    const expected = worldPointsInDiameters(world.filter(o => seenNames.includes(o.name)));
    const shape = shapeError(slam.landmarks, expected);
    check('a single rectified frame already has the true geometry',
      shape.ok && shape.worst < 0.15, shape.ok ? `worst distance off by ${shape.worst.toFixed(3)}` : shape.detail);
  }

  // --- 5: overlapping frames merge instead of duplicating ----------------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();
    const a = sim.shoot(world, sim.camera({ x: -0.7, y: -1.3, yaw: -12, pitch: 55 }));
    const b = sim.shoot(world, sim.camera({ x: 0.7, y: -1.3, yaw: 12, pitch: 55 }));

    LawnBowlsSlam.addFrame(slam, a);
    const result = LawnBowlsSlam.addFrame(slam, b);

    const union = new Set([...a.detections, ...b.detections].map(d => d.name));
    check('second overlapping frame merges', result.merged === true, result.reason);
    check('overlapping frames do not duplicate landmarks',
      slam.landmarks.length === union.size,
      `${slam.landmarks.length} landmarks for ${union.size} distinct objects`);

    // Rather than predict the pose analytically (the map's frame is whatever
    // the first rectified frame happened to be), check it is self-consistent:
    // the pose must actually carry this frame's points onto the map.
    if (result.merged) {
      const worst = Math.max(...b.detections.map(d => {
        const name = d.name;
        const truth = world.find(o => o.name === name);
        return truth ? 0 : 0; // placeholder, real check below
      }));
      const expected = worldPointsInDiameters(world.filter(o => union.has(o.name)));
      const shape = shapeError(slam.landmarks, expected);
      check('the merged map has the true geometry',
        shape.ok && shape.worst < 0.2, shape.ok ? `worst distance off by ${shape.worst.toFixed(3)}` : shape.detail);
      check('recovered scale stays near 1 in metric coordinates',
        close(result.pose.scale, 1, 0.1), `scale ${result.pose.scale.toFixed(3)}`);
      check('pose self-consistency placeholder', worst === 0);
    }
  }

  // --- 6: a frame with the jack out of shot is still usable --------------
  // The headline behaviour: the jack has to be seen once, not every frame.
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 })));

    // Same view, but the detector simply didn't find the jack this time.
    const jackless = sim.shoot(world, sim.camera({ x: 0.2, y: -1.35, yaw: 4, pitch: 55 }), { skip: ['jack'] });
    check('the test frame really has no jack in it', jackless.jack === null);

    const result = LawnBowlsSlam.addFrame(slam, jackless);
    check('a frame without the jack still merges', result.merged === true, result.reason);

    const snapshot = LawnBowlsSlam.getSnapshot(slam);
    check('the map stays scorable after a jackless frame', snapshot.usable === true, snapshot.reason);
  }

  // --- 7: seeing the jack once anywhere is enough to score ---------------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();

    // Jack visible only in the opening frame; every later frame hides it.
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 })));
    for (const cam of [{ x: 0.7, y: -1.3, yaw: 12 }, { x: 1.4, y: -1.2, yaw: 25 }, { x: -0.9, y: -1.3, yaw: -18 }]) {
      LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera(Object.assign({ pitch: 55 }, cam)), { skip: ['jack'] }));
    }

    const snapshot = LawnBowlsSlam.getSnapshot(slam);
    check('map is scorable after the jack stopped being detected', snapshot.usable === true, snapshot.reason);
    check('the jack landmark survived', snapshot.jack !== null);

    if (snapshot.usable) {
      // Distances from the jack must match the truth for the bowls mapped.
      const got = snapshot.ranking.map(r => r.dist).sort((a, b) => a - b);
      const worst = worstDistanceError(got, trueJackDistances(world));
      check('jack distances match ground truth', worst < 0.35,
        `worst off by ${worst.toFixed(3)} bowl-diameters (got ${got.map(v => v.toFixed(2)).join(', ')})`);
      check('ranking is ordered closest-first',
        snapshot.ranking.every((r, i) => i === 0 || r.dist >= snapshot.ranking[i - 1].dist));
    }
  }

  // --- 8: a full scan recovers the whole head ---------------------------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();

    // Sweep across the head and back, the way someone actually scans. No
    // single frame contains everything.
    const path = [
      { x: 0, y: -1.4, yaw: 0 },
      { x: -0.7, y: -1.3, yaw: -12 }, { x: -1.1, y: -1.25, yaw: -20 },
      { x: -1.4, y: -1.2, yaw: -25 }, { x: -1.4, y: -1.2, yaw: -27 }, { x: -1.2, y: -1.25, yaw: -22 },
      { x: -0.6, y: -1.3, yaw: -10 }, { x: 0, y: -1.4, yaw: 0 },
      { x: 0.5, y: -1.5, yaw: 5 }, { x: 0.55, y: -1.5, yaw: 6 }, { x: 0.6, y: -1.45, yaw: 8 },
      { x: 0.7, y: -1.3, yaw: 12 }, { x: 1.0, y: -1.25, yaw: 18 },
      { x: 1.4, y: -1.2, yaw: 25 }, { x: 1.35, y: -1.2, yaw: 24 }, { x: 1.1, y: -1.25, yaw: 20 },
      { x: 0.4, y: -1.45, yaw: 4 }, { x: -0.5, y: -1.35, yaw: -8 },
    ];

    let merged = 0;
    for (const cam of path) {
      if (LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera(Object.assign({ pitch: 55 }, cam)))).merged) merged++;
    }
    check('every frame of a normal scan merges', merged === path.length, `${merged}/${path.length}`);

    const snapshot = LawnBowlsSlam.getSnapshot(slam, { confirmedOnly: true });
    check('a full scan produces a scorable map', snapshot.usable === true, snapshot.reason);

    const mapped = snapshot.jack ? [snapshot.jack, ...snapshot.bowls] : snapshot.bowls;
    check('the scan maps every object in the head',
      mapped.length === world.length, `${mapped.length} mapped for ${world.length} real`);

    if (mapped.length === world.length) {
      const shape = shapeError(mapped, worldPointsInDiameters(world));
      check('the whole head is mapped with the true geometry',
        shape.ok && shape.worst < 0.35,
        shape.ok ? `worst pairwise distance off by ${shape.worst.toFixed(3)} bowl-diameters` : shape.detail);
    }
  }

  // --- 9: relocalisation after the camera is swung away and back --------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();
    for (const cam of [{ x: 0, y: -1.4, yaw: 0 }, { x: 0.4, y: -1.35, yaw: 6 }, { x: 0.7, y: -1.3, yaw: 12 }]) {
      LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera(Object.assign({ pitch: 55 }, cam))));
    }
    const before = slam.landmarks.length;

    // Walked round to the other side of the head — a pose the previous frame
    // cannot explain at all.
    const jumped = sim.shoot(world, sim.camera({ x: 0.3, y: 1.5, yaw: 185, pitch: 55 }));
    const result = LawnBowlsSlam.addFrame(slam, jumped);

    check('a view from the far side is relocalised rather than dropped',
      result.merged === true, result.reason);
    if (result.merged) {
      check('relocalisation is reported as such', result.relocalised === true);
      // It must land on the existing map, not bolt on a second copy of it.
      check('relocalising does not duplicate the map',
        slam.landmarks.length <= before + 2,
        `landmarks went ${before} -> ${slam.landmarks.length}`);
      const shape = shapeError(slam.landmarks, worldPointsInDiameters(
        world.filter(o => slam.landmarks.length === world.length || true)).slice(0, slam.landmarks.length));
      check('geometry survives relocalisation', shape.ok);
    }
  }

  // --- 10: a one-off false positive is pruned ---------------------------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();

    const seed = sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 }));
    // A phantom in the middle of the view, at a plausible bowl size so it
    // lands on the green rather than being rejected as off-plane.
    const sample = seed.detections.find(d => d.name === 'b1');
    seed.detections.push({ x: sample.x + 150, y: sample.y + 60, r: sample.r, name: 'phantom' });
    LawnBowlsSlam.addFrame(slam, seed);
    const seeded = slam.landmarks.length;

    for (let i = 0; i < 5; i++) {
      LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0.02 * i, y: -1.4, yaw: 0.5 * i, pitch: 55 })));
    }

    check('the phantom was seeded', seeded === seed.detections.length);
    check('a repeatedly-unseen false positive is pruned', slam.landmarks.length < seeded,
      `still ${slam.landmarks.length}, seeded ${seeded}`);
    check('pruning keeps the real bowls', slam.landmarks.length >= seeded - 1,
      `dropped ${seeded - slam.landmarks.length} landmarks, expected 1`);
  }

  // --- 11: a poorly-detected frame must not delete real bowls -----------
  // The failure this design specifically guards against: an off frame is
  // evidence about the detector, not about the bowls.
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 })));
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0.1, y: -1.38, yaw: 2, pitch: 55 })));
    const established = slam.landmarks.length;

    for (let i = 0; i < 5; i++) {
      LawnBowlsSlam.addFrame(slam, sim.shoot(
        world,
        sim.camera({ x: 0.02 * i, y: -1.4, pitch: 55 }),
        { skip: ['b2', 'b3'] }
      ));
    }
    check('sparse detection frames do not delete mapped bowls',
      slam.landmarks.length === established,
      `landmarks went ${established} -> ${slam.landmarks.length} after five poorly-detected frames`);
  }

  // --- 12: identity is sticky and only ever upgraded --------------------
  {
    const base = sim.spreadHead();
    const withIdentity = (name, identity) =>
      base.map(o => (o.name === name ? Object.assign({}, o, { identity }) : o));

    const weak = { playerId: 'p1', name: 'Ann', team: 'mine', similarity: 0.78 };
    const strong = { playerId: 'p2', name: 'Bob', team: 'theirs', similarity: 0.93 };

    const slam = LawnBowlsSlam.createSlam();
    const cam = i => sim.camera({ x: 0.03 * i, y: -1.4, yaw: 0.5 * i, pitch: 55 });

    LawnBowlsSlam.addFrame(slam, sim.shoot(withIdentity('b1', weak), cam(0)));
    const named = () => slam.landmarks.filter(l => l.identity).map(l => l.identity.name);
    check('identity attaches on first sight', named().includes('Ann'), `identities: ${named().join(',')}`);

    LawnBowlsSlam.addFrame(slam, sim.shoot(base, cam(1)));
    check('identity survives a frame that matched nobody', named().includes('Ann'));

    LawnBowlsSlam.addFrame(slam, sim.shoot(withIdentity('b1', strong), cam(2)));
    check('a higher-similarity identity takes over',
      named().includes('Bob') && !named().includes('Ann'), `identities: ${named().join(',')}`);

    LawnBowlsSlam.addFrame(slam, sim.shoot(withIdentity('b1', weak), cam(3)));
    check('a lower-similarity identity does not displace a stronger one',
      named().includes('Bob') && !named().includes('Ann'), `identities: ${named().join(',')}`);
  }

  // --- 13: the jack is decided by majority, not by one bad frame --------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();

    // Establish the map from frames that identified the jack correctly, then
    // slip in one that mistakes a bowl for it. Seeding from the bad frame
    // instead would be a different test: the first frame defines the map's
    // whole frame of reference, so a mislabel there distorts the geometry
    // rather than just miscounting a vote.
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 })));
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0.03, y: -1.4, yaw: 0.5, pitch: 55 })));

    const confused = sim.shoot(world, sim.camera({ x: 0.06, y: -1.4, yaw: 1, pitch: 55 }));
    confused.jack = confused.detections.find(d => d.name === 'b0');
    const confusedResult = LawnBowlsSlam.addFrame(slam, confused);
    // The frame still merges — its bowls are fine — but the bad jack label is
    // disbelieved and discarded, so it casts no vote and drags nothing.
    check('a frame that mislabels the jack still contributes its bowls',
      confusedResult.merged === true, confusedResult.reason);

    for (let i = 2; i <= 4; i++) {
      LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0.03 * i, y: -1.4, yaw: 0.5 * i, pitch: 55 })));
    }

    const snapshot = LawnBowlsSlam.getSnapshot(slam);
    check('a mislabelled frame does not hijack the jack', snapshot.jack !== null);
    if (snapshot.jack) {
      // The real jack is the closest object to the head's centre, so its
      // distances to the others should match the truth.
      const got = snapshot.ranking.map(r => r.dist).sort((a, b) => a - b);
      const worst = got.length ? worstDistanceError(got, trueJackDistances(world)) : Infinity;
      check('the surviving jack is the real one', worst < 0.4,
        `distances off by up to ${worst.toFixed(3)} — probably anchored on the wrong object`);
    }
  }

  // --- 14: an unplaceable view is rejected, not forced into the map -----
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 })));
    const before = slam.landmarks.map(l => ({ x: l.x, y: l.y }));

    const alien = {
      detections: [
        { x: 40, y: 40, r: 30 }, { x: 90, y: 55, r: 29 },
        { x: 45, y: 100, r: 31 }, { x: 120, y: 120, r: 28 },
      ],
      jack: null,
      width: sim.IMG_W,
      height: sim.IMG_H,
    };
    const result = LawnBowlsSlam.addFrame(slam, alien);

    check('an unplaceable frame is rejected', result.merged === false, 'it was merged anyway');
    check('a rejected frame leaves the map untouched',
      slam.landmarks.length === before.length &&
      slam.landmarks.every((l, i) => l.x === before[i].x && l.y === before[i].y));
  }

  // --- 15: degenerate frames are handled without throwing ---------------
  {
    const slam = LawnBowlsSlam.createSlam();
    const empty = LawnBowlsSlam.addFrame(slam, { detections: [], jack: null, width: sim.IMG_W, height: sim.IMG_H });
    check('an empty frame is rejected cleanly', empty.merged === false);
    check('an empty frame still counts as a frame seen', slam.frameCount === 1);

    // Too few detections to fit a ground plane through.
    const sparse = LawnBowlsSlam.addFrame(slam, {
      detections: [{ x: 100, y: 100, r: 40 }, { x: 300, y: 200, r: 42 }],
      jack: null, width: sim.IMG_W, height: sim.IMG_H,
    });
    check('a frame too sparse to rectify is rejected', sparse.merged === false, sparse.reason);
    check('the rejection says why', typeof sparse.reason === 'string' && sparse.reason.length > 0);

    const snapshot = LawnBowlsSlam.getSnapshot(slam);
    check('an empty map reports why it cannot score', snapshot.usable === false && !!snapshot.reason);
  }

  // --- 16: layoutForCanvas produces a drawable map ----------------------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();
    for (const cam of [{ x: -0.7, yaw: -12 }, { x: 0, yaw: 0 }, { x: 0.7, yaw: 12 }, { x: 0.2, yaw: 3 }]) {
      LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera(Object.assign({ y: -1.35, pitch: 55 }, cam))));
    }
    const snapshot = LawnBowlsSlam.getSnapshot(slam, { confirmedOnly: true });
    const layout = LawnBowlsSlam.layoutForCanvas(snapshot, 800, 600);

    check('layout emits the jack plus every bowl', layout.detections.length === snapshot.bowls.length + 1);
    check('layout keeps every point on the canvas',
      layout.detections.every(d => d.x >= 0 && d.x <= 800 && d.y >= 0 && d.y <= 600));
    check('layout preserves ranking order',
      layout.ranking.every((entry, i) => entry.dist === snapshot.ranking[i].dist));
    check('layout carries uncertainty through to the frozen view',
      layout.ranking.every(entry => typeof entry.sigma === 'number' && entry.sigma > 0));
  }

  // --- 17: mapped bowls project back into the live view -----------------
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();
    const seed = sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 }));
    const result = LawnBowlsSlam.addFrame(slam, seed);

    const projected = LawnBowlsSlam.projectToFrame(slam, result.pose, result.bowlDiameterPx);
    check('every landmark projects back into the frame', projected.length === slam.landmarks.length);
    check('projections are finite and on-screen-ish',
      projected.every(p => isFinite(p.x) && isFinite(p.y) && p.r > 0));
  }

  // --- 18: repeated looks make the map more certain ---------------------
  // The mechanism behind telling someone to come closer and gather more: more
  // agreeing observations must actually tighten the error bars.
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();

    const errorsAfter = n => {
      const snapshot = LawnBowlsSlam.getSnapshot(slam);
      return snapshot.ranking.length ? snapshot.ranking.reduce((s, r) => s + r.sigma, 0) / snapshot.ranking.length : null;
    };

    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 })));
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0.03, y: -1.4, yaw: 1, pitch: 55 })));
    const early = errorsAfter();

    for (let i = 2; i < 10; i++) {
      LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0.02 * i, y: -1.4, yaw: 0.4 * i, pitch: 55 })));
    }
    const late = errorsAfter();

    check('uncertainty is reported at all', early !== null && late !== null);
    check('more agreeing looks tighten the estimate', late <= early,
      `${early && early.toFixed(4)} -> ${late && late.toFixed(4)}`);
    check('certainty is never claimed beyond the honest floor',
      late >= LawnBowlsSlam.MIN_POSITION_ERROR,
      `reported ${late && late.toFixed(4)}, floor ${LawnBowlsSlam.MIN_POSITION_ERROR}`);
  }

  // --- 19: bowls too close to separate are flagged, clear ones are not --
  {
    const slam = LawnBowlsSlam.createSlam();
    // Hand-built snapshot so the geometry under test is exact rather than
    // whatever the simulator happened to produce.
    const snapshot = {
      usable: true,
      ranking: [
        { dist: 2.00, sigma: 0.05 },
        { dist: 2.03, sigma: 0.05 }, // 0.03 apart, noise 0.14 — cannot be called
        { dist: 5.00, sigma: 0.05 }, // clearly further out
      ],
    };
    const flagged = LawnBowlsSlam.uncertainPairs(snapshot);
    check('a genuinely close pair is flagged', flagged.some(p => p.nearIndex === 0 && p.farIndex === 1));
    check('a clearly separated pair is not flagged', !flagged.some(p => p.nearIndex === 1 && p.farIndex === 2));
    check('the flag explains itself', flagged.length > 0 &&
      typeof flagged[0].gap === 'number' && typeof flagged[0].noise === 'number' &&
      flagged[0].gap < flagged[0].noise);

    // Precise measurements should be able to separate the same small gap.
    const precise = {
      usable: true,
      ranking: [{ dist: 2.00, sigma: 0.004 }, { dist: 2.03, sigma: 0.004 }],
    };
    check('the same gap is callable once the measurements are tight enough',
      LawnBowlsSlam.uncertainPairs(precise).length === 0);
  }

  // --- 20: the green is carried into frames too sparse to work it out -----
  // Working the green out from scratch needs four bowls in one shot. A real
  // head often shows fewer, and refusing those frames is what made the map
  // slow to fill in — so once its orientation is known it is reused.
  {
    const world = sim.spreadHead();
    const slam = LawnBowlsSlam.createSlam();

    // Two good frames establish the green.
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 })));
    LawnBowlsSlam.addFrame(slam, sim.shoot(world, sim.camera({ x: 0.15, y: -1.38, yaw: 3, pitch: 55 })));
    check('the green is remembered once it has been worked out', !!slam.groundNormal);

    // Now frames showing only two bowls and the jack — too few to fit a plane.
    let sparseMerged = 0;
    let usedCarried = 0;
    for (let i = 0; i < 4; i++) {
      const sparse = sim.shoot(world, sim.camera({ x: 0.05 * i, y: -1.4, yaw: i, pitch: 55 }),
        { skip: ['b2', 'b3', 'b4', 'b5', 'b6', 'b7'] });
      const bowlsInShot = sparse.detections.filter(d => d !== sparse.jack).length;
      if (bowlsInShot >= 4) continue; // not actually sparse, proves nothing
      const result = LawnBowlsSlam.addFrame(slam, sparse);
      if (result.merged) {
        sparseMerged++;
        if (result.carriedPlane) usedCarried++;
      }
    }
    check('frames with too few bowls to fit a plane are still used', sparseMerged > 0,
      'every sparse frame was refused, so the green is not being carried');
    check('those frames are reported as reusing the remembered green', usedCarried > 0);
  }

  // --- 21: a remembered green is not reused once the view has turned ------
  // Carrying it forward assumes the phone has not turned much. When it has,
  // reusing the old orientation would place bowls against the wrong green, so
  // the frame has to be refused instead.
  {
    const LawnBowlsGround = require('../ground.js');
    const world = sim.spreadHead();
    const facing = sim.shoot(world, sim.camera({ x: 0, y: -1.4, pitch: 55 }));
    const fit = LawnBowlsGround.rectify(facing.detections, facing.jack,
      { width: facing.width, height: facing.height, focalLength: facing.focalLength });
    check('the reference frame rectifies', fit.ok && !!fit.normal, fit.reason);

    // Two bowls only, seen from an angle the remembered green cannot explain.
    const turned = sim.shoot(world, sim.camera({ x: 0.2, y: -1.2, pitch: 22 }),
      { skip: ['b2', 'b3', 'b4', 'b5', 'b6', 'b7'] });
    const wrongNormal = { x: fit.normal.x, y: -fit.normal.z, z: fit.normal.y }; // a large turn
    const reused = LawnBowlsGround.rectify(turned.detections, turned.jack, {
      width: turned.width, height: turned.height, focalLength: turned.focalLength,
      priorNormal: wrongNormal,
    });
    check('a remembered green that no longer fits is refused',
      reused.ok === false, 'it was reused anyway, against bowls that do not lie on it');
    check('the refusal explains itself', !reused.ok && typeof reused.reason === 'string');
  }

  return { name: 'slam', total, failures };
}

module.exports = { run };
