// Tests for slam.js — camera pose estimation + shared bowl map.
//
// These drive the module through a simulated camera rather than synthetic
// point sets: a ground-truth world layout is defined once, and each "frame"
// is rendered by working out exactly which objects a camera at a given pose
// would see and where they would land in its image. That means a test failure
// points at real behaviour (a pan that loses the map, a bowl deleted because
// one frame detected poorly) rather than at an abstract matrix identity.

const LawnBowlsSlam = require('../slam.js');

const VIEW_W = 1280;
const VIEW_H = 720;
const DIAMETER_PX = 100; // pixels per bowl diameter

// Renders the frame a camera centred on (cam.x, cam.y) and rotated by
// cam.theta would capture: every world object that falls inside the image,
// at its correct pixel position. Also returns the true pose so tests can
// check what the estimator recovered.
function renderFrame(world, cam, opts) {
  const options = opts || {};
  const cos = Math.cos(cam.theta);
  const sin = Math.sin(cam.theta);
  const localCenter = { x: VIEW_W / (2 * DIAMETER_PX), y: VIEW_H / (2 * DIAMETER_PX) };
  const t = {
    x: cam.x - (localCenter.x * cos - localCenter.y * sin),
    y: cam.y - (localCenter.x * sin + localCenter.y * cos),
  };

  const detections = [];
  let jack = null;
  for (const obj of world) {
    if (options.skip && options.skip.includes(obj.name)) continue;
    const dx = obj.x - t.x;
    const dy = obj.y - t.y;
    const lx = dx * cos + dy * sin;
    const ly = -dx * sin + dy * cos;
    const px = lx * DIAMETER_PX;
    const py = ly * DIAMETER_PX;
    if (px < 0 || py < 0 || px > VIEW_W || py > VIEW_H) continue;

    const det = {
      x: px,
      y: py,
      r: obj.isJack ? DIAMETER_PX * 0.3 : DIAMETER_PX / 2,
      name: obj.name,
    };
    if (obj.identity) det.identity = obj.identity;
    detections.push(det);
    if (obj.isJack && !options.hideJack) jack = det;
  }

  return {
    detections,
    jack,
    width: VIEW_W,
    height: VIEW_H,
    truePose: { x: t.x, y: t.y, theta: cam.theta, scale: 1 },
  };
}

// Ground truth expressed in the map's own coordinates. The first frame merged
// defines the world frame, so anything the map reports can be compared
// directly against world points pushed through that frame's inverse pose.
function toMapCoords(firstFrame, worldPoint) {
  return LawnBowlsSlam.applyInversePose(firstFrame.truePose, worldPoint);
}

// The pose the estimator *should* report for a frame. Not the camera's
// absolute pose in the simulator's world: the map's origin is the first
// merged frame's own local frame, so the truth to compare against is the
// camera pose composed through that frame's inverse. Read off by mapping the
// local origin and a local unit vector, rather than composing the transform
// algebra by hand, so the expectation can't share a bug with the code it
// checks.
function expectedMapPose(firstFrame, frame) {
  const origin = toMapCoords(firstFrame, LawnBowlsSlam.applyPose(frame.truePose, { x: 0, y: 0 }));
  const unit = toMapCoords(firstFrame, LawnBowlsSlam.applyPose(frame.truePose, { x: 1, y: 0 }));
  const dx = unit.x - origin.x;
  const dy = unit.y - origin.y;
  return { x: origin.x, y: origin.y, theta: Math.atan2(dy, dx), scale: Math.hypot(dx, dy) };
}

// A rink wider than any single frame can capture: the image spans 12.8 x 7.2
// bowl-diameters, these bowls span roughly 19 x 7. Scanning it necessarily
// means letting bowls leave the shot.
function wideRink() {
  return [
    { name: 'jack', x: 0, y: 0, isJack: true },
    { name: 'b1', x: 1.4, y: 0.9 },
    { name: 'b2', x: -1.8, y: 1.3 },
    { name: 'b3', x: 2.9, y: -1.6 },
    { name: 'b4', x: -3.7, y: -0.8 },
    { name: 'b5', x: 5.6, y: 2.1 },
    { name: 'b6', x: -6.2, y: 1.9 },
    { name: 'b7', x: 8.1, y: -1.2 },
    { name: 'b8', x: -8.8, y: -1.7 },
  ];
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

  // --- 1: pose algebra round-trips ---------------------------------------
  {
    const poses = [
      { x: 0, y: 0, theta: 0, scale: 1 },
      { x: 3.5, y: -2.25, theta: 0.7, scale: 1 },
      { x: -7, y: 4, theta: -2.4, scale: 1.15 },
    ];
    const points = [{ x: 0, y: 0 }, { x: 2, y: -3 }, { x: -5.5, y: 1.25 }];
    let worst = 0;
    for (const pose of poses) {
      for (const p of points) {
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
    const pairs = locals.map(l => ({ local: l, world: LawnBowlsSlam.applyPose(truth, l), weight: 1 }));
    const solved = LawnBowlsSlam.solveSimilarity(pairs);

    check('solveSimilarity returns a pose', solved !== null);
    if (solved) {
      check('solveSimilarity recovers theta', close(solved.theta, truth.theta, 1e-9), `got ${solved.theta}`);
      check('solveSimilarity recovers scale', close(solved.scale, truth.scale, 1e-9), `got ${solved.scale}`);
      check('solveSimilarity recovers translation',
        close(solved.x, truth.x, 1e-9) && close(solved.y, truth.y, 1e-9),
        `got (${solved.x}, ${solved.y})`);
    }
  }

  // --- 3: solveFromPair recovers a known transform from two points --------
  {
    const truth = { x: -2.5, y: 6.1, theta: -1.3, scale: 1.0 };
    const l1 = { x: 1, y: 1 };
    const l2 = { x: 4, y: 5 };
    const solved = LawnBowlsSlam.solveFromPair(l1, LawnBowlsSlam.applyPose(truth, l1), l2, LawnBowlsSlam.applyPose(truth, l2));

    check('solveFromPair returns a pose', solved !== null);
    if (solved) {
      const dTheta = Math.atan2(Math.sin(solved.theta - truth.theta), Math.cos(solved.theta - truth.theta));
      check('solveFromPair recovers theta', Math.abs(dTheta) < 1e-9, `off by ${dTheta}`);
      check('solveFromPair recovers translation',
        close(solved.x, truth.x, 1e-9) && close(solved.y, truth.y, 1e-9),
        `got (${solved.x}, ${solved.y})`);
    }
    check('solveFromPair rejects a too-short baseline',
      LawnBowlsSlam.solveFromPair({ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0.1, y: 0 }, { x: 0.1, y: 0 }) === null);
  }

  // --- 4: first frame seeds the map at the identity pose ------------------
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    const frame = renderFrame(world, { x: 0, y: 0, theta: 0 });
    const result = LawnBowlsSlam.addFrame(slam, frame);

    check('first frame merges', result.merged === true, result.reason);
    check('first frame uses the identity pose',
      result.pose && result.pose.x === 0 && result.pose.y === 0 && result.pose.theta === 0 && result.pose.scale === 1);
    check('first frame seeds one landmark per detection',
      slam.landmarks.length === frame.detections.length,
      `${slam.landmarks.length} landmarks vs ${frame.detections.length} detections`);
  }

  // --- 5: overlapping frames merge instead of duplicating -----------------
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    const a = renderFrame(world, { x: 0, y: 0, theta: 0 });
    const b = renderFrame(world, { x: 2.5, y: 0.4, theta: 0.15 });

    LawnBowlsSlam.addFrame(slam, a);
    const result = LawnBowlsSlam.addFrame(slam, b);

    const namesSeen = new Set([...a.detections, ...b.detections].map(d => d.name));
    check('second overlapping frame merges', result.merged === true, result.reason);
    check('overlapping frames do not duplicate landmarks',
      slam.landmarks.length === namesSeen.size,
      `${slam.landmarks.length} landmarks for ${namesSeen.size} distinct objects`);
    const wantPose = expectedMapPose(a, b);
    check('second frame pose matches the true camera motion',
      result.pose && close(result.pose.x, wantPose.x, 0.05) && close(result.pose.y, wantPose.y, 0.05),
      result.pose ? `got (${result.pose.x.toFixed(3)}, ${result.pose.y.toFixed(3)}) want (${wantPose.x.toFixed(3)}, ${wantPose.y.toFixed(3)})` : 'no pose');
    check('second frame pose matches the true camera rotation',
      result.pose && close(result.pose.theta, wantPose.theta, 0.02),
      result.pose ? `got ${result.pose.theta.toFixed(4)} want ${wantPose.theta.toFixed(4)}` : 'no pose');
  }

  // --- 6: a frame with the jack out of shot is still usable ---------------
  // The headline behaviour. The old jack-anchored fusion discarded any frame
  // without the jack in it, which is exactly what made scanning fiddly.
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 0, y: 0, theta: 0 }));

    const jackless = renderFrame(world, { x: 7.5, y: 0.3, theta: 0.1 });
    check('the test frame really has no jack in it', jackless.jack === null,
      'simulated frame still contains the jack — test setup is wrong');

    const before = slam.landmarks.length;
    const result = LawnBowlsSlam.addFrame(slam, jackless);
    check('a frame without the jack still merges', result.merged === true, result.reason);
    check('a jackless frame still contributes new bowls', slam.landmarks.length > before,
      `landmarks went ${before} -> ${slam.landmarks.length}`);
  }

  // --- 7: seeing the jack once anywhere is enough to score ----------------
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    // Jack visible only in the very first frame; the pan then moves away.
    const frames = [
      { x: 0, y: 0, theta: 0 },
      { x: 3.0, y: 0.2, theta: 0.05 },
      { x: 6.0, y: 0.1, theta: 0.1 },
      { x: 8.5, y: 0.0, theta: 0.05 },
    ];
    frames.forEach(cam => LawnBowlsSlam.addFrame(slam, renderFrame(world, cam)));

    const snapshot = LawnBowlsSlam.getSnapshot(slam);
    check('map is scorable after the jack left the frame', snapshot.usable === true, snapshot.reason);
    check('the jack landmark was identified', snapshot.jack !== null);

    if (snapshot.usable) {
      // Ranking must match true distance order for the bowls actually mapped.
      const truthByName = new Map(world.map(o => [o.name, o]));
      const ranked = snapshot.ranking.map(entry => {
        // Recover which world object this landmark is, by nearest true position.
        let best = null;
        let bestDist = Infinity;
        for (const obj of world) {
          if (obj.isJack) continue;
          const truthInMap = toMapCoords(renderFrame(world, frames[0]), obj);
          const d = Math.hypot(entry.bowl.x - truthInMap.x, entry.bowl.y - truthInMap.y);
          if (d < bestDist) { bestDist = d; best = obj; }
        }
        return { name: best.name, mapDist: entry.dist, trueDist: Math.hypot(best.x, best.y), err: bestDist };
      });

      const wellPlaced = ranked.every(r => r.err < 0.25);
      check('ranked bowls sit at their true positions', wellPlaced,
        ranked.map(r => `${r.name} off by ${r.err.toFixed(3)}`).join(', '));

      const distancesOk = ranked.every(r => close(r.mapDist, r.trueDist, 0.3));
      check('jack distances match ground truth', distancesOk,
        ranked.map(r => `${r.name}: map ${r.mapDist.toFixed(2)} vs true ${r.trueDist.toFixed(2)}`).join(', '));

      const order = ranked.map(r => r.mapDist);
      const sorted = order.slice().sort((p, q) => p - q);
      check('ranking is ordered closest-first', order.every((v, i) => v === sorted[i]));
    }
  }

  // --- 8: a full pan recovers the whole rink -----------------------------
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    const firstFrame = renderFrame(world, { x: 0, y: 0, theta: 0 });

    // Sweep right, come back through the middle, then sweep left — the way
    // someone actually scans, with plenty of overlap but never everything.
    const path = [
      { x: 0, y: 0, theta: 0 },
      { x: 2.2, y: 0.3, theta: 0.06 },
      { x: 4.4, y: 0.1, theta: 0.12 },
      { x: 6.6, y: -0.2, theta: 0.05 },
      { x: 8.2, y: 0.0, theta: -0.03 },
      { x: 5.0, y: 0.2, theta: 0.0 },
      { x: 1.0, y: 0.1, theta: -0.05 },
      { x: -2.5, y: 0.3, theta: -0.1 },
      { x: -5.0, y: 0.0, theta: -0.06 },
      { x: -7.5, y: -0.2, theta: 0.0 },
      { x: -9.0, y: 0.1, theta: 0.04 },
    ];

    let merged = 0;
    for (const cam of path) {
      const r = LawnBowlsSlam.addFrame(slam, renderFrame(world, cam));
      if (r.merged) merged++;
    }

    check('every frame of a normal pan merges', merged === path.length,
      `${merged}/${path.length} merged`);

    const snapshot = LawnBowlsSlam.getSnapshot(slam, { confirmedOnly: true });
    check('full pan produces a scorable map', snapshot.usable === true, snapshot.reason);

    // Every bowl in the rink should be present, at its true position.
    const bowls = world.filter(o => !o.isJack);
    const missing = [];
    let worstErr = 0;
    for (const obj of bowls) {
      const want = toMapCoords(firstFrame, obj);
      let bestDist = Infinity;
      for (const landmark of snapshot.bowls) {
        bestDist = Math.min(bestDist, Math.hypot(landmark.x - want.x, landmark.y - want.y));
      }
      if (bestDist > 0.3) missing.push(`${obj.name} (nearest landmark ${bestDist.toFixed(2)} away)`);
      worstErr = Math.max(worstErr, bestDist);
    }
    check('full pan maps every bowl in the rink', missing.length === 0, missing.join(', '));
    check('mapped positions stay accurate across the pan', worstErr < 0.3,
      `worst position error ${worstErr.toFixed(3)} bowl-diameters`);
    check('full pan invents no extra bowls',
      snapshot.bowls.length === bowls.length,
      `${snapshot.bowls.length} confirmed bowls for ${bowls.length} real ones`);
  }

  // --- 9: relocalisation after the camera is swung away and back ----------
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    const firstFrame = renderFrame(world, { x: 0, y: 0, theta: 0 });
    LawnBowlsSlam.addFrame(slam, firstFrame);
    LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 2.0, y: 0.2, theta: 0.05 }));
    LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 4.0, y: 0.1, theta: 0.05 }));

    // A big rotation the previous pose cannot possibly explain — the phone
    // was turned around and brought back.
    const jumped = renderFrame(world, { x: -1.0, y: 0.0, theta: 2.6 });
    const result = LawnBowlsSlam.addFrame(slam, jumped);

    check('a jumped view is relocalised rather than dropped', result.merged === true, result.reason);
    check('relocalisation is reported as such', result.merged && result.relocalised === true);
    if (result.merged) {
      const want = expectedMapPose(firstFrame, jumped);
      const dTheta = Math.atan2(
        Math.sin(result.pose.theta - want.theta),
        Math.cos(result.pose.theta - want.theta)
      );
      check('relocalised pose recovers the true rotation', Math.abs(dTheta) < 0.08,
        `off by ${dTheta.toFixed(4)} rad`);
      check('relocalised pose recovers the true position',
        close(result.pose.x, want.x, 0.3) && close(result.pose.y, want.y, 0.3),
        `got (${result.pose.x.toFixed(2)}, ${result.pose.y.toFixed(2)}) want (${want.x.toFixed(2)}, ${want.y.toFixed(2)})`);
    }
  }

  // --- 10: a one-off false positive is pruned -----------------------------
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();

    const seed = renderFrame(world, { x: 0, y: 0, theta: 0 });
    // A phantom detection (a shoe, a bag) in the middle of the view, well
    // clear of every real bowl so it can't be confused with one.
    seed.detections.push({ x: VIEW_W / 2 + 260, y: VIEW_H / 2 + 210, r: DIAMETER_PX / 2, name: 'phantom' });
    LawnBowlsSlam.addFrame(slam, seed);
    const seeded = slam.landmarks.length;

    // Keep looking at the same place, cleanly, several times over.
    for (let i = 0; i < 4; i++) {
      LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 0.05 * i, y: 0.02 * i, theta: 0.01 * i }));
    }

    check('the phantom was actually seeded', seeded === seed.detections.length);
    check('a repeatedly-unseen false positive is pruned',
      slam.landmarks.length < seeded,
      `still ${slam.landmarks.length} landmarks, seeded ${seeded}`);

    // And the real bowls must survive that same pruning.
    const firstFrame = renderFrame(world, { x: 0, y: 0, theta: 0 });
    const survived = firstFrame.detections.filter(d => d.name !== 'phantom').every(d => {
      const want = { x: d.x / DIAMETER_PX, y: d.y / DIAMETER_PX };
      return slam.landmarks.some(l => Math.hypot(l.x - want.x, l.y - want.y) < 0.3);
    });
    check('pruning keeps every real bowl', survived);
  }

  // --- 11: a poorly-detected frame must not delete real bowls ------------
  // Regression guard for the failure mode this design specifically avoids:
  // the detector having an off frame is evidence about the detector, not
  // evidence that the bowls have gone.
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 0, y: 0, theta: 0 }));
    LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 0.2, y: 0.1, theta: 0.02 }));
    const established = slam.landmarks.length;

    // Same view, but the detector only finds the jack and two bowls each time.
    for (let i = 0; i < 5; i++) {
      const sparse = renderFrame(world, { x: 0.05 * i, y: 0, theta: 0 }, { skip: ['b2', 'b3', 'b4'] });
      LawnBowlsSlam.addFrame(slam, sparse);
    }

    check('sparse detection frames do not delete mapped bowls',
      slam.landmarks.length === established,
      `landmarks went ${established} -> ${slam.landmarks.length} after five poorly-detected frames`);
  }

  // --- 12: identity is sticky and only ever upgraded ----------------------
  {
    const world = wideRink();
    const withWeak = world.map(o =>
      o.name === 'b1' ? Object.assign({}, o, { identity: { playerId: 'p1', name: 'Ann', team: 'mine', similarity: 0.78 } }) : o);
    const withStrong = world.map(o =>
      o.name === 'b1' ? Object.assign({}, o, { identity: { playerId: 'p2', name: 'Bob', team: 'theirs', similarity: 0.93 } }) : o);

    const slam = LawnBowlsSlam.createSlam();
    LawnBowlsSlam.addFrame(slam, renderFrame(withWeak, { x: 0, y: 0, theta: 0 }));

    const b1Truth = { x: 1.4, y: 0.9 };
    const firstFrame = renderFrame(world, { x: 0, y: 0, theta: 0 });
    const want = toMapCoords(firstFrame, b1Truth);
    const findB1 = () => slam.landmarks.find(l => Math.hypot(l.x - want.x, l.y - want.y) < 0.3);

    check('identity attaches on first sight',
      findB1() && findB1().identity && findB1().identity.name === 'Ann');

    // A frame where that bowl matched nothing must not clear the identity.
    LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 0.2, y: 0.1, theta: 0.02 }));
    check('identity survives a frame that matched nobody',
      findB1() && findB1().identity && findB1().identity.name === 'Ann');

    // A stronger match takes over.
    LawnBowlsSlam.addFrame(slam, renderFrame(withStrong, { x: 0.1, y: 0.05, theta: 0.01 }));
    check('a higher-similarity identity replaces a weaker one',
      findB1() && findB1().identity && findB1().identity.name === 'Bob',
      findB1() && findB1().identity ? `still ${findB1().identity.name}` : 'identity lost');

    // A weaker one does not.
    LawnBowlsSlam.addFrame(slam, renderFrame(withWeak, { x: 0.15, y: 0.08, theta: 0.015 }));
    check('a lower-similarity identity does not displace a stronger one',
      findB1() && findB1().identity && findB1().identity.name === 'Bob');
  }

  // --- 13: the jack is decided by majority vote, not by one bad frame -----
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();

    // First frame mistakes a nearby bowl for the jack.
    const confused = renderFrame(world, { x: 0, y: 0, theta: 0 });
    confused.jack = confused.detections.find(d => d.name === 'b1');
    LawnBowlsSlam.addFrame(slam, confused);

    // Three good frames identify the real one.
    for (let i = 0; i < 3; i++) {
      LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 0.1 * i, y: 0.05 * i, theta: 0.01 * i }));
    }

    const snapshot = LawnBowlsSlam.getSnapshot(slam);
    const firstFrame = renderFrame(world, { x: 0, y: 0, theta: 0 });
    const jackTruth = toMapCoords(firstFrame, { x: 0, y: 0 });
    check('a single mislabelled frame does not hijack the jack',
      snapshot.jack && Math.hypot(snapshot.jack.x - jackTruth.x, snapshot.jack.y - jackTruth.y) < 0.3,
      snapshot.jack ? `jack at (${snapshot.jack.x.toFixed(2)}, ${snapshot.jack.y.toFixed(2)}) want (${jackTruth.x.toFixed(2)}, ${jackTruth.y.toFixed(2)})` : 'no jack');
  }

  // --- 14: an unplaceable view is rejected, not forced into the map ------
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    LawnBowlsSlam.addFrame(slam, renderFrame(world, { x: 0, y: 0, theta: 0 }));
    const before = slam.landmarks.map(l => ({ x: l.x, y: l.y }));

    // Detections whose mutual distances match nothing in the map at any
    // plausible scale — the camera is pointed somewhere else entirely.
    const alien = {
      detections: [
        { x: 40, y: 40, r: DIAMETER_PX / 2 },
        { x: 60, y: 55, r: DIAMETER_PX / 2 },
        { x: 45, y: 70, r: DIAMETER_PX / 2 },
      ],
      jack: null,
      width: VIEW_W,
      height: VIEW_H,
    };
    const result = LawnBowlsSlam.addFrame(slam, alien);

    check('an unplaceable frame is rejected', result.merged === false, 'it was merged anyway');
    check('a rejected frame leaves the map untouched',
      slam.landmarks.length === before.length &&
      slam.landmarks.every((l, i) => l.x === before[i].x && l.y === before[i].y));
  }

  // --- 15: an empty frame is handled without throwing ---------------------
  {
    const slam = LawnBowlsSlam.createSlam();
    const result = LawnBowlsSlam.addFrame(slam, { detections: [], jack: null, width: VIEW_W, height: VIEW_H });
    check('an empty frame is rejected cleanly', result.merged === false);
    check('an empty frame still counts as a frame seen', slam.frameCount === 1);

    const snapshot = LawnBowlsSlam.getSnapshot(slam);
    check('an empty map reports why it cannot score', snapshot.usable === false && !!snapshot.reason);
  }

  // --- 16: layoutForCanvas produces a drawable, correctly-ordered map -----
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    [
      { x: 0, y: 0, theta: 0 },
      { x: 2.5, y: 0.2, theta: 0.05 },
      { x: 5.0, y: 0.1, theta: 0.05 },
      { x: -2.5, y: 0.1, theta: -0.05 },
      { x: -5.0, y: 0.0, theta: -0.05 },
    ].forEach(cam => LawnBowlsSlam.addFrame(slam, renderFrame(world, cam)));

    const snapshot = LawnBowlsSlam.getSnapshot(slam, { confirmedOnly: true });
    const layout = LawnBowlsSlam.layoutForCanvas(snapshot, 800, 600);

    check('layout emits the jack plus every bowl',
      layout.detections.length === snapshot.bowls.length + 1);
    check('layout keeps every point on the canvas',
      layout.detections.every(d => d.x >= 0 && d.x <= 800 && d.y >= 0 && d.y <= 600),
      layout.detections.map(d => `(${d.x.toFixed(0)},${d.y.toFixed(0)})`).join(' '));
    check('layout preserves ranking order',
      layout.ranking.every((entry, i) => entry.dist === snapshot.ranking[i].dist));
    check('layout gives every bowl a drawable radius',
      layout.detections.every(d => d.r > 0));
  }

  // --- 17: mapped bowls project back into the live view -------------------
  // This is what lets the overlay draw a bowl the detector missed this frame.
  {
    const world = wideRink();
    const slam = LawnBowlsSlam.createSlam();
    const seed = renderFrame(world, { x: 0, y: 0, theta: 0 });
    LawnBowlsSlam.addFrame(slam, seed);

    const projected = LawnBowlsSlam.projectToFrame(slam, LawnBowlsSlam.currentPose(slam), DIAMETER_PX);
    check('every landmark projects back into the frame', projected.length === slam.landmarks.length);

    // The seeding frame is the identity pose, so projections must land back
    // exactly on the pixels the detections came from.
    let worst = 0;
    for (const det of seed.detections) {
      let best = Infinity;
      for (const p of projected) best = Math.min(best, Math.hypot(p.x - det.x, p.y - det.y));
      worst = Math.max(worst, best);
    }
    check('projection round-trips to the original pixel positions', worst < 1e-6,
      `worst error ${worst} px`);
  }

  return { name: 'slam', total, failures };
}

module.exports = { run };
