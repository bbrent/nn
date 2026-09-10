// Tests for ground.js — recovering true top-down positions from apparent size.
//
// Driven by a simulated pinhole camera over a real rink: bowls of real size on
// a flat green, a camera at a real height and angle, projected the way an
// actual lens would. What matters is whether the distances the app scores
// from survive that projection, so the assertions are about recovered
// distances against ground truth, not about intermediate matrices.

const LawnBowlsGround = require('../ground.js');

const BOWL_DIAMETER_M = 0.125; // a typical lawn bowl
const BOWL_RADIUS_M = BOWL_DIAMETER_M / 2;
const JACK_RADIUS_M = BOWL_RADIUS_M * LawnBowlsGround.JACK_SIZE_RATIO;
const IMG_W = 1280;
const IMG_H = 720;
const FOCAL = 900;

// Projects a point on the green through a camera at height `z`, standing
// `back` metres away, pitched down by `pitch` degrees from horizontal.
function project(point, cam, focal, realRadius) {
  const dx = point.x - cam.x;
  const dy = point.y - cam.y;
  const dz = -cam.z;
  const c = Math.cos(cam.pitch);
  const s = Math.sin(cam.pitch);
  const forward = dy * c - dz * s;
  const up = dy * s + dz * c;
  if (forward <= 0.05) return null;
  return {
    x: IMG_W / 2 + focal * dx / forward,
    y: IMG_H / 2 - focal * up / forward,
    r: focal * realRadius / forward,
  };
}

function camera(pitchDeg, back, height) {
  return { x: 0, y: -back, z: height, pitch: pitchDeg * Math.PI / 180 };
}

// A head of bowls around the jack, all at distinct distances so ordering
// checks mean something.
function head() {
  const objects = [{ name: 'jack', x: 0, y: 0, isJack: true }];
  const spec = [[0.31, 0.3], [0.52, 1.1], [0.68, 2.0], [0.85, 2.9], [1.05, 3.7], [1.28, 4.6], [1.5, 5.4], [1.72, 0.9]];
  spec.forEach(([radius, angle], i) => {
    objects.push({ name: 'b' + i, x: radius * Math.cos(angle), y: radius * Math.sin(angle) });
  });
  return objects;
}

// Renders what a camera would actually capture, in the detection shape the
// rest of the app uses.
function shoot(world, cam, focal) {
  const detections = [];
  let jack = null;
  for (const obj of world) {
    const p = project(obj, cam, focal || FOCAL, obj.isJack ? JACK_RADIUS_M : BOWL_RADIUS_M);
    if (!p) continue;
    if (p.x < 0 || p.y < 0 || p.x > IMG_W || p.y > IMG_H) continue;
    const det = { x: p.x, y: p.y, r: p.r, name: obj.name };
    detections.push(det);
    if (obj.isJack) jack = det;
  }
  return { detections, jack };
}

// True distance between two world objects, in bowl-diameters — the unit the
// whole app scores in.
function trueDistance(world, a, b) {
  const p = world.find(o => o.name === a);
  const q = world.find(o => o.name === b);
  return Math.hypot(p.x - q.x, p.y - q.y) / BOWL_DIAMETER_M;
}

function run() {
  const failures = [];
  let total = 0;

  function check(label, condition, detail) {
    total++;
    if (!condition) failures.push(detail ? `${label}: ${detail}` : label);
  }

  // Error in each recovered distance from an origin object, as a percentage.
  // The origin defaults to the jack, but any mapped object works — useful when
  // the jack has deliberately been made unusable.
  function distanceErrors(result, world, originName) {
    const origin = originName || 'jack';
    const originPoint = result.points.find(p => p.detection.name === origin);
    if (!originPoint) return [{ name: origin, got: NaN, want: NaN, pct: Infinity }];

    const errors = [];
    for (const point of result.points) {
      if (point.detection.name === origin) continue;
      const got = Math.hypot(point.x - originPoint.x, point.y - originPoint.y);
      const want = trueDistance(world, origin, point.detection.name);
      errors.push({ name: point.detection.name, got, want, pct: Math.abs(100 * (got - want) / want) });
    }
    return errors;
  }

  function rankingCorrect(errors) {
    const byRecovered = [...errors].sort((a, b) => a.got - b.got).map(e => e.name);
    const byTruth = [...errors].sort((a, b) => a.want - b.want).map(e => e.name);
    return byRecovered.every((n, i) => n === byTruth[i]);
  }

  // --- 1: distances survive the projection at every realistic phone angle --
  // The whole point of the module. If this fails the app scores wrongly.
  {
    const world = head();
    for (const [pitch, back, height] of [[90, 0, 2.2], [70, 0.8, 2.0], [55, 1.4, 1.7], [40, 2.0, 1.5], [30, 3.0, 1.5], [22, 4.0, 1.4]]) {
      const { detections, jack } = shoot(world, camera(pitch, back, height));
      const result = LawnBowlsGround.rectify(detections, jack, { width: IMG_W, height: IMG_H, focalLength: FOCAL });

      check(`rectify succeeds at ${pitch}deg`, result.ok, result.reason);
      if (!result.ok) continue;

      const errors = distanceErrors(result, world);
      const worst = Math.max(...errors.map(e => e.pct));
      check(`distances accurate at ${pitch}deg`, worst < 0.5,
        `worst ${worst.toFixed(2)}% (${errors.map(e => `${e.name} ${e.got.toFixed(2)} vs ${e.want.toFixed(2)}`).join(', ')})`);
      check(`ranking correct at ${pitch}deg`, rankingCorrect(errors));
    }
  }

  // --- 2: the jack's smaller size is accounted for ------------------------
  // A jack is about half a bowl across, so the same apparent radius means a
  // very different depth. Treating it as bowl-sized puts the scoring origin
  // at nearly twice its true distance, which corrupts every measurement.
  {
    const world = head();
    const { detections, jack } = shoot(world, camera(40, 2.0, 1.5));

    const correct = LawnBowlsGround.rectify(detections, jack, { width: IMG_W, height: IMG_H, focalLength: FOCAL });
    check('the jack is kept when its size is accounted for',
      correct.points.some(p => p.detection.name === 'jack'));
    check('the jack label is not flagged as wrong when it is right',
      correct.jackRejected === false);
    const worstCorrect = Math.max(...distanceErrors(correct, world).map(e => e.pct));
    check('distances are accurate with the jack correctly sized', worstCorrect < 0.5,
      `worst ${worstCorrect.toFixed(2)}%`);

    // Passing jack:null makes the module treat the jack as an ordinary bowl.
    // Its apparent size then implies a depth roughly twice its real one, so it
    // no longer lies on the green the other bowls define — and rather than
    // quietly folding that error into every measurement, the fit should notice
    // and discard it. Dropping one object is a far better outcome than scoring
    // the whole end from a jack in the wrong place.
    const naive = LawnBowlsGround.rectify(detections, null, { width: IMG_W, height: IMG_H, focalLength: FOCAL });
    check('a wrongly-sized jack is detected and discarded, not silently used',
      naive.ok && !naive.points.some(p => p.detection.name === 'jack'),
      naive.ok ? 'it was kept and folded into the geometry' : naive.reason);

    // Whatever survives must still be right — rejecting the odd one out must
    // not disturb the bowls around it.
    const survivors = distanceErrors(naive, world, 'b0');
    check('the remaining bowls keep their true geometry after the rejection',
      survivors.every(e => e.pct < 1),
      survivors.map(e => `${e.name} ${e.pct.toFixed(1)}%`).join(', '));
  }

  // --- 3: two views of one scene are a rotation, never a mirror -----------
  // The plane normal's sign is arbitrary from the fit alone. If it flips
  // between frames the coordinates mirror, and the next frame looks like a
  // reflection of the map instead of a turn of the camera — which no rigid
  // alignment can reconcile.
  {
    const world = head();
    function signedArea(result) {
      const pick = name => result.points.find(p => p.detection.name === name);
      const a = pick('jack'), b = pick('b0'), c = pick('b3');
      if (!a || !b || !c) return null;
      return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }

    const views = [[60, 1.2, 1.8], [40, 2.0, 1.5], [75, 0.6, 2.0], [30, 3.0, 1.5]]
      .map(([p, b, h]) => shoot(world, camera(p, b, h)))
      .map(({ detections, jack }) => LawnBowlsGround.rectify(detections, jack, { width: IMG_W, height: IMG_H, focalLength: FOCAL }));

    const areas = views.map(signedArea).filter(a => a !== null);
    check('every view produces a usable rectification', areas.length === 4);
    check('all views share one handedness (no mirrored frames)',
      areas.every(a => Math.sign(a) === Math.sign(areas[0])),
      `signs: ${areas.map(a => Math.sign(a)).join(', ')}`);
  }

  // --- 4: the scale is absolute, not per-frame ---------------------------
  // Recovered coordinates are metric in bowl-diameters because the bowl's real
  // size sets the unit. Two frames from very different distances must agree,
  // which is what lets frame-to-frame alignment be a plain rigid transform
  // with no scale left to guess at.
  {
    const world = head();
    const near = LawnBowlsGround.rectify(...(() => { const s = shoot(world, camera(55, 1.0, 1.6)); return [s.detections, s.jack]; })(),
      { width: IMG_W, height: IMG_H, focalLength: FOCAL });
    const far = LawnBowlsGround.rectify(...(() => { const s = shoot(world, camera(55, 3.0, 2.6)); return [s.detections, s.jack]; })(),
      { width: IMG_W, height: IMG_H, focalLength: FOCAL });

    check('both distances rectify', near.ok && far.ok);
    if (near.ok && far.ok) {
      const shared = near.points
        .map(p => p.detection.name)
        .filter(n => far.points.some(q => q.detection.name === n));
      check('the two views share enough bowls to compare', shared.length >= 3, `${shared.length} shared`);

      let worst = 0;
      for (let i = 0; i < shared.length; i++) {
        for (let j = i + 1; j < shared.length; j++) {
          const dn = (r) => {
            const a = r.points.find(p => p.detection.name === shared[i]);
            const b = r.points.find(p => p.detection.name === shared[j]);
            return Math.hypot(a.x - b.x, a.y - b.y);
          };
          const nearDist = dn(near);
          const farDist = dn(far);
          worst = Math.max(worst, Math.abs(100 * (nearDist - farDist) / nearDist));
        }
      }
      check('near and far views agree on absolute distances', worst < 1,
        `worst disagreement ${worst.toFixed(2)}%`);
    }
  }

  // --- 5: the tilt estimate reflects the real camera angle ---------------
  // Used to warn that a view is too oblique to trust for close calls.
  {
    const world = head();
    for (const [pitch, back, height] of [[90, 0, 2.2], [60, 1.2, 1.8], [30, 3.0, 1.5]]) {
      const { detections, jack } = shoot(world, camera(pitch, back, height));
      const result = LawnBowlsGround.rectify(detections, jack, { width: IMG_W, height: IMG_H, focalLength: FOCAL });
      // Camera pitch 90deg = looking straight down = 0deg off the green's normal.
      const expected = 90 - pitch;
      check(`tilt estimate tracks camera angle at ${pitch}deg`,
        result.ok && Math.abs(result.tilt - expected) < 12,
        result.ok ? `got ${result.tilt.toFixed(1)}deg, expected about ${expected}deg` : result.reason);
    }
  }

  // --- 6: a guessed focal length degrades gently -------------------------
  // It cannot be recovered from the scene, so it comes from a default. What
  // matters is that being wrong costs accuracy slowly rather than breaking.
  {
    const world = head();
    const { detections, jack } = shoot(world, camera(40, 2.0, 1.5));

    for (const [assumed, tolerance] of [[700, 8], [800, 4], [900, 0.5], [1000, 4], [1100, 8]]) {
      const result = LawnBowlsGround.rectify(detections, jack, { width: IMG_W, height: IMG_H, focalLength: assumed });
      const errors = distanceErrors(result, world);
      const mean = errors.reduce((s, e) => s + e.pct, 0) / errors.length;
      check(`focal length ${assumed} (true ${FOCAL}) stays usable`, result.ok && mean < tolerance,
        result.ok ? `mean error ${mean.toFixed(1)}%` : result.reason);
      check(`focal length ${assumed} keeps the ranking`, result.ok && rankingCorrect(errors));
    }
  }

  // --- 7: the default focal length is in the right ballpark --------------
  {
    const guessed = LawnBowlsGround.defaultFocalLength(IMG_W, IMG_H);
    check('default focal length suits a phone camera', guessed > 700 && guessed < 1200,
      `got ${guessed} for a ${IMG_W}x${IMG_H} frame`);

    const world = head();
    const { detections, jack } = shoot(world, camera(40, 2.0, 1.5));
    const result = LawnBowlsGround.rectify(detections, jack, { width: IMG_W, height: IMG_H });
    const errors = distanceErrors(result, world);
    const mean = errors.reduce((s, e) => s + e.pct, 0) / errors.length;
    check('the default alone beats not rectifying at all', result.ok && mean < 10,
      result.ok ? `mean error ${mean.toFixed(1)}%` : result.reason);
    check('the default keeps the ranking correct', result.ok && rankingCorrect(errors));
  }

  // --- 8: the residual notices something that isn't on the green ---------
  {
    const world = head();
    const { detections, jack } = shoot(world, camera(50, 1.6, 1.7));
    const clean = LawnBowlsGround.rectify(detections, jack, { width: IMG_W, height: IMG_H, focalLength: FOCAL });

    // A detection whose apparent size puts it well above the green — a hand,
    // or a bowl being carried.
    const polluted = detections.concat([{ x: IMG_W / 2, y: IMG_H / 2, r: detections[1].r * 2.4, name: 'held' }]);
    const dirty = LawnBowlsGround.rectify(polluted, jack, { width: IMG_W, height: IMG_H, focalLength: FOCAL });

    check('a clean head fits the plane tightly', clean.ok && clean.residual < 0.01,
      clean.ok ? `residual ${clean.residual.toExponential(2)}` : clean.reason);
    check('a clean head loses nothing to outlier rejection',
      clean.ok && clean.points.length === detections.length,
      clean.ok ? `kept ${clean.points.length} of ${detections.length}` : clean.reason);

    // The object that isn't on the green is dropped rather than folded in, so
    // the frame it leaves behind is clean instead of merely suspicious.
    check('an object that is not on the green is rejected',
      dirty.ok && !dirty.points.some(p => p.detection.name === 'held'),
      dirty.ok ? 'it was kept and folded into the geometry' : dirty.reason);
    check('rejecting it leaves every real bowl in place',
      dirty.ok && dirty.points.length === detections.length,
      dirty.ok ? `kept ${dirty.points.length} of ${detections.length} real detections` : dirty.reason);

    // And the bowls that remain must be exactly where they were before the
    // intruder appeared — rejection must not perturb the fit.
    if (clean.ok && dirty.ok) {
      const cleanErrors = distanceErrors(clean, world);
      const dirtyErrors = distanceErrors(dirty, world).filter(e => e.name !== 'held');
      const worst = Math.max(...dirtyErrors.map(e => e.pct));
      check('the polluted frame measures as accurately as the clean one', worst < 0.5,
        `worst ${worst.toFixed(2)}% vs ${Math.max(...cleanErrors.map(e => e.pct)).toFixed(2)}% clean`);
    }
  }

  // --- 9: degenerate inputs are refused rather than guessed at -----------
  {
    const tooFew = LawnBowlsGround.rectify(
      [{ x: 100, y: 100, r: 40 }, { x: 300, y: 150, r: 42 }, { x: 500, y: 120, r: 38 }],
      null, { width: IMG_W, height: IMG_H });
    check('three points is refused (a plane through three is meaningless)',
      tooFew.ok === false && !!tooFew.reason);

    const empty = LawnBowlsGround.rectify([], null, { width: IMG_W, height: IMG_H });
    check('an empty frame is refused cleanly', empty.ok === false);

    // Identical detections carry no depth variation to fit a plane through.
    const degenerate = LawnBowlsGround.rectify(
      [0, 1, 2, 3, 4].map(() => ({ x: 640, y: 360, r: 40 })),
      null, { width: IMG_W, height: IMG_H });
    check('coincident detections are refused', degenerate.ok === false, degenerate.reason);

    const zeroRadius = LawnBowlsGround.rectify(
      [{ x: 100, y: 100, r: 0 }, { x: 200, y: 200, r: 0 }, { x: 300, y: 100, r: 0 }, { x: 400, y: 300, r: 0 }],
      null, { width: IMG_W, height: IMG_H });
    check('zero-radius detections are refused', zeroRadius.ok === false);
  }

  // --- 10: rectifying beats not rectifying, on the same frames -----------
  // The direct comparison that justifies the module existing. Model the old
  // path (image pixels over one median diameter for the whole frame) and check
  // it really is worse on identical input.
  {
    const world = head();
    let rectifiedWins = 0;
    let comparisons = 0;

    for (const [pitch, back, height] of [[70, 0.8, 2.0], [55, 1.4, 1.7], [40, 2.0, 1.5], [30, 3.0, 1.5]]) {
      const { detections, jack } = shoot(world, camera(pitch, back, height));

      const radii = detections.filter(d => d !== jack).map(d => d.r).sort((a, b) => a - b);
      const medianDiameter = radii[Math.floor(radii.length / 2)] * 2;
      const oldModel = {
        points: detections.map(d => ({ x: d.x / medianDiameter, y: d.y / medianDiameter, detection: d })),
      };

      const rectified = LawnBowlsGround.rectify(detections, jack, { width: IMG_W, height: IMG_H, focalLength: FOCAL });
      const oldErrors = distanceErrors(oldModel, world);
      const newErrors = distanceErrors(rectified, world);
      const oldMean = oldErrors.reduce((s, e) => s + e.pct, 0) / oldErrors.length;
      const newMean = newErrors.reduce((s, e) => s + e.pct, 0) / newErrors.length;

      comparisons++;
      if (newMean < oldMean) rectifiedWins++;

      check(`rectifying is a clear improvement at ${pitch}deg`, newMean < oldMean / 4,
        `un-rectified ${oldMean.toFixed(1)}% vs rectified ${newMean.toFixed(1)}%`);
      check(`the un-rectified model really does get ${pitch}deg wrong`, !rankingCorrect(oldErrors) || oldMean > 5,
        `un-rectified was fine here (${oldMean.toFixed(1)}%) — the comparison proves nothing`);
    }
    check('rectifying wins at every angle tested', rectifiedWins === comparisons,
      `${rectifiedWins}/${comparisons}`);
  }

  return { name: 'ground', total, failures };
}

module.exports = { run };
