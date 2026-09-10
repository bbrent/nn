// Rectification: turns a frame of image-space detections into true top-down
// positions on the green, using each bowl's apparent size as a depth cue.
//
// Why this exists: everything downstream measures how far each bowl is from
// the jack, and that measurement decides the score. Treating the image as if
// it were already a top-down view — dividing image pixels by one bulk scale
// for the whole frame — only holds when the phone points straight down. At
// any realistic angle the far side of the head is foreshortened, distances
// there come out short, and the ranking can inverte outright. Simulating a
// pinhole camera over a real rink put that at 12% mean distance error with the
// phone fairly overhead and 27% standing back, with the closest-bowl order
// wrong in every tilted case.
//
// The fix uses information the detector already produces and the old path
// threw away. A bowl of known real size at depth Z images with radius
// r = f*R/Z, so Z = f*R/r: every detection's radius is a direct depth
// measurement. That recovers each bowl's full 3D position; since they all sit
// on the green, fitting a plane through them and projecting onto it gives
// genuine top-down coordinates. In simulation that is exact at every tilt.
//
// Two things this needs care with:
//   - The jack is roughly half a bowl's diameter, so the same apparent radius
//     means a very different depth. Using the bowl size for it would place the
//     scoring origin at nearly twice its true distance and corrupt every
//     measurement taken from it.
//   - The recovered coordinates are metric in bowl-diameters, not relative to
//     a per-frame scale, so they are directly comparable between frames taken
//     from different distances. That is what removes the scale guesswork from
//     frame-to-frame alignment.
//
// Focal length cannot be recovered from the scene: scaling f stretches the
// depth axis, and a stretched plane is still a plane, so coplanarity says
// nothing about it (measured — the fit residual is ~1e-17 for every value).
// It therefore comes from a default, which is fine: being 22% wrong still
// leaves distance error at 1-10% and the ranking correct, against 12-27% and
// consistently wrong ranking for the un-rectified path.
//
// Shared with the browser app (window.LawnBowlsGround) and the Node tests.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.LawnBowlsGround = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  // Jack diameter as a fraction of a bowl's. A match jack is about 64mm
  // across; bowls run 116-134mm depending on size, so ~125mm is typical.
  const JACK_SIZE_RATIO = 0.52;
  // Focal length as a fraction of the frame's long edge, when nothing better
  // is known. A phone main camera is around 65-75 degrees horizontally, which
  // puts f/width near 0.7-0.8.
  const DEFAULT_FOCAL_RATIO = 0.75;
  // A plane needs three points, and three exactly-determined ones fit any
  // arrangement perfectly with no way to tell a good fit from a bad one.
  const MIN_POINTS_FOR_PLANE = 4;
  // Beyond this the view is so oblique that depth resolution collapses and
  // the recovered geometry should not be trusted for close calls.
  const STEEP_TILT_DEGREES = 70;
  // How far off the green a detection may sit, relative to how spread out the
  // frame is, before it is treated as something other than what it was taken
  // for. Measured rather than guessed: with realistic detector noise a
  // correctly-sized detection sits at 0.07 (p95 0.17), while one whose assumed
  // size is wrong by the bowl/jack ratio sits around 0.35.
  const MAX_OFF_PLANE = 0.25;
  // Cap on candidate planes tried when choosing one by consensus. A frame
  // holds a handful of detections, so this is comfortably exhaustive in
  // practice (fifteen detections is 455 triples) and only bites on an
  // implausibly crowded frame.
  const CONSENSUS_BUDGET = 2000;

  function defaultFocalLength(width, height) {
    return Math.max(width, height) * DEFAULT_FOCAL_RATIO;
  }

  // Each detection's position in camera-frame coordinates, in units of bowl
  // radius. From u = f*X/Z and r = f*R/Z: X/R = u/r, Y/R = v/r, Z/R = f/r.
  // Image coordinates are taken relative to the principal point (frame
  // centre), which is where the optical axis meets the sensor.
  function recover3D(detections, jack, focal, width, height) {
    const cx = width / 2;
    const cy = height / 2;
    const points = [];
    for (const d of detections) {
      if (!(d.r > 0)) continue;
      // Scale the jack's radius up to what a bowl at the same depth would
      // measure, so one consistent unit covers every detection.
      const equivalentRadius = d === jack ? d.r / JACK_SIZE_RATIO : d.r;
      const u = d.x - cx;
      const v = d.y - cy;
      points.push({
        x: u / equivalentRadius,
        y: v / equivalentRadius,
        z: focal / equivalentRadius,
        detection: d,
      });
    }
    return points;
  }

  // Least-squares plane through the recovered points, as a unit normal and a
  // centroid. Uses the cofactor form rather than a full eigen-decomposition:
  // it is closed-form, needs no iteration, and picks the numerically
  // best-conditioned axis to divide through by.
  function fitPlane(points) {
    const n = points.length;
    let cx = 0, cy = 0, cz = 0;
    for (const p of points) { cx += p.x; cy += p.y; cz += p.z; }
    cx /= n; cy /= n; cz /= n;

    let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (const p of points) {
      const dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
      xx += dx * dx; xy += dx * dy; xz += dx * dz;
      yy += dy * dy; yz += dy * dz; zz += dz * dz;
    }

    const detX = yy * zz - yz * yz;
    const detY = xx * zz - xz * xz;
    const detZ = xx * yy - xy * xy;
    const best = Math.max(detX, detY, detZ);
    if (!(best > 0)) return null; // degenerate: all points collinear

    let normal;
    if (best === detX) normal = { x: detX, y: xz * yz - xy * zz, z: xy * yz - xz * yy };
    else if (best === detY) normal = { x: xz * yz - xy * zz, y: detY, z: xy * xz - yz * xx };
    else normal = { x: xy * yz - xz * yy, y: xy * xz - yz * xx, z: detZ };

    const len = Math.hypot(normal.x, normal.y, normal.z);
    if (!(len > 0)) return null;
    normal = { x: normal.x / len, y: normal.y / len, z: normal.z / len };

    // Orient the normal consistently toward the camera (which sits at the
    // origin looking down +z). Without this the sign can flip between frames,
    // which mirrors the whole frame's coordinates and makes the next frame
    // look like a reflection of the map rather than a rotation of it.
    if (normal.x * cx + normal.y * cy + normal.z * cz > 0) {
      normal = { x: -normal.x, y: -normal.y, z: -normal.z };
    }

    return { centroid: { x: cx, y: cy, z: cz }, normal };
  }

  // RMS out-of-plane residual relative to the in-plane spread. Scale-free, so
  // it reads the same regardless of how far away the camera is: a genuine
  // flat arrangement of bowls scores near zero, while a frame polluted by
  // something well off the green does not.
  function planeResidual(points, plane) {
    let out = 0;
    let along = 0;
    for (const p of points) {
      const dx = p.x - plane.centroid.x;
      const dy = p.y - plane.centroid.y;
      const dz = p.z - plane.centroid.z;
      const perpendicular = dx * plane.normal.x + dy * plane.normal.y + dz * plane.normal.z;
      out += perpendicular * perpendicular;
      along += dx * dx + dy * dy + dz * dz;
    }
    if (!(along > 0)) return 0;
    return Math.sqrt(out / points.length) / Math.sqrt(along / points.length);
  }

  // Fits the green while discarding detections that clearly do not sit on it.
  //
  // A detection's recovered depth is only right if its assumed real size is
  // right, so anything sized differently from what it was taken for lands off
  // the plane. That covers the cases that actually matter: a bowl mistaken for
  // the jack is placed at roughly twice its true distance, and — the case that
  // makes a jack-only check insufficient — whenever that happens the real jack
  // is simultaneously being treated as a bowl, so it lands badly wrong too.
  // Rejecting outliers generally catches both at once, along with anything
  // else in shot that was never a bowl.
  //
  // Points are dropped one round at a time, and never more than a third of the
  // frame: past that the sensible reading is that the fit is wrong, not that
  // most of the bowls are impostors.
  // Exact plane through three points.
  function planeFromThree(a, b, c) {
    const ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
    const vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    if (!(len > 1e-12)) return null; // the three are collinear
    return {
      centroid: { x: (a.x + b.x + c.x) / 3, y: (a.y + b.y + c.y) / 3, z: (a.z + b.z + c.z) / 3 },
      normal: { x: nx / len, y: ny / len, z: nz / len },
    };
  }

  function offPlaneDistance(plane, p) {
    return Math.abs(
      (p.x - plane.centroid.x) * plane.normal.x +
      (p.y - plane.centroid.y) * plane.normal.y +
      (p.z - plane.centroid.z) * plane.normal.z
    );
  }

  // Fits the green while discarding detections that clearly do not sit on it.
  //
  // A detection's recovered depth is only right if its assumed real size is
  // right, so anything sized differently from what it was taken for lands off
  // the plane. That covers the case that actually matters: when a bowl is
  // mistaken for the jack it is placed at roughly twice its true distance —
  // and at the same moment the real jack is being treated as a bowl, so it
  // goes wrong too. Two detections out of six or eight is a third of the
  // frame, which is exactly where fitting everything and trimming the worst
  // offender breaks down: least-squares is dragged so far by the pair that the
  // innocent bowls start looking like the outliers, and trimming removes them
  // instead. Measured on that case, greedy trimming discarded two good bowls,
  // kept both bad ones, and left the survivors 14 bowl-diameters out of place.
  //
  // So the plane is chosen by consensus instead: every triple of detections
  // proposes a plane, and the one the most detections agree with wins. Three
  // points that happen to be the misplaced ones convince nobody, while any
  // three genuinely on the green are backed by every other bowl on it. With
  // the handful of detections in a frame the search is exhaustive and
  // deterministic rather than sampled, so the same frame always resolves the
  // same way.
  function fitPlaneRobustly(points) {
    if (points.length < MIN_POINTS_FOR_PLANE) return null;

    // Scale-free threshold, measured against how spread out the frame is as a
    // whole so it means the same at any camera distance.
    let cx = 0, cy = 0, cz = 0;
    for (const p of points) { cx += p.x; cy += p.y; cz += p.z; }
    cx /= points.length; cy /= points.length; cz /= points.length;
    const spread = Math.sqrt(points.reduce((sum, p) =>
      sum + (p.x - cx) ** 2 + (p.y - cy) ** 2 + (p.z - cz) ** 2, 0) / points.length);
    if (!(spread > 0)) return null;
    const tolerance = MAX_OFF_PLANE * spread;

    let best = null;
    let budget = CONSENSUS_BUDGET;
    for (let i = 0; i < points.length && budget > 0; i++) {
      for (let j = i + 1; j < points.length && budget > 0; j++) {
        for (let k = j + 1; k < points.length && budget > 0; k++) {
          budget--;
          const candidate = planeFromThree(points[i], points[j], points[k]);
          if (!candidate) continue;

          const inliers = points.filter(p => offPlaneDistance(candidate, p) <= tolerance);
          if (inliers.length < MIN_POINTS_FOR_PLANE) continue;
          const error = inliers.reduce((sum, p) => sum + offPlaneDistance(candidate, p), 0) / inliers.length;

          // Most agreement wins; ties go to the tighter fit.
          if (!best || inliers.length > best.inliers.length ||
              (inliers.length === best.inliers.length && error < best.error)) {
            best = { inliers, error };
          }
        }
      }
    }

    if (!best) return null;
    // Refit properly over everything that agreed, rather than keeping the
    // plane through the three that happened to propose it.
    const plane = fitPlane(best.inliers);
    if (!plane) return null;
    return { plane, inliers: best.inliers };
  }

  // Right-handed orthonormal basis spanning the plane. The in-plane axes are
  // arbitrary but the handedness is not — deriving the second axis from the
  // cross product with the normal keeps every frame's coordinates the same
  // chirality, so alignment between frames stays a rotation.
  function planeBasis(normal) {
    const seed = Math.abs(normal.x) < 0.9 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 };
    const dot = seed.x * normal.x + seed.y * normal.y + seed.z * normal.z;
    let e1 = { x: seed.x - dot * normal.x, y: seed.y - dot * normal.y, z: seed.z - dot * normal.z };
    const len = Math.hypot(e1.x, e1.y, e1.z);
    e1 = { x: e1.x / len, y: e1.y / len, z: e1.z / len };
    const e2 = {
      x: normal.y * e1.z - normal.z * e1.y,
      y: normal.z * e1.x - normal.x * e1.z,
      z: normal.x * e1.y - normal.y * e1.x,
    };
    return { e1, e2 };
  }

  // Angle between the camera's optical axis and the green's normal. Zero means
  // looking straight down at it; large means an oblique view where depth is
  // poorly resolved and close calls should not be trusted.
  function tiltDegrees(plane) {
    const towardCamera = -plane.normal.z; // camera looks along +z from the origin
    const clamped = Math.max(-1, Math.min(1, Math.abs(towardCamera)));
    return Math.acos(clamped) * 180 / Math.PI;
  }

  // Where the frame's edges land on the green: the four image corners
  // back-projected onto the fitted plane, in the same coordinates rectify()
  // returns. This is what "was that bowl actually in shot?" should be asked
  // against — a real footprint on the ground rather than a guessed radius.
  //
  // Returns null when the footprint is unbounded, which happens once the view
  // is oblique enough that the horizon is in frame: corner rays then run
  // parallel to the green or meet it behind the camera, and no finite region
  // describes what was seen. Callers should decline to reason about absence
  // in that case rather than invent a boundary.
  function groundFootprint(plane, basis, opts) {
    const { width, height, focalLength, inset } = opts;
    const margin = inset || 0;
    const cx = width / 2;
    const cy = height / 2;
    const corners = [
      { u: margin - cx, v: margin - cy },
      { u: width - margin - cx, v: margin - cy },
      { u: width - margin - cx, v: height - margin - cy },
      { u: margin - cx, v: height - margin - cy },
    ];

    const n = plane.normal;
    const nDotC = n.x * plane.centroid.x + n.y * plane.centroid.y + n.z * plane.centroid.z;

    const polygon = [];
    for (const corner of corners) {
      // Ray from the camera at the origin through this pixel.
      const d = { x: corner.u, y: corner.v, z: focalLength };
      const nDotD = n.x * d.x + n.y * d.y + n.z * d.z;
      if (Math.abs(nDotD) < 1e-9) return null; // parallel to the green
      const t = nDotC / nDotD;
      if (t <= 0) return null; // meets the green behind the camera
      const hit = { x: t * d.x - plane.centroid.x, y: t * d.y - plane.centroid.y, z: t * d.z - plane.centroid.z };
      polygon.push({
        x: (hit.x * basis.e1.x + hit.y * basis.e1.y + hit.z * basis.e1.z) / 2,
        y: (hit.x * basis.e2.x + hit.y * basis.e2.y + hit.z * basis.e2.z) / 2,
      });
    }
    return polygon;
  }

  // Point-in-convex-polygon by consistent edge sign. The footprint of a
  // rectangle projected onto a plane is convex whenever it is bounded at all,
  // which groundFootprint has already established.
  function containsPoint(polygon, point) {
    if (!polygon || polygon.length < 3) return false;
    let positive = false;
    let negative = false;
    for (let i = 0; i < polygon.length; i++) {
      const a = polygon[i];
      const b = polygon[(i + 1) % polygon.length];
      const cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
      if (cross > 0) positive = true;
      if (cross < 0) negative = true;
      if (positive && negative) return false;
    }
    return true;
  }

  // Turns one frame's detections into top-down ground positions, in
  // bowl-diameter units.
  //
  // detections: [{ x, y, r, ... }] in image pixels
  // jack:       one of them, or null when it isn't in this shot
  // opts:       { width, height, focalLength? }
  //
  // Returns { ok, points, tilt, residual, focalLength, reason }, where each
  // point carries the detection it came from so callers can keep identity,
  // jack-ness and confidence attached.
  function rectify(detections, jack, opts) {
    const width = opts.width;
    const height = opts.height;
    const focal = opts.focalLength || defaultFocalLength(width, height);

    const usable = (detections || []).filter(d => d.r > 0);
    if (usable.length < MIN_POINTS_FOR_PLANE) {
      return { ok: false, reason: 'too few detections to work out the ground plane', points: [], focalLength: focal };
    }

    const fit = fitPlaneRobustly(recover3D(usable, jack, focal, width, height));
    if (!fit) {
      return { ok: false, reason: 'detections are collinear — cannot fit the ground', points: [], focalLength: focal };
    }
    const plane = fit.plane;
    const spatial = fit.inliers;
    const jackRejected = !!jack && !spatial.some(p => p.detection === jack);

    const { e1, e2 } = planeBasis(plane.normal);
    const points = spatial.map(p => {
      const dx = p.x - plane.centroid.x;
      const dy = p.y - plane.centroid.y;
      const dz = p.z - plane.centroid.z;
      return {
        // /2 converts bowl-radius units to bowl-diameters.
        x: (dx * e1.x + dy * e1.y + dz * e1.z) / 2,
        y: (dx * e2.x + dy * e2.y + dz * e2.z) / 2,
        detection: p.detection,
      };
    });

    // The visible footprint is inset by roughly a bowl's width so detections
    // clipped by the frame edge don't count as "clearly looked at and absent".
    const medianRadius = usable.map(d => d.r).sort((a, b) => a - b)[Math.floor(usable.length / 2)];
    const footprint = groundFootprint(plane, { e1, e2 }, {
      width,
      height,
      focalLength: focal,
      inset: (opts.insetDiameters === undefined ? 1 : opts.insetDiameters) * medianRadius * 2,
    });

    return {
      ok: true,
      points,
      jackRejected,
      footprint,
      tilt: tiltDegrees(plane),
      residual: planeResidual(spatial, plane),
      focalLength: focal,
      reason: null,
    };
  }

  return {
    JACK_SIZE_RATIO,
    DEFAULT_FOCAL_RATIO,
    MIN_POINTS_FOR_PLANE,
    STEEP_TILT_DEGREES,
    MAX_OFF_PLANE,
    CONSENSUS_BUDGET,
    defaultFocalLength,
    recover3D,
    fitPlane,
    planeBasis,
    planeResidual,
    fitPlaneRobustly,
    tiltDegrees,
    groundFootprint,
    containsPoint,
    rectify,
  };
});
