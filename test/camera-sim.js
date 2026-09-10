// A simulated phone camera over a real rink, shared by the ground and slam
// tests so they cannot drift apart in what they believe a camera does.
//
// Everything is in real units: bowls are 125mm across, a jack is about half
// that, the green is flat, and the camera is a pinhole at a real height and
// angle. Detections come out as pixel positions and radii, exactly the shape
// the detector produces — so a test failure means the pipeline mishandles
// something a real camera would actually do to the scene, rather than
// something an abstract point set would.

const BOWL_DIAMETER_M = 0.125;
const BOWL_RADIUS_M = BOWL_DIAMETER_M / 2;
const JACK_RADIUS_M = BOWL_RADIUS_M * 0.52; // a match jack is roughly half a bowl

const IMG_W = 1280;
const IMG_H = 720;
const FOCAL = 900; // px, about 70 degrees horizontal on this frame size

// pitch: degrees below horizontal (90 = straight down at the green)
// yaw:   degrees the camera is turned about the vertical, for panning
function camera(opts) {
  return {
    x: opts.x || 0,
    y: opts.y || 0,
    z: opts.z === undefined ? 1.6 : opts.z,
    pitch: (opts.pitch === undefined ? 55 : opts.pitch) * Math.PI / 180,
    yaw: (opts.yaw || 0) * Math.PI / 180,
  };
}

// Projects a point on the green (z = 0) into the camera's image.
function project(point, cam, realRadius, focal) {
  const f = focal || FOCAL;
  const dx = point.x - cam.x;
  const dy = point.y - cam.y;
  const dz = -cam.z;

  // Turn the camera about the vertical axis first.
  const cy = Math.cos(cam.yaw);
  const sy = Math.sin(cam.yaw);
  const rx = dx * cy + dy * sy;
  const ry = -dx * sy + dy * cy;

  // Then tip it down toward the green.
  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);
  const forward = ry * cp - dz * sp;
  const up = ry * sp + dz * cp;
  if (forward <= 0.05) return null; // behind the camera or on the horizon

  return {
    x: IMG_W / 2 + f * rx / forward,
    y: IMG_H / 2 - f * up / forward,
    r: f * realRadius / forward,
    depth: forward,
  };
}

// What this camera actually captures, in the detection shape the app uses.
// opts.skip drops named objects, to simulate the detector missing them.
function shoot(world, cam, opts) {
  const options = opts || {};
  const focal = options.focalLength || FOCAL;
  const detections = [];
  let jack = null;

  for (const obj of world) {
    if (options.skip && options.skip.includes(obj.name)) continue;
    const p = project(obj, cam, obj.isJack ? JACK_RADIUS_M : BOWL_RADIUS_M, focal);
    if (!p) continue;
    if (p.x < 0 || p.y < 0 || p.x > IMG_W || p.y > IMG_H) continue;

    const detection = { x: p.x, y: p.y, r: p.r, name: obj.name };
    if (obj.identity) detection.identity = obj.identity;
    detections.push(detection);
    if (obj.isJack && !options.hideJack) jack = detection;
  }

  return { detections, jack, width: IMG_W, height: IMG_H, focalLength: focal };
}

// A head of bowls around the jack, all at distinct distances so that checks on
// ordering actually mean something. Spread wide enough that no single frame
// from a normal scanning position contains all of it.
function spreadHead() {
  const objects = [{ name: 'jack', x: 0, y: 0, isJack: true }];
  const spec = [
    [0.22, 0.4], [0.38, 1.9], [0.55, 3.4], [0.74, 5.1],
    [0.96, 2.6], [1.21, 4.4], [1.5, 0.9], [1.85, 3.0],
  ];
  spec.forEach(([radius, angle], i) => {
    objects.push({ name: 'b' + i, x: radius * Math.cos(angle), y: radius * Math.sin(angle) });
  });
  return objects;
}

// True distance between two objects, in bowl-diameters — the unit the map and
// the scoring both work in.
function trueDistance(world, nameA, nameB) {
  const a = world.find(o => o.name === nameA);
  const b = world.find(o => o.name === nameB);
  return Math.hypot(a.x - b.x, a.y - b.y) / BOWL_DIAMETER_M;
}

module.exports = {
  BOWL_DIAMETER_M,
  BOWL_RADIUS_M,
  JACK_RADIUS_M,
  IMG_W,
  IMG_H,
  FOCAL,
  camera,
  project,
  shoot,
  spreadHead,
  trueDistance,
};
