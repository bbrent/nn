// Pure, DOM-free circle detection + jack classification + relative ranking.
// Shared between the browser app (live camera, window.cv) and the Node test
// harness (static fixture images, @techstark/opencv-js). Callers own/free
// the cv.Mat instances they pass in.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.LawnBowlsDetection = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  const WORK_WIDTH = 800;
  const JACK_RADIUS_RATIO = 0.65; // jack is ~0.54x a bowl's diameter; threshold with margin

  // srcMat: RGBA cv.Mat at any resolution. Returns detections in srcMat's
  // coordinate space (i.e. already scaled back up from the downscaled work image).
  function detectCircles(cv, srcMat) {
    const scale = WORK_WIDTH / srcMat.cols;
    const workHeight = Math.round(srcMat.rows * scale);
    const work = new cv.Mat();
    cv.resize(srcMat, work, new cv.Size(WORK_WIDTH, workHeight), 0, 0, cv.INTER_AREA);

    const gray = new cv.Mat();
    cv.cvtColor(work, gray, cv.COLOR_RGBA2GRAY);
    cv.medianBlur(gray, gray, 5);

    const circles = new cv.Mat();
    cv.HoughCircles(
      gray,
      circles,
      cv.HOUGH_GRADIENT,
      1,
      gray.rows / 12, // min distance between circle centers
      100,            // Canny high threshold
      15,             // accumulator threshold (lower = more false positives)
      4,              // min radius, px at work resolution
      100             // max radius, px at work resolution
    );

    const detections = [];
    for (let i = 0; i < circles.cols; i++) {
      detections.push({
        x: circles.data32F[i * 3] / scale,
        y: circles.data32F[i * 3 + 1] / scale,
        r: circles.data32F[i * 3 + 2] / scale,
      });
    }

    work.delete();
    gray.delete();
    circles.delete();
    return detections;
  }

  // Splits raw detections into a jack + bowls, and ranks bowls by distance
  // to the jack in bowl-diameter units (no absolute scale needed).
  function classifyAndRank(detections) {
    if (detections.length === 0) {
      return { jack: null, bowls: [], ranking: [] };
    }

    const sortedR = [...detections].map(d => d.r).sort((a, b) => a - b);
    const medianR = sortedR[Math.floor(sortedR.length / 2)];
    const jackThreshold = medianR * JACK_RADIUS_RATIO;

    let jack = null;
    const bowls = [];
    for (const d of detections) {
      if (d.r < jackThreshold) {
        if (!jack || d.r < jack.r) {
          if (jack) bowls.push(jack);
          jack = d;
        } else {
          bowls.push(d);
        }
      } else {
        bowls.push(d);
      }
    }

    const avgBowlDiameter = bowls.length
      ? bowls.reduce((sum, b) => sum + b.r * 2, 0) / bowls.length
      : medianR * 2;

    const ranking = bowls.map(b => ({
      bowl: b,
      dist: jack ? Math.hypot(b.x - jack.x, b.y - jack.y) / avgBowlDiameter : null,
    }));
    if (jack) ranking.sort((a, b) => a.dist - b.dist);

    return { jack, bowls, ranking };
  }

  function detectAndRank(cv, srcMat) {
    const detections = detectCircles(cv, srcMat);
    return { detections, ...classifyAndRank(detections) };
  }

  return { WORK_WIDTH, JACK_RADIUS_RATIO, detectCircles, classifyAndRank, detectAndRank };
});
