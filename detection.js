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

  // Hough's own radius floor is 4px (in work-image pixels, see below) — right at
  // that floor, detection is unreliable (misses, merges with neighbors). A frame
  // is only "usable" once the jack clears this floor with real margin; otherwise
  // it should be skipped rather than trusted (e.g. hold the phone closer/more
  // overhead and rescan, or just drop the frame during multi-frame fusion).
  const CONFIDENT_JACK_WORK_RADIUS_PX = 6;

  // If the true jack goes undetected, the smallest *bowl* still gets picked as
  // jack by the size-rank heuristic below — a false positive worse than no jack
  // at all. Guard against it: a real jack sits far below the nearest bowl in
  // size (ratio ~0.45-0.72 across tested scenes); anything close to 1 means
  // we're just looking at two same-size bowls, not a real jack.
  const JACK_TO_NEAREST_BOWL_RATIO = 0.8;

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
      const workR = circles.data32F[i * 3 + 2];
      detections.push({
        x: circles.data32F[i * 3] / scale,
        y: circles.data32F[i * 3 + 1] / scale,
        r: workR / scale,
        workR, // pre-scale radius, work-image px — used to judge detection confidence
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
      return { jack: null, bowls: [], ranking: [], usable: false, reason: 'no circles detected' };
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

    // localX/localY: jack-centered position in bowl-diameter units — the frame's
    // own orientation and origin, not tied to pixels. This is what cross-frame
    // fusion aligns between frames (see fusion.js); dist is just its magnitude.
    const ranking = bowls.map(b => {
      const localX = jack ? (b.x - jack.x) / avgBowlDiameter : null;
      const localY = jack ? (b.y - jack.y) / avgBowlDiameter : null;
      return { bowl: b, dist: jack ? Math.hypot(localX, localY) : null, localX, localY };
    });
    if (jack) ranking.sort((a, b) => a.dist - b.dist);

    let usable = true;
    let reason = null;
    const minBowlWorkR = bowls.length ? Math.min(...bowls.map(b => b.workR)) : null;
    if (!jack) {
      usable = false;
      reason = 'no jack identified';
    } else if (jack.workR < CONFIDENT_JACK_WORK_RADIUS_PX) {
      usable = false;
      reason = 'jack too small in frame to trust — move the phone closer/more overhead';
    } else if (minBowlWorkR !== null && jack.workR / minBowlWorkR > JACK_TO_NEAREST_BOWL_RATIO) {
      usable = false;
      reason = 'no bowl is clearly smaller than the rest — jack may be undetected or out of frame';
    }

    return { jack, bowls, ranking, usable, reason };
  }

  function detectAndRank(cv, srcMat) {
    const detections = detectCircles(cv, srcMat);
    return { detections, ...classifyAndRank(detections) };
  }

  // Detection has a real noise floor (circle-center jitter, jack-position
  // jitter feeding into every distance) — two bowls whose distances differ by
  // less than this, in bowl-diameter units, aren't reliably orderable. This
  // is a rough starting estimate pending real-photo data; see
  // test/fixtures/real/.
  const TIE_EPSILON = 0.15;

  // Real lawn bowls scoring: a team's score is how many of its bowls sit
  // closer to the jack than the other team's closest bowl. Since assignments
  // is parallel to ranking (already sorted closest-first, ranking[i].dist in
  // bowl-diameter units), the score is the leading run of same-team entries
  // — camera can't reliably read faded markings, so a human assigns team
  // membership by tapping, closest bowl first, and can stop as soon as the
  // other team's first bowl shows up.
  //
  // The one boundary that can actually flip the reported score is between
  // the last counted bowl and the first opposing one — if those two are
  // within TIE_EPSILON of each other, the true order isn't something this
  // measurement can resolve, so the result is flagged tooClose rather than
  // stated as certain.
  function computeScore(ranking, assignments, epsilon) {
    if (epsilon === undefined) epsilon = TIE_EPSILON;

    if (assignments.length === 0) {
      return { team: null, count: 0, pending: false, tooClose: false };
    }

    const leadTeam = assignments[0];
    if (!leadTeam) {
      return { team: null, count: 0, pending: true, tooClose: false };
    }

    let count = 0;
    for (let i = 0; i < assignments.length; i++) {
      const team = assignments[i];
      if (team === leadTeam) {
        count++;
      } else if (team === null) {
        return { team: leadTeam, count, pending: true, tooClose: false };
      } else {
        const gap = ranking[i].dist - ranking[i - 1].dist;
        return { team: leadTeam, count, pending: false, tooClose: gap < epsilon };
      }
    }
    return { team: leadTeam, count, pending: false, tooClose: false };
  }

  return {
    WORK_WIDTH,
    JACK_RADIUS_RATIO,
    CONFIDENT_JACK_WORK_RADIUS_PX,
    JACK_TO_NEAREST_BOWL_RATIO,
    TIE_EPSILON,
    detectCircles,
    classifyAndRank,
    detectAndRank,
    computeScore,
  };
});
