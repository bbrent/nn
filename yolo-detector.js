// YOLO-based detection: pure pre/post-processing (letterbox resize math,
// tensor conversion, raw-output decode + NMS) with zero ONNX dependency, so
// it's fully unit-testable in Node without a native runtime. The actual
// session.run() call is a thin wrapper the caller provides an `ort`-shaped
// session for — in the browser that's onnxruntime-web (loaded via CDN
// script tag), no npm/native binary needed there either.
//
// Model contract (yolov8n.onnx, stock COCO weights): input "images"
// [1,3,640,640] float32 RGB/255, output "output0" [1,84,8400] (4 box coords
// + 80 class scores per anchor, no separate objectness channel).
//
// Shared with the browser app (window.LawnBowlsYolo) and the Node test
// harness (decode/NMS validated against Python's real inference output).

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.LawnBowlsYolo = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  const INPUT_SIZE = 640;
  const NUM_ANCHORS = 8400;
  const NUM_CLASSES = 80;

  // COCO class indices for "round bowl-like object". sports ball (32) is
  // the main hit in practice; bowl (45) shows up occasionally too (see
  // test/fixtures/real/ — a lawn bowl's ring/spiral pattern doesn't match
  // any COCO ball's texture well, so the model sometimes reaches for the
  // nearest round-object class instead).
  const DEFAULT_ALLOWED_CLASSES = new Set([32, 45]);

  // Deliberately very low, because on real photographs confidence tracks how
  // dark the bowl is rather than how sure the model is that something round is
  // there. Measured on test/fixtures/real: in one clear, well-lit, nearly
  // overhead shot the blue bowl scored 0.73 and the yellow jack 0.64, while
  // the three black bowls scored 0.17, 0.05 and 0.04 — and every one of those
  // five detections landed within a few pixels of a real object, with nothing
  // spurious at all. A threshold of 0.1 therefore did not filter noise, it
  // filtered out the black bowls specifically, which is most of a real set.
  // Across the five real photos, dropping to this value found half again as
  // many objects (12 detections to 18) and invented none.
  //
  // Letting weak detections through is safe here in a way it would not be in a
  // bare detector, because everything downstream is built to disbelieve them:
  // a frame's detections must agree on one flat green, a landmark must be seen
  // several times before it can be scored, and anything seen once or twice and
  // then looked straight at is dropped.
  const DEFAULT_CONF_THRESHOLD = 0.03;
  const DEFAULT_IOU_THRESHOLD = 0.45;

  // Jack is ~0.54x a bowl's diameter; same physical reasoning as detection.js.
  const JACK_RADIUS_RATIO = 0.65;
  // Jack must be clearly smaller than the nearest bowl, not just barely —
  // same rationale as detection.js's JACK_TO_NEAREST_BOWL_RATIO.
  const JACK_TO_NEAREST_BOWL_RATIO = 0.8;
  // Unlike Hough, every detection here already carries a real model
  // confidence score — a much more direct signal than an absolute pixel-size
  // floor. Rough starting estimate pending real tuning (see
  // test/fixtures/real/): jack detections tend to score lower than bowls
  // (smaller, less textured), so this can't be set too high.
  const MIN_JACK_SCORE = 0.1;

  // Fit srcWidth x srcHeight into a targetSize x targetSize square, preserving
  // aspect ratio, padding the rest — matches Ultralytics' own preprocessing,
  // which the model was trained/exported against.
  function computeLetterbox(srcWidth, srcHeight, targetSize) {
    targetSize = targetSize || INPUT_SIZE;
    const scale = Math.min(targetSize / srcWidth, targetSize / srcHeight);
    const newWidth = Math.round(srcWidth * scale);
    const newHeight = Math.round(srcHeight * scale);
    const padX = (targetSize - newWidth) / 2;
    const padY = (targetSize - newHeight) / 2;
    return { scale, newWidth, newHeight, padX, padY, targetSize };
  }

  // imageData: {data: Uint8ClampedArray (RGBA), width, height} — already
  // letterboxed to targetSize x targetSize by the caller (a canvas draw,
  // browser or node-canvas, both expose the same getImageData() shape).
  // Returns a flat NCHW float32 tensor, normalized to [0,1].
  function imageDataToTensor(imageData) {
    const { data, width, height } = imageData;
    const size = width * height;
    const tensorData = new Float32Array(3 * size);
    for (let i = 0; i < size; i++) {
      tensorData[i] = data[i * 4] / 255;
      tensorData[size + i] = data[i * 4 + 1] / 255;
      tensorData[2 * size + i] = data[i * 4 + 2] / 255;
    }
    return { data: tensorData, dims: [1, 3, height, width] };
  }

  function iou(a, b) {
    const ax1 = a.x - a.w / 2, ay1 = a.y - a.h / 2, ax2 = a.x + a.w / 2, ay2 = a.y + a.h / 2;
    const bx1 = b.x - b.w / 2, by1 = b.y - b.h / 2, bx2 = b.x + b.w / 2, by2 = b.y + b.h / 2;
    const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1);
    const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2);
    const iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    const areaA = a.w * a.h, areaB = b.w * b.h;
    return inter <= 0 ? 0 : inter / (areaA + areaB - inter);
  }

  function nms(candidates, iouThreshold) {
    const sorted = [...candidates].sort((a, b) => b.score - a.score);
    const kept = [];
    for (const c of sorted) {
      if (kept.every(k => iou(c, k) < iouThreshold)) kept.push(c);
    }
    return kept;
  }

  // outputData: flat Float32Array, layout [1,84,8400] row-major (channel-major:
  // outputData[c*8400 + i] is channel c's value for anchor i). Coordinates in
  // the raw output are in the 640x640 letterboxed space; this un-letterboxes
  // them back into the original source image's pixel coordinates.
  function decodeOutput(outputData, letterbox, opts) {
    opts = opts || {};
    const confThreshold = opts.confThreshold !== undefined ? opts.confThreshold : DEFAULT_CONF_THRESHOLD;
    const iouThreshold = opts.iouThreshold !== undefined ? opts.iouThreshold : DEFAULT_IOU_THRESHOLD;
    const allowedClasses = opts.allowedClasses || DEFAULT_ALLOWED_CLASSES;

    const candidates = [];
    for (let i = 0; i < NUM_ANCHORS; i++) {
      let bestScore = -Infinity;
      let bestClass = -1;
      for (let c = 0; c < NUM_CLASSES; c++) {
        const score = outputData[(4 + c) * NUM_ANCHORS + i];
        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }
      if (bestScore < confThreshold) continue;
      if (allowedClasses && !allowedClasses.has(bestClass)) continue;

      const cx = outputData[0 * NUM_ANCHORS + i];
      const cy = outputData[1 * NUM_ANCHORS + i];
      const w = outputData[2 * NUM_ANCHORS + i];
      const h = outputData[3 * NUM_ANCHORS + i];

      candidates.push({
        x: (cx - letterbox.padX) / letterbox.scale,
        y: (cy - letterbox.padY) / letterbox.scale,
        w: w / letterbox.scale,
        h: h / letterbox.scale,
        score: bestScore,
        classIndex: bestClass,
      });
    }

    return nms(candidates, iouThreshold).map(c => ({
      x: c.x,
      y: c.y,
      r: (c.w + c.h) / 4,
      score: c.score,
      classIndex: c.classIndex,
    }));
  }

  // Mirrors detection.js's classifyAndRank shape/logic (jack = smallest
  // circle clearly separated from the rest, ranked by jack-centered
  // distance in bowl-diameter units), swapping Hough's absolute-pixel
  // confidence floor for the model's own per-detection score.
  function classifyAndRank(detections) {
    if (detections.length === 0) {
      return { jack: null, bowls: [], ranking: [], usable: false, reason: 'no detections' };
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

    const ranking = bowls.map(b => {
      const localX = jack ? (b.x - jack.x) / avgBowlDiameter : null;
      const localY = jack ? (b.y - jack.y) / avgBowlDiameter : null;
      return { bowl: b, dist: jack ? Math.hypot(localX, localY) : null, localX, localY };
    });
    if (jack) ranking.sort((a, b) => a.dist - b.dist);

    let usable = true;
    let reason = null;
    const minBowlR = bowls.length ? Math.min(...bowls.map(b => b.r)) : null;
    if (!jack) {
      usable = false;
      reason = 'no jack identified';
    } else if (jack.score < MIN_JACK_SCORE) {
      usable = false;
      reason = 'jack detection confidence too low to trust';
    } else if (minBowlR !== null && jack.r / minBowlR > JACK_TO_NEAREST_BOWL_RATIO) {
      usable = false;
      reason = 'no bowl is clearly smaller than the rest — jack may be undetected or out of frame';
    }

    return { jack, bowls, ranking, usable, reason };
  }

  // Top-level glue mirroring detection.js's detectAndRank(cv, srcMat) shape.
  // imageData640 must already be letterboxed to 640x640 by the caller.
  async function detectAndRank(ort, session, imageData640, letterbox, opts) {
    const tensor = imageDataToTensor(imageData640);
    const feeds = { images: new ort.Tensor('float32', tensor.data, tensor.dims) };
    const results = await session.run(feeds);
    const output = results.output0;
    const detections = decodeOutput(output.data, letterbox, opts);
    return { detections, ...classifyAndRank(detections) };
  }

  return {
    INPUT_SIZE,
    NUM_ANCHORS,
    NUM_CLASSES,
    DEFAULT_ALLOWED_CLASSES,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    JACK_RADIUS_RATIO,
    JACK_TO_NEAREST_BOWL_RATIO,
    MIN_JACK_SCORE,
    computeLetterbox,
    imageDataToTensor,
    iou,
    nms,
    decodeOutput,
    classifyAndRank,
    detectAndRank,
  };
});
