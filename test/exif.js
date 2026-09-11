// Reads just enough EXIF from a JPEG to know what lens took it.
//
// This matters more than it looks. Depth is recovered from how large a bowl
// appears, which only works if the focal length is known, and it cannot be
// recovered from the picture itself (see ground.js). Phones record it, so for
// real photos there is no need to guess at all — and knowing the true value
// is also the only way to find out whether the guess the live app has to make
// is any good.
//
// Deliberately minimal and dependency-free: a handful of tags, no orientation
// handling, no maker notes.

const fs = require('fs');

const TAGS = {
  0x010f: 'make',
  0x0110: 'model',
  0x920a: 'focalLengthMm',
  0xa405: 'focalLength35mm',
  0xa002: 'pixelWidth',
  0xa003: 'pixelHeight',
};

// A 35mm frame is 36mm wide, so the equivalent focal length is a focal length
// expressed as if the sensor were that size — which makes it directly
// convertible to pixels for any image width, whatever the real sensor is.
const FILM_WIDTH_MM = 36;

function findExifStart(buffer) {
  const limit = Math.min(buffer.length - 4, 500000);
  for (let i = 0; i < limit; i++) {
    if (buffer[i] === 0xff && buffer[i + 1] === 0xe1 && buffer.toString('ascii', i + 4, i + 8) === 'Exif') {
      return i + 10; // past the marker, length and "Exif\0\0"
    }
  }
  return -1;
}

function read(filePath) {
  let buffer;
  try {
    buffer = fs.readFileSync(filePath);
  } catch (err) {
    return null;
  }

  const base = findExifStart(buffer);
  if (base < 0) return null;

  const littleEndian = buffer.toString('ascii', base, base + 2) === 'II';
  const u16 = o => (littleEndian ? buffer.readUInt16LE(o) : buffer.readUInt16BE(o));
  const u32 = o => (littleEndian ? buffer.readUInt32LE(o) : buffer.readUInt32BE(o));

  const found = {};
  const seen = new Set();

  function readDirectory(offset, depth) {
    if (depth > 3 || offset <= 0 || seen.has(offset)) return;
    seen.add(offset);
    const start = base + offset;
    if (start + 2 > buffer.length) return;

    const count = u16(start);
    for (let i = 0; i < count; i++) {
      const entry = start + 2 + i * 12;
      if (entry + 12 > buffer.length) return;
      const tag = u16(entry);
      const type = u16(entry + 2);
      const length = u32(entry + 4);

      if (tag === 0x8769) { // the Exif sub-directory, where the lens tags live
        readDirectory(u32(entry + 8), depth + 1);
        continue;
      }
      if (!TAGS[tag]) continue;

      let value = null;
      if (type === 3) value = u16(entry + 8);
      else if (type === 4) value = u32(entry + 8);
      else if (type === 5) {
        const at = base + u32(entry + 8);
        if (at + 8 <= buffer.length) {
          const denominator = u32(at + 4);
          value = denominator ? u32(at) / denominator : null;
        }
      } else if (type === 2) {
        const at = length > 4 ? base + u32(entry + 8) : entry + 8;
        if (at + length <= buffer.length) value = buffer.toString('ascii', at, at + length - 1).trim();
      }
      if (value !== null) found[TAGS[tag]] = value;
    }

    const next = u32(start + 2 + count * 12);
    if (next) readDirectory(next, depth + 1);
  }

  readDirectory(u32(base + 4), 0);
  return Object.keys(found).length ? found : null;
}

// Focal length in pixels for an image processed at the given width. Returns
// null when the photo doesn't say, so callers can fall back to a guess
// knowingly rather than by accident.
function focalLengthPx(exif, processingWidth) {
  if (!exif || !exif.focalLength35mm) return null;
  return (exif.focalLength35mm / FILM_WIDTH_MM) * processingWidth;
}

module.exports = { read, focalLengthPx, FILM_WIDTH_MM };
