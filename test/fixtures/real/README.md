# Real photo fixtures

One folder per end. Every image inside a folder must be **the same head of
bowls, photographed from different positions** — that is exactly what the app
sees during a scan, and it is the only way to test the parts that combine
several views into one map.

```
real/
  end-01/
    view-1.jpg
    view-2.jpg        # same bowls, moved round a bit
    view-3.jpg
    end.json          # optional, see below
```

Run them through the real pipeline with:

```
npm run real-scan            # every end
npm run real-scan end-07     # just one
```

## Taking the photos

- Walk around the head and take 4-8 shots from **different positions**, not
  just different zooms from one spot. Overlap matters more than coverage: each
  shot should share several bowls with the one before it, so the views can be
  tied together.
- The jack only needs to be in **one** of them.
- Hold the phone the way you would when scoring — chest height, angled down.
  Deliberately include some awkward angles; those are the ones worth testing.
- Don't move any bowls between shots.

## end.json (all fields optional)

Nothing here is required — a folder of bare images already tells us whether
the views tie together, how much the map disagrees with itself between views,
and whether the bowl count comes out right. Each field you can add pins down
something more.

```json
{
  "note": "afternoon sun, long shadows across the head",
  "bowls": 4,
  "jackVisible": true,
  "focalLength35mm": 24,
  "measured": [
    { "bowls": ["nearest", "second"], "metresApart": 0.18 },
    { "fromJack": "nearest", "metres": 0.31 }
  ]
}
```

- `bowls` — how many bowls are really in the head, not counting the jack. Lets
  the harness report whether anything was missed or invented.
- `focalLength35mm` — only needed if the photos have no EXIF. Most phones
  record it, and the harness reads it automatically; it matters because depth
  is recovered from apparent size, so the lens has to be known.
- `measured` — anything you can put a tape measure on. This is the valuable
  one: it is the only way to check the app's absolute distances against
  reality rather than against its own consistency. Even a single measurement
  between the two closest bowls is worth having, since those are the ones a
  score actually turns on.
