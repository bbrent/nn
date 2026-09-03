Drop real phone photos of bowls-on-turf here (`.jpg`/`.png`) and run `npm test`.

There's no ground truth for these, so they aren't asserted against — the
harness just prints how many circles it found and whether a jack was
identified, for manual sanity-checking against what you can see in the
photo. This is where lighting, turf texture, motion blur, and real bowl
coloring will actually stress-test the detection parameters tuned against
the synthetic fixtures.
