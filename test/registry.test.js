// Pure-logic tests for registry.js. Uses small hand-built 2D unit vectors at
// known angles (cosine similarity = cos of the angle between them) rather
// than real embeddings, since this only tests the matching/persistence
// logic — the embedding model itself was already validated separately
// against real photos.

const LawnBowlsRegistry = require('../registry.js');

function unitVectorAtAngle(degrees) {
  const rad = (degrees * Math.PI) / 180;
  return new Float32Array([Math.cos(rad), Math.sin(rad)]);
}

function approxEqual(a, b, tol) {
  return Math.abs(a - b) < tol;
}

function run() {
  const failures = [];

  // Case 1: basic add + match — a query close to a registered view matches
  // that player.
  {
    const registry = LawnBowlsRegistry.createRegistry();
    const aliceId = LawnBowlsRegistry.addPlayer(registry, 'Alice', 'mine');
    LawnBowlsRegistry.addGalleryView(registry, aliceId, unitVectorAtAngle(0));

    const query = unitVectorAtAngle(5); // cos(5°) ~ 0.996, well above threshold
    const match = LawnBowlsRegistry.matchBowl(registry, query);
    if (!match || match.playerId !== aliceId) failures.push(`case1: expected match on Alice, got ${JSON.stringify(match)}`);
    if (match && match.team !== 'mine') failures.push(`case1: expected team 'mine', got ${match.team}`);
  }

  // Case 2: a query far from every registered view should not match at all.
  {
    const registry = LawnBowlsRegistry.createRegistry();
    const aliceId = LawnBowlsRegistry.addPlayer(registry, 'Alice', 'mine');
    LawnBowlsRegistry.addGalleryView(registry, aliceId, unitVectorAtAngle(0));

    const query = unitVectorAtAngle(90); // orthogonal, similarity 0
    const match = LawnBowlsRegistry.matchBowl(registry, query);
    if (match !== null) failures.push(`case2: expected no match, got ${JSON.stringify(match)}`);
  }

  // Case 3: matching checks every view in a gallery, not just the first —
  // a player registered from multiple angles should match on whichever
  // view the query is actually closest to.
  {
    const registry = LawnBowlsRegistry.createRegistry();
    const bobId = LawnBowlsRegistry.addPlayer(registry, 'Bob', 'theirs');
    LawnBowlsRegistry.addGalleryView(registry, bobId, unitVectorAtAngle(0));
    LawnBowlsRegistry.addGalleryView(registry, bobId, unitVectorAtAngle(60));

    const query = unitVectorAtAngle(65); // near the second view (60°), far from the first (0°)
    const match = LawnBowlsRegistry.matchBowl(registry, query);
    if (!match || match.playerId !== bobId) failures.push(`case3: expected match on Bob via second gallery view, got ${JSON.stringify(match)}`);
  }

  // Case 4: with two registered players, a query should match whichever is
  // actually closer, not just whoever was registered first.
  {
    const registry = LawnBowlsRegistry.createRegistry();
    const aliceId = LawnBowlsRegistry.addPlayer(registry, 'Alice', 'mine');
    const bobId = LawnBowlsRegistry.addPlayer(registry, 'Bob', 'theirs');
    LawnBowlsRegistry.addGalleryView(registry, aliceId, unitVectorAtAngle(0));
    LawnBowlsRegistry.addGalleryView(registry, bobId, unitVectorAtAngle(80));

    const query = unitVectorAtAngle(75); // closer to Bob's 80° than Alice's 0°
    const match = LawnBowlsRegistry.matchBowl(registry, query);
    if (!match || match.playerId !== bobId) failures.push(`case4: expected match on Bob (closer), got ${JSON.stringify(match)}`);
  }

  // Case 5: removePlayer drops their gallery from future matches.
  {
    const registry = LawnBowlsRegistry.createRegistry();
    const aliceId = LawnBowlsRegistry.addPlayer(registry, 'Alice', 'mine');
    LawnBowlsRegistry.addGalleryView(registry, aliceId, unitVectorAtAngle(0));
    LawnBowlsRegistry.removePlayer(registry, aliceId);

    const match = LawnBowlsRegistry.matchBowl(registry, unitVectorAtAngle(0));
    if (match !== null) failures.push(`case5: expected no match after removal, got ${JSON.stringify(match)}`);
  }

  // Case 6: serialize/deserialize round-trips embeddings and still matches.
  {
    const registry = LawnBowlsRegistry.createRegistry();
    const aliceId = LawnBowlsRegistry.addPlayer(registry, 'Alice', 'mine');
    LawnBowlsRegistry.addGalleryView(registry, aliceId, unitVectorAtAngle(0));

    const restored = LawnBowlsRegistry.deserialize(LawnBowlsRegistry.serialize(registry));
    const match = LawnBowlsRegistry.matchBowl(restored, unitVectorAtAngle(5));
    if (!match || match.name !== 'Alice') failures.push(`case6: expected match on Alice after round-trip, got ${JSON.stringify(match)}`);
    if (!(restored.players[0].gallery[0] instanceof Float32Array)) failures.push('case6: expected gallery views to deserialize as Float32Array');
  }

  // Case 7: proposing a group from a couple of clear examples. Registering a
  // player should not mean sorting every thumbnail by hand — the person picks
  // the pictures that actually show the bowl and the rest are matched against
  // those.
  {
    // Two bowl sets, each with several views scattered a few degrees apart,
    // and the sets far enough apart to be genuinely different-looking.
    const setA = [0, 6, -8, 14].map(unitVectorAtAngle);
    const setB = [70, 76, 64].map(unitVectorAtAngle);
    const all = [...setA, ...setB];

    // One clear example from set A. cos(14 deg) is about 0.97, cos(56 deg)
    // about 0.56, so the threshold sits comfortably between them.
    const proposal = LawnBowlsRegistry.proposeGroup(all, [setA[0]]);
    const chosen = proposal.map(p => p.proposed);

    if (!chosen.slice(0, 4).every(Boolean)) {
      failures.push(`case7: one clear example should propose its whole set, got ${JSON.stringify(chosen)}`);
    }
    if (chosen.slice(4).some(Boolean)) {
      failures.push(`case7: the other player's bowls should not be proposed, got ${JSON.stringify(chosen)}`);
    }
    if (proposal.some(p => typeof p.similarity !== 'number')) {
      failures.push('case7: every candidate should report its similarity so a borderline one can be judged');
    }
    // The example itself scores 1 against itself.
    if (!approxEqual(proposal[0].similarity, 1, 1e-6)) {
      failures.push(`case7: the example should score 1 against itself, got ${proposal[0].similarity}`);
    }
  }

  // Case 8: more examples reach further, and similarity is against the best
  // of them rather than their average. A bowl seen in sun and in shade gives
  // two genuinely different embeddings; averaging them lands between the two
  // and matches neither.
  {
    const near = unitVectorAtAngle(0);
    const far = unitVectorAtAngle(60);
    const between = unitVectorAtAngle(30); // 30 deg from both, 0.87 either way

    const oneExample = LawnBowlsRegistry.proposeGroup([far], [near]);
    if (oneExample[0].proposed) {
      failures.push('case8: 60 degrees away should not be proposed from a single near example');
    }

    const twoExamples = LawnBowlsRegistry.proposeGroup([between], [near, far]);
    if (!twoExamples[0].proposed) {
      failures.push('case8: a view between two examples should be proposed');
    }
    if (!approxEqual(twoExamples[0].similarity, Math.cos(30 * Math.PI / 180), 1e-6)) {
      failures.push(`case8: similarity should be to the nearest example (0.866), got ${twoExamples[0].similarity}`);
    }
  }

  // Case 9: with nothing picked yet there is nothing to propose, and asking
  // must not crash or silently select everything.
  {
    const proposal = LawnBowlsRegistry.proposeGroup([unitVectorAtAngle(0), unitVectorAtAngle(45)], []);
    if (proposal.length !== 2) failures.push('case9: should return one entry per candidate');
    if (proposal.some(p => p.proposed)) failures.push('case9: nothing should be proposed with no examples picked');
    if (proposal.some(p => p.similarity !== null)) {
      failures.push('case9: similarity should be null rather than a made-up number');
    }
  }

  return { name: 'registry', total: 9, failures };
}

module.exports = { run };

if (require.main === module) {
  const result = run();
  if (result.failures.length === 0) {
    console.log(`PASS  registry (${result.total} cases)`);
    process.exit(0);
  } else {
    console.log(`FAIL  registry`);
    result.failures.forEach(f => console.log(`        - ${f}`));
    process.exit(1);
  }
}
