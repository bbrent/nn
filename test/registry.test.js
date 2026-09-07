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

  return { name: 'registry', total: 6, failures };
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
