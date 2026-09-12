// Player registry: each player registers once (name, team, and a gallery of
// embedding views captured from their own bowls at a few angles), and a
// detected bowl is matched against every view in every player's gallery —
// not one signature per player — since registration deliberately captures
// several angles/bowls and matching only needs to hit the closest one.
//
// Team membership genuinely can't be recovered from bowl appearance alone —
// real teams mix players with visually distinct individual styles — so this
// is the actual mechanism, not just a convenience: the registry is a roster
// (which players are on which side), and matching identifies the player,
// which the roster then resolves to a team.
//
// Pure data/logic, no image or ONNX dependency at all — shared with the
// browser app (window.LawnBowlsRegistry) and the Node test harness.

(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.LawnBowlsRegistry = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  // Rough starting estimate from real-photo validation: same-style views
  // scored 0.86-0.93 cosine similarity, different styles scored 0.44-0.67 —
  // comfortably above the gap, pending more real-world tuning.
  const DEFAULT_MATCH_THRESHOLD = 0.75;

  function createRegistry() {
    return { players: [] };
  }

  function addPlayer(registry, name, team) {
    const id = 'p_' + Math.random().toString(36).slice(2, 10);
    registry.players.push({ id, name, team, gallery: [] });
    return id;
  }

  function removePlayer(registry, playerId) {
    registry.players = registry.players.filter(p => p.id !== playerId);
  }

  function addGalleryView(registry, playerId, embedding) {
    const player = registry.players.find(p => p.id === playerId);
    if (!player) return false;
    player.gallery.push(embedding);
    return true;
  }

  function cosineSimilarity(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  }

  // Returns the best match across every player's whole gallery, or null if
  // nothing clears the threshold.
  function matchBowl(registry, embedding, threshold) {
    if (threshold === undefined) threshold = DEFAULT_MATCH_THRESHOLD;

    let best = null;
    let bestSim = -Infinity;
    for (const player of registry.players) {
      for (const view of player.gallery) {
        const sim = cosineSimilarity(embedding, view);
        if (sim > bestSim) {
          bestSim = sim;
          best = player;
        }
      }
    }

    if (!best || bestSim < threshold) return null;
    return { playerId: best.id, name: best.name, team: best.team, similarity: bestSim };
  }

  // Given a few pictures picked out as clear examples of somebody's bowl,
  // works out which of the rest are the same bowl set.
  //
  // This is what makes registering a player bearable. Sorting through every
  // thumbnail by hand is tedious and the pictures are not equally useful —
  // most are half-shadowed, motion-blurred or caught at an angle, and the
  // person can see at a glance which one or two actually show the bowl
  // properly. Those are the ones worth their attention; matching the rest
  // against them is exactly what the appearance embeddings are for.
  //
  // Similarity is taken against the best of the chosen examples rather than
  // their average, for the same reason the registry itself keeps a gallery: a
  // bowl photographed in shade and in sun gives two genuinely different
  // embeddings, and averaging them lands between the two and matches neither.
  //
  // Returns one entry per candidate, in the order given, so callers can show
  // the score alongside each picture rather than only a verdict — with a
  // handful of examples the person is far better placed to judge a borderline
  // one than any threshold is.
  function proposeGroup(embeddings, seedEmbeddings, threshold) {
    if (threshold === undefined) threshold = DEFAULT_MATCH_THRESHOLD;
    return embeddings.map(embedding => {
      let best = -1;
      for (const seed of seedEmbeddings) {
        const similarity = cosineSimilarity(embedding, seed);
        if (similarity > best) best = similarity;
      }
      return {
        similarity: seedEmbeddings.length ? best : null,
        proposed: seedEmbeddings.length > 0 && best >= threshold,
      };
    });
  }

  // localStorage-friendly (de)serialization — embeddings become plain arrays.
  function serialize(registry) {
    return JSON.stringify({
      players: registry.players.map(p => ({
        id: p.id,
        name: p.name,
        team: p.team,
        gallery: p.gallery.map(v => Array.from(v)),
      })),
    });
  }

  function deserialize(json) {
    const parsed = JSON.parse(json);
    return {
      players: parsed.players.map(p => ({
        id: p.id,
        name: p.name,
        team: p.team,
        gallery: p.gallery.map(v => Float32Array.from(v)),
      })),
    };
  }

  return {
    DEFAULT_MATCH_THRESHOLD,
    createRegistry,
    addPlayer,
    removePlayer,
    addGalleryView,
    matchBowl,
    proposeGroup,
    serialize,
    deserialize,
  };
});
