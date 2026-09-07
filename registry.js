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
    serialize,
    deserialize,
  };
});
