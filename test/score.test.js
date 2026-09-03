// Pure-logic tests for computeScore — no images/opencv involved, since this
// is just leading-run arithmetic over a ranking + team-assignment array.

const LawnBowlsDetection = require('../detection.js');

// ranking here only needs .dist per entry, in ascending (closest-first) order.
function ranking(dists) {
  return dists.map(dist => ({ dist }));
}

const cases = [
  {
    ranking: ranking([]),
    assignments: [],
    expect: { team: null, count: 0, pending: false, tooClose: false },
  },
  {
    ranking: ranking([0.5, 1.0, 1.5]),
    assignments: [null, 'mine', 'theirs'],
    expect: { team: null, count: 0, pending: true, tooClose: false },
  },
  {
    // Clear gap (0.6) at the mine/theirs boundary (index 1->2) — confident.
    ranking: ranking([0.5, 1.0, 1.6, 2.5]),
    assignments: ['mine', 'mine', 'theirs', 'mine'],
    expect: { team: 'mine', count: 2, pending: false, tooClose: false },
  },
  {
    ranking: ranking([0.5, 1.0, 1.5]),
    assignments: ['theirs', 'theirs', 'theirs'],
    expect: { team: 'theirs', count: 3, pending: false, tooClose: false },
  },
  {
    ranking: ranking([0.5, 1.0, 1.5]),
    assignments: ['mine', null, 'theirs'],
    expect: { team: 'mine', count: 1, pending: true, tooClose: false },
  },
  {
    ranking: ranking([0.5]),
    assignments: ['mine'],
    expect: { team: 'mine', count: 1, pending: false, tooClose: false },
  },
  {
    // Boundary gap (0.05) well under TIE_EPSILON (0.15) — too close to trust.
    ranking: ranking([0.5, 1.0, 1.05]),
    assignments: ['mine', 'mine', 'theirs'],
    expect: { team: 'mine', count: 2, pending: false, tooClose: true },
  },
  {
    // Tie further out doesn't matter — the decisive boundary (0->1) has a clear gap.
    ranking: ranking([0.5, 1.5, 1.52]),
    assignments: ['mine', 'theirs', 'theirs'],
    expect: { team: 'mine', count: 1, pending: false, tooClose: false },
  },
];

function run() {
  const failures = [];
  cases.forEach(({ ranking: r, assignments, expect }, i) => {
    const got = LawnBowlsDetection.computeScore(r, assignments);
    const matches =
      got.team === expect.team &&
      got.count === expect.count &&
      got.pending === expect.pending &&
      got.tooClose === expect.tooClose;
    if (!matches) {
      failures.push(
        `case ${i} [${assignments.join(',')}]: expected ${JSON.stringify(expect)}, got ${JSON.stringify(got)}`
      );
    }
  });
  return { name: 'computeScore', total: cases.length, failures };
}

module.exports = { run };

if (require.main === module) {
  const result = run();
  if (result.failures.length === 0) {
    console.log(`PASS  computeScore (${result.total} cases)`);
    process.exit(0);
  } else {
    console.log(`FAIL  computeScore`);
    result.failures.forEach(f => console.log(`        - ${f}`));
    process.exit(1);
  }
}
