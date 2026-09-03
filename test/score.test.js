// Pure-logic tests for computeScore — no images/opencv involved, since this
// is just leading-run arithmetic over a team-assignment array.

const LawnBowlsDetection = require('../detection.js');

const cases = [
  { assignments: [], expect: { team: null, count: 0, pending: false } },
  { assignments: [null, 'mine', 'theirs'], expect: { team: null, count: 0, pending: true } },
  { assignments: ['mine', 'mine', 'theirs', 'mine'], expect: { team: 'mine', count: 2, pending: false } },
  { assignments: ['theirs', 'theirs', 'theirs'], expect: { team: 'theirs', count: 3, pending: false } },
  { assignments: ['mine', null, 'theirs'], expect: { team: 'mine', count: 1, pending: true } },
  { assignments: ['mine'], expect: { team: 'mine', count: 1, pending: false } },
];

function run() {
  const failures = [];
  cases.forEach(({ assignments, expect }, i) => {
    const got = LawnBowlsDetection.computeScore(assignments);
    const matches = got.team === expect.team && got.count === expect.count && got.pending === expect.pending;
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
