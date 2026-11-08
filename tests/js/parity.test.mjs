// Checks that the dashboard's JavaScript solver reproduces the Python reference
// trajectories stored in docs/data/parity.json (run: node --test tests/js).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { makeGrid, gaussianIC, simulate, toUniform, lglNodes, MODELS, inverse, matmul } from '../../docs/js/solver.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, '../../docs/data/parity.json'), 'utf8'));

test('LGL nodes are symmetric, ascending and include the end points', () => {
  const { xi, w } = lglNodes(16);
  assert.equal(xi[0], -1);
  assert.equal(xi[16], 1);
  for (let i = 0; i < 8; i++) assert.ok(Math.abs(xi[i] + xi[16 - i]) < 1e-13);
  for (let i = 1; i < xi.length; i++) assert.ok(xi[i] > xi[i - 1]);
  assert.ok(Math.abs(w.reduce((a, b) => a + b, 0) - 2) < 1e-13);
});

test('collocation matrix differentiates a smooth function spectrally', () => {
  const g = makeGrid(32, 1);
  const f = g.x.map((x) => Math.exp(Math.sin(3 * x)));
  const df = g.x.map((x) => 3 * Math.cos(3 * x) * Math.exp(Math.sin(3 * x)));
  let err = 0;
  for (let i = 0; i < g.n; i++) {
    let s = 0;
    for (let j = 0; j < g.n; j++) s += g.D1[i * g.n + j] * f[j];
    err = Math.max(err, Math.abs(s - df[i]));
  }
  assert.ok(err < 1e-8, `derivative error ${err}`);
});

test('matrix inverse', () => {
  const A = new Float64Array([4, 1, 2, 1, 3, 0, 2, 0, 5]);
  const I = matmul(A, inverse(A, 3), 3, 3, 3);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) assert.ok(Math.abs(I[i * 3 + j] - (i === j ? 1 : 0)) < 1e-12);
});

for (const c of fx.cases) {
  test(`JS solver matches Python (${c.model})`, () => {
    const g = makeGrid(fx.M, fx.length);
    const u0 = gaussianIC(g.x, fx.seeds.map((s) => ({ a: s.a, c: s.c * fx.length, s: s.s })));
    const snaps = simulate(g, { u0, D: c.D, rho: c.rho, model: c.model, T: fx.T, nSnap: fx.n_snap, substeps: fx.substeps });
    const { u } = toUniform(g, snaps, fx.n_snap, fx.x.length);
    let err = 0;
    for (let k = 0; k < fx.n_snap; k++) {
      for (let i = 0; i < fx.x.length; i++) {
        // Python stores the unclipped interpolant; the dashboard clips to [0, 1]
        err = Math.max(err, Math.abs(u[k * fx.x.length + i] - Math.min(1, Math.max(0, c.snapshots[k][i]))));
      }
    }
    assert.ok(err < 1e-9, `max deviation ${err}`);
  });
}

test('models registry', () => {
  assert.equal(MODELS.fisher_kpp.f(0.5, 2), 0.5);
  assert.ok(Math.abs(MODELS.gompertz.f(Math.E ** -1, 2) - 2 * Math.E ** -1) < 1e-12);
});
