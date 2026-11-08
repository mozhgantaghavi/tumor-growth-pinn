// End-to-end check of the in-browser inverse problem: simulate, add noise, fit.
import test from 'node:test';
import assert from 'node:assert/strict';
import { makeGrid, gaussianIC, simulate, interpMatrix, matvec, fitLM, smoothIC, rng, frontPosition } from '../../docs/js/solver.js';

const NT = 32, NX = 64;
const LO = [Math.log10(2e-3), Math.log10(0.5)], HI = [Math.log10(5e-2), Math.log10(12)];
const seeds = [{ a: 0.7, c: 0.3, s: 0.07 }];

function observe(D, rho, sigma, seed) {
  const g = makeGrid(64, 1);
  const u0 = gaussianIC(g.x, seeds);
  const snaps = simulate(g, { u0, D, rho, model: 'fisher_kpp', T: 1, nSnap: NT, substeps: 12 });
  const xu = Float64Array.from({ length: NX }, (_, i) => i / (NX - 1));
  const A = interpMatrix(g.x, xu);
  const obs = new Float64Array(NT * NX);
  const r = rng(seed);
  for (let k = 0; k < NT; k++) matvec(A, snaps.subarray(k * g.n, (k + 1) * g.n), NX, g.n, obs.subarray(k * NX, (k + 1) * NX));
  for (let i = 0; i < obs.length; i++) obs[i] = Math.min(1, Math.max(0, obs[i])) + sigma * r.gauss();
  return { obs, xu };
}

function fit(obs, xu) {
  const g = makeGrid(40, 1), A = interpMatrix(g.x, xu);
  const u0 = smoothIC(obs.subarray(0, NX), xu, g);
  const res = (th) => {
    const s = simulate(g, { u0, D: 10 ** th[0], rho: 10 ** th[1], model: 'fisher_kpp', T: 1, nSnap: NT, substeps: 6 });
    const out = new Float64Array(NT * NX);
    for (let k = 0; k < NT; k++) matvec(A, s.subarray(k * g.n, (k + 1) * g.n), NX, g.n, out.subarray(k * NX, (k + 1) * NX));
    for (let i = 0; i < out.length; i++) out[i] -= obs[i];
    return out;
  };
  return fitLM(res, [(LO[0] + HI[0]) / 2, (LO[1] + HI[1]) / 2], LO, HI);
}

test('Levenberg-Marquardt recovers (D, rho) from clean data', () => {
  const { obs, xu } = observe(0.012, 3.5, 0, 1);
  const r = fit(obs, xu);
  assert.ok(Math.abs(r.theta[0] - Math.log10(0.012)) < 0.05, `log10 D off: ${r.theta[0]}`);
  assert.ok(Math.abs(r.theta[1] - Math.log10(3.5)) < 0.05, `log10 rho off: ${r.theta[1]}`);
});

test('fit degrades gracefully under 3% noise', () => {
  const { obs, xu } = observe(0.012, 3.5, 0.03, 5);
  const r = fit(obs, xu);
  assert.ok(Math.abs(r.theta[0] - Math.log10(0.012)) < 0.2);
  assert.ok(Math.abs(r.theta[1] - Math.log10(3.5)) < 0.2);
});

test('simulated front follows the Fisher-KPP speed with Bramson correction', () => {
  // smooth start (a jump would seed Gibbs ringing that the reaction amplifies) and a modest
  // rho * T so round-off in the empty region cannot grow into a spurious front
  const L = 8, D = 0.05, rho = 1, T = 12;
  const g = makeGrid(96, L);
  const u0 = new Float64Array(g.n).map((_, i) => 0.999 * 0.5 * (1 - Math.tanh((g.x[i] - 0.4) / 0.1)));
  const snaps = simulate(g, { u0, D, rho, model: 'fisher_kpp', T, nSnap: 25, substeps: 100 });
  const xu = Float64Array.from({ length: 400 }, (_, i) => (L * i) / 399);
  const A = interpMatrix(g.x, xu);
  const u = new Float64Array(25 * 400);
  for (let k = 0; k < 25; k++) matvec(A, snaps.subarray(k * g.n, (k + 1) * g.n), 400, g.n, u.subarray(k * 400, (k + 1) * 400));
  const pos = frontPosition(u, xu, 25, 400);
  const c = 2 * Math.sqrt(D * rho), lam = Math.sqrt(rho / D), t1 = 6, t2 = 12;
  const expected = c - (3 / (2 * lam)) * Math.log(t2 / t1) / (t2 - t1);
  const speed = (pos[24] - pos[12]) / (t2 - t1);
  assert.ok(Math.abs(speed - expected) / expected < 0.05, `speed ${speed} vs ${expected}`);
});
