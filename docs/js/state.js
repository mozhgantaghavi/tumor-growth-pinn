// State shared between the Simulator and the Inference lab.
import { makeGrid, gaussianIC, simulate, toUniform } from './solver.js';

export const state = {
  model: 'fisher_kpp',
  D: 0.01,
  rho: 4.0,
  T: 1.0,
  M: 48,
  nSnap: 49,
  seeds: [
    { a: 0.7, c: 0.3, s: 0.07 },
    { a: 0.0, c: 0.75, s: 0.08 },
  ],
  sigma: 0.0,
};

const gridCache = new Map();
export function getGrid(M, length = 1) {
  const key = `${M}/${length}`;
  if (!gridCache.has(key)) gridCache.set(key, makeGrid(M, length));
  return gridCache.get(key);
}

/** Number of solver steps per snapshot so that rho * dt stays small. */
export function substepsFor(rho, T, nSnap, floor = 5) {
  const steps = Math.max(240, Math.ceil(40 * rho * T));
  return Math.max(floor, Math.ceil(steps / (nSnap - 1)));
}

export function activeSeeds(seeds = state.seeds) {
  return seeds.filter((s) => s.a > 1e-6);
}

/** Run the forward model with the given overrides; returns node + uniform snapshots. */
export function runForward(opts = {}) {
  const o = { ...state, ...opts };
  const grid = getGrid(o.M);
  const u0 = gaussianIC(grid.x, activeSeeds(o.seeds));
  const substeps = o.substeps || substepsFor(o.rho, o.T, o.nSnap);
  const t0 = performance.now();
  const snaps = simulate(grid, { u0, D: o.D, rho: o.rho, model: o.model, T: o.T, nSnap: o.nSnap, substeps });
  const ms = performance.now() - t0;
  const nx = o.nx || 121;
  const uni = toUniform(grid, snaps, o.nSnap, nx);
  return { grid, snaps, x: uni.x, u: uni.u, nx, nSnap: o.nSnap, ms, u0 };
}
