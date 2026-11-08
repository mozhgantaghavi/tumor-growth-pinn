// Legendre-Gauss-Lobatto spectral solver for u_t = D u_xx + f(u; rho) with no-flux
// walls. This file mirrors src/tumorpinn/{spectral,solver}.py so that the dashboard
// simulations agree with the Python pipeline (see tests/js/parity.test.mjs).
// Matrices are dense, row-major Float64Arrays.

export const MODELS = {
  fisher_kpp: {
    title: 'Fisher–KPP (logistic)',
    f: (u, r) => r * u * (1 - u),
    tex: 'u_t = D\\,u_{xx} + \\rho\\,u\\,(1-u)',
  },
  gompertz: {
    title: 'Gompertz',
    f: (u, r) => -r * u * Math.log(Math.max(u, 1e-8)),
    tex: 'u_t = D\\,u_{xx} - \\rho\\,u\\,\\ln u',
  },
};

// ---------------------------------------------------------------- linear algebra
export function matmul(A, B, n, k, m) {
  const C = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const a = A[i * k + p];
      if (a === 0) continue;
      const bo = p * m, co = i * m;
      for (let j = 0; j < m; j++) C[co + j] += a * B[bo + j];
    }
  }
  return C;
}

export function matvec(A, x, n, m, out) {
  out = out || new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const o = i * m;
    for (let j = 0; j < m; j++) s += A[o + j] * x[j];
    out[i] = s;
  }
  return out;
}

export function inverse(A, n) {
  // Gauss-Jordan with partial pivoting.
  const W = new Float64Array(n * 2 * n);
  const w = 2 * n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) W[i * w + j] = A[i * n + j];
    W[i * w + n + i] = 1;
  }
  for (let c = 0; c < n; c++) {
    let piv = c, best = Math.abs(W[c * w + c]);
    for (let r = c + 1; r < n; r++) {
      const v = Math.abs(W[r * w + c]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-300) throw new Error('singular matrix');
    if (piv !== c) {
      for (let j = 0; j < w; j++) {
        const t = W[c * w + j]; W[c * w + j] = W[piv * w + j]; W[piv * w + j] = t;
      }
    }
    const d = 1 / W[c * w + c];
    for (let j = 0; j < w; j++) W[c * w + j] *= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = W[r * w + c];
      if (f === 0) continue;
      for (let j = 0; j < w; j++) W[r * w + j] -= f * W[c * w + j];
    }
  }
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i * n + j] = W[i * w + n + j];
  return out;
}

export function solveLinear(A, b, n) {
  const Ai = inverse(A, n);
  return matvec(Ai, b, n, n);
}

// ---------------------------------------------------------------- LGL machinery
export function lglNodes(M) {
  const N = M, N1 = M + 1;
  let x = new Float64Array(N1);
  for (let k = 0; k < N1; k++) x[k] = Math.cos((Math.PI * k) / N);
  let PN = new Float64Array(N1), PNm1 = new Float64Array(N1);
  for (let it = 0; it < 100; it++) {
    let maxd = 0;
    for (let j = 0; j < N1; j++) {
      let p0 = 1, p1 = x[j];
      for (let k = 2; k <= N; k++) {
        const p2 = ((2 * k - 1) * x[j] * p1 - (k - 1) * p0) / k;
        p0 = p1; p1 = p2;
      }
      PNm1[j] = p0; PN[j] = p1;
      const xn = x[j] - (x[j] * PN[j] - PNm1[j]) / (N1 * PN[j]);
      maxd = Math.max(maxd, Math.abs(xn - x[j]));
      x[j] = xn;
    }
    if (maxd < 1e-15) break;
  }
  // final evaluation of L_M at the converged nodes, then ascending order
  const xs = new Float64Array(N1), LM = new Float64Array(N1);
  for (let j = 0; j < N1; j++) {
    let p0 = 1, p1 = x[j];
    for (let k = 2; k <= N; k++) {
      const p2 = ((2 * k - 1) * x[j] * p1 - (k - 1) * p0) / k;
      p0 = p1; p1 = p2;
    }
    xs[N - j] = x[j];
    LM[N - j] = N === 1 ? x[j] : p1;
  }
  xs[0] = -1; xs[N] = 1;
  const w = new Float64Array(N1);
  for (let j = 0; j < N1; j++) w[j] = 2 / (N * N1 * LM[j] * LM[j]);
  return { xi: xs, w, LM };
}

export function barycentricWeights(x) {
  const n = x.length;
  const logs = new Float64Array(n), sg = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let l = 0, s = 1;
    for (let k = 0; k < n; k++) {
      if (k === j) continue;
      const d = x[j] - x[k];
      l -= Math.log(Math.abs(d));
      if (d < 0) s = -s;
    }
    logs[j] = l; sg[j] = s;
  }
  let mx = -Infinity;
  for (let j = 0; j < n; j++) mx = Math.max(mx, logs[j]);
  const w = new Float64Array(n);
  for (let j = 0; j < n; j++) w[j] = sg[j] * Math.exp(logs[j] - mx);
  return w;
}

export function interpMatrix(nodes, xq) {
  const n = nodes.length, q = xq.length;
  const w = barycentricWeights(nodes);
  const A = new Float64Array(q * n);
  for (let i = 0; i < q; i++) {
    let hit = -1, sum = 0;
    for (let j = 0; j < n; j++) {
      const d = xq[i] - nodes[j];
      if (Math.abs(d) < 1e-14) { hit = j; break; }
      const t = w[j] / d;
      A[i * n + j] = t; sum += t;
    }
    if (hit >= 0) {
      for (let j = 0; j < n; j++) A[i * n + j] = 0;
      A[i * n + hit] = 1;
    } else {
      for (let j = 0; j < n; j++) A[i * n + j] /= sum;
    }
  }
  return A;
}

export function makeGrid(M, length = 1) {
  const n = M + 1, m = M - 1;
  const { xi, w, LM } = lglNodes(M);
  const scale = 2 / length;
  const D1 = new Float64Array(n * n);
  for (let a = 0; a < n; a++) {
    for (let b = 0; b < n; b++) {
      if (a !== b) D1[a * n + b] = (LM[a] / (LM[b] * (xi[a] - xi[b]))) * scale;
    }
  }
  D1[0] = (-M * (M + 1) / 4) * scale;
  D1[n * n - 1] = (M * (M + 1) / 4) * scale;
  const D2 = matmul(D1, D1, n, n, n);
  // eliminate boundary unknowns: D1[b,b] u_b = -D1[b,int] u_int
  const a00 = D1[0], a01 = D1[M], a10 = D1[M * n], a11 = D1[M * n + M];
  const det = a00 * a11 - a01 * a10;
  const inv = [a11 / det, -a01 / det, -a10 / det, a00 / det];
  const P = new Float64Array(n * m);
  for (let j = 0; j < m; j++) {
    const r0 = D1[j + 1], rM = D1[M * n + j + 1];
    P[0 * m + j] = -(inv[0] * r0 + inv[1] * rM);
    P[M * m + j] = -(inv[2] * r0 + inv[3] * rM);
    P[(j + 1) * m + j] = 1;
  }
  const D2int = D2.subarray(n, n + m * n); // rows 1..M-1
  const L = matmul(D2int, P, m, n, m);
  return {
    M, n, m, length, xi, w: w.map((v) => (v * length) / 2),
    x: xi.map((v) => ((v + 1) * length) / 2), D1, D2, P, L,
  };
}

// ---------------------------------------------------------------- time stepping
export function gaussianIC(x, seeds) {
  const u = new Float64Array(x.length);
  for (const s of seeds) {
    for (let i = 0; i < x.length; i++) u[i] += s.a * Math.exp(-0.5 * ((x[i] - s.c) / s.s) ** 2);
  }
  for (let i = 0; i < u.length; i++) u[i] = Math.min(1, Math.max(0, u[i]));
  return u;
}

/** Returns Float64Array (nSnap * (M+1)): snapshots at the LGL nodes. */
export function simulate(grid, { u0, D, rho, model = 'fisher_kpp', T = 1, nSnap = 32, substeps = 8 }) {
  const { M, n, m, P, L } = grid;
  const f = MODELS[model].f;
  const nsteps = (nSnap - 1) * substeps, h = T / nsteps;
  const A = new Float64Array(m * m), B = new Float64Array(m * m);
  for (let i = 0; i < m * m; i++) {
    A[i] = -0.5 * h * D * L[i];
    B[i] = 0.5 * h * D * L[i];
  }
  for (let i = 0; i < m; i++) { A[i * m + i] += 1; B[i * m + i] += 1; }
  const Ainv = inverse(A, m);
  const G = matmul(Ainv, B, m, m, m);
  const out = new Float64Array(nSnap * n);
  let ui = new Float64Array(m);
  for (let i = 0; i < m; i++) ui[i] = u0[i + 1];
  matvec(P, ui, n, m, out.subarray(0, n));
  let fPrev = new Float64Array(m), fCur = new Float64Array(m), fx = new Float64Array(m);
  for (let i = 0; i < m; i++) fPrev[i] = f(ui[i], rho);
  const t1 = new Float64Array(m), t2 = new Float64Array(m);
  for (let s = 0; s < nsteps; s++) {
    for (let i = 0; i < m; i++) fCur[i] = f(ui[i], rho);
    if (s === 0) fx.set(fCur);
    else for (let i = 0; i < m; i++) fx[i] = 1.5 * fCur[i] - 0.5 * fPrev[i];
    matvec(G, ui, m, m, t1);
    matvec(Ainv, fx, m, m, t2);
    for (let i = 0; i < m; i++) ui[i] = t1[i] + h * t2[i];
    fPrev.set(fCur);
    if ((s + 1) % substeps === 0) matvec(P, ui, n, m, out.subarray(((s + 1) / substeps) * n, ((s + 1) / substeps + 1) * n));
  }
  return out;
}

/** Interpolate node snapshots to a uniform grid of nx points. Returns {x, u, A}. */
export function toUniform(grid, snaps, nSnap, nx) {
  const x = new Float64Array(nx);
  for (let i = 0; i < nx; i++) x[i] = (grid.length * i) / (nx - 1);
  const A = interpMatrix(grid.x, x);
  const u = new Float64Array(nSnap * nx);
  for (let k = 0; k < nSnap; k++) {
    matvec(A, snaps.subarray(k * grid.n, (k + 1) * grid.n), nx, grid.n, u.subarray(k * nx, (k + 1) * nx));
  }
  for (let i = 0; i < u.length; i++) u[i] = Math.min(1, Math.max(0, u[i]));
  return { x, u, A };
}

// ---------------------------------------------------------------- diagnostics
export function frontPosition(u, x, nSnap, nx, level = 0.5) {
  const out = new Float64Array(nSnap).fill(NaN);
  for (let k = 0; k < nSnap; k++) {
    let j = -1;
    for (let i = 0; i < nx; i++) if (u[k * nx + i] >= level) j = i;
    if (j < 0) continue;
    if (j + 1 < nx) {
      const y0 = u[k * nx + j], y1 = u[k * nx + j + 1];
      out[k] = x[j] + ((level - y0) * (x[j + 1] - x[j])) / (y1 - y0);
    } else out[k] = x[j];
  }
  return out;
}

export function burden(grid, snaps, nSnap) {
  const out = new Float64Array(nSnap);
  for (let k = 0; k < nSnap; k++) {
    let s = 0;
    for (let i = 0; i < grid.n; i++) s += grid.w[i] * snaps[k * grid.n + i];
    out[k] = s;
  }
  return out;
}

// ---------------------------------------------------------------- noise + RNG
export function rng(seed) {
  let a = seed >>> 0;
  const uni = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => {
    const u1 = Math.max(uni(), 1e-12), u2 = uni();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
  return { uni, gauss };
}

// ---------------------------------------------------------------- inverse problem
export function legendreVander(xs, deg) {
  const n = xs.length, V = new Float64Array(n * (deg + 1));
  for (let i = 0; i < n; i++) {
    V[i * (deg + 1)] = 1;
    if (deg >= 1) V[i * (deg + 1) + 1] = xs[i];
    for (let k = 2; k <= deg; k++) {
      V[i * (deg + 1) + k] = ((2 * k - 1) * xs[i] * V[i * (deg + 1) + k - 1] - (k - 1) * V[i * (deg + 1) + k - 2]) / k;
    }
  }
  return V;
}

/** Ridge-regularised Legendre fit of the observed t=0 row, sampled at the LGL nodes. */
export function smoothIC(row0, xu, grid, deg = 40, ridge = 1e-5) {
  const q = deg + 1, nx = xu.length;
  const xs = Array.from(xu, (v) => (2 * v) / grid.length - 1);
  const V = legendreVander(xs, deg);
  const A = new Float64Array(q * q), rhs = new Float64Array(q);
  for (let i = 0; i < q; i++) {
    for (let j = 0; j < q; j++) {
      let s = 0;
      for (let r = 0; r < nx; r++) s += V[r * q + i] * V[r * q + j];
      A[i * q + j] = s;
    }
    A[i * q + i] += ridge * i ** 4;
    let s = 0;
    for (let r = 0; r < nx; r++) s += V[r * q + i] * row0[r];
    rhs[i] = s;
  }
  const c = solveLinear(A, rhs, q);
  const Vn = legendreVander(Array.from(grid.xi), deg);
  const u = new Float64Array(grid.n);
  for (let i = 0; i < grid.n; i++) {
    let s = 0;
    for (let k = 0; k < q; k++) s += Vn[i * q + k] * c[k];
    u[i] = Math.min(1, Math.max(0, s));
  }
  return u;
}

/** Levenberg-Marquardt in (log10 D, log10 rho) with box bounds; records the path. */
export function fitLM(resFn, theta0, lo, hi, { maxIter = 40, fdStep = 1e-3, onStep } = {}) {
  let th = theta0.slice();
  let r = resFn(th);
  let cost = 0.5 * r.reduce((a, v) => a + v * v, 0);
  let lam = 1e-2;
  const path = [{ theta: th.slice(), cost }];
  let nfev = 1;
  for (let it = 0; it < maxIter; it++) {
    const J = [];
    for (let p = 0; p < 2; p++) {
      const t2 = th.slice();
      const h = th[p] + fdStep > hi[p] ? -fdStep : fdStep;
      t2[p] += h;
      const r2 = resFn(t2); nfev++;
      const col = new Float64Array(r.length);
      for (let i = 0; i < r.length; i++) col[i] = (r2[i] - r[i]) / h;
      J.push(col);
    }
    const JtJ = [[0, 0], [0, 0]], g = [0, 0];
    for (let a = 0; a < 2; a++) {
      for (let b = 0; b < 2; b++) { let s = 0; for (let i = 0; i < r.length; i++) s += J[a][i] * J[b][i]; JtJ[a][b] = s; }
      let s = 0; for (let i = 0; i < r.length; i++) s += J[a][i] * r[i]; g[a] = s;
    }
    let improved = false;
    for (let tries = 0; tries < 8; tries++) {
      const a00 = JtJ[0][0] * (1 + lam) + 1e-12, a11 = JtJ[1][1] * (1 + lam) + 1e-12, a01 = JtJ[0][1];
      const det = a00 * a11 - a01 * a01;
      const d0 = -(a11 * g[0] - a01 * g[1]) / det, d1 = -(-a01 * g[0] + a00 * g[1]) / det;
      const tn = [Math.min(hi[0], Math.max(lo[0], th[0] + d0)), Math.min(hi[1], Math.max(lo[1], th[1] + d1))];
      const rn = resFn(tn); nfev++;
      const cn = 0.5 * rn.reduce((a, v) => a + v * v, 0);
      if (cn < cost) {
        const rel = (cost - cn) / cost;
        th = tn; r = rn; cost = cn; lam = Math.max(lam / 4, 1e-8); improved = true;
        path.push({ theta: th.slice(), cost });
        if (onStep) onStep(path);
        if (rel < 1e-7) return { theta: th, cost, path, nfev };
        break;
      }
      lam *= 5;
    }
    if (!improved) break;
  }
  return { theta: th, cost, path, nfev };
}
