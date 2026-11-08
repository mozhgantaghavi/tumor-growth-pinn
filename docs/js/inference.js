import { fitLM, interpMatrix, matvec, rng, simulate, smoothIC } from './solver.js';
import { state, getGrid, runForward, substepsFor } from './state.js';
import { $, cssVar, densityScale, divergingScale, draw, fmt, nextFrame, sleep, slider } from './common.js';

const NT = 32, NX = 64, FIT_M = 40, FIT_SUBSTEPS = 6;
// search box in (log10 D, log10 rho)
const LO = [Math.log10(2e-3), Math.log10(0.5)];
const HI = [Math.log10(5e-2), Math.log10(12)];

const S = { obs: null, sigma: 0.03, seed: 1, land: null, start: [(LO[0] + HI[0]) / 2, (LO[1] + HI[1]) / 2], path: null, truth: null, busy: false };
let fitCtx = null;

function setStatus(msg, err = false) {
  const el = $('#inf-status');
  el.textContent = msg;
  el.classList.toggle('err', err);
}
const setProg = (f) => { $('#inf-prog').style.width = `${Math.round(100 * f)}%`; };

function buildFit() {
  const grid = getGrid(FIT_M);
  const xu = Float64Array.from({ length: NX }, (_, i) => i / (NX - 1));
  const A = interpMatrix(grid.x, xu);
  fitCtx = { grid, xu, A, u0: smoothIC(S.obs.subarray(0, NX), xu, grid, 40, 1e-5) };
}

function forwardFit(theta) {
  const { grid, A, u0 } = fitCtx;
  const snaps = simulate(grid, { u0, D: 10 ** theta[0], rho: 10 ** theta[1], model: state.model, T: state.T, nSnap: NT, substeps: FIT_SUBSTEPS });
  const out = new Float64Array(NT * NX);
  for (let k = 0; k < NT; k++) matvec(A, snaps.subarray(k * grid.n, (k + 1) * grid.n), NX, grid.n, out.subarray(k * NX, (k + 1) * NX));
  return out;
}

function residual(theta) {
  const f = forwardFit(theta);
  for (let i = 0; i < f.length; i++) f[i] -= S.obs[i];
  return f;
}

const sumsq = (r) => r.reduce((a, v) => a + v * v, 0);

export function initInference() {
  const box = $('#inf-controls');
  const sig = slider({ id: 'inf-sig', label: 'Noise σ', min: 0, max: 0.1, step: 0.005, value: S.sigma, digits: 3, onInput: (v) => { S.sigma = v; } });
  const seed = document.createElement('div');
  seed.className = 'ctl';
  seed.innerHTML = '<label for="inf-seed">Noise seed</label><input type="number" id="inf-seed" min="1" max="9999" value="1">';
  const res = document.createElement('div');
  res.className = 'ctl';
  res.innerHTML = '<label for="inf-grid">Landscape resolution</label><select id="inf-grid"><option value="15">15 × 15 (fast)</option><option value="21" selected>21 × 21</option><option value="31">31 × 31 (slow)</option></select>';
  box.append(sig.el, seed, res);
  $('#inf-seed').addEventListener('change', (e) => { S.seed = +e.target.value || 1; });
  $('#inf-gen').addEventListener('click', generate);
  $('#inf-land').addEventListener('click', landscape);
  $('#inf-fit').addEventListener('click', fit);
  drawEmpty();
}

/** Deep link helper: index.html?autorun=1#inference runs generate -> landscape -> fit. */
export async function autorunInference() {
  $('#inf-grid').value = '15';
  await generate();
  await landscape();
  await fit();
}

export function rerenderInference() { if (S.obs) drawLandscape(); else drawEmpty(); }

function drawEmpty() {
  draw('inf-land-plot', [], { xaxis: { title: 'log10 D', range: [LO[0], HI[0]] }, yaxis: { title: 'log10 ρ', range: [LO[1], HI[1]] },
    annotations: [{ text: 'Generate data, then compute the landscape', showarrow: false, font: { color: cssVar('--muted') } }] });
}

async function generate() {
  if (S.busy) return;
  S.busy = true;
  setStatus('Simulating ground truth and adding noise…');
  await nextFrame();
  const r = runForward({ M: 64, nSnap: NT, nx: NX, substeps: Math.max(12, substepsFor(state.rho, state.T, NT)) });
  const g = rng(1000 + S.seed);
  S.obs = Float64Array.from(r.u, (v) => v + S.sigma * g.gauss());
  S.clean = r.u;
  S.truth = [Math.log10(state.D), Math.log10(state.rho)];
  S.land = null; S.path = null; S.fitResult = null;
  buildFit();
  const t = Array.from({ length: NT }, (_, k) => (state.T * k) / (NT - 1));
  const x = Array.from(r.x);
  const rows = (a) => Array.from({ length: NT }, (_, k) => Array.from(a.subarray(k * NX, (k + 1) * NX)));
  const hm = (z, scale, zmin, zmax) => [{ type: 'heatmap', x, y: t, z, colorscale: scale, zmin, zmax, showscale: false, zsmooth: false }];
  const lay = { xaxis: { title: 'x' }, yaxis: { title: 't' }, margin: { l: 44, r: 8, t: 6, b: 40 } };
  draw('inf-obs', hm(rows(S.obs), densityScale(), 0, 1), lay);
  draw('inf-fitplot', [], lay);
  draw('inf-res', [], lay);
  $('#inf-table').innerHTML = '';
  $('#inf-insight').style.display = 'none';
  drawLandscape();
  const outOfBox = S.truth.some((v, i) => v < LO[i] || v > HI[i]);
  setStatus(`Data ready: ${NT}×${NX} observations, σ = ${S.sigma.toFixed(3)}.` + (outOfBox ? ' Warning: true parameters lie outside the search box.' : ''), outOfBox);
  S.busy = false;
}

async function landscape() {
  if (S.busy) return;
  if (!S.obs) await generate();
  S.busy = true;
  const G = +$('#inf-grid').value;
  const dl = Array.from({ length: G }, (_, i) => LO[0] + ((HI[0] - LO[0]) * i) / (G - 1));
  const rl = Array.from({ length: G }, (_, i) => LO[1] + ((HI[1] - LO[1]) * i) / (G - 1));
  const z = Array.from({ length: G }, () => new Array(G).fill(0));
  const t0 = performance.now();
  for (let j = 0; j < G; j++) {
    for (let i = 0; i < G; i++) z[j][i] = Math.log10(sumsq(residual([dl[i], rl[j]])) + 1e-12);
    setProg((j + 1) / G);
    setStatus(`Evaluating the spectral solver on the parameter grid… ${Math.round((100 * (j + 1)) / G)}%`);
    await nextFrame();
  }
  S.land = { dl, rl, z };
  drawLandscape();
  setStatus(`Landscape: ${G * G} spectral solves in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
  setProg(0);
  S.busy = false;
}

function drawLandscape() {
  const traces = [];
  if (S.land) {
    traces.push({ type: 'contour', x: S.land.dl, y: S.land.rl, z: S.land.z, colorscale: 'Viridis', reversescale: true,
      contours: { coloring: 'heatmap' }, line: { width: 0.4, color: 'rgba(255,255,255,.35)' }, colorbar: { thickness: 12, title: 'log10 cost' },
      hovertemplate: 'log10 D=%{x:.2f}<br>log10 ρ=%{y:.2f}<br>log10 cost=%{z:.2f}<extra></extra>' });
  }
  if (S.path) {
    traces.push({ x: S.path.map((p) => p.theta[0]), y: S.path.map((p) => p.theta[1]), mode: 'lines+markers', name: 'LM iterates',
      line: { color: '#fff', width: 2 }, marker: { size: 6, color: cssVar('--c-pinn'), line: { color: '#000', width: 1 } } });
  }
  traces.push({ x: [S.start[0]], y: [S.start[1]], mode: 'markers', name: 'start', marker: { symbol: 'x', size: 12, color: cssVar('--c-pinn'), line: { width: 2, color: '#fff' } } });
  if (S.truth) traces.push({ x: [S.truth[0]], y: [S.truth[1]], mode: 'markers', name: 'truth', marker: { symbol: 'star', size: 15, color: '#fff', line: { width: 1.5, color: '#000' } } });
  draw('inf-land-plot', traces, { xaxis: { title: 'log10 D', range: [LO[0], HI[0]] }, yaxis: { title: 'log10 ρ', range: [LO[1], HI[1]] },
    margin: { l: 52, r: 10, t: 8, b: 76 }, legend: { orientation: 'h', y: -0.28 } });
  const node = $('#inf-land-plot');
  if (node.removeAllListeners) node.removeAllListeners('plotly_click');
  node.on('plotly_click', (ev) => {
    const p = ev.points[0];
    S.start = [p.x, p.y];
    drawLandscape();
    setStatus(`Start set to D = ${fmt(10 ** p.x, 4)}, ρ = ${fmt(10 ** p.y, 2)}.`);
  });
}

async function fit() {
  if (S.busy) return;
  if (!S.obs) await generate();
  S.busy = true;
  setStatus('Running Levenberg–Marquardt with the spectral solver in the loop…');
  await nextFrame();
  const t0 = performance.now();
  const r = fitLM(residual, S.start.slice(), LO, HI, { maxIter: 40 });
  const secs = (performance.now() - t0) / 1000;
  for (let k = 1; k <= r.path.length; k++) {
    S.path = r.path.slice(0, k);
    drawLandscape();
    setProg(k / r.path.length);
    await sleep(Math.min(140, 1800 / r.path.length));
  }
  setProg(0);
  S.fitResult = r;
  showResult(r, secs);
  S.busy = false;
}

function jacobianCov(theta, cost) {
  const h = 1e-3, r0 = residual(theta), n = r0.length;
  const J = [0, 1].map((p) => {
    const t2 = theta.slice(); t2[p] += h;
    const r2 = residual(t2);
    return Float64Array.from(r2, (v, i) => (v - r0[i]) / h);
  });
  const a = sumsq(J[0]), b = J[0].reduce((s, v, i) => s + v * J[1][i], 0), d = sumsq(J[1]);
  const det = a * d - b * b;
  const s2 = (2 * cost) / (n - 2);
  const cov = [[(s2 * d) / det, (-s2 * b) / det], [(-s2 * b) / det, (s2 * a) / det]];
  const tr = a + d, disc = Math.sqrt(Math.max(0, (tr / 2) ** 2 - det));
  return { cov, cond: Math.sqrt((tr / 2 + disc) / Math.max(1e-30, tr / 2 - disc)) };
}

function showResult(r, secs) {
  const est = [10 ** r.theta[0], 10 ** r.theta[1]];
  const tru = [state.D, state.rho];
  const { cov, cond } = jacobianCov(r.theta, r.cost);
  const sd = [Math.sqrt(cov[0][0]), Math.sqrt(cov[1][1])];
  const corr = cov[0][1] / (sd[0] * sd[1]);
  const pctSd = (s) => 100 * (10 ** s - 1);
  const row = (name, e, t, s) => `<tr><td>${name}</td><td>${fmt(t, 4)}</td><td>${fmt(e, 4)}</td><td>${fmt((100 * (e - t)) / t, 1)}%</td><td>±${fmt(pctSd(s), 1)}%</td></tr>`;
  $('#inf-table').innerHTML = `<tr><th></th><th>true</th><th>estimate</th><th>error</th><th>±1σ</th></tr>
    ${row('D', est[0], tru[0], sd[0])}${row('ρ', est[1], tru[1], sd[1])}
    <tr><td>speed 2√(Dρ)</td><td>${fmt(2 * Math.sqrt(tru[0] * tru[1]), 3)}</td><td>${fmt(2 * Math.sqrt(est[0] * est[1]), 3)}</td><td>${fmt((100 * (Math.sqrt(est[0] * est[1]) / Math.sqrt(tru[0] * tru[1]) - 1)), 1)}%</td><td></td></tr>
    <tr><td>width √(D/ρ)</td><td>${fmt(Math.sqrt(tru[0] / tru[1]), 4)}</td><td>${fmt(Math.sqrt(est[0] / est[1]), 4)}</td><td>${fmt((100 * (Math.sqrt(est[0] / est[1]) / Math.sqrt(tru[0] / tru[1]) - 1)), 1)}%</td><td></td></tr>`;
  const ins = $('#inf-insight');
  ins.style.display = 'block';
  ins.innerHTML = `Converged in ${r.path.length - 1} accepted steps (${r.nfev} solver calls, ${secs.toFixed(2)} s). Correlation of the two estimates: <b>${fmt(corr, 2)}</b>; Jacobian condition number: <b>${fmt(cond, 1)}</b>. ` +
    (Math.abs(corr) > 0.8 ? 'Strongly correlated &mdash; the landscape has a narrow valley, so D and &rho; trade off against each other while the front speed stays well determined.' : 'The two parameters are reasonably decoupled for this configuration.');
  const f = forwardFit(r.theta);
  const t = Array.from({ length: NT }, (_, k) => (state.T * k) / (NT - 1));
  const x = Array.from(fitCtx.xu);
  const rows = (a) => Array.from({ length: NT }, (_, k) => Array.from(a.subarray(k * NX, (k + 1) * NX)));
  const lay = { xaxis: { title: 'x' }, yaxis: { title: 't' }, margin: { l: 44, r: 8, t: 6, b: 40 } };
  draw('inf-fitplot', [{ type: 'heatmap', x, y: t, z: rows(f), colorscale: densityScale(), zmin: 0, zmax: 1, showscale: false }], lay);
  const resid = Float64Array.from(f, (v, i) => v - S.obs[i]);
  const m = Math.max(0.05, ...resid.map(Math.abs));
  draw('inf-res', [{ type: 'heatmap', x, y: t, z: rows(resid), colorscale: divergingScale(), zmin: -m, zmax: m, showscale: false }], lay);
  setStatus(`RMS residual ${fmt(Math.sqrt(sumsq(resid) / resid.length), 4)} (noise σ = ${S.sigma.toFixed(3)}).`);
}
