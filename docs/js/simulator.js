import { MODELS, burden, frontPosition, rng } from './solver.js';
import { state, runForward } from './state.js';
import { $, cssVar, debounce, densityScale, download, draw, fmt, slider } from './common.js';

const PRESETS = {
  slow: { D: 0.004, rho: 2.0, T: 1.0 },
  fast: { D: 0.02, rho: 8.0, T: 1.0 },
  thin: { D: 0.002, rho: 8.0, T: 1.0 },
  wide: { D: 0.03, rho: 1.0, T: 1.0 },
};

let res = null;
const ctl = {};

export function initSimulator() {
  const box = $('#sim-controls');
  const modelSel = document.createElement('div');
  modelSel.className = 'ctl';
  modelSel.innerHTML = `<label for="sim-model">Growth law</label>
    <select id="sim-model">${Object.entries(MODELS).map(([k, m]) => `<option value="${k}">${m.title}</option>`).join('')}</select>`;
  box.append(modelSel);
  $('#sim-model').addEventListener('change', (e) => { state.model = e.target.value; update(); });

  const upd = debounce(() => update(), 25);
  ctl.D = slider({ id: 'sl-D', label: 'Diffusivity D', min: 0.001, max: 0.05, log: true, value: state.D, digits: 4, onInput: (v) => { state.D = v; upd(); } });
  ctl.rho = slider({ id: 'sl-rho', label: 'Proliferation ρ', min: 0.5, max: 10, log: true, value: state.rho, digits: 2, onInput: (v) => { state.rho = v; upd(); } });
  ctl.T = slider({ id: 'sl-T', label: 'Final time T', min: 0.5, max: 2, step: 0.05, value: state.T, digits: 2, onInput: (v) => { state.T = v; upd(); } });
  ctl.a1 = slider({ id: 'sl-a1', label: 'Seed 1 amplitude', min: 0.1, max: 1, step: 0.05, value: state.seeds[0].a, digits: 2, onInput: (v) => { state.seeds[0].a = v; upd(); } });
  ctl.c1 = slider({ id: 'sl-c1', label: 'Seed 1 position', min: 0.1, max: 0.9, step: 0.01, value: state.seeds[0].c, digits: 2, onInput: (v) => { state.seeds[0].c = v; upd(); } });
  ctl.s1 = slider({ id: 'sl-s1', label: 'Seed 1 width', min: 0.03, max: 0.15, step: 0.005, value: state.seeds[0].s, digits: 3, onInput: (v) => { state.seeds[0].s = v; upd(); } });
  ctl.a2 = slider({ id: 'sl-a2', label: 'Seed 2 amplitude (0 = off)', min: 0, max: 1, step: 0.05, value: state.seeds[1].a, digits: 2, onInput: (v) => { state.seeds[1].a = v; upd(); } });
  ctl.c2 = slider({ id: 'sl-c2', label: 'Seed 2 position', min: 0.1, max: 0.9, step: 0.01, value: state.seeds[1].c, digits: 2, onInput: (v) => { state.seeds[1].c = v; upd(); } });
  ctl.sig = slider({ id: 'sl-sig', label: 'Measurement noise σ (overlay)', min: 0, max: 0.1, step: 0.005, value: state.sigma, digits: 3, onInput: (v) => { state.sigma = v; updateProfile(); } });
  for (const c of [ctl.D, ctl.rho, ctl.T, ctl.a1, ctl.c1, ctl.s1, ctl.a2, ctl.c2, ctl.sig]) box.append(c.el);

  const mctl = document.createElement('div');
  mctl.className = 'ctl';
  mctl.innerHTML = `<label for="sim-M">Spectral degree M <span class="muted small">(try 16: under-resolution)</span></label>
    <select id="sim-M">${[16, 24, 32, 48, 64, 96].map((m) => `<option ${m === state.M ? 'selected' : ''}>${m}</option>`).join('')}</select>`;
  box.append(mctl);
  $('#sim-M').addEventListener('change', (e) => { state.M = +e.target.value; update(); });

  const row = document.createElement('div');
  row.className = 'btn-row';
  row.innerHTML = `<button class="btn" data-p="slow">Slow</button><button class="btn" data-p="fast">Fast</button>
    <button class="btn" data-p="thin">Thin front</button><button class="btn" data-p="wide">Wide front</button>
    <button class="btn" id="sim-csv">Download CSV</button>`;
  box.append(row);
  row.addEventListener('click', (e) => {
    const p = e.target.dataset.p;
    if (p) { Object.assign(state, PRESETS[p]); syncControls(); update(); }
  });
  $('#sim-csv').addEventListener('click', exportCSV);
  $('#sim-t').addEventListener('input', updateProfile);
  update();
}

function syncControls() {
  ctl.D.set(state.D); ctl.rho.set(state.rho); ctl.T.set(state.T);
}

function update() {
  res = runForward({ nx: 121 });
  const tSlider = $('#sim-t');
  const first = !tSlider.dataset.ready;
  tSlider.max = res.nSnap - 1;
  tSlider.dataset.ready = '1';
  if (first || +tSlider.value > res.nSnap - 1) tSlider.value = res.nSnap - 1;
  render();
}

export function currentResult() { return res; }
export function rerenderSimulator() { if (res) render(); }

function render() {
  const { x, u, nx, nSnap, grid, snaps, ms } = res;
  const t = Array.from({ length: nSnap }, (_, k) => (state.T * k) / (nSnap - 1));
  const z = [];
  for (let k = 0; k < nSnap; k++) z.push(Array.from(u.subarray(k * nx, (k + 1) * nx)));
  const front = frontPosition(u, x, nSnap, nx, 0.5);
  draw('sim-heat', [
    { type: 'heatmap', x: Array.from(x), y: t, z, colorscale: densityScale(), zmin: 0, zmax: 1, zsmooth: 'best',
      colorbar: { title: 'u', thickness: 12, len: 0.9 }, hovertemplate: 'x=%{x:.3f}<br>t=%{y:.3f}<br>u=%{z:.3f}<extra></extra>' },
    { type: 'scatter', mode: 'lines', x: Array.from(front), y: t, name: 'front (u = 0.5)',
      line: { color: cssVar('--text'), width: 1.8, dash: 'dot' }, hoverinfo: 'skip' },
  ], { xaxis: { title: 'x' }, yaxis: { title: 't' }, margin: { l: 50, r: 10, t: 10, b: 44 }, showlegend: false });

  // stats
  const fisher = state.model === 'fisher_kpp';
  const c = 2 * Math.sqrt(state.D * state.rho);
  const k0 = Math.floor(nSnap / 2);
  let num = 0, den = 0, n = 0, mt = 0, mx = 0;
  for (let k = k0; k < nSnap; k++) if (Number.isFinite(front[k])) { mt += t[k]; mx += front[k]; n++; }
  let speed = NaN;
  if (n > 3) {
    mt /= n; mx /= n;
    for (let k = k0; k < nSnap; k++) if (Number.isFinite(front[k])) { num += (t[k] - mt) * (front[k] - mx); den += (t[k] - mt) ** 2; }
    speed = num / den;
  }
  const bur = burden(grid, snaps, nSnap);
  $('#sim-stats').innerHTML = [
    ['Theoretical speed', fisher ? fmt(c, 3) : 'n/a', fisher ? '2√(Dρ)' : 'Fisher–KPP only'],
    ['Measured speed', fmt(speed, 3), 'second half of the window'],
    ['Front width scale', fmt(Math.sqrt(state.D / state.rho), 3), '√(D/ρ)'],
    ['Burden ∫u dx at T', fmt(bur[nSnap - 1], 3), `solve ${ms.toFixed(1)} ms · M=${state.M}`],
  ].map(([l, v, s]) => `<div class="card kpi"><span class="l">${l}</span><span class="v">${v}</span><span class="s">${s}</span></div>`).join('');

  // front + burden
  const theory = fisher && Number.isFinite(front[nSnap - 1])
    ? { x: [t[Math.floor(nSnap / 2)], t[nSnap - 1]], y: [front[nSnap - 1] - c * (t[nSnap - 1] - t[Math.floor(nSnap / 2)]), front[nSnap - 1]] } : null;
  const traces = [
    { x: t, y: Array.from(front), name: 'front position', line: { color: cssVar('--c-nls'), width: 2.2 }, mode: 'lines' },
    { x: t, y: Array.from(bur), name: 'burden ∫u dx', yaxis: 'y2', line: { color: cssVar('--c-pinn'), width: 2.2 }, mode: 'lines' },
  ];
  if (theory) traces.push({ x: theory.x, y: theory.y, name: 'slope 2√(Dρ)', mode: 'lines', line: { color: cssVar('--text'), dash: 'dash', width: 1.5 } });
  draw('sim-front', traces, {
    xaxis: { title: 't' }, yaxis: { title: 'front position', range: [0, 1.02] },
    yaxis2: { title: 'burden', overlaying: 'y', side: 'right', showgrid: false, rangemode: 'tozero' },
    margin: { l: 56, r: 56, t: 10, b: 84 }, legend: { orientation: 'h', y: -0.36 },
  });
  updateProfile();
}

function updateProfile() {
  if (!res) return;
  const { x, u, nx, nSnap } = res;
  const k = +$('#sim-t').value;
  const tk = (state.T * k) / (nSnap - 1);
  $('#sim-t-o').textContent = tk.toFixed(2);
  $('#sim-tlabel').textContent = `t = ${tk.toFixed(2)}`;
  const row = Array.from(u.subarray(k * nx, (k + 1) * nx));
  const traces = [
    { x: Array.from(x), y: Array.from(u.subarray(0, nx)), name: 'initial', mode: 'lines', line: { color: cssVar('--muted'), dash: 'dot', width: 1.5 } },
    { x: Array.from(x), y: row, name: 'u(x, t)', mode: 'lines', line: { color: cssVar('--accent'), width: 2.6 } },
  ];
  if (state.sigma > 0) {
    const r = rng(7 + k);
    const xs = [], ys = [];
    for (let i = 0; i < nx; i += 2) { xs.push(x[i]); ys.push(row[i] + state.sigma * r.gauss()); }
    traces.push({ x: xs, y: ys, name: 'noisy samples', mode: 'markers', marker: { size: 5, color: cssVar('--c-pinn') } });
  }
  draw('sim-profile', traces, { xaxis: { title: 'x', range: [0, 1] }, yaxis: { title: 'u', range: [-0.15, 1.15] },
    margin: { l: 50, r: 10, t: 10, b: 84 }, legend: { orientation: 'h', y: -0.36 } });
}

function exportCSV() {
  const { x, u, nx, nSnap } = res;
  const lines = ['t,x,u'];
  for (let k = 0; k < nSnap; k++) {
    const tk = ((state.T * k) / (nSnap - 1)).toFixed(5);
    for (let i = 0; i < nx; i++) lines.push(`${tk},${x[i].toFixed(5)},${u[k * nx + i].toFixed(6)}`);
  }
  download(`simulation_${state.model}_D${state.D.toPrecision(3)}_rho${state.rho.toPrecision(3)}.csv`, lines.join('\n'));
}
