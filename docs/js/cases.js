import { $, METHODS, cssVar, densityScale, draw, fmt, methodColor } from './common.js';

let SC = null;
const S = { i: 0, mode: 'field', k: 31 };
const PANELS = [
  { key: 'clean', title: 'Ground truth' },
  { key: 'obs', title: 'Noisy observation' },
  { key: 'cnn', title: 'CNN estimate (simulated)' },
  { key: 'nls', title: 'Spectral least squares' },
  { key: 'pinn_cold', title: 'PINN' },
  { key: 'pinn_warm', title: 'CNN-init PINN' },
];

export function initCases(data) {
  SC = data;
  const sel = $('#cs-case');
  sel.innerHTML = SC.cases.map((c, i) => `<option value="${i}">test #${c.idx} · σ = ${c.sigma} · D=${fmt(c.true[0], 4)}, ρ=${fmt(c.true[1], 2)}</option>`).join('');
  sel.addEventListener('change', () => { S.i = +sel.value; render(); });
  $('#cs-mode').addEventListener('change', (e) => { S.mode = e.target.value; render(); });
  $('#cs-t').addEventListener('input', (e) => { S.k = +e.target.value; renderProfile(); });
  $('#cs-grid').innerHTML = PANELS.map((p) => `<div class="card"><h3>${p.title}<span class="hint" id="cs-r-${p.key}"></span></h3><div id="cs-p-${p.key}" class="plot short"></div></div>`).join('');
  $('#cs-t').max = SC.t.length - 1;
  S.k = SC.t.length - 1;
  $('#cs-t').value = S.k;
  render();
}

export function rerenderCases() { if (SC) render(); }

const field = (c, key) => (key === 'clean' || key === 'obs' ? c[key] : c.recon[key]);
const rmse = (a, b) => {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) { s += (a[i][j] - b[i][j]) ** 2; n++; }
  return Math.sqrt(s / n);
};

function render() {
  const c = SC.cases[S.i];
  const lay = { xaxis: { title: 'x' }, yaxis: { title: 't' }, margin: { l: 44, r: 8, t: 6, b: 40 } };
  for (const p of PANELS) {
    const f = field(c, p.key);
    if (!f) { draw(`cs-p-${p.key}`, [], { ...lay, annotations: [{ text: 'not run for this case', showarrow: false, font: { color: cssVar('--muted') } }] }); continue; }
    let z = f, zmin = 0, zmax = 1, scale = densityScale();
    if (S.mode === 'error' && p.key !== 'clean') {
      z = f.map((row, i) => row.map((v, j) => Math.abs(v - c.clean[i][j])));
      zmin = 0; zmax = 0.15;
      scale = [[0, '#f8fafc'], [0.5, '#f59e0b'], [1, '#7f1d1d']];
    }
    draw(`cs-p-${p.key}`, [{ type: 'heatmap', x: SC.x, y: SC.t, z, colorscale: scale, zmin, zmax, showscale: false, zsmooth: false,
      hovertemplate: 'x=%{x:.2f}, t=%{y:.2f}<br>%{z:.3f}<extra></extra>' }], lay);
    $(`#cs-r-${p.key}`).textContent = p.key === 'clean' ? '' : `RMSE ${fmt(rmse(f, c.clean), 4)}`;
  }
  renderTable(c);
  renderProfile();
}

function renderProfile() {
  const c = SC.cases[S.i];
  const k = S.k;
  $('#cs-t-o').textContent = SC.t[k].toFixed(2);
  const traces = [
    { x: SC.x, y: c.obs[k], mode: 'markers', name: 'observed', marker: { size: 5, color: cssVar('--muted'), opacity: 0.7 } },
    { x: SC.x, y: c.clean[k], mode: 'lines', name: 'truth', line: { color: cssVar('--truth'), width: 3 } },
  ];
  for (const m of ['cnn', 'nls', 'pinn_cold', 'pinn_warm']) {
    const f = field(c, m);
    if (f) traces.push({ x: SC.x, y: f[k], mode: 'lines', name: METHODS[m].label, line: { color: methodColor(m), width: 2, dash: m === 'cnn' ? 'dot' : 'solid' } });
  }
  draw('cs-profile', traces, { xaxis: { title: 'x' }, yaxis: { title: 'u', range: [-0.15, 1.15] }, margin: { l: 50, r: 10, t: 8, b: 76 } });
}

function renderTable(c) {
  const rows = [['cnn', c.est.cnn], ['nls', c.est.nls], ['pinn_cold', c.est.pinn_cold], ['pinn_warm', c.est.pinn_warm]].filter(([, e]) => e);
  const err = (e, t) => `${fmt((100 * (e - t)) / t, 1)}%`;
  $('#cs-table').innerHTML = `<tr><th>Method</th><th>D</th><th>error</th><th>ρ</th><th>error</th></tr>
    <tr><td>Truth</td><td>${fmt(c.true[0], 4)}</td><td></td><td>${fmt(c.true[1], 3)}</td><td></td></tr>` +
    rows.map(([m, e]) => `<tr><td><span class="dot" style="background:var(${METHODS[m].color})"></span>${METHODS[m].label}</td>
      <td>${fmt(e[0], 4)}</td><td>${err(e[0], c.true[0])}</td><td>${fmt(e[1], 3)}</td><td>${err(e[1], c.true[1])}</td></tr>`).join('');
}
