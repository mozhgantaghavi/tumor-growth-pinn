import { $, cssVar, draw, fmt } from './common.js';

const LAWS = ['exponential', 'logistic', 'gompertz'];
const LAW_LABEL = { exponential: 'Exponential', logistic: 'Logistic', gompertz: 'Gompertz' };
const LAW_COLOR = { exponential: '#64748b', logistic: '#0ea5e9', gompertz: '#d946ef' };
const S = { i: 0, mode: 'holdout', log: false, pinn: true, nls: true };
let R = null;
let ALL = null;

export function initReal(all) {
  ALL = all;
  const keys = Object.keys(all.datasets);
  const dsel = $('#rl-dataset-sel');
  dsel.innerHTML = keys.map((k) => `<option value="${k}">${all.datasets[k].meta.dataset.short}</option>`).join('');
  dsel.addEventListener('change', () => { setDataset(dsel.value); });
  const sel = $('#rl-animal');
  sel.addEventListener('change', () => { S.i = +sel.value; renderAnimal(); });
  $('#rl-mode').addEventListener('click', (e) => {
    const m = e.target.dataset.m;
    if (!m) return;
    S.mode = m;
    $('#rl-mode').querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.m === m)));
    renderAnimal();
  });
  $('#rl-log').addEventListener('change', (e) => { S.log = e.target.checked; renderAnimal(); });
  $('#rl-pinn').addEventListener('change', (e) => { S.pinn = e.target.checked; renderAnimal(); });
  $('#rl-nls').addEventListener('change', (e) => { S.nls = e.target.checked; renderAnimal(); });
  setDataset(keys[0]);
}

function setDataset(key) {
  R = ALL.datasets[key];
  S.i = 0;
  const { meta, summary } = R;
  $('#rl-dataset').innerHTML = `<h3>Dataset</h3>
    <p><b>${meta.dataset.title}</b><br><span class="muted small">${meta.dataset.authors} &middot; ${meta.dataset.license} &middot;
    <a href="${meta.dataset.url}">doi:${meta.dataset.doi}</a></span></p>
    <p class="small">Numbers in a table, not images: caliper volumes (mm³) of tumours in mice, measured every few days.
    ${summary.n_animals} animals with at least 7 valid measurements (${summary.n_points} volumes). For the forecast test the last
    ${meta.holdout} measurements of every animal are withheld.</p>${meta.dataset.note ? `<p class="small muted">${meta.dataset.note}</p>` : ''}`;
  const timed = Object.values(ALL.datasets).map((d) => d.meta.timing).find((t) => t && t.gpu_ms_per_iter);
  const gpu = meta.gpu_name ? `${meta.gpu_name}` : 'CPU';
  $('#rl-gpu').innerHTML = `<h3>GPU-batched PINN</h3>
    <p class="small">All <b>${meta.n_networks.toLocaleString()}</b> small networks (${meta.restarts} restarts &times; 3 growth laws &times; 2 modes &times; ${summary.n_animals} animals,
    ${meta.layers}&times;${meta.hidden} tanh units) are trained at once as one batched model on <b>${gpu}</b>: ${meta.iters.toLocaleString()} iterations in
    <b>${meta.train_seconds.toFixed(0)} s</b>.</p>` + (timed ? `<div class="grid g3">
      <div class="kpi"><span class="v">${timed.gpu_ms_per_iter.toFixed(0)} ms</span><span class="l">GPU / iteration</span></div>
      <div class="kpi"><span class="v">${timed.cpu_ms_per_iter.toFixed(0)} ms</span><span class="l">CPU / iteration</span></div>
      <div class="kpi"><span class="v">${timed.speedup.toFixed(1)}&times;</span><span class="l">GPU speed-up</span></div></div>
      <p class="small muted">Measured on the full breast-carcinoma batch.</p>` : '');
  $('#rl-animal').innerHTML = R.animals.map((a, i) => `<option value="${i}">${a.group ? `${a.group} &middot; ` : ''}mouse ${a.id} &middot; ${a.t.length} measurements</option>`).join('');
  renderAnimal();
  renderSummary();
}

export function rerenderReal() { if (R) { renderAnimal(); renderSummary(); } }

function renderAnimal() {
  const a = R.animals[S.i];
  const hold = S.mode === 'holdout';
  const nTr = hold ? a.n_train_holdout : a.t.length;
  const traces = [];
  LAWS.forEach((law) => {
    const f = a.fits[S.mode][law];
    if (S.pinn) traces.push({ x: a.grid, y: f.pinn.curve, name: `${LAW_LABEL[law]} · PINN`, mode: 'lines', line: { color: LAW_COLOR[law], width: 2.6 } });
    if (S.nls) traces.push({ x: a.grid, y: f.nls.curve, name: `${LAW_LABEL[law]} · least squares`, mode: 'lines', line: { color: LAW_COLOR[law], width: 1.8, dash: 'dash' } });
  });
  traces.push({ x: a.t.slice(0, nTr), y: a.v.slice(0, nTr), name: 'measured', mode: 'markers', marker: { size: 9, color: cssVar('--text') } });
  if (hold) {
    traces.push({ x: a.t.slice(nTr), y: a.v.slice(nTr), name: 'withheld', mode: 'markers',
      marker: { size: 11, symbol: 'circle', line: { width: 2.5, color: '#dc2626' }, color: 'rgba(0,0,0,0)' } });
  }
  const shapes = hold ? [{ type: 'rect', xref: 'x', yref: 'paper', x0: a.t[nTr - 1], x1: a.grid[a.grid.length - 1] * 1.02, y0: 0, y1: 1,
    fillcolor: 'rgba(220,38,38,0.07)', line: { width: 0 } }] : [];
  draw('rl-plot', traces, {
    xaxis: { title: 'days since first measurement' },
    yaxis: { title: 'tumour volume (mm³)', type: S.log ? 'log' : 'linear', rangemode: S.log ? undefined : 'tozero' },
    shapes, margin: { l: 62, r: 10, t: 8, b: 110 }, legend: { orientation: 'h', y: -0.28, font: { size: 11 } },
  });

  const head = `<tr><th>law</th><th>method</th><th>&alpha; /day</th><th>K mm³</th>${hold ? `<th>error</th>` : ''}</tr>`;
  const rows = [];
  LAWS.forEach((law) => {
    ['pinn', 'nls'].forEach((m) => {
      const f = a.fits[S.mode][law][m];
      const err = hold ? `<td>${fmt(100 * f.rel_err.reduce((s, v) => s + v, 0) / f.rel_err.length, 1)}%</td>` : '';
      rows.push(`<tr><td><span class="dot" style="background:${LAW_COLOR[law]}"></span>${LAW_LABEL[law]}</td><td>${m === 'pinn' ? 'PINN' : 'LSQ'}</td>
        <td>${fmt(f.alpha, 4)}</td><td>${f.K ? fmt(f.K, 0) : '—'}</td>${err}</tr>`);
    });
  });
  $('#rl-table').innerHTML = head + rows.join('');
}

function renderSummary() {
  const s = R.summary;
  const col = { pinn: cssVar('--c-pinn'), nls: cssVar('--c-nls') };
  draw('rl-err', ['pinn', 'nls'].map((m) => ({
    type: 'bar', name: m === 'pinn' ? 'PINN (GPU)' : 'least squares', x: LAWS.map((l) => LAW_LABEL[l]),
    y: LAWS.map((l) => s.holdout_error[m][l].median_pct), marker: { color: col[m] },
    text: LAWS.map((l) => `${s.holdout_error[m][l].median_pct.toFixed(1)}%`), textposition: 'outside',
    hovertemplate: '%{x}: median %{y:.1f}%<extra>' + (m === 'pinn' ? 'PINN' : 'least squares') + '</extra>',
  })), { barmode: 'group', yaxis: { title: 'median forecast error (%)', range: [0, 1.2 * Math.max(...['pinn', 'nls'].flatMap((m) => LAWS.map((l) => s.holdout_error[m][l].median_pct)))] }, margin: { l: 56, r: 10, t: 24, b: 80 } });

  draw('rl-win', [
    { type: 'bar', name: 'best BIC (full fit)', x: LAWS.map((l) => LAW_LABEL[l]), y: LAWS.map((l) => s.winner_bic[l]), marker: { color: cssVar('--c-cnn') } },
    { type: 'bar', name: 'best forecast · PINN', x: LAWS.map((l) => LAW_LABEL[l]), y: LAWS.map((l) => s.winner_holdout.pinn[l]), marker: { color: col.pinn } },
    { type: 'bar', name: 'best forecast · least squares', x: LAWS.map((l) => LAW_LABEL[l]), y: LAWS.map((l) => s.winner_holdout.nls[l]), marker: { color: col.nls } },
  ], { barmode: 'group', yaxis: { title: 'number of mice' }, margin: { l: 56, r: 10, t: 24, b: 100 }, legend: { orientation: 'h', y: -0.3, font: { size: 11 } } });

  const traces = LAWS.map((law) => ({
    x: R.animals.map((a) => a.fits.full[law].nls.alpha), y: R.animals.map((a) => a.fits.full[law].pinn.alpha),
    mode: 'markers', name: LAW_LABEL[law], marker: { size: 6, opacity: 0.75, color: LAW_COLOR[law] },
    hovertemplate: 'least squares %{x:.4f}<br>PINN %{y:.4f}<extra>' + LAW_LABEL[law] + '</extra>',
  }));
  traces.unshift({ x: [0.003, 2], y: [0.003, 2], mode: 'lines', showlegend: false, line: { color: cssVar('--muted'), dash: 'dash', width: 1 }, hoverinfo: 'skip' });
  draw('rl-rate', traces, { xaxis: { type: 'log', title: 'rate α, least squares' }, yaxis: { type: 'log', title: 'rate α, PINN' },
    margin: { l: 56, r: 10, t: 24, b: 110 }, legend: { orientation: 'h', y: -0.34, font: { size: 11 } } });
  const agree = LAWS.map((l) => `${LAW_LABEL[l]} r = ${s.rate_agreement[l].pearson_log10.toFixed(2)}`).join(' &middot; ');
  $('#rl-rate-note').innerHTML = `Agreement of log-rates (Pearson): ${agree}. Logistic and Gompertz rates are coupled to the weakly identified carrying capacity K.`;
}
