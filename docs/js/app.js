import { $, $$, loadJSON, renderMath, fmt, mape, METHODS } from './common.js';
import { initSimulator, rerenderSimulator } from './simulator.js';
import { autorunInference, initInference, rerenderInference } from './inference.js';
import { initBenchmark, rerenderBenchmark } from './benchmark.js';
import { initCases, rerenderCases } from './cases.js';
import { initNumerics, rerenderNumerics } from './numerics.js';
import { initReal, rerenderReal } from './real.js';

const data = {};
const ready = new Set();

const TABS = {
  overview: { init: () => {}, rerender: () => {} },
  simulator: { init: initSimulator, rerender: rerenderSimulator },
  inference: { init: initInference, rerender: rerenderInference },
  benchmark: { init: () => initBenchmark(data.benchmark), rerender: rerenderBenchmark },
  cases: { init: () => initCases(data.showcase), rerender: rerenderCases },
  real: { init: () => initReal(data.real), rerender: rerenderReal },
  numerics: { init: () => initNumerics(data.convergence, data.training, data.showcase), rerender: rerenderNumerics },
  method: { init: () => renderMath($('#tab-method')), rerender: () => {} },
};

function show(name) {
  if (!TABS[name]) name = 'overview';
  $$('nav.tabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  $$('section.panel').forEach((s) => s.classList.toggle('active', s.id === `tab-${name}`));
  if (!ready.has(name)) {
    ready.add(name);
    try { TABS[name].init(); } catch (e) { console.error(e); $(`#tab-${name}`).insertAdjacentHTML('afterbegin', `<div class="callout err">Failed to initialise this section: ${e.message}</div>`); }
  }
  requestAnimationFrame(() => {
    $$(`#tab-${name} .js-plotly-plot`).forEach((p) => window.Plotly && Plotly.Plots.resize(p));
  });
  if (location.hash.slice(1) !== name) history.replaceState(null, '', `#${name}`);
  window.scrollTo({ top: 0 });
}

function setupTheme() {
  const btn = $('#theme-toggle');
  btn.addEventListener('click', () => {
    const root = document.documentElement;
    const dark = root.dataset.theme === 'dark' || (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) { /* storage may be unavailable */ }
    for (const name of ready) TABS[name].rerender();
    overview();
  });
}

function methodStat(m, sigma) {
  const o = data.benchmark.methods[m];
  const k = o && Object.keys(o).find((q) => Math.abs(parseFloat(q) - sigma) < 1e-9);
  return k ? o[k] : null;
}

function overview() {
  const meta = data.meta, bm = data.benchmark;
  $('#badges').innerHTML = [
    meta.model.title, 'CPU + optional GPU', `${meta.counts.train.toLocaleString()} training simulations`,
    `${meta.counts.cnn_ensemble}-member CNN ensemble`, `PyTorch ${meta.environment.torch.split('+')[0]}`, 'MIT licence',
  ].map((t) => `<span class="badge">${t}</span>`).join('');

  // every comparison below scores the methods on the same test cases (the first n common to all of them)
  const nCommon = (ms, s) => Math.min(...ms.map((m) => methodStat(m, s)).filter(Boolean).map((e) => e.true.length));
  const err = (m, s, n) => {
    const e = methodStat(m, s);
    if (!e) return null;
    const p = e.pred.slice(0, n), t = e.true.slice(0, n);
    return (mape(p, t, 0) + mape(p, t, 1)) / 2;
  };
  const sig = 0.02;
  const n02 = nCommon(Object.keys(METHODS), sig);
  const cards = Object.keys(METHODS).map((m) => {
    const e = methodStat(m, sig);
    if (!e) return '';
    const sec = m === 'cnn' ? e.ms_per_sample / 1000 : e.seconds_mean;
    const t = sec < 0.1 ? `${(sec * 1000).toFixed(1)} ms` : `${sec.toFixed(1)} s`;
    return `<div class="card kpi"><span class="l"><span class="dot" style="background:var(${METHODS[m].color})"></span>${METHODS[m].label}</span>
      <span class="v">${fmt(err(m, sig, n02), 1)}%</span><span class="s">mean abs. error of D and ρ at σ = ${sig} (${n02} shared test cases) &middot; ${t} per case</span></div>`;
  });
  $('#kpis').innerHTML = cards.join('');

  const get = (m, s) => err(m, s, nCommon(['cnn', 'nls'], s));
  const sec = (m, s) => { const e = methodStat(m, s); return e ? (m === 'cnn' ? e.ms_per_sample / 1000 : e.seconds_mean) : null; };
  const lines = [];
  const cnn0 = get('cnn', 0.0), cnn5 = get('cnn', 0.05), nls0 = get('nls', 0.0), nls5 = get('nls', 0.05);
  if (nls0 !== null && cnn0 !== null) {
    lines.push(`<b>Noise-free data:</b> spectral least squares reaches ${fmt(nls0, 1)}% mean error against ${fmt(cnn0, 1)}% for the CNN &mdash; with the right model inside the loop, the classical fit is hard to beat.`);
  }
  if (nls5 !== null && cnn5 !== null) {
    lines.push(`<b>At σ = 0.05:</b> errors grow to ${fmt(nls5, 1)}% (spectral LSQ) and ${fmt(cnn5, 1)}% (CNN); the CNN, which was trained on noisy data from the same prior, degrades more slowly.`);
  }
  const tc = sec('cnn', 0.02), tn = sec('nls', 0.02), tp = sec('pinn_cold', 0.02), tw = sec('pinn_warm', 0.02);
  if (tc && tn && tp) {
    lines.push(`<b>Speed:</b> the CNN needs ${(tc * 1000).toFixed(2)} ms per tumour &mdash; about ${Math.round(tn / tc).toLocaleString('en-US')}× faster than the spectral fit and ${Math.round(tp / tc).toLocaleString('en-US')}× faster than a PINN.`);
  }
  const ep = err('pinn_cold', 0.02, n02), ew = err('pinn_warm', 0.02, n02);
  if (ep !== null && ew !== null && tp && tw) {
    lines.push(`<b>Warm start:</b> initialising the PINN with the CNN estimate cuts its run time by ${fmt(100 * (1 - tw / tp), 0)}% and changes its mean error from ${fmt(ep, 1)}% to ${fmt(ew, 1)}%.`);
  }
  if (data.real) {
    const laws = ['exponential', 'logistic', 'gompertz'];
    for (const d of Object.values(data.real.datasets)) {
      const r = d.summary, he = r.holdout_error;
      const bestLaw = laws.reduce((a, b) => (he.pinn[a].median_pct <= he.pinn[b].median_pct ? a : b));
      lines.push(`<b>Real data · ${d.meta.dataset.short} (GPU):</b> median forecast error of the last ${r.holdout} measurements of ${r.n_animals} mice with the ${bestLaw[0].toUpperCase() + bestLaw.slice(1)} law is ${fmt(he.pinn[bestLaw].median_pct, 1)}% for the batched PINN, ${fmt(he.nls[bestLaw].median_pct, 1)}% for classical least squares.`);
    }
  }
  $('#takeaways').innerHTML = `<h3>Key results <span class="hint">computed from the exported benchmark files</span></h3><ul>${lines.map((l) => `<li>${l}</li>`).join('')}</ul>
    <p class="small muted">Methods are always compared on identical test cases (the PINN runs use ${meta.counts.pinn_cases}, spectral least squares ${meta.counts.nls_cases}, the CNN ${meta.counts.test}). Run times are single-thread CPU seconds measured while several workers ran concurrently.</p>`;
}

async function main() {
  setupTheme();
  $$('nav.tabs button').forEach((b) => b.addEventListener('click', () => show(b.dataset.tab)));
  window.addEventListener('hashchange', () => show(location.hash.slice(1)));
  const names = ['meta', 'benchmark', 'showcase', 'convergence', 'training'];
  try { data.real = await loadJSON('data/real.json'); } catch (e) { console.warn('real-data results not found', e.message); }
  try {
    const res = await Promise.all(names.map((n) => loadJSON(`data/${n}.json`)));
    names.forEach((n, i) => { data[n] = res[i]; });
    overview();
  } catch (e) {
    console.error(e);
    $('#kpis').innerHTML = `<div class="card callout err">Could not load the exported results (${e.message}). Run <code>python -m tumorpinn all</code> to create <code>docs/data/*.json</code>; the Simulator and Inference lab work without them.</div>`;
  }
  show(location.hash.slice(1) || 'overview');
  if (new URLSearchParams(location.search).has('autorun') && location.hash === '#inference') autorunInference();
}

main();
