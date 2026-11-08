import { $, METHODS, cssVar, draw, fmt, median, methodColor, spearman } from './common.js';

let B = null;
const S = { sigma: 0.02, on: new Set(['cnn', 'nls', 'pinn_cold', 'pinn_warm']) };

const keyFor = (obj, sigma) => Object.keys(obj).find((k) => Math.abs(parseFloat(k) - sigma) < 1e-9);
const entry = (m, sigma) => { const o = B.methods[m]; const k = o && keyFor(o, sigma); return k ? o[k] : null; };

function stats(pred, tru) {
  const n = pred.length;
  const out = { n };
  [['D', 0], ['rho', 1]].forEach(([nm, j]) => {
    const ape = pred.map((p, i) => (100 * Math.abs(p[j] - tru[i][j])) / tru[i][j]);
    out[`mape_${nm}`] = ape.reduce((a, b) => a + b, 0) / n;
    out[`med_${nm}`] = median(ape);
    const lt = tru.map((t) => Math.log10(t[j])), lp = pred.map((p) => Math.log10(p[j]));
    const mu = lt.reduce((a, b) => a + b, 0) / n;
    out[`r2_${nm}`] = 1 - lp.reduce((s, v, i) => s + (v - lt[i]) ** 2, 0) / lt.reduce((s, v) => s + (v - mu) ** 2, 0);
  });
  return out;
}

export function initBenchmark(data) {
  B = data;
  const seg = $('#bm-noise');
  seg.innerHTML = B.levels.map((s) => `<button aria-pressed="${s === S.sigma}" data-s="${s}">σ = ${s}</button>`).join('');
  seg.addEventListener('click', (e) => {
    const s = e.target.dataset.s;
    if (s === undefined) return;
    S.sigma = +s;
    seg.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.s === S.sigma)));
    render();
  });
  const chk = $('#bm-methods');
  chk.innerHTML = Object.entries(METHODS).map(([k, m]) =>
    `<label><span class="dot" style="background:var(${m.color})"></span><input type="checkbox" value="${k}" checked>${m.label}</label>`).join('');
  chk.addEventListener('change', () => {
    S.on = new Set([...chk.querySelectorAll('input:checked')].map((i) => i.value));
    render();
  });
  render();
}

export function rerenderBenchmark() { if (B) render(); }

function render() {
  const avail = Object.keys(METHODS).filter((m) => S.on.has(m) && entry(m, S.sigma));
  scatter('D', 0, avail);
  scatter('rho', 1, avail);
  table(avail);
  noisePlot();
  timePlot();
  uncertainty();
}

function scatter(name, j, avail) {
  const traces = [];
  let lo = Infinity, hi = -Infinity;
  for (const m of avail) {
    const e = entry(m, S.sigma);
    const x = e.true.map((t) => t[j]), y = e.pred.map((p) => p[j]);
    lo = Math.min(lo, ...x, ...y); hi = Math.max(hi, ...x, ...y);
    traces.push({ x, y, mode: 'markers', name: METHODS[m].label, type: 'scattergl',
      marker: { size: m === 'cnn' ? 5 : 7, opacity: m === 'cnn' ? 0.45 : 0.85, color: methodColor(m) },
      hovertemplate: `true=%{x:.4g}<br>est=%{y:.4g}<extra>${METHODS[m].label}</extra>` });
  }
  const ticks = name === 'D' ? [0.003, 0.005, 0.01, 0.02, 0.03] : [1, 2, 3, 5, 8];
  if (isFinite(lo)) traces.unshift({ x: [lo, hi], y: [lo, hi], mode: 'lines', name: 'ideal', line: { color: cssVar('--muted'), dash: 'dash', width: 1.2 }, hoverinfo: 'skip' });
  draw(`bm-scatter-${name}`, traces, {
    xaxis: { type: 'log', title: `true ${name === 'D' ? 'D' : 'ρ'}`, tickmode: 'array', tickvals: ticks, ticktext: ticks.map(String) },
    yaxis: { type: 'log', title: `estimated ${name === 'D' ? 'D' : 'ρ'}`, tickmode: 'array', tickvals: ticks, ticktext: ticks.map(String) },
    margin: { l: 58, r: 10, t: 8, b: 70 },
  });
}

function table(avail) {
  const ns = avail.map((m) => entry(m, S.sigma).true.length);
  const nCommon = ns.length ? Math.min(...ns) : 0;
  $('#bm-subset').textContent = `first ${nCommon} test cases common to the selected methods · σ = ${S.sigma}`;
  const rows = avail.map((m) => {
    const e = entry(m, S.sigma);
    const st = stats(e.pred.slice(0, nCommon), e.true.slice(0, nCommon));
    const sec = m === 'cnn' ? e.ms_per_sample / 1000 : e.seconds_mean;
    return { m, st, sec };
  });
  const score = (r) => r.st.mape_D + r.st.mape_rho;
  const best = rows.length ? rows.reduce((a, b) => (score(a) <= score(b) ? a : b)).m : null;
  $('#bm-table').innerHTML = `<tr><th>Method</th><th>MAPE D</th><th>MAPE ρ</th><th>median APE D</th><th>median APE ρ</th><th>R² log D</th><th>R² log ρ</th><th>time / case</th></tr>` +
    rows.map(({ m, st, sec }) => `<tr class="${m === best ? 'best' : ''}"><td><span class="dot" style="background:var(${METHODS[m].color})"></span>${METHODS[m].label}</td>
      <td>${fmt(st.mape_D, 1)}%</td><td>${fmt(st.mape_rho, 1)}%</td><td>${fmt(st.med_D, 1)}%</td><td>${fmt(st.med_rho, 1)}%</td>
      <td>${fmt(st.r2_D, 3)}</td><td>${fmt(st.r2_rho, 3)}</td><td>${sec < 0.1 ? `${(sec * 1000).toFixed(1)} ms` : `${sec.toFixed(1)} s`}</td></tr>`).join('');
}

function noisePlot() {
  const traces = [];
  for (const m of Object.keys(METHODS)) {
    if (!S.on.has(m)) continue;
    const xs = [], ys = [];
    for (const s of B.levels) {
      const e = entry(m, s);
      if (!e) continue;
      xs.push(s);
      ys.push((e.metrics.mape_D + e.metrics.mape_rho) / 2);
    }
    if (xs.length) traces.push({ x: xs, y: ys, name: METHODS[m].label, mode: 'lines+markers', line: { color: methodColor(m), width: 2.4 }, marker: { size: 7 } });
  }
  draw('bm-noise-plot', traces, { xaxis: { title: 'noise σ' }, yaxis: { title: 'mean APE of D and ρ (%)', rangemode: 'tozero' }, margin: { l: 56, r: 10, t: 8, b: 70 } });
}

function timePlot() {
  const x = [], y = [], colors = [], text = [];
  for (const m of Object.keys(METHODS)) {
    if (!S.on.has(m)) continue;
    let e = entry(m, S.sigma);
    if (!e && (m === 'pinn_cold' || m === 'pinn_warm')) e = entry(m, 0.02);
    if (!e) continue;
    const sec = m === 'cnn' ? e.ms_per_sample / 1000 : e.seconds_mean;
    x.push(METHODS[m].label); y.push(sec); colors.push(methodColor(m));
    text.push(sec < 0.1 ? `${(sec * 1000).toFixed(1)} ms` : `${sec.toFixed(1)} s`);
  }
  draw('bm-time', [{ type: 'bar', x, y, text, textposition: 'outside', marker: { color: colors }, hovertemplate: '%{x}: %{text}<extra></extra>' }],
    { yaxis: { type: 'log', title: 'seconds per case (CPU)', range: [-3.5, 3.2] }, margin: { l: 60, r: 10, t: 24, b: 70 }, xaxis: { tickangle: -15 } });
}

function uncertainty() {
  const e = entry('cnn', S.sigma);
  if (!e) return;
  const err = e.pred.map((p, i) => 0.5 * (Math.abs(Math.log10(p[0] / e.true[i][0])) + Math.abs(Math.log10(p[1] / e.true[i][1]))));
  const sd = e.log10_std.map((s) => 0.5 * (s[0] + s[1]));
  const rho = spearman(sd, err);
  $('#bm-unc-hint').textContent = `Spearman = ${rho.toFixed(2)}`;
  draw('bm-unc', [{ x: sd, y: err, mode: 'markers', type: 'scattergl', marker: { size: 5, opacity: 0.55, color: methodColor('cnn') },
    hovertemplate: 'ensemble std=%{x:.4f}<br>|error|=%{y:.4f}<extra></extra>' }],
  { xaxis: { type: 'log', title: 'ensemble std (log10 units)' }, yaxis: { type: 'log', title: 'mean |error| (log10 units)' }, margin: { l: 60, r: 10, t: 8, b: 60 } });
}
