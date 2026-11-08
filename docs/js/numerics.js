import { $, cssVar, draw, methodColor } from './common.js';

let CV = null, TR = null, SC = null;
let sel = 0;

export function initNumerics(conv, train, showcase) {
  CV = conv; TR = train; SC = showcase;
  const s = $('#nm-pinn-sel');
  s.innerHTML = TR.pinn.map((p, i) => `<option value="${i}">test #${p.idx} · σ=${p.sigma}</option>`).join('');
  s.addEventListener('change', () => { sel = +s.value; renderTraining(); });
  render();
}

export function rerenderNumerics() { if (CV) render(); }

function render() {
  const sp = CV.spatial;
  draw('nm-space', [
    { x: sp.spectral.map((r) => r.dof), y: sp.spectral.map((r) => Math.max(r.error, 1e-16)), name: 'Legendre–Gauss–Lobatto', mode: 'lines+markers', line: { color: methodColor('nls'), width: 2.6 }, marker: { size: 7 } },
    { x: sp.fd2.map((r) => r.dof), y: sp.fd2.map((r) => r.error), name: 'finite differences (2nd order)', mode: 'lines+markers', line: { color: methodColor('pinn_cold'), width: 2.6 }, marker: { size: 7, symbol: 'square' } },
  ], { xaxis: { type: 'log', title: 'number of unknowns' }, yaxis: { type: 'log', title: 'max-norm error at t = T', exponentformat: 'power' }, margin: { l: 66, r: 10, t: 8, b: 70 } });

  const tm = CV.temporal;
  const s0 = tm.rows[0];
  draw('nm-time', [
    { x: tm.rows.map((r) => r.steps), y: tm.rows.map((r) => r.error), name: 'CN / AB2', mode: 'lines+markers', line: { color: methodColor('cnn'), width: 2.6 }, marker: { size: 7 } },
    { x: tm.rows.map((r) => r.steps), y: tm.rows.map((r) => (s0.error * (s0.steps / r.steps) ** 2)), name: 'slope −2 guide', mode: 'lines', line: { color: cssVar('--muted'), dash: 'dash' } },
  ], { xaxis: { type: 'log', title: 'time steps' }, yaxis: { type: 'log', title: 'max-norm error', exponentformat: 'power' }, margin: { l: 66, r: 10, t: 8, b: 70 } });
  const ord = tm.observed_order;
  $('#nm-time-note').textContent = `Observed order between successive refinements: ${ord.map((o) => o.toFixed(2)).join(', ')} (theory: 2). Spatial resolution fixed at M = ${tm.M}.`;
  renderTraining();
}

function renderTraining() {
  const cols = [methodColor('cnn'), methodColor('nls'), methodColor('pinn_cold')];
  const tr = [];
  TR.cnn.forEach((h, i) => {
    tr.push({ x: h.epoch, y: h.train, name: `train · member ${i + 1}`, mode: 'lines', line: { color: cols[i % 3], width: 1.8 } });
    tr.push({ x: h.epoch, y: h.val, name: `val · member ${i + 1}`, mode: 'lines', line: { color: cols[i % 3], width: 1.8, dash: 'dot' } });
  });
  draw('nm-cnn', tr, { xaxis: { title: 'epoch' }, yaxis: { type: 'log', title: 'smooth-L1 loss' }, margin: { l: 56, r: 10, t: 8, b: 90 } });

  const p = TR.pinn[sel];
  if (!p) return;
  // the physics loss is only active after the data-only warm-up, so skip the warm-up zeros
  const L = (h, name, key, color, dash) => {
    const keep = h.iter.map((_, i) => i).filter((i) => h[key][i] > 0);
    return { x: keep.map((i) => h.iter[i]), y: keep.map((i) => h[key][i]), name, mode: 'lines', line: { color, width: 2, dash } };
  };
  draw('nm-pinn', [
    L(p.cold, 'data loss · cold', 'data', methodColor('pinn_cold'), 'solid'),
    L(p.cold, 'physics loss · cold', 'pde', methodColor('pinn_cold'), 'dot'),
    L(p.warm, 'data loss · CNN-init PINN', 'data', methodColor('pinn_warm'), 'solid'),
    L(p.warm, 'physics loss · CNN-init PINN', 'pde', methodColor('pinn_warm'), 'dot'),
  ], { xaxis: { title: 'iteration' }, yaxis: { type: 'log', title: 'loss' }, margin: { l: 56, r: 10, t: 8, b: 90 } });

  const c = SC && SC.cases.find((q) => q.idx === p.idx && Math.abs(q.sigma - p.sigma) < 1e-9);
  const tD = c ? Math.log10(c.true[0]) : null, tR = c ? Math.log10(c.true[1]) : null;
  const traces = [
    { x: p.cold.iter, y: p.cold.log10_D, name: 'cold', mode: 'lines', line: { color: methodColor('pinn_cold'), width: 2.2 }, xaxis: 'x', yaxis: 'y' },
    { x: p.warm.iter, y: p.warm.log10_D, name: 'CNN-init PINN', mode: 'lines', line: { color: methodColor('pinn_warm'), width: 2.2 }, xaxis: 'x', yaxis: 'y' },
    { x: p.cold.iter, y: p.cold.log10_rho, name: 'cold', mode: 'lines', showlegend: false, line: { color: methodColor('pinn_cold'), width: 2.2 }, xaxis: 'x2', yaxis: 'y2' },
    { x: p.warm.iter, y: p.warm.log10_rho, name: 'CNN-init PINN', mode: 'lines', showlegend: false, line: { color: methodColor('pinn_warm'), width: 2.2 }, xaxis: 'x2', yaxis: 'y2' },
  ];
  const shapes = [];
  if (c) {
    shapes.push({ type: 'line', xref: 'x domain', x0: 0, x1: 1, yref: 'y', y0: tD, y1: tD, line: { color: cssVar('--truth'), dash: 'dash', width: 1.4 } });
    shapes.push({ type: 'line', xref: 'x2 domain', x0: 0, x1: 1, yref: 'y2', y0: tR, y1: tR, line: { color: cssVar('--truth'), dash: 'dash', width: 1.4 } });
  }
  draw('nm-traj', traces, {
    grid: { rows: 1, columns: 2, pattern: 'independent' },
    xaxis: { title: 'iteration', domain: [0, 0.47] }, yaxis: { title: 'log10 D (dashed = truth)' },
    xaxis2: { title: 'iteration', domain: [0.53, 1], anchor: 'y2' }, yaxis2: { title: 'log10 ρ (dashed = truth)', anchor: 'x2' },
    shapes, margin: { l: 60, r: 10, t: 8, b: 80 },
  });
}
