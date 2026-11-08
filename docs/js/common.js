// Shared helpers for the dashboard.
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export const METHODS = {
  cnn: { label: 'CNN', color: '--c-cnn' },
  nls: { label: 'Spectral least squares', color: '--c-nls' },
  pinn_cold: { label: 'PINN', color: '--c-pinn' },
  pinn_warm: { label: 'CNN-init PINN', color: '--c-warm' },
};

export const methodColor = (k) => cssVar(METHODS[k].color);

export async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

export function debounce(fn, ms = 60) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));

export function fmt(x, d = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(2);
  return x.toFixed(d);
}

export const pct = (x, d = 1) => (x === null || x === undefined ? '—' : `${x.toFixed(d)}%`);

export function plotLayout(extra = {}) {
  const text = cssVar('--text'), muted = cssVar('--muted'), border = cssVar('--border');
  const base = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: text, size: 12, family: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif' },
    margin: { l: 56, r: 16, t: 24, b: 46 },
    xaxis: { gridcolor: border, zerolinecolor: border, linecolor: border, tickcolor: muted, automargin: true },
    yaxis: { gridcolor: border, zerolinecolor: border, linecolor: border, tickcolor: muted, automargin: true },
    legend: { orientation: 'h', y: -0.22, font: { size: 11 } },
    hoverlabel: { font: { size: 12 } },
  };
  const out = { ...base, ...extra };
  for (const k of ['xaxis', 'yaxis']) out[k] = { ...base[k], ...(extra[k] || {}) };
  return out;
}

export const plotConfig = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'] };

export function draw(id, data, layout, config = {}) {
  const node = typeof id === 'string' ? document.getElementById(id) : id;
  return Plotly.react(node, data, plotLayout(layout), { ...plotConfig, ...config });
}

/** Sequential colour scale for tumour density (0 = healthy, 1 = carrying capacity). */
export function densityScale() {
  const dark = document.documentElement.dataset.theme === 'dark' ||
    (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  return dark
    ? [[0, '#0e1319'], [0.25, '#1b3a5c'], [0.5, '#2f7f9a'], [0.75, '#e6b450'], [1, '#fff3c4']]
    : [[0, '#f4f7fb'], [0.25, '#bcd7ea'], [0.5, '#5ea3c4'], [0.75, '#d9822b'], [1, '#7a1f12']];
}

export function divergingScale() {
  return [[0, '#2563eb'], [0.5, '#f8fafc'], [1, '#dc2626']];
}

export function reshape(flat, rows, cols) {
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) out[i] = Array.from(flat.slice ? flat.slice(i * cols, (i + 1) * cols) : []);
  return out;
}

export function spearman(a, b) {
  const rank = (v) => {
    const idx = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(v.length);
    idx.forEach(([, i], k) => { r[i] = k; });
    return r;
  };
  const ra = rank(a), rb = rank(b), n = a.length;
  const ma = (n - 1) / 2;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (ra[i] - ma) * (rb[i] - ma);
    da += (ra[i] - ma) ** 2;
    db += (rb[i] - ma) ** 2;
  }
  return num / Math.sqrt(da * db);
}

export function mape(pred, true_, j) {
  let s = 0;
  for (let i = 0; i < pred.length; i++) s += Math.abs(pred[i][j] - true_[i][j]) / true_[i][j];
  return (100 * s) / pred.length;
}

export function median(v) {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

export function renderMath(root = document.body) {
  if (window.renderMathInElement) {
    window.renderMathInElement(root, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false,
    });
  }
}

export function download(name, text, mime = 'text/csv') {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** Build a slider control. Returns {el, get, set}. `log` makes it log-scaled. */
export function slider({ id, label, min, max, step, value, log = false, digits = 3, unit = '', onInput }) {
  const wrap = document.createElement('div');
  wrap.className = 'ctl';
  const toPos = (v) => (log ? Math.log10(v) : v);
  const fromPos = (p) => (log ? 10 ** p : p);
  wrap.innerHTML = `<label for="${id}"><span>${label}</span><output id="${id}-o"></output></label>
    <input type="range" id="${id}" min="${toPos(min)}" max="${toPos(max)}" step="${log ? 0.01 : step}" value="${toPos(value)}">`;
  const input = wrap.querySelector('input'), out = wrap.querySelector('output');
  const show = () => { out.textContent = `${fmt(fromPos(+input.value), digits)}${unit}`; };
  input.addEventListener('input', () => { show(); onInput && onInput(fromPos(+input.value)); });
  show();
  return {
    el: wrap,
    get: () => fromPos(+input.value),
    set: (v) => { input.value = toPos(v); show(); },
  };
}
