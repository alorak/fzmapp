// "Eğik atış" page: a physics-informed neural network (PINN) next to the drag-free formula and a
// classical Runge-Kutta solution. Loaded only when the page is opened.
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs';
const G = 9.81;
const $ = id => document.getElementById(id);
const fmt = (value, digits = 1) => new Intl.NumberFormat('tr', {minimumFractionDigits: digits, maximumFractionDigits: digits}).format(value);
const svgNS = 'http://www.w3.org/2000/svg';

let session;
let meta;
let animation;

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(svgNS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  return node;
}

// ---------- the three solutions ----------
function dragFree(v0, theta) {
  const time = 2 * v0 * Math.sin(theta) / G;
  const points = [];
  for (let i = 0; i <= 120; i++) {
    const t = time * i / 120;
    points.push([v0 * Math.cos(theta) * t, Math.max(0, v0 * Math.sin(theta) * t - G * t * t / 2), t]);
  }
  return {points, range: v0 * v0 * Math.sin(2 * theta) / G, height: (v0 * Math.sin(theta)) ** 2 / (2 * G), time};
}

// Classical fourth-order Runge-Kutta integration of a = -g y_hat - k |v| v until the projectile lands.
function rungeKutta(v0, theta, k) {
  const rhs = ([, , vx, vy]) => {
    const speed = Math.hypot(vx, vy);
    return [vx, vy, -k * speed * vx, -G - k * speed * vy];
  };
  const dt = Math.max(1e-4, (2 * v0 * Math.sin(theta) / G) / 4000);
  let state = [0, 0, v0 * Math.cos(theta), v0 * Math.sin(theta)];
  let t = 0;
  let height = 0;
  const points = [[0, 0, 0]];
  for (let step = 0; step < 200000; step++) {
    const k1 = rhs(state);
    const k2 = rhs(state.map((s, i) => s + dt / 2 * k1[i]));
    const k3 = rhs(state.map((s, i) => s + dt / 2 * k2[i]));
    const k4 = rhs(state.map((s, i) => s + dt * k3[i]));
    const next = state.map((s, i) => s + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    if (next[1] < 0) {
      const frac = state[1] / (state[1] - next[1]);
      const x = state[0] + frac * (next[0] - state[0]);
      points.push([x, 0, t + frac * dt]);
      return {points: thin(points), range: x, height, time: t + frac * dt};
    }
    state = next;
    t += dt;
    height = Math.max(height, state[1]);
    points.push([state[0], state[1], t]);
  }
  return {points: thin(points), range: state[0], height, time: t};
}

function thin(points, keep = 160) {
  if (points.length <= keep) return points;
  const step = (points.length - 1) / (keep - 1);
  return Array.from({length: keep}, (_, i) => points[Math.round(i * step)]);
}

// The PINN works in units of v0^2/g (length) and v0/g (time): inputs (tau, theta, kappa), outputs (X, Y).
async function neuralNetwork(v0, theta, k) {
  const kappa = k * v0 * v0 / G;
  const length = v0 * v0 / G;
  const timeUnit = v0 / G;
  const n = 400;
  const tauEnd = 2.1 * Math.sin(theta);
  const input = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    input[i * 3] = tauEnd * i / (n - 1);
    input[i * 3 + 1] = theta;
    input[i * 3 + 2] = kappa;
  }
  const {ort} = session;
  const output = await session.run({inputs: new ort.Tensor('float32', input, [n, 3])});
  const xy = output.position.data;
  const points = [[0, 0, 0]];
  let height = 0;
  for (let i = 1; i < n; i++) {
    const x = xy[i * 2] * length;
    const y = xy[i * 2 + 1] * length;
    const t = input[i * 3] * timeUnit;
    if (y < 0) {
      const [px, py, pt] = points[points.length - 1];
      const frac = py / (py - y);
      const landing = [px + frac * (x - px), 0, pt + frac * (t - pt)];
      points.push(landing);
      return {points, range: landing[0], height, time: landing[2], kappa};
    }
    height = Math.max(height, y);
    points.push([x, y, t]);
  }
  const last = points[points.length - 1];
  return {points, range: last[0], height, time: last[2], kappa};
}

// ---------- drawing ----------
// The drawing uses real CSS pixels so that labels stay readable on a phone.
const PLOT = {width: 820, height: 420, left: 48, right: 14, top: 16, bottom: 40};

// Height follows the shape of the widest trajectory, so a flat throw doesn't leave empty sky.
function fitPlot(plot, maxX, maxY) {
  const width = Math.round(plot.clientWidth) || 820;
  const innerW = width - PLOT.left - PLOT.right;
  const wanted = innerW * maxY / maxX + PLOT.top + PLOT.bottom;
  PLOT.width = width;
  PLOT.height = Math.round(Math.min(width * 0.62, 480, Math.max(200, wanted)));
  plot.setAttribute('viewBox', `0 0 ${PLOT.width} ${PLOT.height}`);
}

function niceStep(span, target = 6) {
  const raw = span / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map(m => m * power).find(step => span / step <= target) ?? power * 10;
}

function draw(curves) {
  const plot = $('proj-plot');
  const shown = curves.filter(curve => curve && curve.visible);
  const maxX = Math.max(1, ...shown.map(curve => curve.range)) * 1.05;
  const maxY = Math.max(1, ...shown.map(curve => curve.height)) * 1.15;
  fitPlot(plot, maxX, maxY);
  // Same scale on both axes so the shape of the trajectory is true.
  const innerW = PLOT.width - PLOT.left - PLOT.right;
  const innerH = PLOT.height - PLOT.top - PLOT.bottom;
  const scale = Math.min(innerW / maxX, innerH / maxY);
  const sx = x => PLOT.left + x * scale;
  const sy = y => PLOT.height - PLOT.bottom - y * scale;
  const items = [];

  const stepX = niceStep(innerW / scale, Math.max(3, Math.round(innerW / 110)));
  for (let x = 0; x <= innerW / scale + 1e-9; x += stepX) {
    items.push(svg('line', {class: 'proj-grid', x1: sx(x), x2: sx(x), y1: PLOT.top, y2: sy(0)}));
    items.push(svg('text', {class: 'proj-tick', x: sx(x), y: sy(0) + 18, 'text-anchor': 'middle'}, fmt(x, stepX < 1 ? 1 : 0)));
  }
  const stepY = niceStep(innerH / scale, Math.max(3, Math.round(innerH / 70)));
  for (let y = stepY; y <= innerH / scale + 1e-9; y += stepY) {
    items.push(svg('line', {class: 'proj-grid', x1: PLOT.left, x2: PLOT.width - PLOT.right, y1: sy(y), y2: sy(y)}));
    items.push(svg('text', {class: 'proj-tick', x: PLOT.left - 8, y: sy(y) + 4, 'text-anchor': 'end'}, fmt(y, stepY < 1 ? 1 : 0)));
  }
  items.push(svg('line', {class: 'proj-ground', x1: PLOT.left, x2: PLOT.width - PLOT.right, y1: sy(0), y2: sy(0)}));
  items.push(svg('text', {class: 'proj-axis', x: PLOT.width - PLOT.right, y: PLOT.height - 6, 'text-anchor': 'end'}, 'yatay uzaklık (m)'));
  items.push(svg('text', {class: 'proj-axis', x: 12, y: PLOT.top + 4, transform: `rotate(-90 12 ${PLOT.top + 4})`, 'text-anchor': 'end'}, 'yükseklik (m)'));

  for (const curve of shown) {
    const d = curve.points.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join(' ');
    items.push(svg('path', {class: `proj-curve ${curve.className}`, d}));
    items.push(svg('circle', {class: `proj-land ${curve.className}`, cx: sx(curve.range), cy: sy(0), r: 4.5}));
  }
  const ball = svg('circle', {class: 'proj-ball', r: 7, cx: sx(0), cy: sy(0), visibility: 'hidden'});
  items.push(ball);
  plot.replaceChildren(...items);
  return {ball, sx, sy};
}

function launch(curve, view) {
  cancelAnimationFrame(animation);
  if (!curve) return;
  const {ball, sx, sy} = view;
  ball.setAttribute('visibility', 'visible');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const duration = reduce ? 0 : Math.min(3500, Math.max(900, curve.time * 1000));
  const start = performance.now();
  const frame = now => {
    const share = duration ? Math.min(1, (now - start) / duration) : 1;
    const t = share * curve.time;
    const points = curve.points;
    let i = points.findIndex(p => p[2] >= t);
    if (i <= 0) i = i === 0 ? 1 : points.length - 1;
    const [x0, y0, t0] = points[i - 1];
    const [x1, y1, t1] = points[i];
    const f = t1 > t0 ? (t - t0) / (t1 - t0) : 1;
    ball.setAttribute('cx', sx(x0 + f * (x1 - x0)));
    ball.setAttribute('cy', sy(y0 + f * (y1 - y0)));
    if (share < 1) animation = requestAnimationFrame(frame);
  };
  animation = requestAnimationFrame(frame);
}

// ---------- page ----------
function readInputs() {
  return {
    v0: Number($('proj-speed').value),
    angle: Number($('proj-angle').value),
    k: Number($('proj-drag').value),
  };
}

function showValues({v0, angle, k}) {
  $('proj-speed-out').textContent = `${fmt(v0, 0)} m/s`;
  $('proj-angle-out').textContent = `${fmt(angle, 0)}°`;
  $('proj-drag-out').textContent = k ? `${fmt(k, 4)} 1/m` : 'yok';
  $('proj-kappa').textContent = fmt(k * v0 * v0 / G, 2);
  for (const preset of document.querySelectorAll('#proj-presets button')) {
    preset.setAttribute('aria-pressed', String(Math.abs(Number(preset.dataset.k) - k) < 1e-6));
  }
}

function fillTable(free, exact, net) {
  const rows = [
    ['range', 'm', 1],
    ['height', 'm', 1],
    ['time', 's', 2],
  ];
  for (const [key, unit, digits] of rows) {
    $(`proj-${key}-free`).textContent = `${fmt(free[key], digits)} ${unit}`;
    $(`proj-${key}-exact`).textContent = `${fmt(exact[key], digits)} ${unit}`;
    $(`proj-${key}-net`).textContent = net ? `${fmt(net[key], digits)} ${unit}` : '–';
    const error = net && exact[key] > 0 ? Math.abs(net[key] - exact[key]) / exact[key] * 100 : null;
    $(`proj-${key}-error`).textContent = error == null ? '–' : `%${fmt(error, error < 1 ? 2 : 1)}`;
  }
}

let lastView = null;
let generation = 0;

async function update({animate = false} = {}) {
  // A newer call replaces an older one that is still waiting for the model.
  const current = ++generation;
  const inputs = readInputs();
  showValues(inputs);
  const theta = inputs.angle * Math.PI / 180;
  const free = dragFree(inputs.v0, theta);
  const exact = rungeKutta(inputs.v0, theta, inputs.k);
  let net = null;
  if (session) {
    try {
      net = await neuralNetwork(inputs.v0, theta, inputs.k);
    } catch (error) {
      $('proj-model-state').textContent = 'Model çalıştırılamadı: ' + error.message;
    }
  }
  if (current !== generation) return;
  const curves = [
    {...free, visible: $('proj-show-free').checked, className: 'free'},
    {...exact, visible: $('proj-show-exact').checked, className: 'exact'},
    net && {...net, visible: $('proj-show-net').checked, className: 'net'},
  ];
  lastView = draw(curves);
  fillTable(free, exact, net);
  if (animate) launch(net ?? exact, lastView);
}

async function loadModel() {
  const state = $('proj-model-state');
  try {
    const [ortModule, metaResponse] = await Promise.all([
      import(ORT_URL),
      fetch('./models/egik-atis.json', {cache: 'no-cache'}),
    ]);
    const ort = ortModule.default ?? ortModule;
    ort.env.wasm.numThreads = 1;
    meta = await metaResponse.json();
    const inference = await ort.InferenceSession.create('./models/egik-atis.onnx', {executionProviders: ['wasm']});
    session = {ort, run: feeds => inference.run(feeds)};
    state.dataset.state = 'ready';
    state.textContent = `Model yüklendi · ${new Intl.NumberFormat('tr').format(meta.parameters)} parametre`;
    const check = meta.check_against_runge_kutta;
    $('proj-accuracy').textContent = `Eğitimden sonra ${check.cases} farklı atışta (açı 10°–85°, her direnç düzeyi) sayısal çözümle karşılaştırıldı. Menzilde ortalama fark %${fmt(check.range.mean_percent, 1)}, en kötü durumda %${fmt(check.range.max_percent, 1)}; en yüksek noktada ortalama %${fmt(check.height.mean_percent, 1)}, uçuş süresinde %${fmt(check.time.mean_percent, 1)}.`;
  } catch (error) {
    state.dataset.state = 'error';
    state.textContent = 'Yapay zekâ modeli yüklenemedi; yalnızca formül ve sayısal çözüm gösteriliyor.';
  }
}

export async function start() {
  for (const id of ['proj-speed', 'proj-angle', 'proj-drag']) $(id).addEventListener('input', () => update());
  for (const id of ['proj-show-free', 'proj-show-exact', 'proj-show-net']) $(id).addEventListener('change', () => update());
  for (const preset of document.querySelectorAll('#proj-presets button')) {
    preset.addEventListener('click', () => {
      $('proj-drag').value = preset.dataset.k;
      update({animate: true});
    });
  }
  $('proj-launch').addEventListener('click', () => update({animate: true}));
  let width = 0;
  new ResizeObserver(([entry]) => {
    const next = Math.round(entry.contentRect.width);
    if (next && next !== width) {
      if (width) update();
      width = next;
    }
  }).observe($('proj-plot'));
  update();
  await loadModel();
  update({animate: true});
}

// ---------- "Nasıl çalışır?" tab: figures drawn from the training record in models/egik-atis.json ----------
const SUPERSCRIPT = {'-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹'};
const power = n => '10' + String(n).split('').map(c => SUPERSCRIPT[c]).join('');
const scientific = value => {
  const exponent = Math.floor(Math.log10(value));
  return `${fmt(value / 10 ** exponent, 1)} × ${power(exponent)}`;
};

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text != null) node.textContent = text;
  return node;
}

function checkTables(check) {
  const box = $('pinn-check');
  const stages = [['Eğitimden önce (rastgele ayarlar)', check.start, 'bad'], ['Eğitimden sonra', check.end, 'good']];
  box.replaceChildren(...stages.map(([title, data, tone]) => {
    const stage = el('div', {class: 'pinn-stage'});
    stage.append(el('h4', {}, title));
    const table = el('table');
    table.innerHTML = '<thead><tr><th scope="col">İvme (m/s²)</th><th scope="col">yatay</th><th scope="col">dikey</th></tr></thead>';
    const body = el('tbody');
    const diff = data.network.map((value, i) => value - data.physics[i]);
    for (const [label, values, cls] of [['Ağın tahmininden', data.network], ['Denklemin istediği', data.physics], ['Fark', diff, tone]]) {
      const row = el('tr');
      row.append(el('th', {scope: 'row'}, label));
      for (const value of values) row.append(el('td', cls ? {class: cls} : {}, fmt(value * G, 2)));
      body.append(row);
    }
    table.append(body);
    stage.append(table);
    return stage;
  }));
}

function lossChart(history) {
  // Drawn at the box's real width so the labels stay readable on a phone.
  const W = Math.max(280, Math.round($('pinn-loss').clientWidth) || 460);
  const H = Math.round(Math.min(260, W * 0.62)), left = 46, right = 12, top = 12, bottom = 40;
  const steps = history.at(-1)[0];
  const logs = history.map(([, value]) => Math.log10(value));
  const lo = Math.floor(Math.min(...logs));
  const hi = Math.ceil(Math.max(...logs));
  const sx = step => left + step / steps * (W - left - right);
  const sy = log => top + (hi - log) / (hi - lo) * (H - top - bottom);
  const items = [];
  for (let p = lo; p <= hi; p++) {
    items.push(svg('line', {class: 'proj-grid', x1: left, x2: W - right, y1: sy(p), y2: sy(p)}));
    items.push(svg('text', {class: 'pinn-chart-label', x: left - 6, y: sy(p) + 4, 'text-anchor': 'end'}, power(p)));
  }
  for (let step = 0; step <= steps; step += 10000) {
    const anchor = step === 0 ? 'start' : step + 10000 > steps ? 'end' : 'middle';
    items.push(svg('text', {class: 'pinn-chart-label', x: sx(step), y: H - bottom + 18, 'text-anchor': anchor}, new Intl.NumberFormat('tr').format(step)));
  }
  items.push(svg('text', {class: 'pinn-chart-label', x: (left + W - right) / 2, y: H - 2, 'text-anchor': 'middle'}, 'eğitim adımı'));
  items.push(svg('line', {class: 'proj-ground', x1: left, x2: W - right, y1: H - bottom, y2: H - bottom}));
  const d = history.map(([step], i) => `${i ? 'L' : 'M'}${sx(step).toFixed(1)} ${sy(logs[i]).toFixed(1)}`).join(' ');
  items.push(svg('path', {d, fill: 'none', stroke: '#1b4f9c', 'stroke-width': 2.2, 'stroke-linejoin': 'round'}));
  const first = history[0][1];
  const last = history.at(-1)[1];
  const chart = svg('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `Fizik kaybı ${fmt(first, 2)} değerinden ${scientific(last)} değerine düştü`});
  chart.append(...items);
  $('pinn-loss').replaceChildren(chart, el('p', {class: 'vis-note'}, `İlk 250 adımda ortalama ${fmt(first, 2)} olan kayıp, son adımlarda ${scientific(last)} değerine indi (yaklaşık ${new Intl.NumberFormat('tr').format(Math.round(first / last / 100) * 100)} kat). Her nokta 250 adımın ortalamasıdır.`));
}

function landed(path) {
  // Cut the path where it first goes below the ground, like the main page does.
  const out = [path[0]];
  for (let i = 1; i < path.length; i++) {
    const [x, y] = path[i];
    if (y < 0) {
      const [px, py] = path[i - 1];
      const f = py / (py - y);
      out.push([px + f * (x - px), 0]);
      return out;
    }
    out.push(path[i]);
  }
  return out;
}

function snapshots(data) {
  const reference = data.runge_kutta_path;
  const maxX = Math.max(...reference.map(p => p[0])) * 1.3;
  const maxY = Math.max(...reference.map(p => p[1])) * 1.9;
  const W = 160, H = 100, pad = 6;
  const scale = Math.min((W - 2 * pad) / maxX, (H - 2 * pad) / maxY);
  const sx = x => pad + x * scale;
  const sy = y => H - pad - y * scale;
  const line = path => path.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join(' ');
  const figures = data.snapshots.map(({step, path}, index) => {
    const figure = el('figure');
    const id = `pinn-clip-${index}`;
    const picture = svg('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${step}. adımda yörünge`});
    const clip = svg('clipPath', {id});
    clip.append(svg('rect', {x: 0, y: 0, width: W, height: H - pad}));
    picture.append(
      clip,
      svg('line', {x1: 0, x2: W, y1: sy(0), y2: sy(0), stroke: '#0f2742', 'stroke-width': 1}),
      svg('path', {d: line(reference), fill: 'none', stroke: 'rgba(27,79,156,.35)', 'stroke-width': 5, 'stroke-linecap': 'round'}),
      svg('path', {d: line(landed(path)), fill: 'none', stroke: '#e0a100', 'stroke-width': 2.2, 'clip-path': `url(#${id})`, 'stroke-linejoin': 'round'}),
    );
    const label = step === 0 ? 'Başlangıç' : `${new Intl.NumberFormat('tr').format(step)}. adım`;
    figure.append(picture, el('figcaption', {}, index === data.snapshots.length - 1 ? `${label} (son)` : label));
    return figure;
  });
  $('pinn-snapshots').replaceChildren(...figures);
}

function exampleTable(data) {
  const {v0} = data.demo;
  const length = v0 * v0 / G;
  const time = v0 / G;
  const table = el('table', {class: 'pinn-example'});
  table.innerHTML = '<thead><tr><th><i>τ</i></th><th><i>X</i></th><th><i>Y</i></th><th>süre</th><th>uzaklık</th><th>yükseklik</th></tr></thead>';
  const body = el('tbody');
  for (const {tau, x, y} of data.example) {
    const row = el('tr');
    row.append(el('td', {}, fmt(tau, 2)), el('td', {}, fmt(x, 3)), el('td', {}, fmt(y, 3)),
      el('td', {class: 'unit'}, `${fmt(tau * time, 2)} s`), el('td', {class: 'unit'}, `${fmt(x * length, 1)} m`),
      el('td', {class: 'unit'}, y < 0 ? 'yerin altında' : `${fmt(y * length, 1)} m`));
    body.append(row);
  }
  table.append(body);
  const label = el('p', {class: 'vis-label'}, `Modelin çıktısından birkaç satır (futbol topu, ${fmt(v0, 0)} m/s, ${data.demo.angle_degrees}°)`);
  label.style.marginBottom = '8px';
  const note = el('p', {class: 'vis-note'}, 'Mavi sütunlar, modelin boyutsuz çıktısının saniyeye ve metreye çevrilmiş hâlidir.');
  $('pinn-example').replaceChildren(label, table, note);
}

function heatColor(share) {
  const stops = [[227, 241, 235], [246, 230, 168], [224, 161, 0], [160, 36, 44]];
  const position = Math.min(1, Math.max(0, share)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(position));
  const f = position - i;
  return `rgb(${stops[i].map((c, k) => Math.round(c + f * (stops[i + 1][k] - c))).join(',')})`;
}

function heatmap(check) {
  const grid = check.range_grid;
  const rows = grid.angles_degrees.length;
  const cols = grid.kappas.length;
  const width = Math.max(280, Math.round($('pinn-heatmap').clientWidth) || 492);
  const top = 0, left = 44, cell = Math.min(56, (width - left - 4) / cols), cellH = Math.min(22, cell * 0.7);
  const W = left + cols * cell + 4, H = top + rows * cellH + 36;
  const maxValue = Math.max(...grid.percent.flat());
  const ceiling = Math.max(0.5, Math.ceil(maxValue * 2) / 2);
  const picture = svg('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Açı ve dirence göre menzil hatası haritası'});
  // Steep throws at the top, like the angle on a protractor.
  grid.angles_degrees.forEach((angle, r) => {
    const y = top + (rows - 1 - r) * cellH;
    if (angle % 15 === 10 || angle === 85) picture.append(svg('text', {class: 'pinn-chart-label', x: left - 6, y: y + cellH - 4, 'text-anchor': 'end'}, `${angle}°`));
    grid.kappas.forEach((kappa, c) => {
      const value = grid.percent[r][c];
      const rect = svg('rect', {x: left + c * cell, y, width: cell - 1, height: cellH - 1, rx: 2, fill: heatColor(value / ceiling)});
      rect.append(svg('title', {}, `${angle}°, κ = ${fmt(kappa, 2)}: %${fmt(value, 2)}`));
      picture.append(rect);
    });
  });
  grid.kappas.forEach((kappa, c) => {
    if (c % 2 === 0) picture.append(svg('text', {class: 'pinn-chart-label', x: left + c * cell + cell / 2, y: top + rows * cellH + 14, 'text-anchor': 'middle'}, fmt(kappa, 1)));
  });
  picture.append(svg('text', {class: 'pinn-chart-label', x: left + cols * cell, y: H - 2, 'text-anchor': 'end'}, 'boyutsuz direnç κ'));
  const key = el('p', {class: 'pinn-heat-key'});
  key.append(el('span', {}, '%0'), el('i'), el('span', {}, `%${fmt(ceiling, 1)}`));
  $('pinn-heatmap').replaceChildren(el('p', {class: 'vis-label'}, 'Menzildeki fark (yüzde)'), picture, key);

  let worst = [0, 0];
  grid.percent.forEach((row, r) => row.forEach((value, c) => {
    if (value > grid.percent[worst[0]][worst[1]]) worst = [r, c];
  }));
  $('pinn-accuracy').textContent = `Eğitimden sonra model, ${rows} açı ve ${cols} direnç düzeyinden oluşan ${check.cases} atışta sınandı. Menzildeki fark ortalama %${fmt(check.range.mean_percent, 2)}; en büyük fark %${fmt(maxValue, 2)} ile ${grid.angles_degrees[worst[0]]}° ve κ = ${fmt(grid.kappas[worst[1]], 1)} atışında. En yüksek noktada ortalama fark %${fmt(check.height.mean_percent, 2)}, uçuş süresinde %${fmt(check.time.mean_percent, 2)}.`;
}

export async function explain() {
  const response = await fetch('./models/egik-atis.json', {cache: 'no-cache'});
  const record = await response.json();
  const data = record.explanation;
  checkTables(data.residual_check);
  lossChart(data.loss_history);
  snapshots(data);
  exampleTable(data);
  heatmap(record.check_against_runge_kutta);
}
