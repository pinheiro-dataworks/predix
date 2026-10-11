/* ─────────────────────────────────────────────────────────────────────────────
   Predix — app.js
   Dashboard application: data loading, filter management, Plotly chart renders.
   ───────────────────────────────────────────────────────────────────────────── */

'use strict';

// ── Global state ──────────────────────────────────────────────────────────────
let DATA          = null;
let currentMonth  = 'all';
let currentCat    = 'all';
let currentTab    = 'overview';
let renderedTabs  = new Set();

// ── Palette ───────────────────────────────────────────────────────────────────
const C = {
  bg:      '#0D0D0F',
  card:    '#1A1A1D',
  acc1:    '#7B5CFA',
  acc2:    '#A78BFA',
  green:   '#34D399',
  amber:   '#F59E0B',
  red:     '#EF4444',
  blue:    '#60A5FA',
  orange:  '#F97316',
  pink:    '#EC4899',
  txt:     '#FFFFFF',
  txt2:    '#9A9AA5',
  txt3:    '#5A5A65',
  border:  'rgba(255,255,255,0.06)',
  gridCol: 'rgba(255,255,255,0.05)',
};

const PALETTE = [C.acc1, C.acc2, C.green, C.amber, C.red, C.blue, C.orange, C.pink];

const SEG_COLORS = {
  'Champions':          C.amber,
  'Loyal':              C.green,
  'Potential Loyalist': C.blue,
  'At Risk':            C.red,
  "Needs Attention":    C.orange,
  "Can't Lose":         C.acc2,
  'Hibernating':        '#6B7280',
  'Lost':               '#374151',
};

// ── Plotly base layout ────────────────────────────────────────────────────────
function baseLayout(overrides = {}) {
  return Object.assign({
    paper_bgcolor: C.card,
    plot_bgcolor:  C.card,
    font:  { family: "'Inter','Segoe UI',sans-serif", color: C.txt, size: 12 },
    margin: { t: 40, r: 20, b: 50, l: 60, pad: 4 },
    xaxis: {
      gridcolor: C.gridCol, zerolinecolor: C.gridCol,
      tickfont: { color: C.txt2, size: 11 }, linecolor: C.border, showline: true,
    },
    yaxis: {
      gridcolor: C.gridCol, zerolinecolor: C.gridCol,
      tickfont: { color: C.txt2, size: 11 }, linecolor: C.border,
    },
    legend: { font: { color: C.txt2, size: 11 }, bgcolor: 'transparent', borderwidth: 0 },
    hoverlabel: {
      bgcolor: '#22222A', bordercolor: C.border,
      font: { family: "'Inter',sans-serif", color: C.txt, size: 12 },
    },
    modebar: { bgcolor: 'transparent', color: C.txt3, activecolor: C.acc1 },
    showlegend: true,
  }, overrides);
}

const CFG = {
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toImage'],
  displaylogo: false,
  responsive: true,
};

// ── Format helpers ────────────────────────────────────────────────────────────
function fmtCurrency(v) {
  if (!v || isNaN(v)) return '$0';
  if (v >= 1e9) return '$' + (v / 1e9).toFixed(2) + 'B';
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(1) + 'K';
  return '$' + v.toFixed(0);
}

function fmtNum(v) {
  if (!v || isNaN(v)) return '0';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + 'K';
  return Math.round(v).toLocaleString();
}

function fmtPct(v) { return (v || 0).toFixed(1) + '%'; }

// ── Data loading ──────────────────────────────────────────────────────────────
async function loadData() {
  try {
    const resp = await fetch('./data/dashboard_data.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    DATA = await resp.json();

    initFilters();
    updateTopbarKPIs();
    showTab('overview', true);
    document.getElementById('loading-overlay').style.display = 'none';
  } catch (err) {
    document.getElementById('loading-overlay').innerHTML =
      `<div style="color:#EF4444;font-size:14px;text-align:center;padding:24px">
        <div style="font-size:32px;margin-bottom:12px">⚠️</div>
        <div><b>Dashboard data not found.</b></div>
        <div style="margin-top:8px;color:#9A9AA5">Run: <code style="background:#1a1a1d;padding:2px 8px;border-radius:4px">python scripts/generate_dashboard_data.py</code></div>
        <div style="margin-top:6px;color:#5A5A65;font-size:12px">${err.message}</div>
      </div>`;
  }
}

// ── Filters ───────────────────────────────────────────────────────────────────
function initFilters() {
  const sel = document.getElementById('category-select');
  (DATA.filters.categories || []).forEach(cat => {
    const opt = document.createElement('option');
    opt.value = cat; opt.textContent = cat;
    sel.appendChild(opt);
  });
}

function setMonth(m) {
  currentMonth = m;
  document.querySelectorAll('[data-month]').forEach(b => b.classList.toggle('active', b.dataset.month === m));
  renderedTabs.clear();
  renderTab(currentTab);
  updateTopbarKPIs();
}

function setCategory(cat) {
  currentCat = cat;
  if (currentTab === 'products') renderBrands();
}

function resetFilters() {
  currentMonth = 'all';
  currentCat   = 'all';
  document.querySelectorAll('[data-month]').forEach(b => b.classList.toggle('active', b.dataset.month === 'all'));
  document.getElementById('category-select').value = 'all';
  renderedTabs.clear();
  renderTab(currentTab);
  updateTopbarKPIs();
}

// ── Tab management ────────────────────────────────────────────────────────────
function showTab(tabId, force = false) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
  document.getElementById(`tab-${tabId}`).classList.add('active');
  const navEl = document.querySelector(`[data-tab="${tabId}"]`);
  if (navEl) navEl.classList.add('active');
  currentTab = tabId;
  if (force || !renderedTabs.has(tabId)) renderTab(tabId);
}

function renderTab(tabId) {
  if (!DATA) return;
  renderedTabs.add(tabId);
  switch (tabId) {
    case 'overview':      renderOverview();      break;
    case 'churn':         renderChurn();         break;
    case 'products':      renderProducts();      break;
    case 'behavior':      renderBehavior();      break;
    case 'ml':            renderML();            break;
    case 'architecture':  renderArchitecture();  break;
  }
}

// ── Topbar KPIs ───────────────────────────────────────────────────────────────
function updateTopbarKPIs() {
  if (!DATA) return;
  const k = DATA.overview[currentMonth].kpis;
  document.getElementById('kpi-revenue').textContent   = fmtCurrency(k.total_revenue);
  document.getElementById('kpi-orders').textContent    = fmtNum(k.total_orders);
  document.getElementById('kpi-customers').textContent = fmtNum(k.total_customers);
  document.getElementById('kpi-churn').textContent     = fmtPct(DATA.churn.churn_rate);
}

// ════════════════════════════════════════════════════════════════════════════════
// OVERVIEW
// ════════════════════════════════════════════════════════════════════════════════
function renderOverview() {
  const d = DATA.overview[currentMonth];
  const k = d.kpis;

  // KPI cards
  document.getElementById('ov-revenue').textContent    = fmtCurrency(k.total_revenue);
  document.getElementById('ov-orders').textContent     = fmtNum(k.total_orders);
  document.getElementById('ov-customers').textContent  = fmtNum(k.total_customers);
  document.getElementById('ov-aov').textContent        = fmtCurrency(k.avg_order_value);
  document.getElementById('ov-conv').textContent       = fmtPct(k.conversion_rate) + ' conversion';

  // Daily Event Volume — multi-layer area
  const de = d.daily_events;
  Plotly.newPlot('chart-daily-events', [
    {
      x: de.dates, y: de.views, name: 'Views',
      type: 'scatter', mode: 'lines', fill: 'tozeroy',
      line: { color: C.blue, width: 1.5 },
      fillcolor: 'rgba(96,165,250,0.08)',
    },
    {
      x: de.dates, y: de.carts, name: 'Cart Adds',
      type: 'scatter', mode: 'lines', fill: 'tozeroy',
      line: { color: C.amber, width: 1.5 },
      fillcolor: 'rgba(245,158,11,0.10)',
    },
    {
      x: de.dates, y: de.purchases, name: 'Purchases',
      type: 'scatter', mode: 'lines', fill: 'tozeroy',
      line: { color: C.green, width: 2 },
      fillcolor: 'rgba(52,211,153,0.12)',
    },
  ], baseLayout({ margin: { t: 30, r: 20, b: 60, l: 70 } }), CFG);

  // Conversion Funnel
  const fn = d.funnel;
  Plotly.newPlot('chart-funnel', [{
    type: 'funnel',
    y: fn.stages,
    x: fn.values,
    textinfo: 'value+percent initial',
    textfont: { color: C.txt, size: 13, family: "'Inter',sans-serif" },
    marker: {
      color: [C.blue, C.amber, C.green],
      line: { width: 0 },
    },
    connector: { line: { color: C.border, width: 1 } },
  }], baseLayout({
    margin: { t: 20, r: 20, b: 20, l: 120 },
    xaxis: { visible: false },
    showlegend: false,
  }), CFG);

  // Event split donut
  const es = d.event_split;
  Plotly.newPlot('chart-event-split', [{
    type: 'pie',
    labels: es.labels,
    values: es.values,
    hole: 0.55,
    marker: { colors: [C.blue, C.amber, C.green], line: { color: C.card, width: 2 } },
    textfont: { color: C.txt, size: 11 },
    hovertemplate: '%{label}<br>%{value:,}<br>%{percent}<extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 20, l: 20 },
    showlegend: true,
    legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.1 },
    annotations: [{
      text: 'Events',
      x: 0.5, y: 0.5, showarrow: false,
      font: { color: C.txt2, size: 13, family: "'Inter',sans-serif" },
    }],
  }), CFG);

  // Top categories by revenue
  const cr = d.category_revenue;
  Plotly.newPlot('chart-cat-revenue', [{
    type: 'bar', orientation: 'h',
    y: cr.categories, x: cr.revenues,
    marker: {
      color: cr.revenues.map((_, i) =>
        `rgba(${hexToRgb(PALETTE[i % PALETTE.length])},${i === 0 ? 1 : 0.7})`
      ),
      line: { width: 0 },
    },
    text: cr.revenues.map(v => fmtCurrency(v)),
    textposition: 'outside',
    textfont: { color: C.txt2, size: 11 },
    hovertemplate: '%{y}<br><b>%{x:,.2f}</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 80, b: 40, l: 110 },
    yaxis: { autorange: 'reversed' },
    xaxis: { tickformat: '$.2s', title: { text: 'Revenue (USD)', font: { color: C.txt3 } } },
    showlegend: false,
  }), CFG);

  // Daily revenue trend
  const dr = d.daily_revenue;
  Plotly.newPlot('chart-daily-revenue', [{
    x: dr.dates, y: dr.revenues,
    type: 'scatter', mode: 'lines', fill: 'tozeroy',
    line: { color: C.green, width: 2, shape: 'spline' },
    fillcolor: 'rgba(52,211,153,0.10)',
    hovertemplate: '%{x}<br><b>%{y:$,.2f}</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 60, l: 80 },
    yaxis: { tickformat: '$,.0f' },
    showlegend: false,
  }), CFG);
}

// ════════════════════════════════════════════════════════════════════════════════
// CHURN INTELLIGENCE
// ════════════════════════════════════════════════════════════════════════════════
function renderChurn() {
  const ch = DATA.churn;

  document.getElementById('ch-rate').textContent     = fmtPct(ch.churn_rate);
  document.getElementById('ch-rate-def').textContent =
    `No purchase in the ${ch.horizon_days} days after ${ch.as_of}`;
  document.getElementById('ch-eligible').textContent = fmtNum(ch.eligible);
  document.getElementById('ch-eligible-def').textContent = `Bought before ${ch.as_of}`;
  document.getElementById('ch-retained').textContent = fmtNum(ch.retained);
  document.getElementById('ch-new').textContent      = fmtNum(ch.new_buyers);

  // Gauge
  Plotly.newPlot('chart-churn-gauge', [{
    type: 'indicator', mode: 'gauge+number',
    value: ch.churn_rate,
    number: { suffix: '%', font: { size: 36, color: C.red } },
    gauge: {
      axis: { range: [0, 100], tickcolor: C.txt3, tickfont: { color: C.txt3 } },
      bar: { color: C.red, thickness: 0.6 },
      bgcolor: C.card,
      bordercolor: C.border,
      steps: [
        { range: [0,  30], color: 'rgba(52,211,153,0.15)' },
        { range: [30, 70], color: 'rgba(245,158,11,0.15)' },
        { range: [70,100], color: 'rgba(239,68,68,0.15)'  },
      ],
      threshold: {
        line: { color: C.red, width: 3 },
        thickness: 0.85, value: ch.churn_rate,
      },
    },
    title: { text: 'Churn Rate', font: { color: C.txt2, size: 13 } },
  }], baseLayout({ margin: { t: 30, r: 40, b: 30, l: 40 }, showlegend: false }), CFG);

  // Cohort stacked bar
  const co = ch.cohort;
  Plotly.newPlot('chart-cohort', [
    {
      name: 'Retained', type: 'bar',
      x: co.categories, y: co.retained,
      marker: { color: C.green, line: { width: 0 } },
    },
    {
      name: 'Churned', type: 'bar',
      x: co.categories, y: co.churned,
      marker: { color: C.red, line: { width: 0 } },
    },
    {
      name: 'New Buyers', type: 'bar',
      x: co.categories, y: co.new,
      marker: { color: C.blue, line: { width: 0 } },
    },
  ], baseLayout({
    barmode: 'stack',
    margin: { t: 20, r: 20, b: 50, l: 70 },
    yaxis: { title: { text: 'Customers', font: { color: C.txt3 } } },
  }), CFG);

  // RFM Map scatter
  const rm = ch.rfm_map;
  const uniqueSegs = [...new Set(rm.segment)];
  const rfmTraces  = uniqueSegs.map(seg => {
    const idx = rm.segment.map((s, i) => s === seg ? i : -1).filter(i => i >= 0);
    return {
      name: seg, type: 'scatter', mode: 'markers',
      x:    idx.map(i => rm.recency[i]),
      y:    idx.map(i => rm.monetary[i]),
      marker: {
        color:   SEG_COLORS[seg] || C.acc1,
        size:    idx.map(i => Math.min(Math.max(rm.frequency[i] * 3 + 5, 5), 20)),
        opacity: 0.72,
        line:    { color: 'rgba(0,0,0,0.3)', width: 1 },
      },
      hovertemplate:
        `<b>${seg}</b><br>Recency: %{x}d<br>Monetary: $%{y:,.0f}<extra></extra>`,
    };
  });

  Plotly.newPlot('chart-rfm-map', rfmTraces, baseLayout({
    margin: { t: 20, r: 20, b: 60, l: 90 },
    xaxis: { title: { text: 'Recency (days since last purchase)', font: { color: C.txt3 } } },
    yaxis: { title: { text: 'Monetary Value (USD)', font: { color: C.txt3 } }, tickformat: '$,.0f' },
    hovermode: 'closest',
  }), CFG);

  // Customer segments bar
  const sg = ch.segments;
  Plotly.newPlot('chart-segments', [{
    type: 'bar', orientation: 'h',
    y: sg.names, x: sg.counts,
    marker: { color: sg.colors, line: { width: 0 } },
    text: sg.counts.map(v => fmtNum(v)),
    textposition: 'outside',
    textfont: { color: C.txt2, size: 11 },
    hovertemplate: '%{y}: <b>%{x:,}</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 80, b: 40, l: 140 },
    yaxis: { autorange: 'reversed' },
    showlegend: false,
  }), CFG);

  // 3D RFM Scatter
  const r3 = ch.rfm_3d;
  const seg3Unique = [...new Set(r3.segment)];
  const traces3d   = seg3Unique.map(seg => {
    const idx = r3.segment.map((s, i) => s === seg ? i : -1).filter(i => i >= 0);
    return {
      name: seg, type: 'scatter3d', mode: 'markers',
      x: idx.map(i => r3.r[i]),
      y: idx.map(i => r3.f[i]),
      z: idx.map(i => r3.m[i]),
      marker: {
        color:   SEG_COLORS[seg] || C.acc1,
        size:    5, opacity: 0.75,
        line:    { color: 'rgba(0,0,0,0.3)', width: 0.5 },
      },
      hovertemplate:
        `<b>${seg}</b><br>Recency: %{x}d<br>Frequency: %{y}<br>Monetary: $%{z:,.0f}<extra></extra>`,
    };
  });

  Plotly.newPlot('chart-rfm-3d', traces3d, {
    paper_bgcolor: C.card,
    scene: {
      bgcolor: C.card,
      xaxis: { title: 'Recency (days)', color: C.txt2, gridcolor: C.gridCol, zerolinecolor: C.gridCol },
      yaxis: { title: 'Frequency',      color: C.txt2, gridcolor: C.gridCol, zerolinecolor: C.gridCol },
      zaxis: { title: 'Monetary ($)',   color: C.txt2, gridcolor: C.gridCol, zerolinecolor: C.gridCol },
      camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } },
    },
    font:   { family: "'Inter',sans-serif", color: C.txt },
    margin: { t: 30, r: 0, b: 0, l: 0 },
    legend: { font: { color: C.txt2, size: 10 }, bgcolor: 'transparent' },
    paper_bgcolor: C.card,
    modebar: { bgcolor: 'transparent', color: C.txt3, activecolor: C.acc1 },
  }, CFG);
}

// ════════════════════════════════════════════════════════════════════════════════
// PRODUCTS & CATEGORIES
// ════════════════════════════════════════════════════════════════════════════════
function renderProducts() {
  const pr = DATA.products[currentMonth];

  // Treemap
  Plotly.newPlot('chart-treemap', [{
    type: 'treemap',
    labels:  pr.treemap.labels,
    values:  pr.treemap.values,
    parents: pr.treemap.labels.map(() => ''),
    textfont: { color: C.txt, size: 12 },
    hovertemplate: '%{label}<br><b>%{value:$,.2f}</b><extra></extra>',
    marker: { colorscale: [[0,'#1a1a35'],[1,C.acc1]], line: { width: 1, color: C.card } },
  }], baseLayout({ margin: { t: 20, r: 10, b: 10, l: 10 }, showlegend: false }), CFG);

  renderBrands();

  // Category funnel (grouped bar)
  const cf = pr.category_funnel;
  Plotly.newPlot('chart-cat-funnel', [
    { name: 'Views',     type: 'bar', x: cf.categories, y: cf.views,     marker: { color: C.blue,  line:{width:0} } },
    { name: 'Cart Adds', type: 'bar', x: cf.categories, y: cf.carts,     marker: { color: C.amber, line:{width:0} } },
    { name: 'Purchases', type: 'bar', x: cf.categories, y: cf.purchases,  marker: { color: C.green, line:{width:0} } },
  ], baseLayout({
    barmode: 'group',
    margin: { t: 20, r: 20, b: 80, l: 70 },
    xaxis: { tickangle: -30 },
    yaxis: { title: { text: 'Events', font: { color: C.txt3 } } },
  }), CFG);

  // Avg price
  const ap = pr.avg_price;
  Plotly.newPlot('chart-avg-price', [{
    type: 'bar',
    x: ap.categories, y: ap.prices,
    marker: { color: PALETTE, line: { width: 0 } },
    text: ap.prices.map(v => '$' + v.toFixed(0)),
    textposition: 'outside', textfont: { color: C.txt2, size: 11 },
    hovertemplate: '%{x}<br><b>$%{y:.2f}</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 80, l: 80 },
    xaxis: { tickangle: -30 },
    yaxis: { title: { text: 'Avg Price (USD)', font: { color: C.txt3 } }, tickformat: '$,.0f' },
    showlegend: false,
  }), CFG);

  // Conversion rate (semantic colours)
  const cv = pr.conversion_rate;
  const convColors = cv.rates.map(r => r > 2 ? C.green : r > 1 ? C.amber : C.red);
  Plotly.newPlot('chart-conv-rate', [{
    type: 'bar', orientation: 'h',
    y: cv.categories, x: cv.rates,
    marker: { color: convColors, line: { width: 0 } },
    text: cv.rates.map(v => v.toFixed(2) + '%'),
    textposition: 'outside', textfont: { color: C.txt2, size: 11 },
    hovertemplate: '%{y}<br><b>%{x:.2f}%</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 80, b: 40, l: 110 },
    yaxis: { autorange: 'reversed' },
    xaxis: { title: { text: 'Conversion Rate (%)', font: { color: C.txt3 } } },
    showlegend: false,
  }), CFG);

  // Top 20 products table
  const tbody = document.getElementById('products-tbody');
  tbody.innerHTML = pr.top_products.map((p, i) =>
    `<tr>
      <td class="num">${i + 1}</td>
      <td class="num">${p.product_id}</td>
      <td class="cat">${p.category}</td>
      <td>${p.brand}</td>
      <td class="num">${fmtCurrency(p.revenue)}</td>
      <td class="num">${fmtNum(p.orders)}</td>
      <td class="num">${fmtCurrency(p.avg_price)}</td>
    </tr>`
  ).join('');
}

function renderBrands() {
  if (!DATA) return;
  const pr  = DATA.products[currentMonth];
  const brd = pr.brands;
  let brands, revenues;
  if (currentCat !== 'all' && brd.by_category[currentCat]) {
    brands   = brd.by_category[currentCat].brands;
    revenues = brd.by_category[currentCat].revenues;
  } else {
    brands   = brd.all_brands;
    revenues = brd.all_revenues;
  }

  Plotly.newPlot('chart-brands', [{
    type: 'bar', orientation: 'h',
    y: brands, x: revenues,
    marker: {
      color: revenues.map((_, i) =>
        `rgba(${hexToRgb(C.acc1)},${Math.max(0.4, 1 - i * 0.04)})`
      ),
      line: { width: 0 },
    },
    text: revenues.map(v => fmtCurrency(v)),
    textposition: 'outside', textfont: { color: C.txt2, size: 11 },
    hovertemplate: '%{y}<br><b>%{x:$,.2f}</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 80, b: 40, l: 100 },
    yaxis: { autorange: 'reversed' },
    xaxis: { tickformat: '$.2s', title: { text: 'Revenue (USD)', font: { color: C.txt3 } } },
    showlegend: false,
  }), CFG);
}

// ════════════════════════════════════════════════════════════════════════════════
// USER BEHAVIOR
// ════════════════════════════════════════════════════════════════════════════════
function renderBehavior() {
  const bh = DATA.behavior[currentMonth];

  // Heatmap
  const hm = bh.heatmap;
  Plotly.newPlot('chart-heatmap', [{
    type: 'heatmap',
    x: hm.hours, y: hm.weekdays, z: hm.z,
    colorscale: [[0,'#0D0D0F'],[0.3,'#2D1B6E'],[0.6,C.acc1],[1,C.acc2]],
    hovertemplate: '%{y} at %{x}:00 → <b>%{z}</b> purchases<extra></extra>',
    showscale: true,
    colorbar: { tickfont: { color: C.txt2, size: 10 }, bgcolor: C.card },
  }], baseLayout({
    margin: { t: 20, r: 80, b: 50, l: 90 },
    xaxis: { title: { text: 'Hour of Day', font: { color: C.txt3 } }, dtick: 2 },
    yaxis: { title: '' },
    showlegend: false,
  }), CFG);

  // Purchases by hour (area curve)
  const hr = bh.hourly;
  Plotly.newPlot('chart-hourly', [{
    x: hr.hours, y: hr.purchases,
    type: 'scatter', mode: 'lines+markers', fill: 'tozeroy',
    line:      { color: C.acc1, width: 2, shape: 'spline' },
    fillcolor: 'rgba(123,92,250,0.10)',
    marker:    { color: C.acc2, size: 5 },
    hovertemplate: 'Hour %{x}:00 → <b>%{y:,}</b> purchases<extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 50, l: 70 },
    xaxis: { title: { text: 'Hour of Day', font: { color: C.txt3 } }, dtick: 4 },
    yaxis: { title: { text: 'Purchases', font: { color: C.txt3 } } },
    showlegend: false,
  }), CFG);

  // Weekday bar
  const wd = bh.weekday;
  Plotly.newPlot('chart-weekday', [{
    type: 'bar',
    x: wd.days, y: wd.purchases,
    marker: { color: wd.purchases.map((_, i) => PALETTE[i % PALETTE.length]), line: { width: 0 } },
    hovertemplate: '%{x}: <b>%{y:,}</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 60, l: 70 },
    xaxis: { tickangle: -20 },
    yaxis: { title: { text: 'Purchases', font: { color: C.txt3 } } },
    showlegend: false,
  }), CFG);

  // Session depth
  const sd = bh.session_depth;
  Plotly.newPlot('chart-session-depth', [{
    type: 'bar',
    x: sd.depth, y: sd.count,
    marker: {
      color: sd.depth.map(d => d >= 15 ? C.green : d >= 5 ? C.acc1 : C.txt3),
      line: { width: 0 },
    },
    hovertemplate: '%{x} events/session → <b>%{y:,}</b> sessions<extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 50, l: 70 },
    xaxis: { title: { text: 'Events per Session', font: { color: C.txt3 } } },
    yaxis: { title: { text: 'Sessions', font: { color: C.txt3 } } },
    showlegend: false,
  }), CFG);

  // Cumulative revenue
  const cm = bh.cumulative_revenue;
  Plotly.newPlot('chart-cumulative', [{
    x: cm.hours, y: cm.cumulative_revenue,
    type: 'scatter', mode: 'lines', fill: 'tozeroy',
    line:      { color: C.blue, width: 2, shape: 'spline' },
    fillcolor: 'rgba(96,165,250,0.10)',
    hovertemplate: 'Hour %{x}:00 → <b>$%{y:,.0f}</b> cumulative<extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 50, l: 80 },
    xaxis: { title: { text: 'Hour of Day', font: { color: C.txt3 } } },
    yaxis: { title: { text: 'Cumulative Revenue', font: { color: C.txt3 } }, tickformat: '$,.0f' },
    showlegend: false,
  }), CFG);

  // Views/Purchases ratio
  const vp = bh.views_purchases_ratio;
  Plotly.newPlot('chart-ratio', [{
    x: vp.dates, y: vp.ratios,
    type: 'scatter', mode: 'lines+markers',
    line:   { color: C.orange, width: 2, shape: 'spline' },
    marker: { color: C.orange, size: 5 },
    fill:      'tozeroy', fillcolor: 'rgba(249,115,22,0.08)',
    hovertemplate: '%{x}: <b>%{y:.3f}%</b> conversion ratio<extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 60, l: 80 },
    yaxis: { title: { text: 'Purchases / Views (%)', font: { color: C.txt3 } } },
    showlegend: false,
  }), CFG);
}

// ════════════════════════════════════════════════════════════════════════════════
// ML INTELLIGENCE
// ════════════════════════════════════════════════════════════════════════════════
function renderML() {
  const ml = DATA.ml;

  // AUC KPI cards: ROC-AUC as the headline, PR-AUC against the churn base rate
  const base = (ml.split.churn_rate_test * 100).toFixed(1);
  [
    ['rec', 'Recency baseline'],
    ['lr',  'Logistic Regression'],
    ['rf',  'Random Forest'],
    ['gb',  'Gradient Boosting'],
  ].forEach(([key, name]) => {
    document.getElementById(`ml-${key}-auc`).textContent = ml.model_auc[name].toFixed(4);
    document.getElementById(`ml-${key}-pr`).textContent  =
      `PR-AUC ${ml.model_pr_auc[name].toFixed(4)} (base rate ${base}%)`;
  });

  // ROC Curves
  const roc = ml.roc_curves;
  const rocColors = {
    'Gradient Boosting': C.green, 'Random Forest': C.blue,
    'Logistic Regression': C.acc2, 'Recency baseline': C.amber,
  };
  const rocTraces = Object.keys(roc).map(name => ({
    x: roc[name].fpr, y: roc[name].tpr,
    type: 'scatter', mode: 'lines', name: `${name} (AUC=${roc[name].auc.toFixed(3)})`,
    line: { color: rocColors[name] || C.acc1, width: 2 },
  }));
  rocTraces.push({
    x: [0,1], y: [0,1], name: 'Random (AUC=0.5)',
    type: 'scatter', mode: 'lines',
    line: { color: C.txt3, width: 1, dash: 'dash' },
  });

  Plotly.newPlot('chart-roc', rocTraces, baseLayout({
    margin: { t: 20, r: 20, b: 60, l: 70 },
    xaxis: { title: { text: 'False Positive Rate', font: { color: C.txt3 } }, range: [0,1] },
    yaxis: { title: { text: 'True Positive Rate',  font: { color: C.txt3 } }, range: [0,1] },
    shapes: [{
      type: 'rect', xref: 'paper', yref: 'paper',
      x0:0, y0:0, x1:1, y1:1,
      line: { color: C.border, width: 1 },
    }],
  }), CFG);

  // Feature importance
  const fi = ml.feature_importance;
  const fiColors = fi.importances.map(v => {
    const max = Math.max(...fi.importances);
    const ratio = v / max;
    return `rgba(${hexToRgb(C.acc1)},${0.4 + ratio * 0.6})`;
  });
  Plotly.newPlot('chart-feature-imp', [{
    type: 'bar', orientation: 'h',
    y: fi.features, x: fi.importances,
    marker: { color: fiColors, line: { width: 0 } },
    text: fi.importances.map(v => (v * 100).toFixed(1) + '%'),
    textposition: 'outside', textfont: { color: C.txt2, size: 11 },
    hovertemplate: '%{y}: <b>%{x:.4f}</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 80, b: 40, l: 160 },
    yaxis: { autorange: 'reversed' },
    showlegend: false,
  }), CFG);

  // Churn score distribution
  const cs = ml.churn_scores;
  const scoreColors = cs.bins.map(b => b < 0.35 ? C.green : b < 0.6 ? C.amber : C.red);
  Plotly.newPlot('chart-churn-scores', [{
    type: 'bar',
    x: cs.bins, y: cs.counts, width: 0.045,
    marker: { color: scoreColors, line: { width: 0 } },
    hovertemplate: 'Score ~%{x:.2f}: <b>%{y:,}</b> customers<extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 20, b: 60, l: 70 },
    xaxis: { title: { text: 'Churn Probability Score', font: { color: C.txt3 } } },
    yaxis: { title: { text: 'Customers', font: { color: C.txt3 } } },
    showlegend: false,
    shapes: [
      { type:'line', x0:0.35, y0:0, x1:0.35, y1:1, yref:'paper', line:{color:C.amber, width:1.5, dash:'dot'} },
      { type:'line', x0:0.6,  y0:0, x1:0.6,  y1:1, yref:'paper', line:{color:C.red,   width:1.5, dash:'dot'} },
    ],
    annotations: [
      { x:0.18, y:1, yref:'paper', text:'Low Risk', showarrow:false, font:{color:C.green, size:11} },
      { x:0.48, y:1, yref:'paper', text:'Medium',   showarrow:false, font:{color:C.amber, size:11} },
      { x:0.80, y:1, yref:'paper', text:'High Risk', showarrow:false, font:{color:C.red,  size:11} },
    ],
  }), CFG);

  // Category affinity matrix
  const am = ml.affinity_matrix;
  Plotly.newPlot('chart-affinity', [{
    type: 'heatmap',
    x: am.categories, y: am.categories, z: am.z,
    colorscale: [[0,'#0D0D0F'],[0.3,'#1a1060'],[0.7,C.acc1],[1,C.acc2]],
    text: am.z.map(row => row.map(v => v.toFixed(0) + '%')),
    texttemplate: '%{text}',
    textfont: { color: C.txt, size: 10 },
    hovertemplate: '%{y} → %{x}<br><b>%{z:.1f}%</b> affinity<extra></extra>',
    colorbar: { tickfont: { color: C.txt2, size: 10 } },
  }], baseLayout({
    margin: { t: 20, r: 80, b: 80, l: 100 },
    xaxis: { tickangle: -30 },
    showlegend: false,
  }), CFG);

  // Upsell scores
  const us = ml.upsell;
  Plotly.newPlot('chart-upsell', [{
    type: 'bar', orientation: 'h',
    y: us.categories, x: us.scores,
    marker: {
      color: us.scores.map((_, i) => `rgba(${hexToRgb(C.orange)},${1 - i * 0.07})`),
      line: { width: 0 },
    },
    text: us.scores.map(v => fmtNum(v)),
    textposition: 'outside', textfont: { color: C.txt2, size: 11 },
    hovertemplate: '%{y}<br><b>Opportunity Score: %{x:,}</b><extra></extra>',
  }], baseLayout({
    margin: { t: 20, r: 80, b: 40, l: 110 },
    yaxis: { autorange: 'reversed' },
    xaxis: { title: { text: 'Upsell Opportunity Score', font: { color: C.txt3 } } },
    showlegend: false,
  }), CFG);

  // Segment action table
  const riskBadge = { 'Low':'badge-green','Medium':'badge-amber','High':'badge-red','Very High':'badge-red' };
  const upsellBadge = { 'High':'badge-green','Medium':'badge-blue','Low':'badge-gray','Very Low':'badge-gray' };
  const tbody = document.getElementById('segment-tbody');
  tbody.innerHTML = (ml.segment_scores || []).map(s =>
    `<tr>
      <td style="font-weight:600;color:${SEG_COLORS[s.segment]||'#fff'}">${s.segment}</td>
      <td class="num">${fmtNum(s.count)}</td>
      <td><span class="badge ${riskBadge[s.churn_risk]||'badge-gray'}">${s.churn_risk}</span></td>
      <td><span class="badge ${upsellBadge[s.upsell_priority]||'badge-gray'}">${s.upsell_priority}</span></td>
    </tr>`
  ).join('');
}

// ════════════════════════════════════════════════════════════════════════════════
// ARCHITECTURE (static — just populate AUC reference value)
// ════════════════════════════════════════════════════════════════════════════════
function renderArchitecture() {
  if (!DATA) return;
  const auc = DATA.ml.gb_auc;
  const el  = document.getElementById('arch-gb-auc');
  if (el) el.textContent = auc.toFixed(4) + ' ROC-AUC';
}

// ── Utility ───────────────────────────────────────────────────────────────────
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3), 16);
  const g = parseInt(hex.slice(3,5), 16);
  const b = parseInt(hex.slice(5,7), 16);
  return `${r},${g},${b}`;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadData);
