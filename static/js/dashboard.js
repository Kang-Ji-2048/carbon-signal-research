/**
 * Climate Finance Dashboard -- Client-side rendering
 *
 * Loads detail.json + meta.json, then recreates the full interactive
 * dashboard using Plotly.js.  Mirrors the Python Dash callbacks in
 * dashboard/app.py but runs entirely in the browser.
 *
 * JSON record keys (short form for smaller payload):
 *   y = year, r = region, c = country, s = sector,
 *   i = instrument_type, a = amount_usd_mn, src = source
 */

/* ================================================================
   Global state
   ================================================================ */
let DATA = [];
let META = {};
let FILTERS = {
  yearMin: 2013,
  yearMax: 2025,
  regions: [],
  sectors: [],
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

/* ================================================================
   Bootstrap
   ================================================================ */
document.addEventListener("DOMContentLoaded", async () => {
  try {
    const [detailResp, metaResp] = await Promise.all([
      fetch("data/detail.json"),
      fetch("data/meta.json"),
    ]);
    DATA = await detailResp.json();
    META = await metaResp.json();
  } catch (err) {
    document.getElementById("loading-overlay").innerHTML =
      `<p class="text-danger">Failed to load data: ${err.message}</p>`;
    return;
  }

  initFilters();
  renderAll();

  document.getElementById("loading-overlay").classList.add("hidden");
  setTimeout(() => {
    document.getElementById("loading-overlay").style.display = "none";
  }, 400);
});

/* ================================================================
   Filters
   ================================================================ */
function initFilters() {
  const years = META.years;
  const yearMin = document.getElementById("year-min");
  const yearMax = document.getElementById("year-max");

  yearMin.min = years[0];
  yearMin.max = years[years.length - 1];
  yearMin.value = years[0];
  yearMax.min = years[0];
  yearMax.max = years[years.length - 1];
  yearMax.value = years[years.length - 1];

  FILTERS.yearMin = years[0];
  FILTERS.yearMax = years[years.length - 1];

  document.getElementById("year-min-label").textContent = years[0];
  document.getElementById("year-max-label").textContent = years[years.length - 1];

  yearMin.addEventListener("input", () => {
    let v = parseInt(yearMin.value);
    if (v > parseInt(yearMax.value)) v = parseInt(yearMax.value);
    yearMin.value = v;
    FILTERS.yearMin = v;
    document.getElementById("year-min-label").textContent = v;
    renderAll();
  });
  yearMax.addEventListener("input", () => {
    let v = parseInt(yearMax.value);
    if (v < parseInt(yearMin.value)) v = parseInt(yearMin.value);
    yearMax.value = v;
    FILTERS.yearMax = v;
    document.getElementById("year-max-label").textContent = v;
    renderAll();
  });

  buildFilterPills("region-filter", META.regions, (selected) => {
    FILTERS.regions = selected;
    renderAll();
  });

  buildFilterPills("sector-filter", META.sectors, (selected) => {
    FILTERS.sectors = selected;
    renderAll();
  });
}

function buildFilterPills(containerId, items, onChange) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  items.forEach((item) => {
    const pill = document.createElement("label");
    pill.className = "filter-pill";
    pill.innerHTML = `<input type="checkbox" value="${item}"> ${item}`;
    pill.addEventListener("click", () => {
      const cb = pill.querySelector("input");
      setTimeout(() => {
        pill.classList.toggle("active", cb.checked);
        const selected = Array.from(
          container.querySelectorAll("input:checked")
        ).map((el) => el.value);
        onChange(selected);
      }, 0);
    });
    container.appendChild(pill);
  });
}

/* ================================================================
   Data helpers
   ================================================================ */
function getFilteredData() {
  return DATA.filter((d) => {
    if (d.y < FILTERS.yearMin || d.y > FILTERS.yearMax) return false;
    if (FILTERS.regions.length > 0 && !FILTERS.regions.includes(d.r))
      return false;
    if (FILTERS.sectors.length > 0 && !FILTERS.sectors.includes(d.s))
      return false;
    return true;
  });
}

function groupBy(data, keyFn) {
  const map = new Map();
  for (const d of data) {
    const key = keyFn(d);
    if (!map.has(key)) map.set(key, { total: 0, count: 0 });
    const g = map.get(key);
    g.total += d.a;
    g.count += 1;
  }
  return map;
}

function groupBy2(data, key1Fn, key2Fn) {
  const map = new Map();
  for (const d of data) {
    const k1 = key1Fn(d);
    const k2 = key2Fn(d);
    if (!map.has(k1)) map.set(k1, new Map());
    const inner = map.get(k1);
    inner.set(k2, (inner.get(k2) || 0) + d.a);
  }
  return map;
}

function fmtNum(n) {
  return n.toLocaleString("en-US", { maximumFractionDigits: 1 });
}

/* ================================================================
   Render all
   ================================================================ */
function renderAll() {
  const df = getFilteredData();
  renderKPIs(df);
  renderTrendChart(df);
  renderSectorPie(df);
  renderRegionBar(df);
  renderInstrumentTreemap(df);
  renderHeatmap(df);
  renderSourceChart(df);
}

/* ================================================================
   KPI Cards
   ================================================================ */
function renderKPIs(df) {
  const total = df.reduce((s, d) => s + d.a, 0);
  const countries = new Set(df.map((d) => d.c)).size;
  const flows = df.length;
  const avg = flows > 0 ? total / flows : 0;

  document.getElementById("kpi-total").textContent = `$${fmtNum(total / 1000)}B`;
  document.getElementById("kpi-countries").textContent = countries;
  document.getElementById("kpi-flows").textContent = fmtNum(flows);
  document.getElementById("kpi-avg").textContent = `$${fmtNum(avg)}M`;
}

/* ================================================================
   1. Trend area chart (year x sector)
   ================================================================ */
function renderTrendChart(df) {
  const grouped = groupBy2(df, (d) => d.s, (d) => d.y);
  const years = META.years.filter(
    (y) => y >= FILTERS.yearMin && y <= FILTERS.yearMax
  );
  const traces = [];

  for (const [sector, yearMap] of grouped) {
    traces.push({
      x: years,
      y: years.map((y) => +(yearMap.get(y) || 0).toFixed(1)),
      name: sector,
      type: "scatter",
      mode: "lines",
      fill: "tonexty",
      stackgroup: "one",
      line: { color: META.sectorColors[sector] || "#95a5a6" },
    });
  }

  const layout = {
    title: "Climate Finance Trends by Sector",
    xaxis: { title: "Year" },
    yaxis: { title: "USD Millions" },
    legend: { orientation: "h", y: -0.25 },
    margin: { t: 40, b: 80, l: 60, r: 20 },
  };

  Plotly.react("trend-chart", traces, layout, PLOTLY_CONFIG);
}

/* ================================================================
   2. Sector donut chart
   ================================================================ */
function renderSectorPie(df) {
  const grouped = groupBy(df, (d) => d.s);
  const sectors = Array.from(grouped.keys());
  const values = sectors.map((s) => +grouped.get(s).total.toFixed(1));
  const colors = sectors.map((s) => META.sectorColors[s] || "#95a5a6");

  const trace = {
    labels: sectors,
    values: values,
    type: "pie",
    hole: 0.4,
    marker: { colors: colors },
    textinfo: "percent+label",
    textposition: "inside",
  };

  const layout = {
    title: "Sector Distribution",
    margin: { t: 40, b: 40, l: 20, r: 20 },
    showlegend: false,
  };

  Plotly.react("sector-pie", [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   3. Region grouped bar chart
   ================================================================ */
function renderRegionBar(df) {
  const grouped = groupBy2(df, (d) => d.r, (d) => d.y);
  const years = META.years.filter(
    (y) => y >= FILTERS.yearMin && y <= FILTERS.yearMax
  );
  const traces = [];

  for (const [region, yearMap] of grouped) {
    traces.push({
      x: years,
      y: years.map((y) => +(yearMap.get(y) || 0).toFixed(1)),
      name: region,
      type: "bar",
      marker: { color: META.regionColors[region] || "#95a5a6" },
    });
  }

  const layout = {
    title: "Investment by Region Over Time",
    barmode: "group",
    xaxis: { title: "Year" },
    yaxis: { title: "USD Millions" },
    legend: { orientation: "h", y: -0.3 },
    margin: { t: 40, b: 80, l: 60, r: 20 },
  };

  Plotly.react("region-bar", traces, layout, PLOTLY_CONFIG);
}

/* ================================================================
   4. Instrument treemap
   ================================================================ */
function renderInstrumentTreemap(df) {
  const grouped = groupBy2(df, (d) => d.i, (d) => d.s);

  const labels = ["All"];
  const parents = [""];
  const values = [0];

  for (const [instrument, sectorMap] of grouped) {
    labels.push(instrument);
    parents.push("All");
    values.push(0);

    for (const [sector, total] of sectorMap) {
      labels.push(`${sector} (${instrument})`);
      parents.push(instrument);
      values.push(+total.toFixed(1));
    }
  }

  const trace = {
    type: "treemap",
    labels: labels,
    parents: parents,
    values: values,
    textinfo: "label+value",
    branchvalues: "total",
  };

  const layout = {
    title: "Finance by Instrument Type & Sector",
    margin: { t: 40, b: 20, l: 10, r: 10 },
  };

  Plotly.react("instrument-chart", [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   5. Heatmap (region x sector)
   ================================================================ */
function renderHeatmap(df) {
  const grouped = groupBy2(df, (d) => d.r, (d) => d.s);
  const regions = META.regions.filter(
    (r) => FILTERS.regions.length === 0 || FILTERS.regions.includes(r)
  );
  const sectors = META.sectors;

  const z = regions.map((r) => {
    const sectorMap = grouped.get(r) || new Map();
    return sectors.map((s) => +(sectorMap.get(s) || 0).toFixed(1));
  });

  const trace = {
    x: sectors,
    y: regions,
    z: z,
    type: "heatmap",
    colorscale: "YlGn",
    colorbar: { title: "USD Millions" },
  };

  const layout = {
    title: "Investment Heatmap: Region x Sector",
    xaxis: { title: "Sector", tickangle: -30 },
    yaxis: { title: "Region" },
    margin: { t: 40, b: 100, l: 160, r: 20 },
  };

  Plotly.react("heatmap", [trace], layout, PLOTLY_CONFIG);
}

/* ================================================================
   6. Source comparison line chart
   ================================================================ */
function renderSourceChart(df) {
  const grouped = groupBy2(df, (d) => d.src, (d) => d.y);
  const years = META.years.filter(
    (y) => y >= FILTERS.yearMin && y <= FILTERS.yearMax
  );
  const traces = [];

  const sourceColors = {
    OECD: "#2c3e50",
    IRENA: "#e67e22",
    "World Bank": "#3498db",
  };

  for (const [source, yearMap] of grouped) {
    traces.push({
      x: years,
      y: years.map((y) => +(yearMap.get(y) || 0).toFixed(1)),
      name: source,
      type: "scatter",
      mode: "lines+markers",
      line: { color: sourceColors[source] || "#95a5a6" },
      marker: { size: 6 },
    });
  }

  const layout = {
    title: "Data Source Comparison",
    xaxis: { title: "Year" },
    yaxis: { title: "USD Millions" },
    margin: { t: 40, b: 40, l: 60, r: 20 },
  };

  Plotly.react("source-chart", traces, layout, PLOTLY_CONFIG);
}
