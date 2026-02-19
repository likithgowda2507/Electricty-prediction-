const MN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const C = {
    indigo: '#6366f1', indigoL: 'rgba(99,102,241,0.15)',
    emerald: '#10b981', emeraldL: 'rgba(16,185,129,0.15)',
    amber: '#f59e0b', amberL: 'rgba(245,158,11,0.15)',
    rose: '#f43f5e', roseL: 'rgba(244,63,94,0.15)',
    sky: '#0ea5e9', violet: '#8b5cf6', teal: '#14b8a6',
    slate: 'rgba(148,163,184,0.4)', fuchsia: '#d946ef'
};
const charts = {};
let historyData = null;

const dOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { usePointStyle: true, font: { size: 11, family: 'Inter' }, padding: 14 } } },
    scales: { y: { grid: { color: '#f1f5f9' }, ticks: { font: { size: 11 } } }, x: { grid: { display: false }, ticks: { font: { size: 11, weight: 500 } } } }
};

function dc(k) { if (charts[k]) { charts[k].destroy(); charts[k] = null; } }
function grad(ctx, c1, c2) { const g = ctx.createLinearGradient(0, 0, 0, 350); g.addColorStop(0, c1); g.addColorStop(1, c2); return g; }

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('current-date').textContent = new Date().toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    fetchHistory();
});

// ---- NAV ----
let forecastLoaded = false;
let solarLoaded = false;
function showSection(name) {
    document.querySelectorAll('[id^="section-"]').forEach(el => el.classList.add('hidden'));
    document.getElementById('section-' + name).classList.remove('hidden');
    document.querySelectorAll('.nav-link').forEach(el => { el.classList.remove('active'); el.classList.add('text-slate-400'); });
    event.currentTarget.classList.add('active');
    event.currentTarget.classList.remove('text-slate-400');
    if (name === 'analysis' && historyData) loadBuildingData();
    if (name === 'trends') loadTrendsSection();
    if (name === 'prediction' && !forecastLoaded) { forecastLoaded = true; runYearlyForecast(); }
    if (name === 'solar' && !solarLoaded) { solarLoaded = true; runSolarAnalysis(); }
}

// ===== OVERVIEW =====
async function fetchHistory() {
    try {
        const res = await fetch('/api/history');
        historyData = await res.json();
        const d = historyData;
        document.getElementById('kpi-consumption').innerText = `${(d.kpi.total_consumption / 1e3).toFixed(1)} MWh`;
        document.getElementById('kpi-pf').innerText = d.kpi.avg_pf.toFixed(2);
        document.getElementById('kpi-demand').innerText = `${d.kpi.max_demand.toFixed(0)} kW`;
        document.getElementById('kpi-interruption').innerText = `${(d.kpi.interruption / 60).toFixed(0)} Hrs`;
        renderConsumptionChart(d);
        renderBillChart(d);
        renderBuildingPie(d);
        renderYearlyTrend(d);
    } catch (err) { console.error("History error:", err); }
}

function renderConsumptionChart(d) {
    const ctx = document.getElementById('consumptionChart').getContext('2d'); dc('consumption');
    charts.consumption = new Chart(ctx, { type: 'line', data: { labels: d.months.map(m => MN[m - 1]), datasets: [{ label: 'Energy (kWh)', data: d.consumption, borderColor: C.indigo, backgroundColor: grad(ctx, C.indigoL, 'rgba(99,102,241,0)'), borderWidth: 3, tension: 0.4, fill: true, pointRadius: 5, pointBackgroundColor: '#fff', pointBorderColor: C.indigo, pointBorderWidth: 2 }] }, options: { ...dOpts, plugins: { legend: { display: false } } } });
}
function renderBillChart(d) {
    const ctx = document.getElementById('billChart').getContext('2d'); dc('bill');
    charts.bill = new Chart(ctx, { type: 'bar', data: { labels: d.months.map(m => MN[m - 1]), datasets: [{ label: 'Avg Bill (₹)', data: d.monthly_bill_avg, backgroundColor: C.emerald, borderRadius: 8, barPercentage: 0.6 }] }, options: { ...dOpts, plugins: { legend: { display: false } } } });
}
function renderBuildingPie(d) {
    const ctx = document.getElementById('buildingPieChart').getContext('2d'); dc('buildingPie');
    const labels = d.buildings.map(b => b.Building_ID), data = d.buildings.map(b => b.Energy_Consumption_kWh);
    charts.buildingPie = new Chart(ctx, { type: 'doughnut', data: { labels, datasets: [{ data, backgroundColor: [C.indigo, C.emerald, C.amber], borderWidth: 0, hoverOffset: 10 }] }, options: { responsive: true, maintainAspectRatio: false, cutout: '72%', plugins: { legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true, font: { size: 12 } } } } } });
}
function renderYearlyTrend(d) {
    const ctx = document.getElementById('yearlyTrendChart').getContext('2d'); dc('yearlyTrend');
    charts.yearlyTrend = new Chart(ctx, { type: 'bar', data: { labels: d.yearly.years.map(y => String(Math.round(y))), datasets: [{ label: 'Energy (kWh)', data: d.yearly.consumption, backgroundColor: C.indigo, borderRadius: 6, yAxisID: 'y' }, { label: 'Bill (₹)', data: d.yearly.bill, type: 'line', borderColor: C.emerald, borderWidth: 3, tension: 0.4, pointRadius: 5, pointBackgroundColor: '#fff', pointBorderColor: C.emerald, pointBorderWidth: 2, yAxisID: 'y1', fill: false }] }, options: { ...dOpts, scales: { y: { grid: { color: '#f1f5f9' }, position: 'left' }, y1: { grid: { display: false }, position: 'right' }, x: { grid: { display: false } } } } });
}

// ===== BUILDING ANALYSIS =====
function loadBuildingData() {
    if (!historyData) return;
    const bid = document.getElementById('building-select').value;
    const bm = historyData.building_monthly[bid];
    if (!bm) return;
    const labels = bm.months.map(m => MN[m - 1]);

    // Consumption
    const c1 = document.getElementById('buildingConsumptionChart').getContext('2d'); dc('bCons');
    charts.bCons = new Chart(c1, { type: 'bar', data: { labels, datasets: [{ label: `${bid} Energy (kWh)`, data: bm.consumption, backgroundColor: C.indigo, borderRadius: 8 }] }, options: { ...dOpts, plugins: { legend: { display: false } } } });

    // Bill
    const c2 = document.getElementById('buildingBillChart').getContext('2d'); dc('bBill');
    charts.bBill = new Chart(c2, { type: 'bar', data: { labels, datasets: [{ label: `${bid} Bill (₹)`, data: bm.bill, backgroundColor: C.emerald, borderRadius: 8 }] }, options: { ...dOpts, plugins: { legend: { display: false } } } });

    // Voltage
    const c3 = document.getElementById('bVoltageChart').getContext('2d'); dc('bVolt');
    charts.bVolt = new Chart(c3, { type: 'line', data: { labels, datasets: [{ label: 'Voltage (V)', data: bm.voltage, borderColor: C.amber, borderWidth: 3, tension: 0.4, pointRadius: 4, pointBackgroundColor: C.amber, fill: false }] }, options: { ...dOpts, plugins: { legend: { display: false } } } });

    // Current
    const c4 = document.getElementById('bCurrentChart').getContext('2d'); dc('bCurr');
    charts.bCurr = new Chart(c4, { type: 'line', data: { labels, datasets: [{ label: 'Current (A)', data: bm.current, borderColor: C.sky, borderWidth: 3, tension: 0.4, pointRadius: 4, pointBackgroundColor: C.sky, fill: false }] }, options: { ...dOpts, plugins: { legend: { display: false } } } });

    // PF
    const c5 = document.getElementById('bPFChart').getContext('2d'); dc('bPF');
    charts.bPF = new Chart(c5, { type: 'line', data: { labels, datasets: [{ label: 'Power Factor', data: bm.pf, borderColor: C.violet, borderWidth: 3, tension: 0.4, pointRadius: 4, pointBackgroundColor: C.violet, fill: false }] }, options: { ...dOpts, plugins: { legend: { display: false } }, scales: { ...dOpts.scales, y: { ...dOpts.scales.y, min: 0.85, max: 0.98 } } } });

    // Demand + Peak Load
    const c6 = document.getElementById('bDemandChart').getContext('2d'); dc('bDem');
    charts.bDem = new Chart(c6, { type: 'bar', data: { labels, datasets: [{ label: 'Max Demand (kW)', data: bm.demand, backgroundColor: C.rose, borderRadius: 6 }, { label: 'Peak Load (kW)', data: bm.peak_load, backgroundColor: C.sky, borderRadius: 6 }] }, options: dOpts });

    // All buildings comparison
    const c7 = document.getElementById('allBuildingsCompare').getContext('2d'); dc('allB');
    const ds = []; const colors = [C.indigo, C.emerald, C.amber]; let i = 0;
    for (const [id, data] of Object.entries(historyData.building_monthly)) {
        ds.push({ label: id, data: data.consumption, borderColor: colors[i % 3], backgroundColor: 'transparent', borderWidth: 3, tension: 0.4, pointRadius: 4, pointBackgroundColor: colors[i % 3] }); i++;
    }
    charts.allB = new Chart(c7, { type: 'line', data: { labels: MN, datasets: ds }, options: dOpts });
}

// ===== TRENDS & IMPACT =====
async function loadTrendsSection() {
    try {
        const [impRes, fiRes] = await Promise.all([fetch('/api/impact-analysis'), fetch('/api/feature-importance')]);
        const impact = await impRes.json();
        const fi = await fiRes.json();
        renderImpactCharts(impact);
        renderFeatureChart(fi);
    } catch (err) { console.error("Trends error:", err); }
}

function renderImpactCharts(d) {
    // Weather Impact
    const c1 = document.getElementById('weatherImpactChart').getContext('2d'); dc('weather');
    charts.weather = new Chart(c1, { type: 'bar', data: { labels: d.weather.labels, datasets: [{ label: 'Avg Energy (kWh)', data: d.weather.consumption, backgroundColor: [C.sky, C.emerald, C.amber, C.rose, C.fuchsia], borderRadius: 10, barPercentage: 0.6 }] }, options: { ...dOpts, plugins: { legend: { display: false } } } });

    // Interruption Impact
    const c2 = document.getElementById('interruptImpactChart').getContext('2d'); dc('intImp');
    charts.intImp = new Chart(c2, { type: 'bar', data: { labels: d.interruptions.labels, datasets: [{ label: 'Avg Energy (kWh)', data: d.interruptions.consumption, backgroundColor: C.rose, borderRadius: 8 }, { label: 'Avg Bill (₹)', data: d.interruptions.bill, backgroundColor: C.amber, borderRadius: 8 }] }, options: dOpts });

    // Seasonal
    const c3 = document.getElementById('seasonalChart').getContext('2d'); dc('season');
    charts.season = new Chart(c3, { type: 'bar', data: { labels: d.seasonal.labels, datasets: [{ label: 'Avg Energy (kWh)', data: d.seasonal.consumption, backgroundColor: [C.sky, C.amber, C.emerald], borderRadius: 10, yAxisID: 'y' }, { label: 'Avg Bill (₹)', data: d.seasonal.bill, type: 'line', borderColor: C.rose, borderWidth: 3, tension: 0.4, pointRadius: 6, pointBackgroundColor: C.rose, yAxisID: 'y1', fill: false }] }, options: { ...dOpts, scales: { y: { grid: { color: '#f1f5f9' }, position: 'left' }, y1: { grid: { display: false }, position: 'right' }, x: { grid: { display: false } } } } });

    // Weekday vs Weekend
    const c4 = document.getElementById('weekdayChart').getContext('2d'); dc('weekday');
    charts.weekday = new Chart(c4, { type: 'bar', data: { labels: d.weekday_weekend.labels, datasets: [{ label: 'Avg Energy (kWh)', data: d.weekday_weekend.consumption, backgroundColor: [C.indigo, C.violet], borderRadius: 12, barPercentage: 0.5 }, { label: 'Avg Bill (₹)', data: d.weekday_weekend.bill, backgroundColor: [C.emerald, C.teal], borderRadius: 12, barPercentage: 0.5 }] }, options: dOpts });

    // Rainfall
    const c5 = document.getElementById('rainfallChart').getContext('2d'); dc('rain');
    charts.rain = new Chart(c5, { type: 'bar', data: { labels: d.rainfall.labels, datasets: [{ label: 'Avg Energy (kWh)', data: d.rainfall.consumption, backgroundColor: [C.sky, C.indigo, C.violet, C.fuchsia], borderRadius: 10, barPercentage: 0.6 }] }, options: { ...dOpts, plugins: { legend: { display: false } } } });
}

function renderFeatureChart(fi) {
    const c6 = document.getElementById('featureChart').getContext('2d'); dc('feat');

    // Pair features with both energy and bill importance
    const paired = fi.features.map((f, i) => ({
        name: f.replace(/_/g, ' '),
        energy: fi.energy_importance[i] || 0,
        bill: fi.bill_importance ? (fi.bill_importance[i] || 0) : 0
    }));

    // Sort by energy importance and take top 12
    paired.sort((a, b) => b.energy - a.energy);
    const top = paired.slice(0, 12).reverse(); // reverse for horizontal chart (top item at top)

    // Color gradient from amber to indigo
    const barColors = top.map((_, i) => {
        const ratio = i / (top.length - 1);
        const r = Math.round(99 + (245 - 99) * (1 - ratio));
        const g = Math.round(102 + (158 - 102) * (1 - ratio));
        const b = Math.round(241 + (11 - 241) * (1 - ratio));
        return `rgba(${r},${g},${b},0.85)`;
    });

    charts.feat = new Chart(c6, {
        type: 'bar',
        data: {
            labels: top.map(t => t.name),
            datasets: [
                {
                    label: 'Energy Model',
                    data: top.map(t => +(t.energy * 100).toFixed(2)),
                    backgroundColor: barColors,
                    borderRadius: 6,
                    barPercentage: 0.7
                },
                {
                    label: 'Bill Model',
                    data: top.map(t => +(t.bill * 100).toFixed(2)),
                    backgroundColor: 'rgba(16,185,129,0.6)',
                    borderRadius: 6,
                    barPercentage: 0.7
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { left: 5, right: 15 } },
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, font: { size: 11, family: 'Inter' }, padding: 14 } },
                tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%` } }
            },
            scales: {
                x: {
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { size: 11 }, callback: v => v + '%' },
                    title: { display: true, text: 'Importance (%)', font: { size: 11, weight: '500' }, color: '#64748b' }
                },
                y: {
                    grid: { display: false },
                    ticks: { font: { size: 11, weight: 500, family: 'Inter' }, color: '#334155' }
                }
            }
        }
    });
}

// ===== FORECAST (single-month, cached) =====
let forecastCache = null; // stores full 12-month result
let lastForecastKey = '';  // "building|year"

async function runYearlyForecast() {
    const building = document.getElementById('pred-building').value;
    const year = document.getElementById('pred-year').value;
    const key = `${building}|${year}`;

    // Use cache if same building+year
    if (forecastCache && lastForecastKey === key) {
        displaySelectedMonth();
        return;
    }

    document.getElementById('pred-loading').classList.remove('hidden');
    ['pred-summary', 'qa-result'].forEach(id => document.getElementById(id).classList.add('hidden'));
    try {
        const res = await fetch('/api/yearly-forecast', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ building_id: building, year: parseInt(year) })
        });
        const result = await res.json();
        if (!res.ok) { alert('Error: ' + result.error); return; }
        forecastCache = result;
        lastForecastKey = key;
        document.getElementById('pred-loading').classList.add('hidden');
        displaySelectedMonth();
    } catch (err) {
        document.getElementById('pred-loading').classList.add('hidden');
        alert('Failed to connect.');
    }
}

function onForecastInputChange() {
    forecastCache = null;
    lastForecastKey = '';
    forecastLoaded = false;
    runYearlyForecast();
}

function onMonthChange() {
    if (forecastCache) displaySelectedMonth();
    else runYearlyForecast();
}

function displaySelectedMonth() {
    if (!forecastCache) return;
    const selMonth = parseInt(document.getElementById('pred-month').value);
    const year = document.getElementById('pred-year').value;
    const building = document.getElementById('pred-building').value;
    const m = forecastCache.monthly.find(x => x.month === selMonth);
    if (!m) return;

    const monthName = MN[selMonth - 1];
    const fullMonthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

    // Summary cards
    document.getElementById('sum-energy').innerText = `${m.energy_pred.toLocaleString('en-IN', { maximumFractionDigits: 0 })} kWh`;
    document.getElementById('sum-bill').innerText = `\u20b9 ${m.bill_pred.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
    document.getElementById('sum-month-label').innerText = `${monthName} ${year}`;
    document.getElementById('sum-prev-energy').innerText = `${m.prev_energy.toLocaleString('en-IN', { maximumFractionDigits: 0 })} kWh`;
    document.getElementById('sum-prev-label').innerText = `${monthName} ${parseInt(year) - 1}`;

    const diff = m.prev_energy > 0 ? ((m.energy_pred - m.prev_energy) / m.prev_energy * 100) : 0;
    const changeEl = document.getElementById('sum-change');
    const changeLabel = document.getElementById('sum-change-label');
    changeEl.innerText = `${diff > 0 ? '+' : ''}${diff.toFixed(1)}%`;
    if (diff > 2) { changeEl.className = 'text-3xl font-extrabold mt-2 text-rose-600'; changeLabel.innerText = 'Higher than ' + (parseInt(year) - 1); changeLabel.className = 'text-xs mt-1 text-rose-500'; }
    else if (diff < -2) { changeEl.className = 'text-3xl font-extrabold mt-2 text-emerald-600'; changeLabel.innerText = 'Lower than ' + (parseInt(year) - 1); changeLabel.className = 'text-xs mt-1 text-emerald-500'; }
    else { changeEl.className = 'text-3xl font-extrabold mt-2 text-slate-500'; changeLabel.innerText = 'Similar to ' + (parseInt(year) - 1); changeLabel.className = 'text-xs mt-1 text-slate-400'; }

    // Comparison charts (2025 vs 2026 for this single month)
    const c1 = document.getElementById('forecastEnergyChart').getContext('2d'); dc('fcE');
    charts.fcE = new Chart(c1, {
        type: 'bar', data: {
            labels: [`${monthName} ${parseInt(year) - 1}`, `${monthName} ${year}`],
            datasets: [{ label: 'Energy (kWh)', data: [m.prev_energy, m.energy_pred], backgroundColor: [C.slate, C.indigo], borderRadius: 12, barPercentage: 0.5 }]
        }, options: { ...dOpts, plugins: { legend: { display: false } } }
    });
    const c2 = document.getElementById('forecastBillChart').getContext('2d'); dc('fcB');
    charts.fcB = new Chart(c2, {
        type: 'bar', data: {
            labels: [`${monthName} ${parseInt(year) - 1}`, `${monthName} ${year}`],
            datasets: [{ label: 'Bill (\u20b9)', data: [m.prev_bill, m.bill_pred], backgroundColor: [C.slate, C.emerald], borderRadius: 12, barPercentage: 0.5 }]
        }, options: { ...dOpts, plugins: { legend: { display: false } } }
    });

    // Detail table (single row)
    const tbody = document.getElementById('pred-table-body'); tbody.innerHTML = '';
    const tc = diff > 2 ? 'text-rose-600' : diff < -2 ? 'text-emerald-600' : 'text-slate-400';
    const icon = diff > 2 ? 'fa-arrow-up' : diff < -2 ? 'fa-arrow-down' : 'fa-minus';
    const label = diff > 2 ? 'Higher' : diff < -2 ? 'Lower' : 'Same';
    tbody.innerHTML = `<tr class="hover:bg-slate-50 transition">
        <td class="px-6 py-4 font-semibold">${fullMonthNames[selMonth - 1]}</td>
        <td class="px-6 py-4 text-right font-bold text-indigo-600">${m.energy_pred.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
        <td class="px-6 py-4 text-right font-bold text-emerald-600">\u20b9${m.bill_pred.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
        <td class="px-6 py-4 text-right text-slate-500">${m.prev_energy.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
        <td class="px-6 py-4 text-right text-slate-500">\u20b9${m.prev_bill.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</td>
        <td class="px-6 py-4 text-right text-slate-500">${m.temp}\u00b0C</td>
        <td class="px-6 py-4 text-right font-semibold text-amber-600">\u20b9${(m.tariff_rate || 0).toFixed(2)}</td>
        <td class="px-6 py-4 text-center"><span class="inline-flex items-center gap-1 text-xs font-bold ${tc}"><i class="fa-solid ${icon}"></i> ${label} (${diff.toFixed(1)}%)</span></td>
    </tr>`;

    document.getElementById('pred-summary').classList.remove('hidden');

    // Auto-trigger business insight for the selected month
    autoBusinessInsight(building, parseInt(year), selMonth);
}

// Auto-run business insight after forecast
async function autoBusinessInsight(building, year, month) {
    try {
        const res = await fetch('/api/business-qa', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ building_id: building, month: month, year_a: year - 1, year_b: year, ref_bill: null })
        });
        const d = await res.json();
        if (!res.ok) return;
        renderBusinessInsight(d);
    } catch (err) { console.error('QA auto error:', err); }
}

// ===== BUSINESS INSIGHT (auto-rendered) =====
function renderBusinessInsight(d) {
    const card = document.getElementById('qa-verdict-card');
    card.className = 'rounded-2xl p-8 text-white shadow-xl relative overflow-hidden ';
    if (d.verdict === 'HIGHER') card.className += 'bg-gradient-to-br from-rose-500 to-red-600';
    else if (d.verdict === 'LOWER') card.className += 'bg-gradient-to-br from-emerald-500 to-teal-600';
    else card.className += 'bg-gradient-to-br from-amber-500 to-orange-500';

    document.getElementById('qa-verdict-text').innerText = `${d.month_name} ${d.year_b} bill will be ${d.verdict}`;
    document.getElementById('qa-ref-bill').innerText = `\u20b9${d.actual_bill_a.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
    document.getElementById('qa-pred-bill').innerText = `\u20b9${d.predicted_bill_b.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
    document.getElementById('qa-pct-change').innerText = `${d.pct_change > 0 ? '+' : ''}${d.pct_change}%`;
    document.getElementById('qa-temp').innerText = `${d.assumed_conditions.temp}\u00b0C`;

    const factorsDiv = document.getElementById('qa-factors'); factorsDiv.innerHTML = '';
    d.factors.forEach(f => {
        const color = f.impact === 'negative' ? 'border-rose-200 bg-rose-50 text-rose-700' : f.impact === 'positive' ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-slate-50 text-slate-600';
        const icon = f.impact === 'negative' ? 'fa-circle-xmark text-rose-500' : f.impact === 'positive' ? 'fa-circle-check text-emerald-500' : 'fa-circle-minus text-slate-400';
        factorsDiv.innerHTML += `<div class="flex items-start space-x-3 p-4 rounded-xl border ${color}"><i class="fa-solid ${icon} text-lg mt-0.5"></i><div><p class="font-semibold text-sm">${f.factor}</p><p class="text-xs mt-0.5 opacity-80">${f.detail}</p></div></div>`;
    });

    document.getElementById('qa-result').classList.remove('hidden');
}

// ===== SOLAR ANALYSIS =====
async function runSolarAnalysis() {
    const building = document.getElementById('solar-building').value;
    const capacity = parseFloat(document.getElementById('solar-capacity').value) || 0;

    document.getElementById('solar-loading').classList.remove('hidden');
    document.getElementById('solar-results').classList.add('hidden');

    try {
        const res = await fetch('/api/solar-analysis', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ building_id: building, year: 2026, panel_capacity_kw: capacity })
        });
        const d = await res.json();
        if (!res.ok) { alert('Error: ' + d.error); return; }
        document.getElementById('solar-loading').classList.add('hidden');
        renderSolarResults(d);
        renderSolarCharts(d);
    } catch (err) {
        document.getElementById('solar-loading').classList.add('hidden');
        alert('Failed to load solar analysis.');
        console.error(err);
    }
}

function renderSolarResults(d) {
    const s = d.summary;
    const INR = v => '₹' + Number(v).toLocaleString('en-IN', { maximumFractionDigits: 0 });

    // System info badges
    document.getElementById('solar-sys-capacity').innerText = d.panel_capacity_kw;
    document.getElementById('solar-sys-building').innerText = `${d.building} — ${d.building_type}`;
    document.getElementById('solar-sys-area').innerText = Number(d.building_area_sqft).toLocaleString('en-IN');

    // KPI cards
    document.getElementById('solar-kpi-gen').innerText = `${Number(s.total_generation_kwh).toLocaleString('en-IN')} kWh`;
    document.getElementById('solar-kpi-offset').innerText = `${s.overall_solar_offset_pct}% of consumption`;
    document.getElementById('solar-kpi-save').innerText = INR(s.total_savings_rs);
    document.getElementById('solar-kpi-billpct').innerText = `${s.bill_reduction_pct}% bill reduction`;
    document.getElementById('solar-kpi-payback').innerText = `${s.payback_years} years`;
    document.getElementById('solar-kpi-cost').innerText = `Installation: ${INR(s.installation_cost)}`;
    document.getElementById('solar-kpi-co2').innerText = `${s.annual_co2_saved_tons} tons`;
    document.getElementById('solar-kpi-trees').innerText = `≈ ${Number(s.trees_equivalent).toLocaleString('en-IN')} trees planted`;

    // ROI card
    document.getElementById('roi-cost').innerText = INR(s.installation_cost);
    document.getElementById('roi-savings').innerText = INR(s.total_savings_rs);
    document.getElementById('roi-payback').innerText = `${s.payback_years} yrs`;
    const profit25yr = (s.total_savings_rs * 25) - s.installation_cost;
    document.getElementById('roi-profit').innerText = INR(profit25yr);
    document.getElementById('roi-best').innerText = s.best_month ? MN[s.best_month - 1] : '--';

    // Monthly table
    const tbody = document.getElementById('solar-table-body'); tbody.innerHTML = '';
    let totals = { solar: 0, cons: 0, net: 0, oldBill: 0, newBill: 0, saved: 0, co2: 0 };
    d.monthly.forEach(m => {
        totals.solar += m.solar_generation_kwh;
        totals.cons += m.avg_consumption_kwh;
        totals.net += m.net_consumption_kwh;
        totals.oldBill += m.avg_bill;
        totals.newBill += m.reduced_bill;
        totals.saved += m.total_savings;
        totals.co2 += m.co2_saved_kg;
        const offsetColor = m.solar_offset_pct > 80 ? 'text-emerald-600' : m.solar_offset_pct > 40 ? 'text-amber-600' : 'text-slate-500';
        tbody.innerHTML += `<tr class="hover:bg-slate-50 transition">
            <td class="px-4 py-3 font-semibold">${MN[m.month - 1]}</td>
            <td class="px-4 py-3 text-right font-bold text-amber-600">${m.solar_generation_kwh.toLocaleString('en-IN')}</td>
            <td class="px-4 py-3 text-right text-slate-600">${m.avg_consumption_kwh.toLocaleString('en-IN')}</td>
            <td class="px-4 py-3 text-right text-slate-500">${m.net_consumption_kwh.toLocaleString('en-IN')}</td>
            <td class="px-4 py-3 text-right font-semibold ${offsetColor}">${m.solar_offset_pct}%</td>
            <td class="px-4 py-3 text-right text-slate-400">${INR(m.avg_bill)}</td>
            <td class="px-4 py-3 text-right font-bold text-indigo-600">${INR(m.reduced_bill)}</td>
            <td class="px-4 py-3 text-right font-bold text-emerald-600">${INR(m.total_savings)}</td>
            <td class="px-4 py-3 text-right text-green-600">${m.co2_saved_kg.toLocaleString('en-IN')}</td>
        </tr>`;
    });
    // Total row
    tbody.innerHTML += `<tr class="bg-slate-100 font-bold">
        <td class="px-4 py-3">TOTAL</td>
        <td class="px-4 py-3 text-right text-amber-700">${Math.round(totals.solar).toLocaleString('en-IN')}</td>
        <td class="px-4 py-3 text-right">${Math.round(totals.cons).toLocaleString('en-IN')}</td>
        <td class="px-4 py-3 text-right">${Math.round(totals.net).toLocaleString('en-IN')}</td>
        <td class="px-4 py-3 text-right text-emerald-700">${s.overall_solar_offset_pct}%</td>
        <td class="px-4 py-3 text-right">${INR(totals.oldBill)}</td>
        <td class="px-4 py-3 text-right text-indigo-700">${INR(totals.newBill)}</td>
        <td class="px-4 py-3 text-right text-emerald-700">${INR(totals.saved)}</td>
        <td class="px-4 py-3 text-right text-green-700">${Math.round(totals.co2).toLocaleString('en-IN')}</td>
    </tr>`;

    document.getElementById('solar-results').classList.remove('hidden');
}

function renderSolarCharts(d) {
    const labels = d.monthly.map(m => MN[m.month - 1]);

    // 1. Solar Generation vs Consumption
    const c1 = document.getElementById('solarGenChart').getContext('2d'); dc('solarGen');
    charts.solarGen = new Chart(c1, {
        type: 'bar', data: {
            labels,
            datasets: [
                { label: 'Solar Generation (kWh)', data: d.monthly.map(m => m.solar_generation_kwh), backgroundColor: '#f59e0b', borderRadius: 8, barPercentage: 0.45, categoryPercentage: 0.8 },
                { label: 'Grid Consumption (kWh)', data: d.monthly.map(m => m.avg_consumption_kwh), backgroundColor: '#6366f1', borderRadius: 8, barPercentage: 0.45, categoryPercentage: 0.8 }
            ]
        }, options: { ...dOpts }
    });

    // 2. Bill Savings
    const c2 = document.getElementById('solarSavingsChart').getContext('2d'); dc('solarSave');
    charts.solarSave = new Chart(c2, {
        type: 'bar', data: {
            labels,
            datasets: [
                { label: 'Original Bill (₹)', data: d.monthly.map(m => m.avg_bill), backgroundColor: 'rgba(148,163,184,0.5)', borderRadius: 8 },
                { label: 'Reduced Bill (₹)', data: d.monthly.map(m => m.reduced_bill), backgroundColor: '#10b981', borderRadius: 8 },
                { label: 'Savings (₹)', data: d.monthly.map(m => m.total_savings), type: 'line', borderColor: '#f43f5e', borderWidth: 3, tension: 0.4, pointRadius: 5, pointBackgroundColor: '#f43f5e', fill: false, yAxisID: 'y1' }
            ]
        }, options: { ...dOpts, scales: { y: { grid: { color: '#f1f5f9' }, position: 'left' }, y1: { grid: { display: false }, position: 'right' }, x: { grid: { display: false } } } }
    });

    // 3. Solar Offset %
    const c3 = document.getElementById('solarOffsetChart').getContext('2d'); dc('solarOff');
    const offsets = d.monthly.map(m => m.solar_offset_pct);
    const barColors = offsets.map(v => v > 80 ? '#10b981' : v > 50 ? '#f59e0b' : v > 30 ? '#3b82f6' : '#94a3b8');
    charts.solarOff = new Chart(c3, {
        type: 'bar', data: {
            labels,
            datasets: [{ label: 'Solar Offset %', data: offsets, backgroundColor: barColors, borderRadius: 10, barPercentage: 0.6 }]
        }, options: {
            ...dOpts, plugins: { legend: { display: false } },
            scales: { y: { grid: { color: '#f1f5f9' }, max: 120, ticks: { callback: v => v + '%' } }, x: { grid: { display: false } } }
        }
    });

    // 4. CO2 Saved
    const c4 = document.getElementById('solarCO2Chart').getContext('2d'); dc('solarCO2');
    charts.solarCO2 = new Chart(c4, {
        type: 'line', data: {
            labels,
            datasets: [{
                label: 'CO₂ Saved (kg)', data: d.monthly.map(m => m.co2_saved_kg),
                borderColor: '#22c55e', backgroundColor: grad(c4, 'rgba(34,197,94,0.2)', 'rgba(34,197,94,0)'),
                borderWidth: 3, tension: 0.4, fill: true, pointRadius: 5, pointBackgroundColor: '#fff', pointBorderColor: '#22c55e', pointBorderWidth: 2
            }]
        }, options: { ...dOpts, plugins: { legend: { display: false } } }
    });
}
