"""Serve an interactive page comparing every model evaluated so far, read from results/.

    uv run python scripts/dashboard.py              # http://localhost:8050
    uv run python scripts/dashboard.py --port 9000

Drag the false-cut-in budget to see, for each model, the setting its development sweep would pick
at that budget (highest recall, ties to the lower median detection latency) and the scores it
reaches there. The cross-validated and held-out scores are shown as they are on disk. Rerun
`scripts/evaluate.py` and reload the page to see new results.
"""

import argparse
import json
import math
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from turn_detector.evaluation import MAX_FALSE_CUT_IN_RATE, Scores
from turn_detector.report import PUBLISHED_REFERENCES, ModelResults, load_all

# The budgets the slider steps through.
BUDGETS = [i / 400 for i in range(201)]  # 0 to 0.5 in steps of 0.0025


def finite(value: float) -> float | None:
    return None if math.isnan(value) else value


def scores_json(scores: Scores | None) -> dict | None:
    if scores is None:
        return None
    return {key: finite(value) if isinstance(value, float) else value for key, value in asdict(scores).items()}


def best_per_budget(sweep: list[Scores]) -> list[dict | None]:
    """At each budget, the sweep's highest-recall setting within it, ties to the lower median
    detection latency: TurnBench's operating-point rule, applied to all development conversations."""
    ordered = sorted(sweep, key=lambda scores: scores.false_cut_in_rate)
    best, chosen, i = None, [], 0
    for budget in BUDGETS:
        while i < len(ordered) and ordered[i].false_cut_in_rate <= budget:
            candidate = ordered[i]
            if best is None or (candidate.recall, -candidate.detection_latency_p50_ms) > (
                best.recall,
                -best.detection_latency_p50_ms,
            ):
                best = candidate
            i += 1
        chosen.append(scores_json(best))
    return chosen


def model_json(results: ModelResults) -> dict:
    return {
        "name": results.model.name,
        "colour": results.model.colour,
        "knobs": list(results.model.knob_names),
        "settings": len(results.sweep),
        "best": best_per_budget(results.sweep),
        "cross_validated": scores_json(results.cross_validation.scores),
        "chosen_setting": list(results.cross_validation.setting),
        "fold_settings": [list(fold.setting) for fold in results.cross_validation.folds],
        "held_out": scores_json(results.held_out),
    }


def page() -> str:
    data = {
        "budgets": BUDGETS,
        "default_budget": MAX_FALSE_CUT_IN_RATE,
        "models": [model_json(results) for results in load_all()],
        "references": [asdict(reference) for reference in PUBLISHED_REFERENCES],
    }
    return PAGE.replace("__DATA__", json.dumps(data))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = page().encode()  # read results/ on every load, so a reload shows new runs
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = parser.parse_args()
    page()  # fail now, not on the first request, if results/ can't be read
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Results dashboard on {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Turn Detector Results</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  :root { --ink: #1a1a19; --muted: #6b6a64; --grid: #e4e3dc; --bg: #fbfbf9; --card: #ffffff; }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--ink);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  main { max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  h2 { font-size: 16px; margin: 0 0 8px; }
  p.note { color: var(--muted); margin: 4px 0 12px; max-width: 80ch; }
  section { background: var(--card); border: 1px solid var(--grid); border-radius: 8px; padding: 16px; margin-top: 16px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px 24px; align-items: center; }
  .controls input[type=range] { width: min(420px, 100%); }
  .budget { font-variant-numeric: tabular-nums; font-weight: 600; font-size: 18px; min-width: 4ch; display: inline-block; }
  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
  .chart { height: 380px; }
  .table-wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th:first-child, td:first-child, th.left, td.left { text-align: left; }
  th { color: var(--muted); font-weight: 500; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
  .ref td { color: var(--muted); }
  .over { color: #b3261e; }
  @media (max-width: 480px) { .charts { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>Turn detector results</h1>
  <p class="note">Every model with results in <code>results/models/</code>. Rerun <code>scripts/evaluate.py</code> and reload to update. Click a legend entry to hide a model.</p>

  <section>
    <div class="controls">
      <label for="budget">False-cut-in budget</label>
      <input id="budget" type="range">
      <span class="budget" id="budget-value"></span>
      <button id="reset" type="button">Reset to 0.10</button>
    </div>
    <p class="note">At each budget, each model's operating point is its highest-recall setting whose false-cut-in rate is within the budget, ties going to the lower median latency. Picked and scored on all development conversations (not cross-validated), so these are optimistic.</p>
    <div class="table-wrap"><table id="at-budget"></table></div>
  </section>

  <section>
    <div class="charts">
      <div><h2>Recall</h2><div id="recall" class="chart"></div></div>
      <div><h2>Detection latency p50</h2><div id="latency" class="chart"></div></div>
    </div>
    <p class="note">Lines: the operating point at each budget. Circles: cross-validated scores; diamonds: held-out scores; grey crosses: published TurnBench results (test set, not comparable on equal terms). Markers sit at their actual false-cut-in rate.</p>
  </section>

  <section>
    <h2>Chosen setting: cross-validated and held-out</h2>
    <p class="note">The setting chosen by cross-validation at a 0.10 budget. Cross-validated: development conversations, each fold scored at the setting chosen without it. Held-out: the 12 held-out conversations, scored once by the final model (red: over the 0.10 budget).</p>
    <div class="table-wrap"><table id="final"></table></div>
  </section>
</main>

<script>
const DATA = __DATA__;
const fmt = (v, d) => v === null || v === undefined ? "—" : v.toFixed(d);
const ms = v => fmt(v, 0);
const setting = (m, s) => s.map((v, i) => `${m.knobs[i]} = ${+v.toFixed(4)}`).join(", ");
const lat = s => `${ms(s.detection_latency_p10_ms)} / ${ms(s.detection_latency_p50_ms)} / ${ms(s.detection_latency_p90_ms)}`;
const swatch = c => `<span class="swatch" style="background:${c}"></span>`;

const slider = document.getElementById("budget");
slider.min = 0; slider.max = DATA.budgets.length - 1; slider.step = 1;
const defaultIndex = DATA.budgets.findIndex(b => Math.abs(b - DATA.default_budget) < 1e-9);
slider.value = defaultIndex;

const layout = (yTitle, yRange) => ({
  margin: {l: 56, r: 12, t: 8, b: 44}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: {size: 12, color: "#1a1a19"}, hovermode: "closest",
  xaxis: {title: "false-cut-in rate", range: [0, 0.5], gridcolor: "#e4e3dc", zeroline: false},
  yaxis: {title: yTitle, range: yRange, gridcolor: "#e4e3dc", zeroline: false},
  legend: {orientation: "h", y: -0.2}, shapes: [],
});

function lineTraces(metric) {
  const traces = [];
  for (const m of DATA.models) {
    const xs = [], ys = [], text = [];
    m.best.forEach((s, i) => {
      if (!s) return;
      xs.push(DATA.budgets[i]); ys.push(s[metric]);
      text.push(`${m.name}<br>budget ${DATA.budgets[i].toFixed(4)}<br>${setting(m, s.setting)}<br>recall ${fmt(s.recall, 3)}, FCR ${fmt(s.false_cut_in_rate, 3)}<br>latency p10/p50/p90 ${lat(s)} ms`);
    });
    traces.push({x: xs, y: ys, name: m.name, mode: "lines", line: {color: m.colour, width: 2, shape: "hv"}, legendgroup: m.name, text, hovertemplate: "%{text}<extra></extra>"});
    for (const [key, symbol, label] of [["cross_validated", "circle", "cross-validated"], ["held_out", "diamond", "held-out"]]) {
      const s = m[key];
      if (!s) continue;
      traces.push({x: [s.false_cut_in_rate], y: [s[metric]], mode: "markers", legendgroup: m.name, showlegend: false,
        marker: {color: m.colour, symbol, size: 11, line: {color: "#fff", width: 1.5}},
        hovertemplate: `${m.name}, ${label}<br>${setting(m, s.setting)}<br>recall ${fmt(s.recall, 3)}, FCR ${fmt(s.false_cut_in_rate, 3)}<br>latency p10/p50/p90 ${lat(s)} ms<extra></extra>`});
    }
  }
  traces.push({x: DATA.references.map(r => r.false_cut_in_rate), y: DATA.references.map(r => r[metric]),
    name: "published", mode: "markers", marker: {color: "#9a988f", symbol: "x", size: 10},
    text: DATA.references.map(r => `${r.name} (published, TurnBench test set)<br>recall ${fmt(r.recall, 3)}, FCR ${fmt(r.false_cut_in_rate, 3)}<br>latency p10/p50/p90 ${lat(r)} ms`),
    hovertemplate: "%{text}<extra></extra>"});
  return traces;
}

const recallLayout = layout("recall", [0, 1.02]);
const latencyLayout = layout("detection latency (ms)", [-100, 3100]);
Plotly.newPlot("recall", lineTraces("recall"), recallLayout, {responsive: true, displaylogo: false});
Plotly.newPlot("latency", lineTraces("detection_latency_p50_ms"), latencyLayout, {responsive: true, displaylogo: false});

function update() {
  const i = +slider.value, budget = DATA.budgets[i];
  document.getElementById("budget-value").textContent = budget.toFixed(i % 4 === 0 ? 2 : i % 2 === 0 ? 3 : 4);
  const line = {type: "line", xref: "x", yref: "paper", x0: budget, x1: budget, y0: 0, y1: 1, line: {color: "#6b6a64", dash: "dot", width: 1}};
  Plotly.relayout("recall", {shapes: [line]});
  Plotly.relayout("latency", {shapes: [line]});
  const rows = DATA.models.map(m => {
    const s = m.best[i];
    if (!s) return `<tr><td>${swatch(m.colour)}${m.name}</td><td class="left" colspan="4">no setting within this budget</td></tr>`;
    return `<tr><td>${swatch(m.colour)}${m.name}</td><td class="left">${setting(m, s.setting)}</td><td>${fmt(s.recall, 3)}</td><td>${fmt(s.false_cut_in_rate, 3)}</td><td>${lat(s)}</td></tr>`;
  });
  document.getElementById("at-budget").innerHTML =
    `<tr><th>Model</th><th class="left">Setting</th><th>Recall</th><th>False-cut-in rate</th><th>Latency p10 / p50 / p90 (ms)</th></tr>` + rows.join("");
}

function finalTable() {
  const rows = [];
  for (const m of DATA.models) {
    const cv = m.cross_validated, ho = m.held_out;
    const ranges = m.knobs.map((_, k) => {
      const vs = m.fold_settings.map(s => s[k]);
      const lo = Math.min(...vs), hi = Math.max(...vs);
      return lo === hi ? `${+lo.toFixed(4)}` : `${+lo.toFixed(4)}–${+hi.toFixed(4)}`;
    }).join(", ");
    const over = ho && ho.false_cut_in_rate > DATA.default_budget ? "over" : "";
    rows.push(`<tr><td>${swatch(m.colour)}${m.name}</td><td class="left">${setting(m, m.chosen_setting)}<br><small style="color:var(--muted)">per fold: ${ranges}</small></td>
      <td>${fmt(cv.recall, 3)}</td><td>${fmt(cv.false_cut_in_rate, 3)}</td><td>${lat(cv)}</td>
      <td>${ho ? fmt(ho.recall, 3) : "—"}</td><td class="${over}">${ho ? fmt(ho.false_cut_in_rate, 3) : "—"}</td><td>${ho ? lat(ho) : "not scored yet"}</td></tr>`);
  }
  for (const r of DATA.references) {
    rows.push(`<tr class="ref"><td>${r.name} (published)</td><td class="left">TurnBench test set</td><td colspan="3"></td><td>${fmt(r.recall, 3)}</td><td>${fmt(r.false_cut_in_rate, 3)}</td><td>${lat(r)}</td></tr>`);
  }
  document.getElementById("final").innerHTML =
    `<tr><th>Model</th><th class="left">Setting</th><th colspan="3" style="text-align:center">Cross-validated (development)</th><th colspan="3" style="text-align:center">Held-out</th></tr>
     <tr><th></th><th></th><th>Recall</th><th>FCR</th><th>Latency p10/p50/p90 (ms)</th><th>Recall</th><th>FCR</th><th>Latency p10/p50/p90 (ms)</th></tr>` + rows.join("");
}

slider.addEventListener("input", update);
document.getElementById("reset").addEventListener("click", () => { slider.value = defaultIndex; update(); });
finalTable();
update();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
