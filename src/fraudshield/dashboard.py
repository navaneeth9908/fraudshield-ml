"""Dependency-light analyst dashboard and deterministic decision presentation."""

# ruff: noqa: E501, RUF001

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class DecisionView:
    """Human-readable presentation of a calibrated score and operating threshold."""

    decision: str
    probability_percent: str
    threshold_percent: str
    margin_percent: str
    summary: str


def build_decision_view(*, probability: float, threshold: float) -> DecisionView:
    """Describe a score without changing the deployed decision policy."""
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and between 0 and 1")
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    decision = "review" if probability >= threshold else "approve"
    margin = probability - threshold
    if decision == "review":
        summary = "At or above the operating threshold; route to human review."
    else:
        summary = "below the operating threshold; approve under this demo policy."
    return DecisionView(
        decision=decision,
        probability_percent=f"{probability:.2%}",
        threshold_percent=f"{threshold:.2%}",
        margin_percent=f"{margin * 100:+.2f} pp",
        summary=summary,
    )


def render_dashboard() -> str:
    """Render the self-contained local analyst application."""
    return _DASHBOARD_HTML


_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FraudShield analyst console</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #f7f8ff;
      --muted: #aab1c5;
      --surface: #13182a;
      --surface-2: #191f34;
      --line: #2b3450;
      --cyan: #72e6ff;
      --violet: #a78bfa;
      --green: #64e6a5;
      --amber: #ffc66d;
      --red: #ff7b8b;
      --shadow: 0 22px 70px rgba(0, 0, 0, .32);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 9% 10%, rgba(64, 207, 255, .14), transparent 32rem),
        radial-gradient(circle at 91% 4%, rgba(167, 139, 250, .16), transparent 28rem),
        #090d18;
      font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: var(--cyan); }
    button, textarea { font: inherit; }
    .shell { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 42px 0 64px; }
    header { display: flex; gap: 28px; align-items: flex-start; justify-content: space-between; margin-bottom: 28px; }
    .eyebrow { margin: 0 0 8px; color: var(--cyan); font-size: 12px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    h1 { margin: 0; max-width: 760px; font-size: clamp(34px, 6vw, 66px); line-height: .98; letter-spacing: -.055em; }
    .lede { max-width: 760px; margin: 18px 0 0; color: var(--muted); font-size: 17px; }
    .status {
      display: inline-flex; align-items: center; gap: 9px; flex: none; padding: 10px 14px;
      border: 1px solid var(--line); border-radius: 999px; background: rgba(19, 24, 42, .78);
      color: var(--muted); box-shadow: var(--shadow);
    }
    .status::before { width: 9px; height: 9px; border-radius: 50%; background: var(--amber); content: ""; box-shadow: 0 0 18px currentColor; }
    .status.ready { color: var(--green); border-color: rgba(100, 230, 165, .4); }
    .status.ready::before { background: var(--green); }
    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }
    .card { border: 1px solid var(--line); border-radius: 20px; background: linear-gradient(145deg, rgba(25, 31, 52, .96), rgba(15, 20, 36, .96)); box-shadow: var(--shadow); }
    .metric { grid-column: span 3; min-height: 132px; padding: 20px; }
    .metric-label { display: block; margin-bottom: 13px; color: var(--muted); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
    .metric-value { display: block; overflow-wrap: anywhere; font-size: 22px; font-weight: 750; letter-spacing: -.025em; }
    .metric-note { display: block; margin-top: 7px; color: var(--muted); font-size: 12px; }
    .workbench { grid-column: span 6; padding: 24px; }
    h2 { margin: 0 0 7px; font-size: 21px; letter-spacing: -.02em; }
    h3 { margin: 0 0 7px; font-size: 16px; }
    .subtle { margin: 0 0 18px; color: var(--muted); }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
    button {
      cursor: pointer; border: 0; border-radius: 10px; padding: 10px 14px; color: #07101a;
      background: linear-gradient(135deg, var(--cyan), #9bf4dd); font-weight: 800;
    }
    button.secondary { border: 1px solid var(--line); color: var(--ink); background: var(--surface); font-weight: 650; }
    button:disabled { cursor: wait; opacity: .55; }
    button:focus-visible, textarea:focus-visible { outline: 3px solid rgba(114, 230, 255, .35); outline-offset: 2px; }
    textarea {
      width: 100%; min-height: 280px; resize: vertical; border: 1px solid var(--line); border-radius: 13px;
      padding: 14px; color: #dfe7ff; background: #0a0f1d; font: 12px/1.5 "SFMono-Regular", Consolas, monospace;
    }
    .batch textarea { min-height: 280px; }
    .action-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 12px; }
    .hint { color: var(--muted); font-size: 12px; }
    .error { min-height: 24px; margin: 12px 0 0; color: var(--red); font-weight: 650; white-space: pre-wrap; }
    .result { display: none; margin-top: 15px; border: 1px solid var(--line); border-radius: 14px; padding: 16px; background: #0c1220; }
    .result.visible { display: block; }
    .result-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .decision { border-radius: 999px; padding: 5px 9px; color: #07101a; background: var(--green); font-size: 12px; font-weight: 900; text-transform: uppercase; }
    .decision.review { background: var(--amber); }
    .score { margin: 12px 0 4px; font-size: 34px; font-weight: 800; letter-spacing: -.04em; }
    .meter { position: relative; overflow: hidden; height: 8px; border-radius: 999px; background: #242c42; }
    .meter-fill { width: 0; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--green), var(--amber), var(--red)); transition: width .25s ease; }
    .threshold-copy, .reason { color: var(--muted); font-size: 12px; }
    .table-wrap { overflow: auto; margin-top: 15px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    th { color: var(--muted); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; }
    .context { grid-column: span 7; padding: 24px; }
    .context-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 11px; margin-top: 16px; }
    .context-item { min-height: 118px; border: 1px solid var(--line); border-radius: 14px; padding: 15px; background: rgba(7, 12, 23, .48); }
    .context-item strong { display: block; margin: 4px 0; font-size: 24px; color: var(--cyan); }
    .guardrails { grid-column: span 5; padding: 24px; }
    .guardrails ul { margin: 12px 0 0; padding-left: 20px; color: var(--muted); }
    .guardrails li + li { margin-top: 8px; }
    footer { margin-top: 22px; color: var(--muted); font-size: 12px; text-align: center; }
    @media (max-width: 880px) {
      header { display: block; }
      .status { margin-top: 20px; }
      .metric { grid-column: span 6; }
      .workbench, .context, .guardrails { grid-column: 1 / -1; }
    }
    @media (max-width: 560px) {
      .shell { width: min(100% - 20px, 1180px); padding-top: 26px; }
      .metric { grid-column: 1 / -1; }
      .context-grid { grid-template-columns: 1fr; }
      .workbench, .context, .guardrails { padding: 18px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <p class="eyebrow">Decision intelligence · local only</p>
        <h1>FraudShield analyst console</h1>
        <p class="lede">Explore calibrated fraud risk, the deployed review threshold, and all-or-nothing batch evidence. Scores recommend analyst review; they never authorize or block a payment.</p>
      </div>
      <div id="model-readiness" class="status" aria-live="polite">Checking model readiness…</div>
    </header>

    <section class="grid" aria-label="Model status">
      <article class="card metric"><span class="metric-label">Deployed model</span><span id="model-name" class="metric-value">—</span><span id="model-version" class="metric-note">Waiting for metadata</span></article>
      <article class="card metric"><span class="metric-label">Calibration</span><span id="calibration" class="metric-value">—</span><span class="metric-note">Calibrated fraud probability</span></article>
      <article class="card metric"><span class="metric-label">Operating point</span><span id="threshold" class="metric-value">—</span><span class="metric-note">Minimum expected-loss threshold</span></article>
      <article class="card metric"><span class="metric-label">Input contract</span><span id="feature-count" class="metric-value">30 features</span><span class="metric-note">Time, V1–V28, Amount</span></article>

      <article class="card workbench">
        <h2>Score one transaction</h2>
        <p class="subtle">Edit a synthetic feature vector or load a preset. The target label is never accepted.</p>
        <div class="toolbar">
          <button id="preset-neutral" class="secondary" type="button">Neutral synthetic</button>
          <button id="preset-elevated" class="secondary" type="button">Elevated synthetic</button>
        </div>
        <label for="single-input" class="metric-label">Transaction JSON</label>
        <textarea id="single-input" spellcheck="false" aria-describedby="single-help"></textarea>
        <div class="action-row"><span id="single-help" class="hint">Exact 30-feature schema; finite numeric values only.</span><button id="score-single" type="button">Score risk</button></div>
        <p id="single-error" class="error" role="alert"></p>
        <section id="single-result" class="result" aria-live="polite" aria-label="Single score result">
          <div class="result-head"><strong>Calibrated fraud probability</strong><span id="single-decision" class="decision"></span></div>
          <div id="single-probability" class="score">—</div>
          <div class="meter" aria-hidden="true"><div id="single-meter" class="meter-fill"></div></div>
          <p id="single-threshold" class="threshold-copy"></p>
          <p id="single-reason" class="reason"></p>
        </section>
      </article>

      <article class="card workbench batch">
        <h2>Score a review batch</h2>
        <p class="subtle">Submit unique synthetic transactions and inspect deterministic row accounting and ordered decisions.</p>
        <label for="batch-input" class="metric-label">Transactions JSON array</label>
        <textarea id="batch-input" spellcheck="false" aria-describedby="batch-help"></textarea>
        <div class="action-row"><span id="batch-help" class="hint">Bounded to 1,000 rows; failures return no partial scores.</span><button id="score-batch" type="button">Score batch</button></div>
        <p id="batch-error" class="error" role="alert"></p>
        <section id="batch-result" class="result" aria-live="polite" aria-label="Batch score result">
          <div class="result-head"><strong>Batch evidence</strong><span id="batch-summary" class="hint"></span></div>
          <div class="table-wrap"><table><thead><tr><th>Transaction</th><th>Probability</th><th>Decision</th><th>Threshold</th></tr></thead><tbody id="batch-rows"></tbody></table></div>
        </section>
      </article>

      <article class="card context">
        <p class="eyebrow">Backtest context</p>
        <h2>Why the threshold is not 50%</h2>
        <p class="subtle">The checked-in holdout evidence prices errors asymmetrically and selects the threshold with minimum expected loss. These values explain the portfolio experiment; they are not a live cost calculator.</p>
        <div class="context-grid">
          <div class="context-item"><span class="metric-label">Missed fraud</span><strong>$500</strong><span class="hint">False-negative cost assumption</span></div>
          <div class="context-item"><span class="metric-label">False alert</span><strong>$25</strong><span class="hint">Analyst review cost assumption</span></div>
          <div class="context-item"><span class="metric-label">Review capacity</span><strong>1,000 alerts</strong><span class="hint">Fixed holdout analyst budget</span></div>
        </div>
      </article>

      <aside class="card guardrails">
        <p class="eyebrow">Responsible use</p>
        <h2>Decision support, not autopilot</h2>
        <ul>
          <li>No automatic payment blocking; a <strong>review</strong> decision requires human investigation.</li>
          <li>The two-day 2013 dataset is not representative of current production traffic.</li>
          <li>PCA-anonymized V1–V28 features limit explanation and fairness analysis.</li>
          <li>Model readiness verifies a local artifact contract, not governance or production approval.</li>
          <li>This demo is not a production payment authorization service.</li>
        </ul>
      </aside>
    </section>
    <footer>Runs against the same validated FastAPI scoring contract used by offline batch and container paths.</footer>
  </main>

  <script>
    const featureNames = ["Time", ...Array.from({length: 28}, (_, i) => `V${i + 1}`), "Amount"];
    const neutralFeatures = Object.fromEntries(featureNames.map(name => [name, 0]));
    neutralFeatures.Time = 84600;
    neutralFeatures.Amount = 49.95;
    const elevatedFeatures = {...neutralFeatures, Time: 120000, V1: -2.1, V3: -3.4, V10: -5.2, V12: -6.4, V14: -9.1, V17: -7.3, Amount: 239.90};
    const singleInput = document.querySelector("#single-input");
    const batchInput = document.querySelector("#batch-input");

    function pretty(value) { return JSON.stringify(value, null, 2); }
    function setText(selector, value) { document.querySelector(selector).textContent = value; }
    function explainError(payload, fallback) {
      if (!payload) return fallback;
      const detail = payload.detail;
      if (typeof detail === "string") return detail;
      if (detail) return JSON.stringify(detail, null, 2);
      return fallback;
    }
    async function requestJSON(path, options = {}) {
      let response;
      try { response = await fetch(path, options); }
      catch (error) { throw new Error(`Cannot reach the local scoring service: ${error.message}`); }
      let payload = null;
      try { payload = await response.json(); } catch (_) { /* graceful non-JSON error */ }
      if (!response.ok) throw new Error(explainError(payload, `${response.status} ${response.statusText}`));
      return payload;
    }
    async function refreshReadiness() {
      const status = document.querySelector("#model-readiness");
      try {
        await requestJSON("/ready");
        const info = await requestJSON("/model-info");
        status.textContent = "Model ready";
        status.classList.add("ready");
        setText("#model-name", info.model_name);
        setText("#model-version", info.model_version);
        setText("#calibration", info.calibration_method);
        setText("#threshold", `${(info.decision_threshold * 100).toFixed(3)}%`);
        setText("#feature-count", `${info.feature_count} features`);
      } catch (error) {
        status.textContent = "Model unavailable";
        status.classList.remove("ready");
        setText("#model-name", "Not loaded");
        setText("#model-version", `${error.message} — build and configure a local bundle.`);
      }
    }
    function parseObject(text, label) {
      let value;
      try { value = JSON.parse(text); }
      catch (error) { throw new Error(`${label} is not valid JSON: ${error.message}`); }
      if (!value || Array.isArray(value) || typeof value !== "object") throw new Error(`${label} must be a JSON object.`);
      return value;
    }
    function transaction(id, features) { return {transaction_id: id, features}; }
    function showPrediction(payload) {
      const result = document.querySelector("#single-result");
      const badge = document.querySelector("#single-decision");
      const probability = Number(payload.fraud_probability);
      badge.textContent = payload.decision;
      badge.classList.toggle("review", payload.decision === "review");
      setText("#single-probability", `${(probability * 100).toFixed(3)}%`);
      document.querySelector("#single-meter").style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
      setText("#single-threshold", `Threshold ${(payload.threshold * 100).toFixed(3)}% · margin ${((probability - payload.threshold) * 100).toFixed(3)} percentage points`);
      setText("#single-reason", payload.reason);
      result.classList.add("visible");
    }
    async function scoreSingle() {
      const button = document.querySelector("#score-single");
      setText("#single-error", "");
      button.disabled = true;
      try {
        const features = parseObject(singleInput.value, "Transaction");
        const payload = await requestJSON("/score", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(transaction("dashboard-single", features))});
        showPrediction(payload);
      } catch (error) {
        document.querySelector("#single-result").classList.remove("visible");
        setText("#single-error", error.message);
      } finally { button.disabled = false; }
    }
    async function scoreBatch() {
      const button = document.querySelector("#score-batch");
      setText("#batch-error", "");
      button.disabled = true;
      try {
        let transactions;
        try { transactions = JSON.parse(batchInput.value); }
        catch (error) { throw new Error(`Batch is not valid JSON: ${error.message}`); }
        if (!Array.isArray(transactions)) throw new Error("Batch must be a JSON array of transaction objects.");
        const payload = await requestJSON("/score-batch", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({transactions})});
        setText("#batch-summary", `${payload.batch.scored_rows}/${payload.batch.input_rows} scored · ${payload.batch.failed_rows} failed`);
        const body = document.querySelector("#batch-rows");
        body.replaceChildren();
        for (const prediction of payload.predictions) {
          const row = document.createElement("tr");
          for (const value of [prediction.transaction_id ?? "—", `${(prediction.fraud_probability * 100).toFixed(3)}%`, prediction.decision, `${(prediction.threshold * 100).toFixed(3)}%`]) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
          }
          body.appendChild(row);
        }
        document.querySelector("#batch-result").classList.add("visible");
      } catch (error) {
        document.querySelector("#batch-result").classList.remove("visible");
        setText("#batch-error", error.message);
      } finally { button.disabled = false; }
    }

    singleInput.value = pretty(neutralFeatures);
    batchInput.value = pretty([transaction("demo-neutral", neutralFeatures), transaction("demo-elevated", elevatedFeatures)]);
    document.querySelector("#preset-neutral").addEventListener("click", () => { singleInput.value = pretty(neutralFeatures); });
    document.querySelector("#preset-elevated").addEventListener("click", () => { singleInput.value = pretty(elevatedFeatures); });
    document.querySelector("#score-single").addEventListener("click", scoreSingle);
    document.querySelector("#score-batch").addEventListener("click", scoreBatch);
    refreshReadiness();
  </script>
</body>
</html>
"""
