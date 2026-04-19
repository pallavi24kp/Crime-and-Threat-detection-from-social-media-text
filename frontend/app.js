/**
 * app.js — CrimeShield Frontend Logic
 * Communicates with the Flask API at http://localhost:5000
 */

const API_BASE = "http://localhost:5000";

// ============================================================
// State
// ============================================================
let analysisHistory = JSON.parse(localStorage.getItem("cs_history") || "[]");

// ============================================================
// DOM references
// ============================================================
const inputText      = document.getElementById("input-text");
const charCount      = document.getElementById("char-count");
const analyzeBtn     = document.getElementById("analyze-btn");
const analyzeBtnText = document.getElementById("analyze-btn-text");
const clearBtn       = document.getElementById("clear-btn");

const resultsIdle    = document.getElementById("results-idle");
const resultsLoading = document.getElementById("results-loading");
const resultsContent = document.getElementById("results-content");

const verdictBanner  = document.getElementById("verdict-banner");
const verdictIcon    = document.getElementById("verdict-icon");
const verdictTitle   = document.getElementById("verdict-title");
const verdictSummary = document.getElementById("verdict-summary");
const confValue      = document.getElementById("confidence-value");
const catGrid        = document.getElementById("categories-grid");
const metaTime       = document.getElementById("meta-time");
const metaFlagged    = document.getElementById("meta-flagged");

const batchText      = document.getElementById("batch-text");
const batchLineCount = document.getElementById("batch-line-count");
const batchBtn       = document.getElementById("batch-btn");
const batchBtnText   = document.getElementById("batch-btn-text");
const batchResults   = document.getElementById("batch-results");

const historyList    = document.getElementById("history-list");
const clearHistBtn   = document.getElementById("clear-history-btn");

const statusDot      = document.getElementById("status-dot");
const statusLabel    = document.getElementById("status-label");

const statSamples    = document.getElementById("stat-samples");

const copyExampleBtn = document.getElementById("copy-example-btn");
const exampleCode    = document.getElementById("example-code");

// ============================================================
// API helpers
// ============================================================
async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error || `HTTP ${res.status}`);
  }
  return res.json();
}

// ============================================================
// Status check
// ============================================================
async function checkHealth() {
  try {
    await apiFetch("/api/health");
    statusDot.className   = "status-dot online";
    statusLabel.textContent = "API Online";
  } catch {
    statusDot.className   = "status-dot offline";
    statusLabel.textContent = "API Offline";
  }
}

async function loadStats() {
  try {
    const data = await apiFetch("/api/stats");
    const meta = data.model || {};
    if (meta.sample_size) {
      statSamples.textContent = (meta.sample_size / 1000).toFixed(0) + "K";
    }
  } catch { /* stats non-critical */ }
}

// ============================================================
// Single analyzer
// ============================================================
function setAnalyzing(loading) {
  analyzeBtn.disabled = loading;
  analyzeBtnText.textContent = loading ? "Analyzing…" : "Analyze Text";
  if (loading) {
    resultsIdle.classList.add("hidden");
    resultsContent.classList.add("hidden");
    resultsLoading.classList.remove("hidden");
  }
}

function renderResults(data) {
  resultsLoading.classList.add("hidden");

  // Verdict banner
  const isHarmful = data.is_harmful;
  verdictBanner.className = `verdict-banner ${isHarmful ? "harmful" : "clean"}`;
  verdictIcon.textContent = isHarmful ? "🚨" : "✅";
  verdictTitle.textContent = isHarmful ? "Harmful Content Detected" : "Content Appears Safe";
  verdictSummary.textContent = data.summary;
  confValue.textContent = data.overall_confidence.toFixed(1) + "%";

  // Categories
  catGrid.innerHTML = "";
  (data.labels || []).forEach((lbl, i) => {
    const pct = lbl.confidence;
    const row = document.createElement("div");
    row.className = "category-row";
    row.style.animationDelay = `${i * 0.05}s`;
    row.innerHTML = `
      <div class="category-name">
        <span class="category-dot" style="background:${lbl.color}"></span>
        ${lbl.name}
      </div>
      <div class="prog-track">
        <div class="prog-fill" style="width:${pct}%;background:${lbl.color};animation-delay:${i*0.07}s"></div>
      </div>
      <div class="category-pct ${lbl.flagged ? 'flagged' : ''}">${pct}%</div>
    `;
    catGrid.appendChild(row);
  });

  // Meta
  metaTime.textContent    = `⏱ ${data.processing_time_ms} ms`;
  metaFlagged.textContent = `🏷 ${data.flagged_count} / ${data.labels.length} categories flagged`;

  resultsContent.classList.remove("hidden");
}

async function runAnalysis() {
  const text = inputText.value.trim();
  if (!text) { inputText.focus(); return; }

  setAnalyzing(true);
  try {
    const result = await apiFetch("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    renderResults(result);
    addToHistory(text, result);
  } catch (err) {
    resultsLoading.classList.add("hidden");
    resultsIdle.classList.remove("hidden");
    resultsIdle.querySelector(".idle-message").textContent =
      `Error: ${err.message}. Make sure the Flask server is running on port 5000.`;
  } finally {
    setAnalyzing(false);
  }
}

// ============================================================
// Batch analyzer
// ============================================================
async function runBatch() {
  const lines = batchText.value
    .split("\n")
    .map(l => l.trim())
    .filter(Boolean);

  if (!lines.length) { batchText.focus(); return; }
  if (lines.length > 50) {
    alert("Maximum 50 texts per batch. Please reduce the number of lines.");
    return;
  }

  batchBtn.disabled = true;
  batchBtnText.textContent = "Analyzing…";
  batchResults.classList.remove("hidden");
  batchResults.innerHTML = `<div class="results-loading" style="padding:2rem;display:flex;gap:1rem;align-items:center;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius)">
    <div class="spinner"></div><span style="color:var(--text-secondary)">Running batch analysis on ${lines.length} texts…</span>
  </div>`;

  try {
    const data = await apiFetch("/api/batch", {
      method: "POST",
      body: JSON.stringify({ texts: lines }),
    });
    renderBatchResults(data);
  } catch (err) {
    batchResults.innerHTML = `<div class="empty-state">Error: ${err.message}</div>`;
  } finally {
    batchBtn.disabled = false;
    batchBtnText.textContent = "Run Batch Analysis";
  }
}

function renderBatchResults(data) {
  const harmPct = data.total > 0 ? ((data.harmful_count / data.total) * 100).toFixed(0) : 0;

  let html = `
    <div class="batch-summary-card">
      <div class="batch-summary-stat">
        <div class="batch-summary-value" style="color:var(--text-primary)">${data.total}</div>
        <div class="batch-summary-label">Total Analyzed</div>
      </div>
      <div class="batch-summary-stat">
        <div class="batch-summary-value" style="color:var(--red)">${data.harmful_count}</div>
        <div class="batch-summary-label">Harmful</div>
      </div>
      <div class="batch-summary-stat">
        <div class="batch-summary-value" style="color:var(--green)">${data.clean_count}</div>
        <div class="batch-summary-label">Clean</div>
      </div>
      <div class="batch-summary-stat">
        <div class="batch-summary-value" style="color:var(--amber)">${harmPct}%</div>
        <div class="batch-summary-label">Harm Rate</div>
      </div>
    </div>
  `;

  (data.results || []).forEach((r, i) => {
    if (r.error) {
      html += `<div class="batch-item"><span style="color:var(--text-muted)">#${i+1}</span><span style="color:var(--red);font-size:.8rem">${r.error}</span></div>`;
      return;
    }
    const cats = (r.labels || []).filter(l => l.flagged).map(l => l.name).join(", ") || "None";
    const cls  = r.is_harmful ? "harmful" : "clean";
    html += `
      <div class="batch-item ${cls}" data-index="${i}" style="animation-delay:${i*0.04}s">
        <div class="batch-item-indicator"></div>
        <div style="flex:1;min-width:0">
          <div class="batch-item-text">${escHtml(r.text)}</div>
          <div class="batch-item-cats">${r.is_harmful ? "⚠ " + cats : "✓ Safe"}</div>
        </div>
        <div class="batch-item-score">${r.overall_confidence.toFixed(0)}%</div>
        <div class="batch-item-index">#${i+1}</div>
      </div>
    `;
  });

  batchResults.innerHTML = html;
}

// ============================================================
// History
// ============================================================
function addToHistory(text, result) {
  const entry = {
    id:         Date.now(),
    text:       text.slice(0, 200),
    isHarmful:  result.is_harmful,
    confidence: result.overall_confidence,
    flagged:    result.flagged_count,
    labels:     result.labels.length,
    time:       new Date().toLocaleTimeString(),
  };
  analysisHistory.unshift(entry);
  if (analysisHistory.length > 50) analysisHistory = analysisHistory.slice(0, 50);
  localStorage.setItem("cs_history", JSON.stringify(analysisHistory));
  renderHistory();
}

function renderHistory() {
  if (!analysisHistory.length) {
    historyList.innerHTML = `<div class="empty-state"><p>No analyses yet. Start by analyzing some text above.</p></div>`;
    return;
  }

  historyList.innerHTML = analysisHistory.map((entry, i) => `
    <div class="history-item ${entry.isHarmful ? "harmful" : "clean"}"
         style="animation-delay:${i*0.03}s"
         data-text="${escHtml(entry.text)}"
         onclick="loadHistoryEntry(${entry.id})"
         role="button"
         aria-label="Load analysis for: ${escHtml(entry.text.slice(0,40))}">
      <div class="history-item-text">${escHtml(entry.text)}</div>
      <div class="history-item-meta">${entry.time} · ${entry.flagged}/${entry.labels} categories</div>
      <div class="history-item-badge">${entry.isHarmful ? "⚠ Harmful" : "✓ Safe"}</div>
    </div>
  `).join("");
}

function loadHistoryEntry(id) {
  const entry = analysisHistory.find(e => e.id === id);
  if (!entry) return;
  inputText.value = entry.text;
  updateCharCount();
  document.getElementById("analyzer").scrollIntoView({ behavior: "smooth" });
  setTimeout(() => runAnalysis(), 400);
}

// ============================================================
// Utilities
// ============================================================
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function updateCharCount() {
  const len = inputText.value.length;
  charCount.textContent = `${len.toLocaleString()} / 10,000`;
  charCount.style.color = len > 9000 ? "var(--red)" : len > 7500 ? "var(--amber)" : "var(--text-muted)";
}

function updateBatchLineCount() {
  const lines = batchText.value.split("\n").filter(l => l.trim()).length;
  batchLineCount.textContent = `${lines} line${lines !== 1 ? "s" : ""}`;
  batchLineCount.style.color = lines > 45 ? "var(--red)" : "var(--text-muted)";
}

// ============================================================
// Event listeners
// ============================================================
inputText.addEventListener("input", updateCharCount);
batchText.addEventListener("input", updateBatchLineCount);

analyzeBtn.addEventListener("click", runAnalysis);
clearBtn.addEventListener("click", () => {
  inputText.value = "";
  updateCharCount();
  resultsIdle.classList.remove("hidden");
  resultsLoading.classList.add("hidden");
  resultsContent.classList.add("hidden");
  resultsIdle.querySelector(".idle-message").textContent =
    "Enter text on the left and click Analyze to see the threat assessment.";
});

// Submit on Ctrl/Cmd+Enter
inputText.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runAnalysis();
});

// Example buttons
document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    inputText.value = btn.dataset.text;
    updateCharCount();
    inputText.focus();
  });
});

batchBtn.addEventListener("click", runBatch);

clearHistBtn.addEventListener("click", () => {
  analysisHistory = [];
  localStorage.removeItem("cs_history");
  renderHistory();
});

copyExampleBtn.addEventListener("click", () => {
  navigator.clipboard.writeText(exampleCode.textContent).then(() => {
    copyExampleBtn.textContent = "Copied!";
    setTimeout(() => copyExampleBtn.textContent = "Copy", 2000);
  });
});

// Smooth nav highlighting on scroll
const navLinks  = document.querySelectorAll(".nav-link");
const sections  = ["analyzer", "batch", "about"].map(id => document.getElementById(id));
window.addEventListener("scroll", () => {
  const scrollY = window.scrollY + 120;
  let active = "analyzer";
  sections.forEach(sec => { if (sec && sec.offsetTop <= scrollY) active = sec.id; });
  navLinks.forEach(lnk => {
    lnk.classList.toggle("active", lnk.getAttribute("href") === `#${active}`);
  });
}, { passive: true });

// ============================================================
// Init
// ============================================================
(async function init() {
  updateCharCount();
  updateBatchLineCount();
  renderHistory();
  await checkHealth();
  await loadStats();
  setInterval(checkHealth, 30000);
})();
