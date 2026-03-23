/* ================================================================
   PneumoVision — Frontend Logic
================================================================ */

const API_BASE = "http://localhost:8000";

// Firebase web config (saved for reference)
const firebaseConfig = {
  apiKey: "AIzaSyCcPhT-2C_v1qMEsd8WBm1hjKzNq7LZ2Bo",
  authDomain: "pneumovision-b890b.firebaseapp.com",
  projectId: "pneumovision-b890b",
  storageBucket: "pneumovision-b890b.firebasestorage.app",
  messagingSenderId: "222118042373",
  appId: "1:222118042373:web:8347e393b5bfd22e05356c"
};


let selectedFile = null;
let lastResult   = null;
let loadTimers   = [];

// ── Navigation ────────────────────────────────────────────────
document.querySelectorAll(".sidenav-item").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    switchPanel(item.dataset.panel);
  });
});

function switchPanel(name) {
  document.querySelectorAll(".sidenav-item").forEach(i => i.classList.toggle("active", i.dataset.panel === name));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.getElementById(`panel-${name}`).classList.add("active");
  const titles = { upload: ["New Analysis","Upload a chest X-ray image to begin"], history: ["History","Past analyses stored in Firebase"], about: ["About","How PneumoVision works"] };
  document.getElementById("page-title").textContent = titles[name][0];
  document.getElementById("page-sub").textContent   = titles[name][1];
  if (name === "history") loadHistory();
}

// ── File Handling ─────────────────────────────────────────────
function handleDragOver(e) { e.preventDefault(); document.getElementById("dropzone").classList.add("dragging"); }
function handleDragLeave() { document.getElementById("dropzone").classList.remove("dragging"); }
function handleDrop(e) {
  e.preventDefault();
  document.getElementById("dropzone").classList.remove("dragging");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) setFile(f);
}
function handleFileSelect(e) { if (e.target.files[0]) setFile(e.target.files[0]); }

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById("preview-img").src = ev.target.result;
    document.getElementById("scan-img").src    = ev.target.result;
    document.getElementById("dz-idle").classList.add("hidden");
    document.getElementById("dz-preview").classList.remove("hidden");
    document.getElementById("analyse-btn").disabled = false;
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null; lastResult = null;
  document.getElementById("dz-idle").classList.remove("hidden");
  document.getElementById("dz-preview").classList.add("hidden");
  document.getElementById("analyse-btn").disabled = true;
  document.getElementById("results-pane").classList.add("hidden");
  document.getElementById("empty-state").classList.remove("hidden");
  document.getElementById("file-input").value = "";
}

// ── Analysis ──────────────────────────────────────────────────
async function analyseImage() {
  if (!selectedFile) return;
  showLoading();
  const formData = new FormData();
  formData.append("file", selectedFile);
  try {
    const res = await fetch(`${API_BASE}/predict`, { method:"POST", body: formData });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    lastResult = data;
    hideLoading();
    renderResults(data);
  } catch (err) {
    hideLoading();
    alert(`Analysis failed: ${err.message}\n\nMake sure the FastAPI server is running on port 8000.`);
  }
}

// ── Loading ───────────────────────────────────────────────────
const DELAYS = [300, 1100, 2200, 3400, 4500];
function showLoading() {
  document.getElementById("loading-modal").classList.remove("hidden");
  for (let i=1;i<=5;i++) { const el=document.getElementById(`mstep-${i}`); el.classList.remove("active","done"); }
  DELAYS.forEach((d,i) => {
    loadTimers.push(setTimeout(() => {
      if (i > 0) { document.getElementById(`mstep-${i}`).classList.remove("active"); document.getElementById(`mstep-${i}`).classList.add("done"); }
      document.getElementById(`mstep-${i+1}`).classList.add("active");
    }, d));
  });
}
function hideLoading() {
  loadTimers.forEach(clearTimeout); loadTimers=[];
  document.getElementById("loading-modal").classList.add("hidden");
}

// ── Render Results ────────────────────────────────────────────
function renderResults(data) {
  const cls = data.prediction;
  const slug = cls.toLowerCase().replace(/[^a-z]/g,"");
  const pillClass = { "covid19":"pred-covid", "normal":"pred-normal", "pneumonia":"pred-pneumonia" }[slug] || "";

  // Banner
  document.getElementById("result-pill").textContent = cls;
  document.getElementById("result-pill").className = `result-pill ${pillClass}`;
  document.getElementById("result-main-label").textContent = cls;
  document.getElementById("result-sub-label").textContent = `Severity: ${data.severity}  ·  PIS: ${data.pis_score}%`;
  document.getElementById("conf-pct").textContent = `${data.confidence.toFixed(1)}%`;

  // Images
  document.getElementById("img-original").src = `data:image/png;base64,${data.original_b64}`;
  document.getElementById("img-gradcam").src  = `data:image/png;base64,${data.gradcam_b64}`;

  // Probabilities
  const probColours = { "COVID-19":"#DC2626","Normal":"#16A34A","Pneumonia":"#D97706" };
  const probList = document.getElementById("prob-list");
  probList.innerHTML = Object.entries(data.probabilities)
    .sort((a,b) => b[1]-a[1])
    .map(([k,v]) => `
      <div class="prob-item">
        <div class="prob-row">
          <span class="prob-name">${k}</span>
          <span class="prob-pct">${v.toFixed(1)}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width:0%;background:${probColours[k]||'#2563EB'}" data-w="${v}"></div>
        </div>
      </div>`).join("");
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-fill").forEach(el => { el.style.width = el.dataset.w + "%"; });
  });

  // BLAA
  const blaa = data.blaa;
  setTimeout(() => {
    document.getElementById("lbar-left").style.height  = Math.min(blaa.left_lung_pct * 1.8, 100) + "%";
    document.getElementById("lbar-right").style.height = Math.min(blaa.right_lung_pct * 1.8, 100) + "%";
  }, 100);
  document.getElementById("lpct").textContent  = blaa.left_lung_pct + "%";
  document.getElementById("rpct").textContent  = blaa.right_lung_pct + "%";
  document.getElementById("asym-val").textContent = blaa.asymmetry_score.toFixed(1) + "%";
  document.getElementById("pis-val").textContent  = data.pis_score + "%";
  document.getElementById("pattern-note").textContent = blaa.pattern;

  // Recommendations
  document.getElementById("recs-list").innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join("");

  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("results-pane").classList.remove("hidden");
  document.getElementById("results-pane").scrollIntoView({ behavior:"smooth", block:"start" });
}

// ── Image toggle ──────────────────────────────────────────────
function showOrig(btn) {
  document.getElementById("img-original").classList.add("show");
  document.getElementById("img-gradcam").classList.remove("show");
  document.getElementById("img-badge").textContent = "Original";
  document.querySelectorAll(".vtog").forEach((b,i)=>b.classList.toggle("active",i===0));
}
function showGrad(btn) {
  document.getElementById("img-gradcam").classList.add("show");
  document.getElementById("img-original").classList.remove("show");
  document.getElementById("img-badge").textContent = "GradCAM";
  document.querySelectorAll(".vtog").forEach((b,i)=>b.classList.toggle("active",i===1));
}

// ── History ───────────────────────────────────────────────────
async function loadHistory() {
  const el = document.getElementById("history-list");
  el.innerHTML = '<p class="section-sub" style="padding:32px 0">Loading...</p>';
  try {
    const data = await (await fetch(`${API_BASE}/history`)).json();
    if (!data.length) { el.innerHTML = '<p class="section-sub" style="padding:32px 0">No records found yet.</p>'; return; }
    const c = {"COVID-19":"#DC2626","Normal":"#16A34A","Pneumonia":"#D97706"};
    el.innerHTML = data.map(r => `
      <div class="history-item">
        <span class="hist-pred" style="color:${c[r.prediction]||'#2563EB'}">${r.prediction}</span>
        <span class="hist-conf">${r.confidence?.toFixed(1)}% confidence</span>
        <span class="hist-sev sev-${(r.severity||"none").toLowerCase()}">${r.severity||"—"}</span>
        <span class="hist-ts">${fmtTs(r.timestamp)}</span>
      </div>`).join("");
  } catch { el.innerHTML = '<p class="section-sub" style="padding:32px 0">Could not load — is the backend running?</p>'; }
}

function fmtTs(ts) {
  try { return new Date(ts).toLocaleString("en-IN",{dateStyle:"medium",timeStyle:"short"}); } catch { return ts||"—"; }
}

// ── Download Report ───────────────────────────────────────────
function downloadReport() {
  if (!lastResult) return;
  const d = lastResult, ts = new Date().toLocaleString("en-IN");
  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"/><title>PneumoVision Report</title>
<style>
  body{font-family:'Segoe UI',sans-serif;background:#F5F7FA;color:#0F172A;margin:0;padding:40px;max-width:820px;margin:auto}
  .header{background:#1E3A5F;color:#fff;padding:28px 36px;border-radius:10px;margin-bottom:24px}
  h1{margin:0 0 4px;font-size:22px;letter-spacing:-.3px}.sub{font-size:12px;opacity:.6;letter-spacing:.5px}
  .pred-block{display:inline-block;padding:8px 20px;border-radius:6px;font-weight:700;font-size:18px;margin:16px 0}
  .covid{background:#FEF2F2;color:#DC2626}.normal{background:#F0FDF4;color:#16A34A}.pneumonia{background:#FFFBEB;color:#D97706}
  .section{background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:20px 24px;margin-bottom:16px}
  h2{font-size:12px;letter-spacing:1px;color:#94A3B8;margin:0 0 14px;text-transform:uppercase}
  table{width:100%;border-collapse:collapse}td{padding:8px 12px;border-bottom:1px solid #F1F5F9;font-size:13px}
  td:first-child{color:#64748B;width:55%}td:last-child{font-weight:600;font-family:monospace}
  ul{margin:0;padding-left:18px}li{margin:7px 0;font-size:13px;color:#475569}
  .warn{background:#FFFBEB;border:1px solid #FDE68A;border-radius:6px;padding:10px 16px;font-size:12px;color:#92400E;margin-top:20px}
  .footer{text-align:center;font-size:11px;color:#94A3B8;margin-top:28px;letter-spacing:.5px}
</style></head><body>
<div class="header"><div class="sub">PNEUMOVISION · AI PULMONARY DIAGNOSTICS</div><h1>Diagnostic Report</h1><div class="sub">Generated ${ts}</div></div>
<div class="section"><h2>Primary Finding</h2>
<div class="pred-block ${d.prediction.toLowerCase().replace(/[^a-z]/g,'')}">${d.prediction}</div>
<table><tr><td>Confidence</td><td>${d.confidence.toFixed(1)}%</td></tr><tr><td>Severity</td><td>${d.severity}</td></tr><tr><td>Pulmonary Involvement Score</td><td>${d.pis_score}%</td></tr></table></div>
<div class="section"><h2>Class Probabilities</h2><table>
${Object.entries(d.probabilities).map(([k,v])=>`<tr><td>${k}</td><td>${v.toFixed(2)}%</td></tr>`).join("")}
</table></div>
<div class="section"><h2>Bilateral Lung Asymmetry Analysis (BLAA)</h2><table>
<tr><td>Left Lung Involvement</td><td>${d.blaa.left_lung_pct}%</td></tr>
<tr><td>Right Lung Involvement</td><td>${d.blaa.right_lung_pct}%</td></tr>
<tr><td>Asymmetry Score</td><td>${d.blaa.asymmetry_score.toFixed(1)}%</td></tr>
<tr><td>Pattern</td><td>${d.blaa.pattern}</td></tr></table></div>
<div class="section"><h2>Clinical Notes</h2><ul>${d.recommendations.map(r=>`<li>${r}</li>`).join("")}</ul></div>
<div class="warn">⚠ This report is AI-generated for academic/research purposes only. It is not a clinical diagnosis. Consult a qualified radiologist or physician for medical decisions.</div>
<div class="footer">PneumoVision v2.0 · DenseNet121 · GradCAM + BLAA</div>
</body></html>`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([html],{type:"text/html"}));
  a.download = `PneumoVision_Report_${Date.now()}.html`;
  a.click();
}
