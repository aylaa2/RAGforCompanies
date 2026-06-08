/* ===================================================================
   Scutul Carpatic front-end logic
   Ask    : one /query call, full pipeline (semantic + bm25 + reranker)
   Compare: three /query calls with different toggles, side by side
   =================================================================== */

function esc(s) {
  return String(s).replace(/[&<>"']/g, m =>
    ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));
}

async function postQuery(query, useBm25, useReranker, useIterative) {
  const body = { query };
  if (useBm25 !== undefined) body.use_bm25 = useBm25;
  if (useReranker !== undefined) body.use_reranker = useReranker;
  if (useIterative !== undefined) body.use_iterative = useIterative;
  const r = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

function errorHTML(msg) {
  return '<div class="error"><b>Ceva n-a mers.</b><br>' + esc(msg) +
    '<br><span style="color:var(--muted)">Verifică: serverul pornit, Ollama pornit și ingestul rulat.</span></div>';
}

/* ---------------- navigation ---------------- */
const nav = document.getElementById("nav");
nav.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-view]");
  if (!b) return;
  nav.querySelectorAll("button").forEach(x => x.classList.remove("active"));
  document.querySelectorAll(".view").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  document.getElementById("view-" + b.dataset.view).classList.add("active");
});

/* ---------------- Ask ---------------- */
const askForm = document.getElementById("askForm");
const qInput = document.getElementById("q");
const askBtn = document.getElementById("ask");
const askOut = document.getElementById("askOut");

async function ask() {
  const q = qInput.value.trim();
  if (!q) return;
  askBtn.disabled = true;
  askOut.innerHTML = '<div class="loading"><span class="spinner"></span>Caut în documente și pregătesc răspunsul…</div>';
  try {
    const d = await postQuery(q);   // no toggles -> server defaults = full pipeline
    let h = '<div class="answer"><p class="tag">Răspuns</p><div class="body">' +
            esc(d.answer || "(fără răspuns)") + "</div></div>";
    const cs = d.chunks || [];
    if (cs.length) {
      h += '<div class="sources"><p class="tag">Surse</p>';
      cs.forEach((c, i) => {
        h += "<details><summary><span class='idx'>" + String(i + 1).padStart(2, "0") +
             "</span><span class='src'>" + esc(c.source || "sursă") +
             "</span><span class='chev'>›</span></summary><div class='tx'>" +
             esc(c.text || "") + "</div></details>";
      });
      h += "</div>";
    }
    askOut.innerHTML = h;
  } catch (e) {
    askOut.innerHTML = errorHTML(e.message);
  } finally {
    askBtn.disabled = false;
  }
}
askForm.addEventListener("submit", (e) => { e.preventDefault(); ask(); });
document.getElementById("chips").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  qInput.value = b.textContent;
  ask();
});

/* ---------------- Compare ---------------- */
const CONFIGS = [
  { name: "Semantic",   bm25: false, rer: false, iter: false, stages: [1, 0, 0, 0] },
  { name: "+ BM25",     bm25: true,  rer: false, iter: false, stages: [1, 1, 0, 0] },
  { name: "+ Reranker", bm25: true,  rer: true,  iter: false, stages: [1, 1, 1, 0] },
  { name: "+ Iterativ", bm25: true,  rer: true,  iter: true,  stages: [1, 1, 1, 1] },
];
const cmpForm = document.getElementById("cmpForm");
const cmpGrid = document.getElementById("cmpGrid");
const crun = document.getElementById("crun");
const legend = document.getElementById("legend");

function colSkeleton(cfg, idx) {
  const stages = cfg.stages.map(s => '<span class="' + (s ? "on" : "") + '"></span>').join("");
  const best = idx === CONFIGS.length - 1;
  return '<div class="col' + (best ? " best" : "") + '" id="col' + idx + '">' +
    '<div class="col-head"><div class="cfg">' + esc(cfg.name) +
      (best ? '<span class="badge">cea mai bună</span>' : "") + "</div>" +
      '<div class="ministages">' + stages + "</div></div>" +
    '<div class="col-ans muted" id="ans' + idx + '"><span class="loading"><span class="spinner"></span>se calculează…</span></div>' +
    '<div class="col-chunks" id="ch' + idx + '"><div class="lbl">Surse găsite</div></div></div>';
}

function chunksHTML(chunks, prevSources) {
  return chunks.map((c, i) => {
    const src = c.source || "sursă";
    let flag = "";
    if (prevSources) {
      const prevRank = prevSources.indexOf(src);
      if (prevRank === -1) flag = '<span class="cflag new">nou</span>';
      else if (i < prevRank) flag = '<span class="cflag up">▲ urcat</span>';
    }
    return '<div class="chunk"><span class="rank">' + (i + 1) + "</span>" +
           '<span class="csrc">' + esc(src) + "</span>" + flag + "</div>";
  }).join("");
}

async function compare() {
  const q = document.getElementById("cq").value.trim();
  if (!q) return;
  crun.disabled = true;
  legend.hidden = true;
  cmpGrid.innerHTML = CONFIGS.map(colSkeleton).join("");

  const sourceLists = [];
  try {
    for (let i = 0; i < CONFIGS.length; i++) {
      const c = CONFIGS[i];
      const d = await postQuery(q, c.bm25, c.rer, c.iter);   // sequential: avoids racing the global config
      const chunks = d.chunks || [];
      const prev = i > 0 ? sourceLists[i - 1] : null;
      document.getElementById("ans" + i).outerHTML =
        '<div class="col-ans" id="ans' + i + '">' + esc(d.answer || "(fără răspuns)") + "</div>";
      document.getElementById("ch" + i).innerHTML =
        '<div class="lbl">Surse găsite</div>' + chunksHTML(chunks, prev);
      sourceLists.push(chunks.map(c => c.source || "sursă"));
    }
    legend.hidden = false;
  } catch (e) {
    cmpGrid.innerHTML = errorHTML(e.message);
  } finally {
    crun.disabled = false;
  }
}
cmpForm.addEventListener("submit", (e) => { e.preventDefault(); compare(); });

/* ---------------- Evaluate (RAGAS results -> chart) ---------------- */
const EV_COLORS = ["#b9b1c2", "#12b3a6", "#ff5d73", "#7c5cff"];  // semantic · +bm25 · +reranker · +iterativ
const EV_INK = "#241f2e", EV_FAINT = "#a59cae", EV_LINE = "#e2dccf";
const EV_FONT = "Inter, sans-serif";

async function renderEval() {
  let data;
  try {
    const r = await fetch("/assets/eval_results.json");
    data = await r.json();
  } catch (e) { return; }

  // KPI cards: best precision, best faithfulness, and reranker uplift.
  const m = data.metrics, last = data.configs.length - 1;
  const get = (label) => (m.find(x => x.key === label) || { vals: [0, 0, 0] }).vals;
  const cp = get("Precizie context"), fa = get("Fidelitate");
  const cards = [
    { k: "Precizie context", v: cp[last].toFixed(2), d: "+" + Math.round((cp[last] - cp[0]) * 100) + "% vs. Semantic", good: true },
    { k: "Fidelitate", v: fa[last].toFixed(2), d: "+" + Math.round((fa[last] - fa[0]) * 100) + "% vs. Semantic", good: true },
    { k: "Variante comparate", v: data.configs.length, d: data.configs.join(" · "), good: true },
  ];
  document.getElementById("evCards").innerHTML = cards.map(c =>
    '<div class="card"><div class="k">' + esc(c.k) + '</div><div class="v">' + esc(String(c.v)) +
    '</div><div class="d up-good">' + esc(c.d) + '</div></div>').join("");

  document.getElementById("barChart").innerHTML = barChart(data);
  document.getElementById("evTable").innerHTML = evalTable(data);
}

function barChart(data) {
  const W = 640, H = 270, padL = 34, padB = 42, padT = 10, padR = 12;
  const groups = data.metrics, gn = groups.length, cn = data.configs.length;
  const gw = (W - padL - padR) / gn, bw = gw * 0.64 / cn, gap = (gw - bw * cn) / 2;
  const y = (v) => padT + (H - padT - padB) * (1 - v);
  let s = '<svg viewBox="0 0 ' + W + " " + H + '" width="100%" style="max-width:' + W + 'px">';
  [0, .25, .5, .75, 1].forEach(t => { const yy = y(t);
    s += '<line x1="' + padL + '" x2="' + (W - padR) + '" y1="' + yy + '" y2="' + yy + '" stroke="' + EV_LINE + '"/>' +
         '<text x="' + (padL - 8) + '" y="' + (yy + 3) + '" fill="' + EV_FAINT + '" font-size="9" text-anchor="end" font-family="' + EV_FONT + '">' + t.toFixed(2) + '</text>'; });
  groups.forEach((g, gi) => { const gx = padL + gi * gw + gap;
    g.vals.forEach((v, ci) => { const x = gx + ci * bw, yy = y(v), hh = (H - padT - padB) - (yy - padT);
      s += '<rect x="' + (x + 2) + '" y="' + yy + '" width="' + (bw - 3) + '" height="' + hh + '" rx="3" fill="' + EV_COLORS[ci % EV_COLORS.length] + '"/>'; });
    s += '<text x="' + (gx + bw * cn / 2) + '" y="' + (H - padB + 18) + '" fill="' + EV_INK + '" font-size="10" text-anchor="middle" font-family="' + EV_FONT + '">' + esc(g.key) + '</text>'; });
  return s + "</svg>";
}

function evalTable(data) {
  let h = "<table><thead><tr><th>Variantă</th>" +
    data.metrics.map(x => "<th>" + esc(x.key) + "</th>").join("") + "</tr></thead><tbody>";
  data.configs.forEach((c, ci) => {
    h += "<tr" + (ci === data.configs.length - 1 ? ' class="best"' : "") + "><td>" + esc(c) + "</td>";
    data.metrics.forEach(x => {
      const v = x.vals[ci], base = x.vals[0];
      const delta = ci > 0 ? '<span class="delta">+' + (v - base).toFixed(2) + "</span>" : "";
      h += '<td><span class="cell">' + delta + v.toFixed(2) + "</span></td>";
    });
    h += "</tr>";
  });
  return h + "</tbody></table>";
}

renderEval();
