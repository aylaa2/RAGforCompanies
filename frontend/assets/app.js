/* ===================================================================
   Scutul Carpatic — front-end logic
   Ask    : one /query call, full pipeline (semantic + bm25 + reranker)
   Compare: three /query calls with different toggles, side by side
   =================================================================== */

function esc(s) {
  return String(s).replace(/[&<>"']/g, m =>
    ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));
}

async function postQuery(query, useBm25, useReranker) {
  const body = { query };
  if (useBm25 !== undefined) body.use_bm25 = useBm25;
  if (useReranker !== undefined) body.use_reranker = useReranker;
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
  { name: "Semantic",     bm25: false, rer: false, stages: [1, 0, 0] },
  { name: "+ BM25",       bm25: true,  rer: false, stages: [1, 1, 0] },
  { name: "+ Reranker",   bm25: true,  rer: true,  stages: [1, 1, 1] },
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
      const d = await postQuery(q, c.bm25, c.rer);   // sequential: avoids racing the global config
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
