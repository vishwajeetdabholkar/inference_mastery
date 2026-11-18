#!/usr/bin/env python3
"""Render every module into ONE self-contained index.html (works offline, over file://).

The markdown is embedded as JSON and rendered in the browser by a vendored copy of
marked (site/vendor/marked.min.js, MIT). Re-run after editing any .md file:

    python3 site/build_site.py          # writes ../index.html next to README.md
"""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGES = [  # (file, sidebar label) in reading order
    ("README.md", "Overview"),
    ("00_setup.md", "00 · Setup"),
    ("01_engines_and_serving.md", "01 · Engines & Serving"),
    ("02_inference_optimization.md", "02 · Optimization"),
    ("03_models_gateways_harnesses.md", "03 · Models & Gateways"),
    ("04_production_lifecycle.md", "04 · Production"),
    ("05_case_studies.md", "05 · Case Studies"),
    ("06_interview_drills.md", "06 · Interview Drills"),
]


def safe_js(text):
    # a literal "</script" would close the inline <script> tag early ("<\/" is equivalent in JS)
    return text.replace("</script", "<\\/script")


def safe_json(obj):
    # \u003c is valid JSON for "<", so no "</script" or "<!--" can appear in the embedded data
    return json.dumps(obj).replace("<", "\\u003c")


def main():
    docs = {}
    for name, label in PAGES:
        docs[name] = {"label": label, "md": (ROOT / name).read_text()}
    marked = (ROOT / "site" / "vendor" / "marked.min.js").read_text()
    html = TEMPLATE.replace("/*MARKED*/", safe_js(marked)).replace("/*DOCS*/", safe_json(docs))
    out = ROOT / "index.html"
    out.write_text(html)
    assert "/*MARKED*/" not in html and "/*DOCS*/" not in html
    assert len(re.findall(r"</script>", html)) == 3, "unexpected </script> count: escaping broke"
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, {len(docs)} pages)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inference Mastery</title>
<style>
:root{--bg:#fbfaf7;--panel:#f2efe8;--ink:#1d1f23;--muted:#666a73;--line:#e2ddd2;--accent:#b4432b;--code:#f0ece3;--link:#1f5f9e}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--panel:#1c1f23;--ink:#e6e4df;--muted:#9a9ea6;--line:#2c3036;--accent:#e2735a;--code:#22262b;--link:#7fb2e5}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
.layout{display:grid;grid-template-columns:270px minmax(0,1fr) 230px;min-height:100vh}
nav.side{position:sticky;top:0;height:100vh;overflow:auto;background:var(--panel);border-right:1px solid var(--line);padding:22px 14px}
nav.side h1{font-size:15px;letter-spacing:.06em;text-transform:uppercase;margin:0 6px 4px}
nav.side .sub{font-size:12px;color:var(--muted);margin:0 6px 18px}
nav.side a{display:block;padding:7px 10px;border-radius:7px;color:var(--ink);text-decoration:none;font-size:14.5px}
nav.side a:hover{background:var(--line)}
nav.side a.active{background:var(--accent);color:#fff}
nav.side input{width:100%;padding:8px 10px;margin:0 0 14px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--ink);font:inherit;font-size:14px}
main{padding:40px 56px 120px;max-width:920px}
aside.toc{position:sticky;top:0;height:100vh;overflow:auto;padding:40px 18px;font-size:13px;border-left:1px solid var(--line)}
aside.toc b{display:block;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-size:11px;margin-bottom:8px}
aside.toc a{display:block;color:var(--muted);text-decoration:none;padding:3px 0}
aside.toc a:hover{color:var(--accent)}
h1,h2,h3{line-height:1.25;scroll-margin-top:16px}
main h1{font-size:30px;margin-top:0}
main h2{font-size:22px;margin-top:2.2em;padding-bottom:6px;border-bottom:1px solid var(--line)}
main h3{font-size:17.5px;margin-top:1.7em}
a{color:var(--link)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:14px 16px;border-radius:9px;overflow:auto;position:relative}
pre code{padding:0;background:none}
pre button{position:absolute;top:8px;right:8px;font-size:12px;padding:3px 8px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--muted);cursor:pointer;opacity:0;transition:opacity .15s}
pre:hover button{opacity:1}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:14.5px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}
th{background:var(--panel)}
blockquote{margin:1em 0;padding:4px 16px;border-left:4px solid var(--accent);background:var(--panel);border-radius:0 8px 8px 0}
details{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 14px;margin:1em 0}
summary{cursor:pointer;font-weight:600}
li input[type=checkbox]{margin-right:8px;transform:scale(1.15);cursor:pointer}
.pager{display:flex;justify-content:space-between;margin-top:60px;padding-top:20px;border-top:1px solid var(--line)}
.pager a{text-decoration:none;font-weight:600}
.progress{font-size:12px;color:var(--muted);margin:-8px 6px 14px}
@media (max-width:1150px){.layout{grid-template-columns:240px minmax(0,1fr)}aside.toc{display:none}}
@media (max-width:760px){.layout{display:block}nav.side{position:static;height:auto}main{padding:24px 18px 80px}}
</style>
</head>
<body>
<div class="layout">
  <nav class="side">
    <h1>Inference Mastery</h1>
    <p class="sub">Production LLM inference runbook</p>
    <input id="q" placeholder="Search all modules…" autocomplete="off">
    <div class="progress" id="progress"></div>
    <div id="nav"></div>
  </nav>
  <main id="content"></main>
  <aside class="toc" id="toc"></aside>
</div>
<script>/*MARKED*/</script>
<script id="docs" type="application/json">/*DOCS*/</script>
<script>
const DOCS = JSON.parse(document.getElementById('docs').textContent);
const ORDER = Object.keys(DOCS);
const slug = s => s.toLowerCase().replace(/<[^>]+>/g,'').replace(/[^\w\s-]/g,'').trim().replace(/\s+/g,'-');
const store = k => 'im-check:' + k;

function render(page, anchor) {
  if (!DOCS[page]) page = ORDER[0];
  const main = document.getElementById('content');
  main.innerHTML = marked.parse(DOCS[page].md, {gfm: true});
  // heading ids + outline
  const toc = ['<b>On this page</b>'];
  main.querySelectorAll('h2,h3').forEach(h => {
    h.id = slug(h.textContent);
    if (h.tagName === 'H2') toc.push(`<a href="#${page}::${h.id}">${h.textContent}</a>`);
  });
  document.getElementById('toc').innerHTML = toc.length > 1 ? toc.join('') : '';
  // links between modules: "02_x.md#y" -> in-page route
  main.querySelectorAll('a[href]').forEach(a => {
    const m = a.getAttribute('href').match(/^([\w.-]+\.md)(?:#(.*))?$/);
    if (m && DOCS[m[1]]) a.setAttribute('href', `#${m[1]}${m[2] ? '::' + m[2] : ''}`);
    else if (/^https?:/.test(a.getAttribute('href'))) a.target = '_blank';
  });
  // copy buttons on code blocks
  main.querySelectorAll('pre').forEach(pre => {
    const b = document.createElement('button'); b.textContent = 'copy';
    b.onclick = () => navigator.clipboard.writeText(pre.innerText.replace(/copy$/, '').trim())
      .then(() => { b.textContent = 'copied'; setTimeout(() => b.textContent = 'copy', 1200); });
    pre.appendChild(b);
  });
  // checklists persist in localStorage
  main.querySelectorAll('li input[type=checkbox]').forEach((cb, i) => {
    const key = store(page + ':' + i);
    cb.disabled = false; cb.checked = localStorage.getItem(key) === '1';
    cb.onchange = () => { localStorage.setItem(key, cb.checked ? '1' : '0'); updateProgress(); };
  });
  // prev / next
  const i = ORDER.indexOf(page), prev = ORDER[i - 1], next = ORDER[i + 1];
  main.insertAdjacentHTML('beforeend', `<div class="pager"><span>${prev ? `<a href="#${prev}">← ${DOCS[prev].label}</a>` : ''}</span>
    <span>${next ? `<a href="#${next}">${DOCS[next].label} →</a>` : ''}</span></div>`);
  document.querySelectorAll('#nav a').forEach(a => a.classList.toggle('active', a.dataset.page === page));
  document.title = DOCS[page].label + ' · Inference Mastery';
  const target = anchor && document.getElementById(anchor);
  target ? target.scrollIntoView() : window.scrollTo(0, 0);
}

function updateProgress() {
  const keys = Object.keys(localStorage).filter(k => k.startsWith('im-check:'));
  const done = keys.filter(k => localStorage.getItem(k) === '1').length;
  document.getElementById('progress').textContent = done ? `${done} checklist item${done > 1 ? 's' : ''} done` : '';
}

function drawNav(filter) {
  const q = (filter || '').toLowerCase();
  document.getElementById('nav').innerHTML = ORDER
    .filter(p => !q || DOCS[p].md.toLowerCase().includes(q) || DOCS[p].label.toLowerCase().includes(q))
    .map(p => {
      const hits = q ? (DOCS[p].md.toLowerCase().split(q).length - 1) : 0;
      return `<a href="#${p}" data-page="${p}">${DOCS[p].label}${hits ? ` <small>(${hits})</small>` : ''}</a>`;
    }).join('') || '<p class="sub">No matches</p>';
  const [page] = location.hash.slice(1).split('::');
  document.querySelectorAll('#nav a').forEach(a => a.classList.toggle('active', a.dataset.page === (page || ORDER[0])));
}

function route() {
  const [page, anchor] = decodeURIComponent(location.hash.slice(1)).split('::');
  render(page || ORDER[0], anchor);
}
document.getElementById('q').addEventListener('input', e => drawNav(e.target.value));
window.addEventListener('hashchange', route);
drawNav(); updateProgress(); route();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
