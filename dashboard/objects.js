// Shared helpers for the objects/runs pages. Everything user- or agent-provided is inserted with
// textContent (el()) or esc(); hrefs from data go through safeUrl().
const $ = (s, r = document) => r.querySelector(s);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

function safeUrl(u) {
  try { const x = new URL(u, location.href); return ["http:", "https:"].includes(x.protocol) ? x.href : null; }
  catch { return null; }
}

async function api(path, opts = {}) {
  if (opts.json !== undefined) {
    opts.body = JSON.stringify(opts.json);
    opts.headers = {"Content-Type": "application/json", ...(opts.headers || {})};
    delete opts.json;
  }
  const r = await fetch(path, opts);
  const text = await r.text();
  let data; try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!r.ok) throw new Error((data && data.detail) || `${r.status} ${r.statusText}`);
  return data;
}

const dataUrl = p => p ? "/objdata/" + p.split("/").map(encodeURIComponent).join("/") : null;
const thumbUrl = p => p && p.startsWith("photos/") ? dataUrl(p.replace(/\.jpg$/, ".thumb.jpg")) : dataUrl(p);
const fmtDims = d => Array.isArray(d) && d.length ? d.map(x => +(+x).toFixed(1)).join(" × ") + " mm" : "dims unknown";
const fmtMs = ms => ms == null ? "" : ms < 1000 ? `${ms} ms` : ms < 60000 ? `${(ms / 1000).toFixed(1)} s` : `${(ms / 60000).toFixed(1)} min`;
const fmtTime = t => t ? new Date(t).toLocaleString([], {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit"}) : "";
const STATUS_TAG = {verified: "good", model_found: "acc", identified: "acc", draft: "warn", rejected: "crit"};

function img(src, empty) {
  return src ? el("div", {}, el("img", {src, alt: "", loading: "lazy", onerror: e => { e.target.replaceWith(empty); }}))
             : el("div", {text: empty});
}

function notice(container, msg, isErr = true) {
  const n = el("div", {class: "notice" + (isErr ? " err" : ""), text: msg});
  container.prepend(n);
  setTimeout(() => n.remove(), 6000);
}

function navHeader(active) {
  const h = el("header", {},
    el("h1", {text: "objsearch"}),
    el("nav", {},
      el("a", {href: "/", text: "Box"}),
      el("a", {href: "/objects", class: active === "objects" ? "on" : null, text: "Objects"}),
      el("a", {href: "/objects/identify", class: active === "identify" ? "on" : null, text: "Identify"}),
      el("a", {href: "/objects/parts", class: active === "parts" ? "on" : null, text: "Parts"}),
      el("a", {href: "/objects/runs", class: active === "runs" ? "on" : null, text: "Runs"})),
    el("span", {id: "hdrInfo", class: "muted", style: "margin-left:auto"}));
  document.body.prepend(h);
}

// replaceChildren() that skips null/false (conditional children), instead of printing "null".
function fill(node, ...kids) { node.replaceChildren(...kids.flat().filter(k => k != null && k !== false)); }

// Card image slot: your own photo if there is one, otherwise the public reference image (labelled).
function photoOrRef(o) {
  if (o.photo_path) return img(thumbUrl(o.photo_path), "no photo");
  if (!o.ref_path) return img(null, "no photo");
  const d = img(thumbUrl(o.ref_path), "no photo");
  d.title = o.ref_attribution || "public reference image";
  d.append(el("span", {class: "refbadge", text: "reference"}));
  return d;
}

function credit(r) {
  const lic = `CC ${(r.license || "").toUpperCase()} ${r.license_version || ""}`.trim();
  const page = safeUrl(r.landing_url), lurl = safeUrl(r.license_url);
  return el("div", {class: "credit"},
    page ? el("a", {href: page, target: "_blank", rel: "noopener noreferrer", text: r.title || "source"}) : (r.title || ""),
    r.creator ? ` by ${r.creator}` : "", " · ",
    lurl ? el("a", {href: lurl, target: "_blank", rel: "noopener noreferrer", text: lic}) : lic,
    r.provider ? ` · via ${r.provider.replace("openverse/", "Openverse / ")}` : "");
}

// SO-ARM101 graspability (objsearch/grasp.py): good / borderline / no, with reasons on hover.
const GRIP = {good: ["good", "gripper: good"], borderline: ["warn", "gripper: borderline"], no: ["crit", "gripper: can't pick"]};
function gripTag(o) {
  const g = o.grasp || {};
  if (!g.verdict) return el("span", {class: "tag", text: "gripper: size unknown"});
  const [cls, text] = GRIP[g.verdict];
  return el("span", {class: "tag " + cls, text,
    title: [`${g.height_mm} mm tall, ${g.grasp_width_mm} mm across the jaws`, ...g.reasons].join("\n")});
}

// Model quality tag for cards: approximate / missing parts recorded when the 3D model was built.
function modelTag(o) {
  const f = (o.model_scores && o.model_scores.flags) || [];
  const approx = f.filter(x => x.startsWith("approximate part")).length, missing = f.filter(x => x.startsWith("missing part")).length;
  if (!o.asset_count) return null;
  if (!approx && !missing) return el("span", {class: "tag good", text: "3D model", title: "3D model with no flagged parts"});
  return el("span", {class: "tag warn", text: `3D: ${approx ? approx + " approx" : ""}${approx && missing ? ", " : ""}${missing ? missing + " missing" : ""}`,
    title: f.join("\n")});
}
