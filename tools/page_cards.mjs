// Export each card of the pipeline design page as its own image, for articles: PNG (2x) and SVG (vector, via PDF).
//   node tools/page_cards.mjs [PAGE.html] [OUT_DIR] [--dark] [--scale 2] [--all]
// Defaults: docs/design/20261107-pipeline.html -> pieces/_bench/review/pipeline-cards/
// Cards: every article.stage (Who's who, Words used, tasks 0-8). --all adds the detail strips (one per task),
// the hardware and ComfyUI sections. Each card is rendered alone at the page's own width (cards are 720 px),
// so a card longer than the screen comes out whole. Needs google-chrome and pdftocairo (poppler); no npm packages.
import { spawn, execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync, mkdtempSync, rmSync } from "node:fs";
import { resolve, join } from "node:path";
import { tmpdir } from "node:os";

const args = process.argv.slice(2);
const flag = (f) => { const i = args.indexOf(f); if (i < 0) return null; args.splice(i, 1); return true; };
const opt = (f, d) => { const i = args.indexOf(f); if (i < 0) return d; const v = args[i + 1]; args.splice(i, 2); return v; };
const DARK = flag("--dark"), ALL = flag("--all"), SCALE = Number(opt("--scale", 2));
const PAGE = resolve(args[0] || "docs/design/20261107-pipeline.html");
const OUT = resolve(args[1] || "pieces/_bench/review/pipeline-cards");
mkdirSync(OUT, { recursive: true });
const SEL = ALL ? "article.stage, div.phase, section.hw, section#comfyui" : "article.stage";

const prof = mkdtempSync(join(tmpdir(), "cards-chrome-"));
const chrome = spawn("google-chrome", ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${prof}`,
  "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=1200,900", "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
const wsUrl = await new Promise((ok, no) => {
  let buf = ""; chrome.stderr.on("data", (d) => { buf += d; const m = buf.match(/ws:\/\/\S+/); if (m) ok(m[0]); });
  setTimeout(() => no(new Error("chrome did not start")), 20000);
});
const port = new URL(wsUrl).port;
const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: "PUT" })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((ok) => ws.addEventListener("open", ok));
let id = 0; const wait = new Map(); const events = [];
ws.addEventListener("message", (m) => { const d = JSON.parse(m.data); if (d.id && wait.has(d.id)) { wait.get(d.id)(d); wait.delete(d.id); } else events.push(d); });
const cdp = (method, params = {}) => new Promise((ok, no) => { const n = ++id; wait.set(n, (d) => d.error ? no(new Error(`${method}: ${d.error.message}`)) : ok(d.result)); ws.send(JSON.stringify({ id: n, method, params })); });
const js = async (expr) => (await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result.value;

await cdp("Page.enable"); await cdp("Runtime.enable");
await cdp("Emulation.setDeviceMetricsOverride", { width: 1200, height: 900, deviceScaleFactor: SCALE, mobile: false });
await cdp("Emulation.setEmulatedMedia", { media: "screen" });
const load = async () => {
  events.length = 0; await cdp("Page.navigate", { url: "file://" + PAGE });
  for (let t = 0; t < 200 && !events.some((e) => e.method === "Page.loadEventFired"); t++) await new Promise((r) => setTimeout(r, 50));
  await js(`document.documentElement.dataset.theme = ${JSON.stringify(DARK ? "dark" : "light")}; document.fonts.ready.then(() => 1)`);
};
await load();
const cards = await js(`[...document.querySelectorAll(${JSON.stringify(SEL)})].map((el, i) => {
  const h = el.querySelector("h2, h3"), n = el.querySelector(".st-h i, .num");
  return { i, id: el.id || "", title: h ? h.textContent.trim() : "card", n: n ? n.textContent.trim() : "", kind: el.matches("div.phase") ? "detail" : "" };
})`);
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

// Keep only card i (with its ancestors' classes, so the page's selectors still apply); zero the ancestors' spacing.
const isolate = (i) => js(`(() => {
  const el = document.querySelectorAll(${JSON.stringify(SEL)})[${i}];
  let node = el.cloneNode(true);
  for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
    const c = a.cloneNode(false); c.style.cssText += ";margin:0;padding:0;max-width:none;border:0"; c.appendChild(node); node = c;
  }
  document.body.replaceChildren(node);
  document.body.style.cssText += ";margin:0;padding:0";
  const card = node.querySelector ? (node.matches(${JSON.stringify(SEL)}) ? node : node.querySelector(${JSON.stringify(SEL)})) : node;
  card.style.margin = "0";
  const top = document.documentElement; top.style.background = getComputedStyle(document.body).backgroundColor;
  return document.fonts.ready.then(() => { const r = card.getBoundingClientRect();
    card.style.width = r.width + "px";
    return { x: r.left, y: r.top, w: Math.ceil(r.width), h: Math.ceil(r.height) }; });
})()`);

const made = [];
for (const c of cards) {
  await load();
  const r = await isolate(c.i);
  const name = (c.id && /^stage-/.test(c.id) ? `task-${c.n}-` : c.kind ? `detail-` : "") + slug(c.title);
  const png = await cdp("Page.captureScreenshot", { format: "png", captureBeyondViewport: true,
    clip: { x: r.x, y: r.y, width: r.w, height: r.h, scale: 1 } });
  writeFileSync(join(OUT, `${name}.png`), Buffer.from(png.data, "base64"));
  // Vector: centre the card on a page exactly its size, print to PDF, convert to SVG (text becomes outlines).
  await js(`(() => { const s = document.createElement("style");
    s.textContent = "@page { size: ${r.w}px ${r.h + 600}px; margin: 0 } html, body { background: transparent !important }";
    document.head.appendChild(s); window.scrollTo(0, 0); return 1; })()`);
  const pdf = await cdp("Page.printToPDF", { printBackground: true, preferCSSPageSize: true, pageRanges: "1",
    marginTop: 0, marginBottom: 0, marginLeft: 0, marginRight: 0 });
  const pdfPath = join(prof, `${name}.pdf`);
  writeFileSync(pdfPath, Buffer.from(pdf.data, "base64"));
  // Print lays out slightly taller than screen, so the page has spare height; measure the card in print, then crop.
  const svgPath = join(OUT, `${name}.svg`);
  execFileSync("pdftocairo", ["-svg", pdfPath, svgPath]);
  // Crop height: the last pixel row of the printed page that isn't the empty page fill (rendered at 96 dpi = CSS px).
  execFileSync("pdftoppm", ["-r", "96", "-gray", "-singlefile", pdfPath, pdfPath]);
  const pgm = readFileSync(pdfPath + ".pgm"), hdr = pgm.toString("latin1", 0, 64).split(/\s+/);
  const [pw, phh] = [Number(hdr[1]), Number(hdr[2])], px = pgm.subarray(pgm.length - pw * phh);
  const fill = px[(phh - 1) * pw + Math.floor(pw / 2)];
  let H = phh; while (H > 1 && px.subarray((H - 1) * pw, H * pw).every((v) => Math.abs(v - fill) < 3)) H--;
  const pt = (v) => (v * 0.75).toFixed(2);
  const svg = readFileSync(svgPath, "utf8").replace(/<svg([^>]*?) width="[^"]*" height="[^"]*" viewBox="[^"]*"/,
    `<svg$1 width="${pt(r.w)}pt" height="${pt(H)}pt" viewBox="0 0 ${pt(r.w)} ${pt(H)}"`);
  writeFileSync(svgPath, svg);
  made.push(`${name}  ${r.w}x${r.h}`);
  console.log(`${name}.png/.svg  ${r.w}x${r.h} px (PNG at ${SCALE}x)`);
}
writeFileSync(join(OUT, "README.txt"), `Made by tools/page_cards.mjs from ${PAGE}\n${new Date().toISOString()}\n` + made.join("\n") + "\n");
ws.close(); chrome.kill();
await new Promise((ok) => { chrome.on("exit", ok); setTimeout(ok, 5000); });
rmSync(prof, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
console.log(`-> ${OUT}`);
