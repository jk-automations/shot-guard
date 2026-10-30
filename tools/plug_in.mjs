// Wire Shot Guard after ANY workflow's video output, in the real ComfyUI frontend, and screenshot it. Nothing is queued.
// Finds the SaveVideo (or other video sink) node, takes whatever feeds it, and adds Spec -> 5 checks -> Gate ->
// Report + Retry on that same video. Works on API-format graphs and on normal (UI) workflow files.
// node tools/plug_in.mjs <comfy_url> <workflow.json> <out_workflow.json> <screenshot.png>
import { chromium } from "playwright";
import fs from "fs";

const [url, src, outPath, shot] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(src, "utf8"));
const isUi = Array.isArray(data.nodes);
const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1800, height: 1000 } })).newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await page.route("**/prompt", (r) => (r.request().method() === "POST" ? r.abort() : r.continue()));   // never queue
await page.goto(url, { waitUntil: "load", timeout: 120000 });
await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
const info = await page.evaluate(async ({ data, isUi }) => {
  const app = window.app;
  if (isUi) { const d = structuredClone(data); if (d.extra) { delete d.extra.linearMode; } await app.loadGraphData(d, true, true, "plug-in"); }
  else await app.loadApiJson(data, "plug-in");
  const g = app.graph;
  if (!isUi && g.arrange) g.arrange(80);
  const SINKS = ["SaveVideo", "VHS_VideoCombine", "PreviewVideo"];
  // the most downstream sink (rightmost after arrange): the final cut, not a preview of one shot
  const sink = g.nodes.filter((n) => SINKS.includes(n.type)).sort((a, b) => b.pos[0] - a.pos[0])[0];
  if (!sink) return { error: "no video sink found", types: [...new Set(g.nodes.map((n) => n.type))] };
  const inSlot = sink.inputs.findIndex((i) => i.type === "VIDEO" || i.name === "video" || i.name === "images");
  const link = g.links[sink.inputs[inSlot].link] || g.links.get?.(sink.inputs[inSlot].link);
  const from = g.getNodeById(link.origin_id), fromSlot = link.origin_slot;
  const x0 = Math.max(...g.nodes.map((n) => n.pos[0] + n.size[0])) + 120, y0 = sink.pos[1];
  const add = (type, x, y, title) => { const n = LiteGraph.createNode(type); n.pos = [x, y]; if (title) n.title = title; g.add(n); return n; };
  const spec = add("ShotGuardSpec", x0, y0, "ShotGuard Spec");
  const names = [["ShotGuardDialogue", "dialogue"], ["ShotGuardIdentity", "identity"], ["ShotGuardBlocking", "blocking"],
    ["ShotGuardText", "text"], ["ShotGuardCuts", "cuts"], ["ShotGuardFacts", "facts"]];
  const gate = add("ShotGuardGate", x0 + 900, y0);
  let y = y0;
  for (const [type, key] of names) {
    const c = add(type, x0 + 480, y);
    from.connect(fromSlot, c, c.findInputSlot("video"));
    spec.connect(0, c, c.findInputSlot("spec"));
    c.connect(0, gate, gate.findInputSlot(key));
    y += c.size[1] + 40;
  }
  spec.connect(0, gate, gate.findInputSlot("spec"));
  const rep = add("ShotGuardReport", x0 + 1300, y0);
  gate.connect(0, rep, rep.findInputSlot("result"));
  const ret = add("ShotGuardRetry", x0 + 1300, y0 + 260);
  gate.connect(0, ret, ret.findInputSlot("result"));
  from.connect(fromSlot, ret, ret.findInputSlot("video"));
  // Frame the producer node, its sink and the new Shot Guard block.
  const box = [from, sink, spec, gate, rep, ret];
  const minX = Math.min(...box.map((n) => n.pos[0])) - 60, minY = Math.min(...box.map((n) => n.pos[1])) - 80;
  const maxX = Math.max(...box.map((n) => n.pos[0] + n.size[0])) + 60, maxY = Math.max(y, ...box.map((n) => n.pos[1] + n.size[1])) + 60;
  const c = app.canvas, W = c.canvas.width / (window.devicePixelRatio || 1), H = c.canvas.height / (window.devicePixelRatio || 1);
  const s = Math.min(W / (maxX - minX), H / (maxY - minY), 1.0);
  c.low_quality_zoom_threshold = 0.3; c.min_font_size_for_lod = 1;   // keep full detail when zoomed out
  if (window.app.ui?.settings?.setSettingValueAsync) { try { await window.app.ui.settings.setSettingValueAsync("LiteGraph.Canvas.MinFontSizeForLOD", 1); } catch {} }
  c.ds.scale = s; c.ds.offset = [-minX + (W / s - (maxX - minX)) / 2, -minY + (H / s - (maxY - minY)) / 2];
  c.setDirty(true, true);
  const w = g.serialize();
  return { w, from: from.type, sink: sink.type, nodes: g.nodes.length,
           missing: g.nodes.filter((n) => n.constructor && n.constructor.name === "LGraphNode" && !LiteGraph.registered_node_types[n.type]).map((n) => n.type) };
}, { data, isUi });
if (info.error) { console.log(JSON.stringify(info)); process.exit(1); }
fs.writeFileSync(outPath, JSON.stringify(info.w, null, 1));
await page.waitForTimeout(2500);
await page.screenshot({ path: shot });
console.log(`video from ${info.from} (also feeding ${info.sink}); nodes ${info.nodes}; unregistered ${JSON.stringify(info.missing)}; page errors ${errors.length} ${errors.slice(0, 2)}`);
await browser.close();
