// Turn the API-format graph into a normal ComfyUI workflow with App Mode set up, using ComfyUI's own frontend (so the
// saved file is exactly what the editor writes). Then reload it in a fresh browser, screenshot App Mode as proof.
// Pattern from tiny-world / underdog-story tools/make_app.mjs.
// node tools/make_app.mjs <comfy_url> <api.json> <out_workflow.json> <screenshot.png> [--run]
//   --run: also press Run in App Mode and wait for the Gate verdict (CPU, a few minutes).
import { chromium } from "playwright";
import fs from "fs";

const [url, apiPath, outPath, shot, ...flags] = process.argv.slice(2);
const RUN = flags.includes("--run");
const api = JSON.parse(fs.readFileSync(apiPath, "utf8"));
const ids = (cls) => Object.entries(api).filter(([, n]) => n.class_type === cls).map(([k]) => k);
const spec = ids("ShotGuardSpec")[0];
const take = ids("LoadVideo")[0] || ids("ShotGuardMockGenerator")[0];
const takeWidget = ids("LoadVideo")[0] ? "file" : "takes";
const ref = ids("LoadImage")[0];
// What a user sees in App Mode: the take, the contract, the retry limits. Everything else stays locked in the graph.
const INPUTS = [
  [take, takeWidget], [spec, "take_name"], [spec, "script"], [spec, "character_1"], [spec, "side_1"], [ref, "image"],
  [spec, "character_2"], [spec, "side_2"], [spec, "allowed_text"], [spec, "locked_facts"], [spec, "max_retries"], [spec, "credit_cap"],
  [spec, "cost_per_take"], [ids("ShotGuardReport")[0], "blur_faces"], [ids("ShotGuardRetry")[0], "enabled"],
];
const OUTPUTS = [ids("ShotGuardReport")[0], ids("PreviewAny")[0], ids("ShotGuardRetry")[0]];
const LABELS = { file: "Your take (any generator's video)", takes: "Mock generator takes (seed picks one)",
  take_name: "Take name", script: "Script, in order (Speaker: line)", character_1: "Character 1 name",
  side_1: "Character 1 stays on", image: "Character 1 reference image", character_2: "Character 2 name (optional)",
  side_2: "Character 2 stays on", allowed_text: "Words allowed on screen (one per line)",
  locked_facts: "Locked facts for the vision judge (optional, one per line)", max_retries: "Max retries",
  credit_cap: "Credit cap (0 = none)", cost_per_take: "Cost per take", blur_faces: "Blur faces on the report",
  enabled: "Auto-retry with fix hints" };

const browser = await chromium.launch();
const errors = [];
const open = async (ctx) => {
  const p = await ctx.newPage();
  p.on("pageerror", (e) => errors.push(e.message));
  if (!RUN) await p.route("**/prompt", (r) => (r.request().method() === "POST" ? r.abort() : r.continue()));
  await p.goto(url, { waitUntil: "load", timeout: 120000 });
  await p.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
  return p;
};
const page = await open(await browser.newContext({ viewport: { width: 1440, height: 1200 } }));
const wf = await page.evaluate(async ({ api, INPUTS, OUTPUTS, LABELS }) => {
  await window.app.loadApiJson(api, "shot-guard");
  const g = window.app.graph;
  if (g.arrange) g.arrange(80);
  for (const node of g.nodes) {
    const a = api[String(node.id)];
    if (a && a._meta && a._meta.title) node.title = a._meta.title;
    for (const w of node.widgets || []) {
      if (w.name === "control_after_generate") w.value = "fixed";
      if (INPUTS.some(([n, name]) => String(n) === String(node.id) && name === w.name) && LABELS[w.name]) {
        w.label = LABELS[w.name];
        for (const inp of node.inputs || []) if (inp.widget && inp.widget.name === w.name) inp.label = LABELS[w.name];
      }
    }
  }
  const w = g.serialize();
  w.extra = w.extra || {};
  w.extra.linearMode = true;
  w.extra.linearData = { inputs: INPUTS.map(([n, name]) => [Number(n), name]), outputs: OUTPUTS.map(Number) };
  return w;
}, { api, INPUTS, OUTPUTS, LABELS });
const missing = INPUTS.filter(([n]) => !wf.nodes.find((x) => x.id === Number(n)));
const unknown = wf.nodes.filter((n) => n.type && !n.widgets_values && !n.inputs).map((n) => n.type);
console.log(`nodes ${wf.nodes.length}, links ${wf.links.length}, app inputs ${INPUTS.length}, outputs ${OUTPUTS.length}, missing ${JSON.stringify(missing)} ${JSON.stringify(unknown)}`);
fs.writeFileSync(outPath, JSON.stringify(wf, null, 1));
await page.close();
// Open the saved file the way a new user would: a fresh browser context, then load it. It opens straight in App Mode.
const page2 = await open(await browser.newContext({ viewport: { width: 1440, height: 1200 } }));
await page2.evaluate(async (w) => { await window.app.loadGraphData(w, true, true, "shot-guard"); }, wf);
await page2.waitForTimeout(4000);
try { await page2.click("text=Skip", { timeout: 5000 }); } catch {}   // first-visit App Mode tour
await page2.waitForTimeout(1500);
const nodesNow = await page2.evaluate(() => window.app.graph.nodes.length);
const body = await page2.evaluate(() => document.body.innerText);
const inApp = /Your take|Mock generator takes/.test(body) && /Character 1 stays on/.test(body);
await page2.screenshot({ path: shot });
console.log("reloaded nodes", nodesNow, "app mode visible", inApp, "page errors", errors.length, errors.slice(0, 3));
if (RUN) {
  // Press Run like a user, then wait until this ComfyUI's queue is empty (Retry may queue more attempts).
  await page2.getByRole("button", { name: /^Run/ }).first().click();
  const t0 = Date.now();
  let seen = false, idle = 0, runs = 0;
  while (Date.now() - t0 < 1800000 && idle < 3) {
    await page2.waitForTimeout(5000);
    let n = 1;
    try { const q = await (await page2.request.get(url + "/queue")).json(); n = q.queue_running.length + q.queue_pending.length; }
    catch { continue; }   // the server can be busy mid-run; ask again
    if (n) { seen = true; idle = 0; } else if (seen) idle += 1;
  }
  try { runs = Object.keys(await (await page2.request.get(url + "/history")).json()).length; } catch (e) { runs = "unreadable: " + e.message.slice(0, 80); }
  await page2.waitForTimeout(2000);
  await page2.screenshot({ path: shot.replace(/\.png$/, "-result.png") });
  const text = await page2.evaluate(() => document.body.innerText);
  console.log("runs in history", runs, "seconds", Math.round((Date.now() - t0) / 1000));
  console.log((text.match(/(PASS|RETRY|FAIL)[^\n]{0,160}/g) || []).slice(0, 6).join(" | "));
}
await browser.close();
