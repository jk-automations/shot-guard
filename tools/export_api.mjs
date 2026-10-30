// Load a UI workflow in the real frontend and print its API-format prompt (nothing is queued).
import { chromium } from "playwright";
import fs from "fs";
const [url, src, out] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage();
await page.route("**/prompt", (r) => (r.request().method() === "POST" ? r.abort() : r.continue()));
await page.goto(url, { waitUntil: "load" });
await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 120000 });
const api = await page.evaluate(async (d) => { await window.app.loadGraphData(d, true, true, "x"); const p = await window.app.graphToPrompt(); return p.output; },
  JSON.parse(fs.readFileSync(src, "utf8")));
fs.writeFileSync(out, JSON.stringify(api, null, 1));
console.log(JSON.stringify(api).slice(0, 900));
await browser.close();
