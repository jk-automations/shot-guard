// Open a Shot Guard HTML report from disk, click each timeline flag, check the video jumps there. No network allowed.
import { chromium } from "playwright";
const file = process.argv[2];
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 1400 } });
const net = [];
page.on("request", (r) => { if (!r.url().startsWith("file:")) net.push(r.url()); });
const errs = [];
page.on("pageerror", (e) => errs.push(e.message));
await page.goto("file://" + file);
await page.waitForFunction(() => { const v = document.getElementById("v"); return !v || v.readyState >= 1; }, null, { timeout: 20000 });
const flags = await page.$$eval(".flag", (fs) => fs.map((f) => parseFloat(f.dataset.t)));
const got = [];
for (let k = 0; k < flags.length; k++) {
  await page.locator(".flag").nth(k).click();
  await page.waitForTimeout(300);
  got.push(await page.evaluate(() => document.getElementById("v").currentTime));
}
await page.evaluate(() => document.getElementById("v").pause());
await page.screenshot({ path: process.argv[3], fullPage: false });
console.log(JSON.stringify({ flags, got, net, errs }));
await browser.close();
