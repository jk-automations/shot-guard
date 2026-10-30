// Screen-record a Shot Guard HTML report in Chrome: the take plays, then each flag on the timeline is clicked.
// node tools/capture_html.mjs <report.html> <out.webm>
import { chromium } from "playwright";
import fs from "fs";
import path from "path";
const [file, out] = process.argv.slice(2);
const dir = path.dirname(out);
const browser = await chromium.launch({ channel: "chrome" });   // Chrome plays H.264; the bundled Chromium does not
const ctx = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1,
                                       recordVideo: { dir, size: { width: 1280, height: 720 } } });
const page = await ctx.newPage();
await page.goto("file://" + path.resolve(file));
// fit the player, timeline and finding cards in one 720p frame, and keep the page still while flags are clicked
await page.addStyleTag({ content: "body{zoom:.82} main{max-width:1500px;padding-top:12px} .media{max-height:380px} h2{margin:12px 0 6px} .chips{margin:10px 0}" });
await page.evaluate(() => { Element.prototype.scrollIntoView = () => {}; });
await page.waitForFunction(() => document.getElementById("v").readyState >= 2, null, { timeout: 20000 });
await page.evaluate(() => { const v = document.getElementById("v"); v.muted = true; v.play(); });
await page.waitForTimeout(1500);
const n = await page.locator(".flag").count();
for (let k = 0; k < n; k++) {
  await page.locator(".flag").nth(k).click();
  await page.waitForTimeout(1500);
}
const v = page.video();
await ctx.close();
await browser.close();
fs.renameSync(await v.path(), out);
console.log(out, n, "flags clicked");
