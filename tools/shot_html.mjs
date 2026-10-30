// Screenshot a Shot Guard HTML report in Chrome after clicking its first major flag (for the README).
import { chromium } from "playwright";
import path from "path";
const [file, out] = process.argv.slice(2);
const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
await page.goto("file://" + path.resolve(file));
await page.addStyleTag({ content: "body{zoom:.85} .media{max-height:420px}" });
await page.evaluate(() => { Element.prototype.scrollIntoView = () => {}; });
await page.waitForFunction(() => document.getElementById("v").readyState >= 2, null, { timeout: 20000 });
await page.locator(".flag.major").first().click();
await page.waitForTimeout(700);
await page.evaluate(() => document.getElementById("v").pause());
await page.waitForTimeout(400);
await page.screenshot({ path: out });
console.log(out);
await browser.close();
