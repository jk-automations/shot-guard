"""Clickable report: one self-contained HTML page per take. The video plays at the top; every finding is a button on the
timeline and a card below it, and clicking either jumps the video to that second. The attention curve is drawn under
the timeline. Inline CSS and JS, no network: open the file from disk or share the folder."""
import html
import json
import os
import shutil

from .report import CHECK_COLOR, receipt
from .video import is_image, read_frames

ORDER = ("dialogue", "speaker", "identity", "blocking", "text", "cuts", "glitch", "audio", "facts", "sequence", "attention")


def _rgb(c):
    return "rgb(%d,%d,%d)" % c


def _t(i):
    t = i.get("frame_t") if i.get("frame_t") is not None else i.get("t0")
    return None if t is None else float(t)


def render(result, out_dir, prefix="shotguard", video_path=None):
    """Write <prefix>.html and copy the take next to it as <prefix>.<ext>. Returns the HTML path."""
    os.makedirs(out_dir, exist_ok=True)
    src = video_path or result["video"]
    ext = os.path.splitext(src)[1].lower() or ".mp4"
    media = prefix + ext
    if os.path.abspath(src) != os.path.abspath(os.path.join(out_dir, media)):
        shutil.copyfile(src, os.path.join(out_dir, media))
    still = is_image(src)
    dur = 0.0 if still else (read_frames(src, every=1.0, max_side=32).duration or 1.0)
    r = result
    flags = []
    for k, i in enumerate(r["issues"]):
        t = _t(i)
        flags.append({"k": k, "check": i["check"], "kind": i["kind"].replace("_", " "), "severity": i["severity"], "t": t,
                      "t0": i.get("t0"), "t1": i.get("t1"), "message": i["message"],
                      "color": _rgb(CHECK_COLOR.get(i["check"], (120, 112, 104)))})
    curve = r["checks"].get("attention", {}).get("metrics", {}).get("curve")
    am = r["checks"].get("attention", {}).get("metrics", {})
    chips = []
    for c in ORDER:
        ch = r["checks"].get(c)
        if ch is None:
            continue
        st = ch["status"]
        st = "warn" if st == "pass" and ch.get("issues") else st
        chips.append(f'<span class="chip {st}">{c}<b>{"OK" if st == "pass" else st.upper()}</b></span>')
    svg = ""
    if curve and dur:
        pts = " ".join(f"{100 * t / dur:.2f},{30 - 28 * v:.2f}" for t, v in zip(curve["t"], curve["attention"]))
        svg = f'<svg class="curve" viewBox="0 0 100 30" preserveAspectRatio="none"><polyline points="{pts}"/></svg>'
    marks = "".join(
        f'<button class="flag {f["severity"]}" style="left:{100 * f["t"] / dur:.2f}%;background:{f["color"]}" data-t="{f["t"]:.3f}" '
        f'data-k="{f["k"]}" title="{html.escape(f["check"])}: {html.escape(f["message"])}"></button>'
        for f in flags if f["t"] is not None and dur)
    cards = "".join(
        f'<article class="card" id="c{f["k"]}" data-t0="{"" if f["t0"] is None else f["t0"]}" data-t1="{"" if f["t1"] is None else f["t1"]}">'
        f'<div class="tag" style="color:{f["color"]}">{html.escape(f["check"].upper())} · {html.escape(f["kind"])}'
        f'<span class="sev {f["severity"]}">{f["severity"]}</span></div><p>{html.escape(f["message"])}</p>'
        + (f'<button class="seek" data-t="{f["t"]:.3f}">Play from {f["t"]:.1f} s</button>' if f["t"] is not None and not still else "")
        + "</article>" for f in flags)
    hints = "".join(f"<li>{html.escape(h)}</li>" for h in r["hints"])
    player = (f'<img class="media" src="{html.escape(media)}" alt="the take">' if still else
              f'<video class="media" id="v" src="{html.escape(media)}" controls playsinline preload="metadata"></video>')
    hook = f'<span>hook <b>{am["hook"]}</b>/100</span><span>attention <b>{am["attention"]}</b>/100</span>' if am.get("hook") is not None else ""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shot Guard: {html.escape(r['spec'].get('name', 'take'))}</title><style>
:root{{--bg:#f6f2ec;--panel:#fffcf8;--ink:#26221f;--muted:#786f68;--line:#ded6cc;--pass:#568054;--retry:#b0803e;--fail:#aa523e;--accent:#826496}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1e1b18;--panel:#2a2622;--ink:#f0ebe3;--muted:#aaa094;--line:#3e3832}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 48px}}header{{display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center;justify-content:space-between}}
.brand{{font-size:13px;letter-spacing:.12em;color:var(--muted);font-weight:700}}h1{{margin:2px 0 0;font-size:28px}}
.verdict{{font-weight:800;font-size:22px;color:#fff;padding:8px 18px;border-radius:12px;background:var(--{r['verdict'].lower()})}}
.meta{{display:flex;flex-wrap:wrap;gap:16px;color:var(--muted);margin:10px 0 16px}}.meta b{{color:var(--ink)}}
.media{{display:block;width:100%;max-height:70vh;object-fit:contain;background:#000;border-radius:12px}}
.track{{position:relative;height:30px;margin:14px 0 0;background:var(--line);border-radius:6px}}
.flag{{position:absolute;top:3px;width:14px;height:24px;margin-left:-7px;border:2px solid var(--panel);border-radius:5px;cursor:pointer;padding:0}}
.flag.minor{{opacity:.7}}.flag:focus-visible,.seek:focus-visible{{outline:3px solid var(--accent);outline-offset:2px}}
.head{{position:absolute;top:-4px;width:3px;height:38px;background:var(--ink);left:0;pointer-events:none}}
.curve{{display:block;width:100%;height:46px;margin-top:4px}}.curve polyline{{fill:none;stroke:var(--accent);stroke-width:.6;vector-effect:non-scaling-stroke}}
.legend{{font-size:13px;color:var(--muted);display:flex;justify-content:space-between}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}}.chip{{border:2px solid var(--line);border-radius:8px;padding:3px 10px;font-size:14px}}
.chip b{{margin-left:6px}}.chip.pass b{{color:var(--pass)}}.chip.warn b{{color:var(--retry)}}.chip.fail b{{color:var(--fail)}}.chip.skip{{opacity:.5}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}}
.card{{background:var(--panel);border:2px solid var(--line);border-radius:12px;padding:12px 14px;transition:border-color .2s}}
.card.live{{border-color:var(--accent)}}.card p{{margin:6px 0 10px}}.tag{{font-weight:700;font-size:13px;display:flex;justify-content:space-between;gap:8px}}
.sev{{color:var(--muted);font-weight:600}}.sev.major{{color:var(--fail)}}
.seek{{font:inherit;font-size:14px;border:0;border-radius:8px;padding:6px 12px;background:var(--ink);color:var(--bg);cursor:pointer}}
h2{{font-size:18px;margin:26px 0 8px}}ol{{padding-left:20px}}footer{{margin-top:28px;color:var(--muted);font-size:13px}}
</style></head><body><main>
<header><div><div class="brand">SHOT GUARD</div><h1>{html.escape(r['spec'].get('name', 'take'))}</h1></div><div class="verdict">{r['verdict']}</div></header>
<div class="meta"><span>score <b>{r['score']}</b></span>{hook}<span>attempt <b>{r['attempt']}</b></span><span>{html.escape(r['why'])}</span></div>
{player}
{'' if still else '<div class="track" id="track"><div class="head" id="head"></div>' + marks + '</div>' + svg + '<div class="legend"><span>0 s</span><span>attention</span><span>' + f'{dur:.1f} s' + '</span></div>'}
<div class="chips">{''.join(chips)}</div>
<h2>{len(flags)} finding{'s' if len(flags) != 1 else ''}</h2><div class="cards">{cards or '<p>All checks passed.</p>'}</div>
<h2>{'Fix hints for the next take' if hints else 'No fixes needed'}</h2><ol>{hints}</ol>
<footer>Shot Guard · checked {html.escape(r.get('time', ''))} · a JSON receipt sits next to this page</footer>
</main><script>
const v=document.getElementById("v"),head=document.getElementById("head"),dur={dur:.3f};
function go(t){{if(!v)return;v.currentTime=t;v.play().catch(()=>{{}});}}
document.querySelectorAll("[data-t]").forEach(b=>b.addEventListener("click",()=>go(parseFloat(b.dataset.t))));
document.querySelectorAll(".flag").forEach(b=>b.addEventListener("click",()=>{{const c=document.getElementById("c"+b.dataset.k);if(c)c.scrollIntoView({{block:"nearest",behavior:"smooth"}});}}));
if(v)v.addEventListener("timeupdate",()=>{{const t=v.currentTime;if(head&&dur)head.style.left=(100*t/dur)+"%";
document.querySelectorAll(".card").forEach(c=>{{const a=parseFloat(c.dataset.t0),b=parseFloat(c.dataset.t1||c.dataset.t0);
c.classList.toggle("live",!isNaN(a)&&t>=a-0.1&&t<=Math.max(isNaN(b)?a:b,a+1.5));}});}});
</script></body></html>"""
    out = os.path.join(out_dir, prefix + ".html")
    with open(out, "w") as f:
        f.write(page)
    with open(os.path.join(out_dir, prefix + ".json"), "w") as f:
        json.dump(receipt(result), f, indent=1)
    return out


def index(d, out_dir, title="Dailies"):
    """Dailies index page: every take in rank order, each linking to its own report."""
    rows = []
    for k, x in enumerate(d["rows"]):
        r = x["result"]
        name = os.path.splitext(x["take"])[0]
        render(r, out_dir, name)
        am = r["checks"].get("attention", {}).get("metrics", {})
        worst = next((i["message"] for i in r["issues"] if i["severity"] == "major"), "no major problems")
        rows.append(f'<a class="row" href="{html.escape(name)}.html"><span class="n">{k + 1}</span><span class="v {r["verdict"].lower()}">{r["verdict"]}</span>'
                    f'<span class="t"><b>{html.escape(x["take"])}</b> · score {r["score"]}'
                    + (f' · hook {am["hook"]}' if am.get("hook") is not None else "")
                    + (" · auto-trimmed" if x.get("fixed_by_trim") else "") + f'<br><small>{html.escape(worst)}</small></span></a>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shot Guard dailies</title><style>
:root{{--bg:#f6f2ec;--panel:#fffcf8;--ink:#26221f;--muted:#786f68;--line:#ded6cc;--pass:#568054;--retry:#b0803e;--fail:#aa523e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1e1b18;--panel:#2a2622;--ink:#f0ebe3;--muted:#aaa094;--line:#3e3832}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:900px;margin:0 auto;padding:24px 16px}}.brand{{font-size:13px;letter-spacing:.12em;color:var(--muted);font-weight:700}}
.row{{display:flex;gap:14px;align-items:center;background:var(--panel);border:2px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0;color:inherit;text-decoration:none}}
.row:hover,.row:focus-visible{{border-color:var(--ink)}}.n{{font-size:24px;color:var(--muted);width:28px}}small{{color:var(--muted)}}
.v{{color:#fff;font-weight:800;border-radius:8px;padding:4px 10px}}.v.pass{{background:var(--pass)}}.v.retry{{background:var(--retry)}}.v.fail{{background:var(--fail)}}
</style></head><body><main><div class="brand">SHOT GUARD · DAILIES</div><h1>{html.escape(title)}</h1><p>{html.escape(d['meter']['line'])}</p>
{''.join(rows)}</main></body></html>"""
    out = os.path.join(out_dir, "index.html")
    with open(out, "w") as f:
        f.write(page)
    return out
