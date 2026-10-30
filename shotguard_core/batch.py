"""Dailies: every take in a folder checked in one go, ranked best first on one page, with the credits meter on top.
A take uses <name>.json next to it as its spec when there is one (a whole episode in one folder), else the shared
spec. A failing take whose only problem is a hidden cut is auto-trimmed and checked again."""
import json
import os

from PIL import Image, ImageDraw

from . import best, credits, report, spec as spec_mod, trim
from .report import CHECK_COLOR, INK, LINE, MUTED, PAPER, VERDICT, _thumb, _wrap, font

TAKES = (".mp4", ".mov", ".webm", ".mkv", ".jpg", ".jpeg", ".png")


def takes_in(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(TAKES) and not f.startswith(".") and "-trimmed" not in f)


def spec_for(path, shared):
    own = os.path.splitext(path)[0] + ".json"
    return spec_mod.load(own) if os.path.exists(own) else dict(shared, _cache={})


def run_folder(folder, shared_spec=None, out_dir=None, auto_trim=True, which=None, log=print):
    from . import run
    out_dir = out_dir or os.path.join(folder, "shot_guard")
    os.makedirs(out_dir, exist_ok=True)
    rows, fixes = [], 0
    for path in takes_in(folder):
        sp = spec_for(path, shared_spec or spec_mod.make())
        r = run(path, sp, which)
        row = {"take": os.path.basename(path), "result": r, "fixed_by_trim": None}
        if auto_trim and not r["passed"]:
            p = trim.plan(r, sp["thresholds"]["trim_min_s"])
            if p and p["ok"]:
                piece = trim.cut(path, p["t0"], p["t1"], os.path.join(out_dir, os.path.splitext(row["take"])[0] + "-trimmed.mp4"))
                r2 = run(piece, sp, which)
                if r2["passed"]:
                    row.update(fixed_by_trim={"note": p["note"], "path": piece}, result=r2)
                    fixes += 1
        log(f"{row['take']:28s} {r['verdict']:5s} {r['score']:5}" + (f" -> trimmed: {row['result']['verdict']}" if row["fixed_by_trim"] else ""))
        rows.append(row)
    _, order = best.rank([x["result"] for x in rows])
    ranked = [rows[k] for k in order]
    # the meter counts every generated take as it came out of the generator, before any trim
    m = credits.meter([x["result"] if not x["fixed_by_trim"] else dict(x["result"], passed=False) for x in rows], fixes)
    return {"folder": folder, "rows": ranked, "meter": m}


def summary(d):
    lines = [d["meter"]["line"]]
    for k, x in enumerate(d["rows"]):
        r = x["result"]
        worst = next((i["message"] for i in r["issues"] if i["severity"] == "major"), "no major problems")
        lines.append(f"{k + 1}. {x['take']}: {r['verdict']} {r['score']}" + (" (auto-trimmed)" if x["fixed_by_trim"] else "") + f" - {worst}")
    return "\n".join(lines)


def render(d, out_dir, prefix="dailies", blur=True, width=1600, title=None):
    """One ranked page (PNG) plus a JSON of every take's receipt."""
    os.makedirs(out_dir, exist_ok=True)
    pad, row_h, th_h = 40, 220, 190
    f_big, f_mid, f_small, f_bold = font(44, True), font(24), font(20), font(24, True)
    H = 200 + len(d["rows"]) * (row_h + 14) + pad
    page = Image.new("RGB", (width, H), PAPER)
    dr = ImageDraw.Draw(page)
    dr.rectangle([0, 0, width, 170], fill=INK)
    dr.text((pad, 28), "SHOT GUARD · DAILIES", font=f_bold, fill=(200, 190, 178))
    dr.text((pad, 62), title or os.path.basename(os.path.normpath(d["folder"])), font=f_big, fill=PAPER)
    dr.text((pad, 124), d["meter"]["line"], font=f_mid, fill=(200, 190, 178))
    y = 190
    for k, x in enumerate(d["rows"]):
        r = x["result"]
        dr.rounded_rectangle([pad, y, width - pad, y + row_h], radius=12, fill=(255, 252, 248), outline=LINE, width=2)
        dr.text((pad + 18, y + 18), f"{k + 1}", font=f_big, fill=MUTED)
        mid = 0.0 if r["video"].lower().endswith((".jpg", ".jpeg", ".png")) else None
        bad = next((i for i in r["issues"] if i["severity"] == "major" and (i.get("frame_t") or i.get("t0")) is not None), None)
        t = (bad.get("frame_t") or bad.get("t0")) if bad else (mid if mid is not None else 1.0)
        thumb = _thumb(r["video"], t, th_h, blur, (bad or {}).get("box") if bad and bad["check"] != "text" else None,
                       CHECK_COLOR.get((bad or {}).get("check"), None))
        thumb = thumb.resize((min(thumb.width, 340), th_h)) if thumb.width > 340 else thumb
        page.paste(thumb, (pad + 90, y + 15))
        x0 = pad + 90 + thumb.width + 30
        v = r["verdict"]
        vw = dr.textlength(v, font=f_bold) + 32
        dr.rounded_rectangle([x0, y + 20, x0 + vw, y + 58], radius=8, fill=VERDICT[v])
        dr.text((x0 + 16, y + 26), v, font=f_bold, fill=PAPER)
        am = r["checks"].get("attention", {}).get("metrics", {})
        hook = f"  ·  hook {am['hook']}" if am.get("hook") is not None else ""
        dr.text((x0 + vw + 18, y + 26), f"{x['take']}  ·  score {r['score']}{hook}", font=f_bold, fill=INK)
        msg = ("auto-trimmed: " + x["fixed_by_trim"]["note"]) if x["fixed_by_trim"] else \
            (bad["message"] if bad else next((i["message"] for i in r["issues"]), "all checks passed"))
        col = CHECK_COLOR.get(bad["check"], INK) if bad else (VERDICT["PASS"] if r["passed"] else INK)
        for n, ln in enumerate(_wrap(dr, msg, f_mid, width - pad - x0 - 20)[:2]):
            dr.text((x0, y + 76 + n * 32), ln, font=f_mid, fill=col)
        if r["hints"] and not r["passed"]:
            hint = _wrap(dr, "next take: " + r["hints"][0], f_small, width - pad - x0 - 20)[0]
            dr.text((x0, y + 150), hint, font=f_small, fill=MUTED)
        y += row_h + 14
    png = os.path.join(out_dir, prefix + ".png")
    page.save(png, optimize=True)
    js = os.path.join(out_dir, prefix + ".json")
    with open(js, "w") as f:
        json.dump({"meter": d["meter"], "takes": [{"take": x["take"], "fixed_by_trim": x["fixed_by_trim"],
                                                   "receipt": report.receipt(x["result"])} for x in d["rows"]]},
                  f, indent=1, default=report._jsonable)
    return page, png, js
