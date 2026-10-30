"""Credits meter: what the takes cost and what Shot Guard saved. A bad take caught here never reaches the edit, and a
take fixed by Auto-trim (or a passing take kept by Best of N) is a retry nobody pays for. Units are the spec's
cost_per_take (credits, dollars, whatever the generator charges); with no cost set it counts takes."""


def meter(results, trim_fixes=0):
    """results: gate results, one per generated take. trim_fixes: takes Auto-trim turned into a pass."""
    cost = max((r["spec"]["cost_per_take"] for r in results), default=0.0)
    caught = sum(not r["passed"] for r in results)
    m = {"takes": len(results), "passed": len(results) - caught, "caught": caught, "retries_saved": trim_fixes,
         "cost_per_take": cost, "spent": round(len(results) * cost, 4), "saved": round(trim_fixes * cost, 4)}
    m["line"] = line(m)
    return m


def _amount(n, cost):
    return f"{n * cost:g} credits" if cost else f"{n} take{'s' if n != 1 else ''}"


def line(m):
    out = [f"{m['takes']} take{'s' if m['takes'] != 1 else ''} checked", f"{m['caught']} caught before the edit"]
    if m["retries_saved"]:
        out.append(f"{m['retries_saved']} fixed by auto-trim: {_amount(m['retries_saved'], m['cost_per_take'])} saved")
    if m["cost_per_take"]:
        out.append(f"spent {m['spent']:g} credits")
    return " · ".join(out)
