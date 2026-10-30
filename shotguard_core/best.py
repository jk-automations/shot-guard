"""Best of N: rank several takes of the same shot by their gate results. A take with no major problem beats any take
with one; then the higher score; then the earlier take (cheaper to have stopped there)."""


def rank(results):
    order = sorted(range(len(results)), key=lambda k: (not results[k]["passed"], -results[k]["score"], k))
    return [results[k] for k in order], order


def summary(results):
    ranked, order = rank(results)
    lines = []
    for place, (k, r) in enumerate(zip(order, ranked)):
        worst = next((i["message"] for i in r["issues"] if i["severity"] == "major"), "no major problems")
        lines.append(f"{place + 1}. take {k + 1}: {r['verdict']} score {r['score']} ({worst})")
    return "\n".join(lines)
