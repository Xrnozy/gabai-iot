"""Turn run logs and accuracy tests into graphs and a one-page report (Phase M).

    py -m gabai.tools.make_report --run data/run1.csv --run data/run2.csv --targets data/target_tests.csv --range data/range_test.csv

Writes data/reports/<name>/: PNG graphs, report.md and report.html (one self-contained page, images embedded).
A number that was not measured is printed as "not measured", never as 0.
"""
import argparse
import base64
import csv
import os
import time

from gabai.report import metrics, plots

LIMITS = """- **Tethered proof of concept.** The Pi sends over the network to a PC; speech is played by the PC. It is not wearable yet.
- **Supervised use only.** No claim is made that this is reliable enough for a blind person to rely on.
- **Known weak cases:** glass and mirrors, thin poles and wires, black or very dark objects, low light, shiny floors, and objects closer than 0.23 m (the stereo model's disparity limit); drop-offs are only reported as working where the accuracy table shows a hit rate.
- **Latency is capture to the clip being handed to the sound card.** The sound itself starts a little later; cross-check with a screen-and-audio recording (see docs/phase-M-manual-test.md).
- **The 400 ms in the spec is a target, not a result.** Compare it with the measured p50 and p95 above."""


def fmt(s, key="p50", unit="", digits=0):
    if s is None:
        return "not measured"
    v = s[key]
    return f"{v:.{digits}f}{unit}"


def pair(s, unit="", digits=0):
    return "not measured" if s is None else f"{s['p50']:.{digits}f} / {s['p95']:.{digits}f}{unit}"


def run_rows(name, s):
    alerts = ", ".join(f"{k} x{v}" for k, v in sorted(s["alerts"].items())) or "none"
    m = s["marks"]
    marks = "no marks file" if m is None else f"{m.get('false_alarm', 0)} false, {m.get('missed', 0)} missed"
    pct1 = lambda v: "not measured" if v is None else f"{v:.1f}%"
    return [
        ("Run", name),
        ("Frames / duration", f"{s['frames']} / " + ("not measured" if s["duration_s"] is None else f"{s['duration_s']:.0f} s")),
        ("Frames in, FPS (p50)", fmt(s["fps_in"], digits=1)),
        ("Capture to PC receipt, ms (p50 / p95)", pair(s["latency_ms"])),
        ("Capture to speech, ms (p50 / p95, n)", "not measured" if s["speech_ms"] is None else f"{pair(s['speech_ms'])} (n={s['speech_ms']['n']})"),
        ("Depth model, ms (p50 / p95)", pair(s["model_ms"])),
        ("Whole loop, ms (p50 / p95)", pair(s["loop_ms"])),
        ("Pixels with valid depth (p50)", fmt(s["valid_pct"], unit="%")),
        ("L/R camera pairing offset, ms (p50 / p95)", pair(s["pair_dt_ms"], digits=1)),
        ("Round-trip time, ms (p50 / p95); clock error is about half", pair(s["rtt_ms"])),
        ("Longest gap between frames / gaps over 0.5 s", "not measured" if s["max_gap_s"] is None else f"{s['max_gap_s']:.2f} s / {s['stalls_over_500ms']}"),
        ('"tracking delayed" announcements', "not measured" if s["tracking_delayed_spoken"] is None else str(s["tracking_delayed_spoken"])),
        ("Frames with a zone at the model's 0.23 m limit", "not measured" if s["near_limit_frame_pct"] is None else f"{s['near_limit_frame_pct']:.0f}%"),
        ("Alerts (each counted once)", alerts),
        ("Can't-see-ahead frames", f"{s['cant_see_frame_pct']:.1f}%"),
        ("Drop-ahead frames", f"{s['drop_frame_pct']:.1f}%"),
        ("Supervisor marks", marks),
        ("Pi temperature max", "not measured" if s["pi_temp_max"] is None else f"{s['pi_temp_max']:.0f} C"),
        ("Pi under-voltage now (share of frames)", pct1(s["under_voltage_now_pct"])),
        ("Pi under-voltage since boot", "not measured" if s["under_voltage_ever"] is None else ("YES" if s["under_voltage_ever"] else "no")),
        ("Pi free memory min", "not measured" if s["mem_min_mb"] is None else f"{s['mem_min_mb']:.0f} MB"),
        ("PC GPU temperature max", "not measured" if s["gpu_temp_max"] is None else f"{s['gpu_temp_max']:.0f} C"),
    ]


def md_table(rows):
    return "\n".join(["| Metric | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows])


def load_range(path):
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append((float(r["true_m"]), float(r["est_median_m"]), float(r["est_p10_m"]), float(r["est_p90_m"])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="a run CSV written by demo_viewer --log (repeatable)")
    ap.add_argument("--targets", help="data/target_tests.csv from gabai.tools.target_test")
    ap.add_argument("--range", dest="range_csv", help="data/range_test.csv from gabai.tools.range_test")
    ap.add_argument("--title", default="Gabai Lite: measured results")
    ap.add_argument("--out", help="output folder (default data/reports/<timestamp>)")
    a = ap.parse_args()
    out = a.out or os.path.join("data", "reports", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    sections, images, not_measured = [], [], []
    for path in a.run:
        name = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(path):
            not_measured.append(f"{name}: run not done (no log file {path})")
            continue
        df = metrics.load_run(path)
        s = metrics.summarize(df, metrics.load_marks(path + ".marks.csv"))
        rows = run_rows(name, s)
        not_measured += [f"{name}: {k}" for k, v in rows if v == "not measured"]
        sections.append(f"## Run: {name}\n\n{md_table(rows)}\n")
        for fn, label in ((plots.latency, "latency"), (plots.speech, "speech"), (plots.pipeline, "pipeline"),
                          (plots.zones, "zones"), (plots.health, "health")):
            png = fn(df, os.path.join(out, f"{name}_{label}.png"))
            if png:
                images.append((f"{name}: {label}", png))

    targets = None
    if a.targets and os.path.exists(a.targets):
        from gabai.tools.target_test import load_targets, summarize_targets
        targets = load_targets(a.targets)
        rows = ["| Target | n | no depth | depth on box | median error | worst error |", "|---|---|---|---|---|---|"]
        for lab, s in summarize_targets(targets).items():
            e = lambda v: "none" if v is None else f"{v:+.1f}%"
            rows.append(f"| {lab} | {s['n']} | {s['no_depth']} | {s['valid_pct']:.0f}% | {e(s['median_err_pct'])} | {e(s['worst_err_pct'])} |")
        if targets:
            sections.append("## Depth accuracy on real targets (tape-measured)\n\n" + "\n".join(rows) + "\n")
        else:
            not_measured.append("depth accuracy on real targets (target_tests.csv has no measurements)")
    else:
        not_measured.append("depth accuracy on real targets (no target_tests.csv)")
    ranges = load_range(a.range_csv) if a.range_csv and os.path.exists(a.range_csv) else None
    if ranges is None:
        not_measured.append("checkerboard range accuracy (no range_test.csv)")
    png = plots.accuracy(targets, ranges, os.path.join(out, "accuracy.png"))
    if png:
        images.append(("Depth error vs real distance", png))

    if not a.run:
        not_measured.append("no run logs given (--run)")
    head = f"# {a.title}\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} from {len(a.run)} run log(s).\n"
    nm = "## Not measured\n\n" + ("\n".join(f"- {x}" for x in not_measured) if not_measured else "Everything in this report was measured.") + "\n"
    md = head + "\n" + "\n".join(sections) + "\n## Graphs\n\n" + "\n\n".join(f"![{t}]({os.path.basename(p)})" for t, p in images) \
        + "\n\n" + nm + "\n## Known limits\n\n" + LIMITS + "\n"
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    def embed(p):
        return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()
    import html
    page = ["<!doctype html><meta charset='utf-8'><title>Gabai Lite report</title>",
            "<style>body{font:14px system-ui;max-width:980px;margin:20px auto;padding:0 12px}table{border-collapse:collapse;margin:8px 0}"
            "td,th{border:1px solid #bbb;padding:3px 8px;text-align:left}img{max-width:100%}h1,h2{margin:18px 0 6px}</style>"]
    page.append(f"<h1>{html.escape(a.title)}</h1><p>{html.escape(head.splitlines()[2])}</p>")
    for sec in sections:
        lines = sec.strip().splitlines()
        page.append(f"<h2>{html.escape(lines[0].lstrip('# '))}</h2><table>")
        for ln in lines[2:]:
            if ln.startswith("|---"):
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            tag = "th" if ln is lines[2] else "td"
            page.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        page.append("</table>")
    page.append("<h2>Graphs</h2>" + "".join(f"<p><b>{html.escape(t)}</b><br><img src='{embed(p)}'></p>" for t, p in images))
    page.append("<h2>Not measured</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in not_measured) + "</ul>" if not_measured else "<h2>Not measured</h2><p>Everything was measured.</p>")
    page.append("<h2>Known limits</h2><ul>" + "".join(f"<li>{html.escape(x.lstrip('- ').replace('**', ''))}</li>" for x in LIMITS.splitlines()) + "</ul>")
    with open(os.path.join(out, "report.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(page))
    print(f"report written to {out}  (report.html is one self-contained page, report.md is the same text)")


if __name__ == "__main__":
    main()
