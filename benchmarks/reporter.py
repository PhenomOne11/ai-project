"""
Benchmark report rendering: console, JSON, and HTML.
"""

import json
import html
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from benchmarks.suite import SuiteResult


# =============================================================================
# CONSOLE
# =============================================================================

def render_console(suite: SuiteResult, width: int = 100) -> str:
    """Format suite results as a colored console report."""
    lines = []
    lines.append("=" * width)
    lines.append(f"  NEURALSTREAM-CORE · BENCHMARK SUITE")
    lines.append(f"  Duration: {suite.duration_s:.2f}s · "
                 f"{len(suite.results)} benchmarks")
    lines.append("=" * width)
    lines.append("")

    # Group by category
    by_cat = suite.by_category()
    verdicts = {v.name: v for v in suite.verdicts}

    for cat, results in sorted(by_cat.items()):
        lines.append(f"  ┌─ {cat.upper()} " + "─" * (width - len(cat) - 5))
        lines.append(
            f"  │  {'Benchmark':<30} "
            f"{'mean':>8} {'p95':>8} {'p99':>8} "
            f"{'rssΔ':>8} "
            f"{'verdict':>10}"
        )
        lines.append(f"  │  " + "─" * (width - 5))

        for r in results:
            if r.error:
                lines.append(f"  │  {r.name:<30} ERROR: {r.error[:50]}")
                continue

            v = verdicts.get(r.name)
            if v is None:
                symbol = "  —"
            elif not v.passed:
                symbol = "  ✗ FAIL"
            elif v.reason == "warning-level regression":
                symbol = "  ⚠ WARN"
            else:
                symbol = "  ✓ PASS"

            lines.append(
                f"  │  {r.name:<30} "
                f"{r.mean_ms:>7.3f}ms "
                f"{r.p95_ms:>7.3f}ms "
                f"{r.p99_ms:>7.3f}ms "
                f"{r.rss_delta_mb:>+7.2f} "
                f"{symbol:>10}"
            )

        lines.append(f"  └" + "─" * (width - 3))
        lines.append("")

    # Verdict summary
    lines.append("=" * width)
    if suite.all_passed:
        lines.append(f"  ✓ ALL {suite.n_passed} BENCHMARKS PASSED")
    else:
        lines.append(f"  ✗ SUITE FAILED · {suite.n_failed} regressions")
        for v in suite.verdicts:
            if not v.passed:
                lines.append(f"      • {v.name}: {v.reason}")
                for metric, level in v.levels.items():
                    if level != "nominal":
                        delta = v.deltas.get(metric, 0.0)
                        lines.append(f"          {metric}: {delta:+.1f}% ({level})")
    lines.append("=" * width)

    return "\n".join(lines)


# =============================================================================
# JSON
# =============================================================================

def render_json(suite: SuiteResult) -> str:
    return json.dumps(suite.to_dict(), indent=2, default=str)


def save_json(suite: SuiteResult, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_json(suite), encoding="utf-8")


# =============================================================================
# HTML
# =============================================================================

def render_html(suite: SuiteResult, title: str = "NeuralStream Benchmark Report") -> str:
    """Generate a standalone HTML report with a cyber-themed style."""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdicts = {v.name: v for v in suite.verdicts}
    by_cat = suite.by_category()

    def _verdict_badge(v):
        if v is None:
            return '<span class="badge neutral">—</span>'
        if not v.passed:
            return '<span class="badge danger">FAIL</span>'
        if v.reason == "warning-level regression":
            return '<span class="badge warning">WARN</span>'
        return '<span class="badge success">PASS</span>'

    def _metric_class(level):
        return {
            'nominal': 'ok',
            'warning': 'warn',
            'critical': 'crit',
        }.get(level, 'ok')

    # Category rows
    category_rows = []
    for cat, results in sorted(by_cat.items()):
        rows = []
        for r in results:
            v = verdicts.get(r.name)
            cells = ""
            if v:
                for metric in ['mean_ms', 'p95_ms', 'rss_peak_mb']:
                    pct = v.deltas.get(metric, 0.0)
                    level = v.levels.get(metric, 'nominal')
                    cells += (
                        f'<td class="delta {_metric_class(level)}">'
                        f'{pct:+.1f}%</td>'
                    )
            else:
                cells = "<td>—</td><td>—</td><td>—</td>"

            error_row = ""
            if r.error:
                error_row = (
                    f'<tr><td colspan="9" class="error">{html.escape(r.error)}</td></tr>'
                )

            rows.append(f"""
            <tr>
                <td class="name">{html.escape(r.name)}</td>
                <td class="num">{r.mean_ms:.3f}</td>
                <td class="num">{r.p95_ms:.3f}</td>
                <td class="num">{r.p99_ms:.3f}</td>
                <td class="num">{r.throughput_per_s:.1f}</td>
                <td class="num">{r.rss_peak_mb:.2f}</td>
                {cells}
                <td style="text-align:center">{_verdict_badge(v)}</td>
            </tr>
            {error_row}
            """)

        category_rows.append(f"""
            <h2>{cat.upper()}</h2>
            <table>
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>mean ms</th>
                        <th>p95 ms</th>
                        <th>p99 ms</th>
                        <th>ops/s</th>
                        <th>RSS peak MB</th>
                        <th>Δ mean</th>
                        <th>Δ p95</th>
                        <th>Δ RSS</th>
                        <th>Verdict</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """)

    overall_status = (
        f'<span class="badge success">ALL {suite.n_passed} PASSED</span>'
        if suite.all_passed
        else f'<span class="badge danger">{suite.n_failed} REGRESSIONS</span>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
    body {{
        background: #050810; color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        margin: 0; padding: 40px 32px; line-height: 1.5;
    }}
    h1 {{ font-size: 1.8rem; margin-bottom: 4px;
           background: linear-gradient(90deg, #38bdf8, #a78bfa);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           background-clip: text; }}
    h2 {{ font-size: 1.05rem; color: #94a3b8; margin-top: 32px;
           text-transform: uppercase; letter-spacing: 0.1em; }}
    .meta {{ color: #64748b; font-size: 0.85rem;
             font-family: 'JetBrains Mono', monospace; margin-bottom: 20px; }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
              font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; }}
    .badge.success {{ background: rgba(16, 185, 129, 0.2); color: #10b981;
                      border: 1px solid #10b981; }}
    .badge.warning {{ background: rgba(245, 158, 11, 0.2); color: #f59e0b;
                      border: 1px solid #f59e0b; }}
    .badge.danger {{ background: rgba(239, 68, 68, 0.2); color: #ef4444;
                     border: 1px solid #ef4444; }}
    .badge.neutral {{ background: rgba(148, 163, 184, 0.2); color: #cbd5e1;
                      border: 1px solid #64748b; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px;
             font-size: 0.85rem; }}
    th {{ text-align: left; padding: 10px 12px; color: #94a3b8;
          font-size: 0.72rem; text-transform: uppercase;
          letter-spacing: 0.08em; border-bottom: 1px solid #1e293b; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid rgba(30, 41, 59, 0.5); }}
    td.name {{ font-family: 'JetBrains Mono', monospace; color: #e2e8f0; }}
    td.num {{ font-family: 'JetBrains Mono', monospace; color: #cbd5e1;
              text-align: right; }}
    .delta.ok {{ color: #10b981; }}
    .delta.warn {{ color: #f59e0b; }}
    .delta.crit {{ color: #ef4444; font-weight: 700; }}
    tr:hover {{ background: rgba(56, 189, 248, 0.05); }}
    .error {{ color: #ef4444; font-family: 'JetBrains Mono', monospace;
              font-size: 0.8rem; padding: 8px 12px; }}
    .footer {{ margin-top: 40px; color: #64748b; font-size: 0.75rem;
               font-family: 'JetBrains Mono', monospace; }}
</style>
</head>
<body>
<h1>NeuralStream-Core · Benchmark Report</h1>
<div class="meta">
    Generated {generated} · duration {suite.duration_s:.2f}s ·
    {len(suite.results)} benchmarks · {overall_status}
</div>
{''.join(category_rows)}
<div class="footer">
    Regressions are flagged when a metric exceeds its baseline
    by more than the configured tolerance (mean latency 15%, memory 20%).
</div>
</body>
</html>
"""


def save_html(suite: SuiteResult, path: str, title: Optional[str] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(suite, title or "NeuralStream Benchmark Report"),
                 encoding="utf-8")