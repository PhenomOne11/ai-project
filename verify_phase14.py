"""
Phase 14 Verification Script
Validates all theme primitives and layout helpers.
"""

import sys


def verify_palettes():
    print("=" * 60)
    print("  PHASE 14: Palette System")
    print("=" * 60)

    from dashboard.theme import (
        PALETTES, get_palette, list_palettes, Palette
    )

    names = list_palettes()
    print(f"  ✅ Registered palettes: {names}")

    for name in names:
        p = get_palette(name)
        assert isinstance(p, Palette)
        assert len(p.chart_sequence) >= 4
        print(f"      [{name:<12}] "
              f"accent={p.accent_primary:<8} "
              f"secondary={p.accent_secondary:<8} "
              f"chart_colors={len(p.chart_sequence)}")

    # Fallback test
    fallback = get_palette("nonexistent")
    assert fallback.name == "cyber_dark"
    print(f"  ✅ Fallback palette works: {fallback.name}")

    return True


def verify_css():
    print("\n" + "=" * 60)
    print("  PHASE 14: CSS Generator")
    print("=" * 60)

    from dashboard.theme import generate_css, get_palette

    for pname in ["cyber_dark", "plasma", "mono", "solar"]:
        p = get_palette(pname)
        css = generate_css(p)
        assert "<style>" in css and "</style>" in css
        assert p.accent_primary in css

        # Key sections
        for section in ["ROOT VARIABLES", "METRIC CARDS",
                        "BUTTONS", "TABS", "ANIMATIONS"]:
            assert section in css, f"Missing section: {section}"

        print(f"  ✅ {pname:<12}: {len(css):,} chars, all sections present")

    return True


def verify_components():
    print("\n" + "=" * 60)
    print("  PHASE 14: Component Library")
    print("=" * 60)

    from dashboard.theme import (
        status_badge, metric_card, section_header, divider,
        chip, kv_row, alert_box, sparkline_card, progress_bar,
    )

    # Each function must return non-empty HTML string
    checks = [
        ("status_badge", status_badge("LIVE", "active", icon="●")),
        ("status_badge_crit", status_badge("ALERT", "critical", icon="⚠")),
        ("metric_card", metric_card("Batches", "1,204", "+12")),
        ("section_header", section_header("⚡", "Engine")),
        ("divider", divider()),
        ("chip", chip("FP16")),
        ("kv_row", kv_row("Precision", "FP16")),
        ("alert_box", alert_box("Info", "All good.", level="info")),
        ("sparkline_card", sparkline_card(
            "Latency", "2.48ms", [1.0, 2.0, 1.5, 2.2, 1.8, 2.4]
        )),
        ("progress_bar", progress_bar("Buffer", 18, 64)),
    ]

    for name, html in checks:
        assert isinstance(html, str) and len(html) > 10, f"{name} returned empty"
        assert "<" in html and ">" in html, f"{name} not HTML"
        print(f"  ✅ {name:<18} → {len(html):>5} chars")

    # Verify sparkline SVG structure
    svg = sparkline_card("x", "1.0", [1.0, 2.0, 3.0])
    assert "<svg" in svg and "path" in svg
    print(f"  ✅ sparkline SVG structure verified")

    return True


def verify_plotly():
    print("\n" + "=" * 60)
    print("  PHASE 14: Plotly Theming")
    print("=" * 60)

    import plotly.graph_objects as go
    import numpy as np
    from dashboard.theme import apply_theme, get_palette, accent_color, severity_color

    p = get_palette("cyber_dark")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(20), y=np.random.randn(20),
        mode="lines+markers", name="test"
    ))
    fig = apply_theme(fig, p, height=400)

    # Verify layout was set
    assert fig.layout.height == 400
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor.startswith("rgba")
    assert len(fig.layout.colorway) >= 4

    print(f"  ✅ apply_theme height: {fig.layout.height}")
    print(f"  ✅ colorway: {list(fig.layout.colorway)[:4]}")
    print(f"  ✅ paper_bgcolor: {fig.layout.paper_bgcolor}")

    # Test accent color helper
    c1 = accent_color(p, 0)
    c2 = accent_color(p, 1)
    c5 = accent_color(p, 5)
    print(f"  ✅ accent_color: [0]={c1}, [1]={c2}, [5]={c5}")
    assert c1 != c2

    # Test severity color mapping
    for level in ["nominal", "warning", "critical", "emergency"]:
        col = severity_color(p, level)
        assert col.startswith("#")
        print(f"  ✅ severity_color('{level}') = {col}")

    return True


def verify_layout():
    print("\n" + "=" * 60)
    print("  PHASE 14: Layout Helpers")
    print("=" * 60)

    from dashboard.layouts.shell import sync_engine_state

    # Engine state mapping
    for state in ["UNINITIALIZED", "BOOTING", "STREAMING", "PAUSED", "FAULT"]:
        level, text = sync_engine_state(state)
        print(f"  ✅ {state:<15} → level='{level}', text='{text}'")
        assert level in ("neutral", "info", "active", "paused", "critical")

    # Fallback for unknown state
    level, text = sync_engine_state("WEIRD_STATE")
    print(f"  ✅ Unknown state → ({level}, {text})")
    assert level == "neutral"

    return True


def verify_preview_app_imports():
    print("\n" + "=" * 60)
    print("  PHASE 14: Preview App Imports")
    print("=" * 60)

    # Verify that the preview app module's imports all resolve
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "preview", "dashboard/preview.py"
        )
        # We can't fully execute it (it calls st.set_page_config at module level),
        # but we can verify the imports resolve by checking the AST.
        import ast
        source = open("dashboard/preview.py", "r", encoding="utf-8").read()
        tree = ast.parse(source)
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        print(f"  ✅ Preview app has {len(imports)} import statements")

        # Verify each import resolves
        for node in imports:
            if isinstance(node, ast.ImportFrom):
                mod_name = node.module
                try:
                    __import__(mod_name)
                except Exception as e:
                    raise AssertionError(f"Import failed: {mod_name} → {e}")
        print(f"  ✅ All preview imports resolve")

        return True
    except FileNotFoundError:
        print("  ⚠️  preview.py not found — skipping")
        return True
    except Exception as e:
        print(f"  ❌ Preview verification failed: {e}")
        return False


def check_files_exist():
    print("\n" + "=" * 60)
    print("  PHASE 14: File Checklist")
    print("=" * 60)

    from pathlib import Path
    files = [
        "dashboard/theme/palette.py",
        "dashboard/theme/css.py",
        "dashboard/theme/components.py",
        "dashboard/theme/plots.py",
        "dashboard/theme/__init__.py",
        "dashboard/layouts/shell.py",
        "dashboard/layouts/grid.py",
        "dashboard/layouts/__init__.py",
        "dashboard/preview.py",
    ]
    all_ok = True
    for f in files:
        exists = Path(f).is_file()
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"  {status}  {f}")
    return all_ok


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 14 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Palette System", verify_palettes()))
    results.append(("CSS Generator", verify_css()))
    results.append(("Component Library", verify_components()))
    results.append(("Plotly Theming", verify_plotly()))
    results.append(("Layout Helpers", verify_layout()))
    results.append(("Preview App", verify_preview_app_imports()))
    results.append(("File Checklist", check_files_exist()))

    print("\n" + "=" * 60)
    print("  PHASE 14 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 14 COMPLETE — Ready for Phase 15!")
        print("\n  ┌──────────────────────────────────────────────┐")
        print("  │  You can now preview the theme live:         │")
        print("  │                                              │")
        print("  │     streamlit run dashboard/preview.py        │")
        print("  │                                              │")
        print("  │  Next: Phase 15 — 3D Latent Space Explorer   │")
        print("  └──────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)