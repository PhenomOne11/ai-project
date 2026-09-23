"""
Phase 21 Verification Script
"""

import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np


def verify_state():
    print("=" * 60)
    print("  PHASE 21: TopologyState")
    print("=" * 60)

    from dashboard.panels.topology import (
        TopologyState, TopologyNode, TopologyEdge,
    )

    s = TopologyState(activity_window_s=5.0)
    s.add_node(TopologyNode(id="a", label="A", category="model",
                            x=0, y=0, phase="P3"))
    s.add_node(TopologyNode(id="b", label="B", category="analysis",
                            x=2, y=0, phase="P5"))
    s.add_edge(TopologyEdge("a", "b", "flow"))

    print(f"  ✅ Nodes: {len(s.nodes)}, Edges: {len(s.edges)}")

    s.update_status("a", "warning")
    s.update_metric("a", "latency", "2.4 ms")
    print(f"  ✅ Status counts: {s.status_counts()}")
    print(f"  ✅ Overall health: {s.overall_health()}")

    # Activity
    for _ in range(10):
        s.mark_active("a")
    for _ in range(3):
        s.mark_active("b")

    print(f"  ✅ Activity rate a: {s.activity_rate('a'):.3f} Hz")
    print(f"  ✅ Category activity: {s.category_activity()}")
    print(f"  ✅ Active nodes: {s.active_nodes(min_rate=0.1)}")

    return True


def verify_default_topology():
    print("\n" + "=" * 60)
    print("  PHASE 21: Default Topology Seeding")
    print("=" * 60)

    from dashboard.panels.topology import TopologyState, seed_default_topology

    s = TopologyState()
    seed_default_topology(s)

    print(f"  ✅ Total nodes:  {len(s.nodes)}")
    print(f"  ✅ Total edges:  {len(s.edges)}")
    print(f"  ✅ Categories:   {s.categories()}")

    by_cat = s.nodes_by_category()
    for cat, nodes in sorted(by_cat.items()):
        print(f"      {cat:<10}: {len(nodes)} nodes")

    print(f"  ✅ Health: {s.overall_health()}")
    assert s.overall_health() == "nominal"
    assert len(s.nodes) >= 20

    return True


def verify_figures():
    print("\n" + "=" * 60)
    print("  PHASE 21: Figure Builders")
    print("=" * 60)

    from dashboard.panels.topology import (
        TopologyState, seed_default_topology,
        build_topology_figure, build_activity_timeline,
        build_category_activity_bar,
    )
    from dashboard.theme import get_palette

    s = TopologyState(activity_window_s=10.0)
    seed_default_topology(s)
    for _ in range(25):
        s.mark_active("engine_learner")
        s.mark_active("model_edge")
    for _ in range(10):
        s.mark_active("analysis_drift")

    p = get_palette("cyber_dark")

    fig1 = build_topology_figure(s, p, height=620)
    print(f"  ✅ Topology graph: {len(fig1.data)} traces, height={fig1.layout.height}")

    fig2 = build_activity_timeline(s, p, height=240)
    print(f"  ✅ Activity timeline: {len(fig2.data)} trace(s)")

    fig3 = build_category_activity_bar(s, p, height=240)
    print(f"  ✅ Category activity: {len(fig3.data)} trace(s)")

    # All palettes
    for pname in ["cyber_dark", "plasma", "mono", "solar"]:
        pal = get_palette(pname)
        build_topology_figure(s, pal)
        build_activity_timeline(s, pal)
        build_category_activity_bar(s, pal)
    print(f"  ✅ All 4 palettes render successfully")

    return True


def verify_simulation():
    print("\n" + "=" * 60)
    print("  PHASE 21: Activity Simulation")
    print("=" * 60)

    from dashboard.panels.topology import (
        TopologyState, seed_default_topology, _simulate_activity,
        _randomize_statuses,
    )

    s = TopologyState(activity_window_s=30.0)
    seed_default_topology(s)

    _simulate_activity(s, duration_s=30.0)
    active = s.active_nodes(min_rate=0.05)
    print(f"  ✅ Simulated activity for {len(active)} nodes")
    print(f"      Top activity: {sorted([(nid, s.activity_rate(nid)) for nid in active], key=lambda x: x[1], reverse=True)[:3]}")

    # Status randomization
    _randomize_statuses(s)
    counts = s.status_counts()
    print(f"  ✅ Randomized status counts: {counts}")

    # Set all back
    s.set_all_status("nominal")
    print(f"  ✅ All nominal: {s.status_counts()}")

    return True


def verify_render_integration():
    print("\n" + "=" * 60)
    print("  PHASE 21: Render Integration")
    print("=" * 60)

    from dashboard.panels.topology import (
        TopologyState, seed_default_topology, render,
    )

    # Mock streamlit module to test the render pipeline
    class MockColumn:
        def __init__(self): self._st = MockST()
        def __enter__(self): return self._st
        def __exit__(self, *args): pass

    class MockST:
        def __init__(self):
            self.markdown_calls = 0
            self.plotly_calls = 0
            self.metric_calls = 0
            self.button_calls = 0
            self.selectbox_val = None
            self.dataframe_calls = 0
        def markdown(self, *a, **k): self.markdown_calls += 1; return self
        def plotly_chart(self, *a, **k): self.plotly_calls += 1; return self
        def metric(self, *a, **k): self.metric_calls += 1; return self
        def button(self, *a, **k): self.button_calls += 1; return False
        def selectbox(self, *a, **k):
            # Return first option
            opts = k.get('options') or a[0]
            self.selectbox_val = opts[0] if opts else None
            return self.selectbox_val
        def slider(self, *a, **k):
            return k.get('value', 620)
        def checkbox(self, *a, **k):
            return k.get('value', True)
        def columns(self, n, **k):
            return [MockColumn() for _ in range(n if isinstance(n, int) else len(n))]
        def info(self, *a, **k): return 0
        def warning(self, *a, **k): return 0
        def caption(self, *a, **k): return 0
        def toast(self, *a, **k): return 0
        def rerun(self): pass
        def dataframe(self, *a, **k): self.dataframe_calls += 1

    mock = MockST()
    st = None  # not used

    from dashboard.theme import get_palette
    palette = get_palette("cyber_dark")

    state = TopologyState()
    seed_default_topology(state)

    render(mock, palette=palette, state=state,
           key_prefix="test_topo", show_demo_button=True)

    print(f"  ✅ Render executed without error")
    print(f"      markdown calls:  {mock.markdown_calls}")
    print(f"      plotly calls:    {mock.plotly_calls}")
    print(f"      dataframe calls: {mock.dataframe_calls}")

    assert mock.plotly_calls >= 3
    assert mock.markdown_calls > 0
    assert mock.dataframe_calls >= 1

    return True


def verify_file_checklist():
    print("\n" + "=" * 60)
    print("  PHASE 21: File Checklist")
    print("=" * 60)

    from pathlib import Path
    files = [
        "dashboard/panels/topology.py",
        "dashboard/panels/__init__.py",
        "dashboard/preview_panels.py",
        "tests/unit/test_topology.py",
    ]
    all_ok = True
    for f in files:
        exists = Path(f).is_file()
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"  {status}  {f}")
    return all_ok


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 21: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_topology.py",
         "-v", "--tb=short"],
        capture_output=True, text=True, encoding="utf-8"
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 21 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("TopologyState", verify_state()))
    results.append(("Default Topology", verify_default_topology()))
    results.append(("Figure Builders", verify_figures()))
    results.append(("Activity Simulation", verify_simulation()))
    results.append(("Render Integration", verify_render_integration()))
    results.append(("File Checklist", verify_file_checklist()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 21 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  ┌──────────────────────────────────────────────────────┐")
        print("  │  🎉 PHASE 21 COMPLETE — ALL PANELS BUILT             │")
        print("  │                                                       │")
        print("  │  Preview the panel live:                              │")
        print("  │                                                       │")
        print("  │      streamlit run dashboard/preview_panels.py         │")
        print("  │                                                       │")
        print("  │  Choose 'Topology View' in the sidebar.               │")
        print("  │                                                       │")
        print("  │  NEXT: Phase 22 — Advanced Benchmark Suite            │")
        print("  └──────────────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)