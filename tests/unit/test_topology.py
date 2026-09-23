"""
Unit Tests for dashboard.panels.topology
"""

import time
import numpy as np
import pytest


class TestTopologyNode:
    def test_construction(self):
        from dashboard.panels.topology import TopologyNode
        n = TopologyNode(id="x", label="X", category="model", x=1.0, y=2.0)
        assert n.id == "x"
        assert n.status == "nominal"
        assert n.importance == 1.0

    def test_to_dict(self):
        from dashboard.panels.topology import TopologyNode
        n = TopologyNode(id="x", label="X", category="model")
        d = n.to_dict()
        assert d['id'] == 'x'
        assert d['category'] == 'model'


class TestTopologyEdge:
    def test_construction(self):
        from dashboard.panels.topology import TopologyEdge
        e = TopologyEdge("a", "b", label="flow")
        assert e.src == "a"
        assert e.dst == "b"
        assert e.dashed is False

    def test_to_dict(self):
        from dashboard.panels.topology import TopologyEdge
        e = TopologyEdge("a", "b", dashed=True)
        d = e.to_dict()
        assert d['dashed'] is True


class TestTopologyState:
    def test_empty_init(self):
        from dashboard.panels.topology import TopologyState
        s = TopologyState()
        assert len(s.nodes) == 0
        assert len(s.edges) == 0
        assert s.overall_health() == "nominal"

    def test_add_node(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        assert "a" in s.nodes

    def test_add_edge_skips_duplicates(self):
        from dashboard.panels.topology import TopologyState, TopologyNode, TopologyEdge
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        s.add_node(TopologyNode(id="b", label="B", category="model"))
        s.add_edge(TopologyEdge("a", "b"))
        s.add_edge(TopologyEdge("a", "b"))  # duplicate
        assert len(s.edges) == 1

    def test_remove_node_removes_edges(self):
        from dashboard.panels.topology import TopologyState, TopologyNode, TopologyEdge
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        s.add_node(TopologyNode(id="b", label="B", category="model"))
        s.add_edge(TopologyEdge("a", "b"))
        s.remove_node("a")
        assert "a" not in s.nodes
        assert len(s.edges) == 0

    def test_update_status(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        s.update_status("a", "warning")
        assert s.nodes["a"].status == "warning"
        # Invalid status should not change
        s.update_status("a", "invalid")
        assert s.nodes["a"].status == "warning"

    def test_update_metric(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        s.update_metric("a", "latency", "2.4 ms")
        assert s.nodes["a"].metrics["latency"] == "2.4 ms"

    def test_set_all_status(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        for i in range(3):
            s.add_node(TopologyNode(id=f"n{i}", label="N", category="model"))
        s.set_all_status("critical")
        assert all(n.status == "critical" for n in s.nodes.values())

    def test_mark_active_and_rate(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState(activity_window_s=1.0)
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        # 20 events marked → rate should be ~20 Hz
        for _ in range(20):
            s.mark_active("a")
        rate = s.activity_rate("a")
        assert rate > 5.0

    def test_mark_active_ignores_unknown(self):
        from dashboard.panels.topology import TopologyState
        s = TopologyState()
        s.mark_active("unknown")
        assert len(s._activity_events) == 0

    def test_active_nodes(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState(activity_window_s=10.0)
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        s.add_node(TopologyNode(id="b", label="B", category="model"))
        for _ in range(5):
            s.mark_active("a")
        active = s.active_nodes(min_rate=0.05)
        assert "a" in active
        # b had no activity
        assert "b" not in active

    def test_category_activity(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState(activity_window_s=10.0)
        s.add_node(TopologyNode(id="m1", label="M", category="model"))
        s.add_node(TopologyNode(id="a1", label="A", category="analysis"))
        for _ in range(10):
            s.mark_active("m1")
        for _ in range(5):
            s.mark_active("a1")
        cat_act = s.category_activity()
        assert cat_act.get('model', 0) > cat_act.get('analysis', 0)

    def test_activity_histogram(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState(activity_window_s=10.0, activity_bins=10)
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        for _ in range(20):
            s.mark_active("a")
        counts, edges = s.activity_histogram()
        assert counts.shape == (10,)
        assert edges.shape == (11,)

    def test_nodes_by_category(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        s.add_node(TopologyNode(id="m1", label="M1", category="model"))
        s.add_node(TopologyNode(id="m2", label="M2", category="model"))
        s.add_node(TopologyNode(id="p1", label="P1", category="panel"))
        by_cat = s.nodes_by_category()
        assert len(by_cat['model']) == 2
        assert len(by_cat['panel']) == 1

    def test_status_counts(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        for i in range(5):
            s.add_node(TopologyNode(id=f"n{i}", label="N", category="model"))
        s.update_status("n0", "critical")
        s.update_status("n1", "warning")
        counts = s.status_counts()
        assert counts['critical'] == 1
        assert counts['warning'] == 1
        assert counts['nominal'] == 3

    def test_overall_health(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        assert s.overall_health() == "nominal"
        s.update_status("a", "warning")
        assert s.overall_health() == "warning"
        s.update_status("a", "critical")
        assert s.overall_health() == "critical"

    def test_to_dataframe_like(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model", phase="P3"))
        rows = s.to_dataframe_like()
        assert len(rows) == 1
        assert rows[0]['ID'] == 'a'
        assert rows[0]['Phase'] == 'P3'

    def test_reset_activity(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        s.mark_active("a")
        s.reset_activity()
        assert s.activity_rate("a") == 0.0

    def test_reset(self):
        from dashboard.panels.topology import TopologyState, TopologyNode
        s = TopologyState()
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        s.reset()
        assert len(s.nodes) == 0
        assert len(s.edges) == 0


class TestSeedDefaultTopology:
    def test_seed_populates(self):
        from dashboard.panels.topology import TopologyState, seed_default_topology
        s = TopologyState()
        seed_default_topology(s)
        assert len(s.nodes) > 20
        assert len(s.edges) > 20

    def test_seed_has_all_categories(self):
        from dashboard.panels.topology import TopologyState, seed_default_topology
        s = TopologyState()
        seed_default_topology(s)
        cats = s.categories()
        for c in ['source', 'engine', 'model', 'analysis', 'infra', 'panel']:
            assert c in cats

    def test_seed_has_key_nodes(self):
        from dashboard.panels.topology import TopologyState, seed_default_topology
        s = TopologyState()
        seed_default_topology(s)
        for nid in ['engine_learner', 'model_edge', 'analysis_drift',
                    'infra_ipc', 'panel_topology']:
            assert nid in s.nodes

    def test_seed_all_nominal(self):
        from dashboard.panels.topology import TopologyState, seed_default_topology
        s = TopologyState()
        seed_default_topology(s)
        assert s.overall_health() == "nominal"

    def test_seed_idempotent(self):
        from dashboard.panels.topology import TopologyState, seed_default_topology
        s = TopologyState()
        seed_default_topology(s)
        n1 = len(s.nodes)
        seed_default_topology(s)
        n2 = len(s.nodes)
        assert n1 == n2


class TestFigureBuilders:
    def test_topology_empty(self):
        from dashboard.panels.topology import build_topology_figure, TopologyState
        from dashboard.theme import get_palette
        fig = build_topology_figure(TopologyState(), get_palette("cyber_dark"))
        assert fig is not None

    def test_topology_basic(self):
        from dashboard.panels.topology import (
            build_topology_figure, TopologyState, seed_default_topology,
        )
        from dashboard.theme import get_palette
        s = TopologyState()
        seed_default_topology(s)
        fig = build_topology_figure(s, get_palette("cyber_dark"), height=500)
        assert fig.layout.height == 500
        # Should have at least the base node + edge traces
        assert len(fig.data) >= 2

    def test_topology_with_active_glow(self):
        from dashboard.panels.topology import (
            build_topology_figure, TopologyState, seed_default_topology,
        )
        from dashboard.theme import get_palette
        s = TopologyState(activity_window_s=10.0)
        seed_default_topology(s)
        for _ in range(10):
            s.mark_active("engine_learner")
        fig = build_topology_figure(s, get_palette("cyber_dark"), show_active_glow=True)
        # Should have glow trace AND highlight trace
        assert len(fig.data) >= 3

    def test_activity_timeline_empty(self):
        from dashboard.panels.topology import build_activity_timeline, TopologyState
        from dashboard.theme import get_palette
        fig = build_activity_timeline(TopologyState(), get_palette("cyber_dark"))
        assert fig is not None

    def test_activity_timeline_basic(self):
        from dashboard.panels.topology import (
            build_activity_timeline, TopologyState, TopologyNode,
        )
        from dashboard.theme import get_palette
        s = TopologyState(activity_window_s=5.0)
        s.add_node(TopologyNode(id="a", label="A", category="model"))
        for _ in range(30):
            s.mark_active("a")
        fig = build_activity_timeline(s, get_palette("cyber_dark"))
        assert len(fig.data) == 1

    def test_category_activity_bar(self):
        from dashboard.panels.topology import (
            build_category_activity_bar, TopologyState, seed_default_topology,
        )
        from dashboard.theme import get_palette
        s = TopologyState(activity_window_s=10.0)
        seed_default_topology(s)
        for _ in range(10):
            s.mark_active("engine_learner")
        fig = build_category_activity_bar(s, get_palette("cyber_dark"))
        assert len(fig.data) == 1

    def test_all_palettes(self):
        from dashboard.panels.topology import (
            TopologyState, seed_default_topology,
            build_topology_figure, build_activity_timeline,
            build_category_activity_bar,
        )
        from dashboard.theme import get_palette
        s = TopologyState()
        seed_default_topology(s)
        for _ in range(10):
            s.mark_active("engine_learner")

        for pname in ["cyber_dark", "plasma", "mono", "solar"]:
            pal = get_palette(pname)
            build_topology_figure(s, pal)
            build_activity_timeline(s, pal)
            build_category_activity_bar(s, pal)


class TestPackageExports:
    def test_exports(self):
        from dashboard.panels import (
            TopologyNode, TopologyEdge, TopologyState,
            seed_default_topology, build_topology_figure,
            build_activity_timeline, build_category_activity_bar,
            render_topology,
        )
        assert callable(render_topology)
        assert TopologyState is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])