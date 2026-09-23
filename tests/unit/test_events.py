"""
Unit Tests for core.events
"""

import time
import pytest


class TestEvent:
    def test_construction(self):
        from core.events.event import Event, EventPriority
        e = Event(kind="test.kind", payload={"x": 1})
        assert e.kind == "test.kind"
        assert e.priority == EventPriority.INFO
        assert e.payload == {"x": 1}
        assert len(e.event_id) == 10

    def test_to_dict(self):
        from core.events.event import make_event, EventPriority
        e = make_event("foo", {"a": 1}, EventPriority.CRITICAL, source="tester")
        d = e.to_dict()
        assert d['kind'] == "foo"
        assert d['priority'] == "critical"
        assert d['source'] == "tester"
        assert d['payload'] == {"a": 1}


class TestEventBusSubscribe:
    def test_exact_topic_match(self):
        from core.events.bus import EventBus
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda e: received.append(e))

        bus.emit("test.event", {"v": 1})
        bus.emit("test.other", {"v": 2})

        assert len(received) == 1
        assert received[0].payload["v"] == 1

    def test_wildcard_star(self):
        from core.events.bus import EventBus
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))

        bus.emit("a.b")
        bus.emit("c.d")
        bus.emit("anything.goes")

        assert len(received) == 3

    def test_prefix_wildcard(self):
        from core.events.bus import EventBus
        bus = EventBus()
        received = []
        bus.subscribe("drift.*", lambda e: received.append(e))

        bus.emit("drift.nominal")
        bus.emit("drift.critical")
        bus.emit("precision.transition")

        assert len(received) == 2

    def test_multiple_listeners(self):
        from core.events.bus import EventBus
        bus = EventBus()
        hits = {'a': 0, 'b': 0}
        bus.subscribe("e", lambda e: hits.__setitem__('a', hits['a'] + 1))
        bus.subscribe("e", lambda e: hits.__setitem__('b', hits['b'] + 1))

        bus.emit("e")
        assert hits['a'] == 1
        assert hits['b'] == 1

    def test_unsubscribe_returns_function(self):
        from core.events.bus import EventBus
        bus = EventBus()
        received = []
        unsub = bus.subscribe("x", lambda e: received.append(e))

        bus.emit("x")
        assert len(received) == 1

        unsub()
        bus.emit("x")
        assert len(received) == 1  # no new hits

    def test_unsubscribe_all(self):
        from core.events.bus import EventBus
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        bus.subscribe("a", lambda e: None)
        removed = bus.unsubscribe_all()
        assert removed == 3


class TestEventBusPriority:
    def test_priority_filter(self):
        from core.events.bus import EventBus
        from core.events.event import EventPriority
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e),
                      priority_filter=EventPriority.WARNING)

        bus.emit("test", priority=EventPriority.DEBUG)
        bus.emit("test", priority=EventPriority.INFO)
        bus.emit("test", priority=EventPriority.WARNING)
        bus.emit("test", priority=EventPriority.CRITICAL)

        assert len(received) == 2  # Only WARNING and CRITICAL
        assert received[0].priority == EventPriority.WARNING
        assert received[1].priority == EventPriority.CRITICAL


class TestEventBusRobustness:
    def test_handler_exception_isolated(self):
        from core.events.bus import EventBus
        bus = EventBus()

        # Register a deliberately broken handler + a good handler
        bus.subscribe("boom", lambda e: (_ for _ in ()).throw(RuntimeError("bad")))
        bus.subscribe("boom", lambda e: None)

        # Should NOT raise — bus swallows and logs
        delivered = bus.emit("boom")
        assert delivered == 1  # only the good handler's delivery counts
        assert bus.stats()['handler_errors'] == 1

        # Bus should still be usable afterward
        received = []
        bus.subscribe("ok", lambda e: received.append(e))
        bus.emit("ok")
        assert len(received) == 1

    def test_stats_tracking(self):
        from core.events.bus import EventBus
        bus = EventBus()
        bus.subscribe("test", lambda e: None)

        for _ in range(5):
            bus.emit("test")

        stats = bus.stats()
        assert stats['published'] == 5
        assert stats['delivered'] == 5


class TestEventBusHistory:
    def test_history_accumulates(self):
        from core.events.bus import EventBus
        bus = EventBus(history_size=10)
        for i in range(5):
            bus.emit("test", {"i": i})

        history = bus.history()
        assert len(history) == 5
        assert history[0].payload['i'] == 0
        assert history[-1].payload['i'] == 4

    def test_history_ring_bounded(self):
        from core.events.bus import EventBus
        bus = EventBus(history_size=5)
        for i in range(20):
            bus.emit("t", {"i": i})

        history = bus.history()
        assert len(history) == 5
        assert history[0].payload['i'] == 15

    def test_history_filter_by_prefix(self):
        from core.events.bus import EventBus
        bus = EventBus()
        bus.emit("drift.nominal")
        bus.emit("precision.transition")
        bus.emit("drift.critical")

        drift_events = bus.history(topic_prefix="drift")
        assert len(drift_events) == 2

    def test_clear_history(self):
        from core.events.bus import EventBus
        bus = EventBus()
        bus.emit("x")
        bus.emit("y")
        bus.clear_history()
        assert len(bus.history()) == 0


class TestEventBusIntegration:
    def test_default_bus_singleton(self):
        from core.events.bus import get_default_bus
        b1 = get_default_bus()
        b2 = get_default_bus()
        assert b1 is b2

    def test_event_kind_enum(self):
        from core.events.bus import EventBus
        from core.events.event import EventKind
        bus = EventBus()
        received = []
        bus.subscribe(EventKind.DRIFT_CRITICAL.value,
                      lambda e: received.append(e))

        bus.emit(EventKind.DRIFT_CRITICAL.value, {"layer": "enc.0"})
        assert len(received) == 1

    def test_subscribers_count(self):
        from core.events.bus import EventBus
        bus = EventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)

        subs = bus.subscribers()
        assert subs['a'] == 2
        assert subs['b'] == 1

    def test_realistic_drift_flow(self):
        """Simulate the typical dashboard subscription pattern."""
        from core.events.bus import EventBus
        from core.events.event import EventKind

        bus = EventBus()
        critical_events = []
        all_drift_events = []

        bus.subscribe(EventKind.DRIFT_CRITICAL.value,
                      lambda e: critical_events.append(e))
        bus.subscribe("drift.*",
                      lambda e: all_drift_events.append(e))

        bus.emit(EventKind.DRIFT_NOMINAL.value, {"score": 0.05})
        bus.emit(EventKind.DRIFT_WARNING.value, {"score": 0.20})
        bus.emit(EventKind.DRIFT_CRITICAL.value, {"score": 0.45})
        bus.emit(EventKind.DRIFT_EMERGENCY.value, {"score": 0.85})

        assert len(all_drift_events) == 4
        assert len(critical_events) == 1
        assert critical_events[0].payload['score'] == 0.45


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])