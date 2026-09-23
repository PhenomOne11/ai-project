"""
Phase 13 Verification Script
"""

import sys
import time


def verify_basic_pubsub():
    print("=" * 60)
    print("  PHASE 13: Basic Pub/Sub")
    print("=" * 60)

    from core.events import EventBus

    bus = EventBus()
    received = []
    bus.subscribe("test.event", lambda e: received.append(e))

    bus.emit("test.event", {"msg": "hello"})
    bus.emit("test.other", {"msg": "ignored"})

    print(f"  ✅ Published 2, received {len(received)}")
    print(f"      Payload: {received[0].payload}")

    assert len(received) == 1
    return True


def verify_wildcards():
    print("\n" + "=" * 60)
    print("  PHASE 13: Wildcard Subscriptions")
    print("=" * 60)

    from core.events import EventBus

    bus = EventBus()
    all_events = []
    drift_only = []
    critical_only = []

    bus.subscribe("*", lambda e: all_events.append(e))
    bus.subscribe("drift.*", lambda e: drift_only.append(e))
    bus.subscribe("drift.critical", lambda e: critical_only.append(e))

    bus.emit("drift.nominal")
    bus.emit("drift.warning")
    bus.emit("drift.critical")
    bus.emit("precision.transition")
    bus.emit("adaptation.step")

    print(f"  ✅ All events (*):       {len(all_events)}")
    print(f"  ✅ Drift events (drift.*): {len(drift_only)}")
    print(f"  ✅ Critical (exact):     {len(critical_only)}")

    assert len(all_events) == 5
    assert len(drift_only) == 3
    assert len(critical_only) == 1
    return True


def verify_priority():
    print("\n" + "=" * 60)
    print("  PHASE 13: Priority Filtering")
    print("=" * 60)

    from core.events import EventBus, EventPriority

    bus = EventBus()
    warns_and_up = []
    crits_only = []

    bus.subscribe("any.*", lambda e: warns_and_up.append(e),
                  priority_filter=EventPriority.WARNING)
    bus.subscribe("any.*", lambda e: crits_only.append(e),
                  priority_filter=EventPriority.CRITICAL)

    bus.emit("any.debug", priority=EventPriority.DEBUG)
    bus.emit("any.info", priority=EventPriority.INFO)
    bus.emit("any.warn", priority=EventPriority.WARNING)
    bus.emit("any.crit", priority=EventPriority.CRITICAL)

    print(f"  ✅ WARN+ received:  {len(warns_and_up)}")
    print(f"  ✅ CRIT-only:       {len(crits_only)}")

    assert len(warns_and_up) == 2
    assert len(crits_only) == 1
    return True


def verify_robustness():
    print("\n" + "=" * 60)
    print("  PHASE 13: Handler Error Isolation")
    print("=" * 60)

    from core.events import EventBus

    bus = EventBus()
    good_calls = []

    def bad_handler(event):
        raise ValueError("intentional failure")

    def good_handler(event):
        good_calls.append(event)

    bus.subscribe("test", bad_handler)
    bus.subscribe("test", good_handler)

    # Should not raise
    for i in range(5):
        bus.emit("test", {"i": i})

    print(f"  ✅ 5 bad handlers thrown, 5 good handlers survived: {len(good_calls)}")
    stats = bus.stats()
    print(f"      Stats: published={stats['published']}, "
          f"delivered={stats['delivered']}, "
          f"errors={stats['handler_errors']}")

    assert len(good_calls) == 5
    assert stats['handler_errors'] == 5
    return True


def verify_history():
    print("\n" + "=" * 60)
    print("  PHASE 13: Event History")
    print("=" * 60)

    from core.events import EventBus

    bus = EventBus(history_size=20)
    for i in range(15):
        bus.emit("test.event", {"i": i})

    history = bus.history()
    print(f"  ✅ History has {len(history)} events")
    print(f"      Latest: {history[-1].payload}")

    # Filter
    for i in range(5):
        bus.emit("other.event", {"j": i})

    test_only = bus.history(topic_prefix="test")
    print(f"  ✅ Filtered by prefix 'test': {len(test_only)}")

    # Bounded
    for _ in range(30):
        bus.emit("filler")
    print(f"  ✅ After 50 total, history bounded to: {len(bus.history())}")

    assert len(bus.history()) == 20
    return True


def verify_integration():
    print("\n" + "=" * 60)
    print("  PHASE 13: Realistic Integration Flow")
    print("=" * 60)

    from core.events import EventBus, EventKind, EventPriority

    bus = EventBus()
    alerts = {'drift': 0, 'precision': 0, 'adaptation': 0}

    bus.subscribe("drift.*", lambda e: alerts.__setitem__('drift', alerts['drift'] + 1))
    bus.subscribe("precision.*", lambda e: alerts.__setitem__('precision', alerts['precision'] + 1))
    bus.subscribe("adaptation.*", lambda e: alerts.__setitem__('adaptation', alerts['adaptation'] + 1))

    # Simulate pipeline events
    bus.emit(EventKind.ENGINE_BOOT.value)
    bus.emit(EventKind.ENGINE_START.value)
    bus.emit(EventKind.BATCH_PROCESSED.value, {"batch_id": 1})

    bus.emit(EventKind.DRIFT_WARNING.value, {"score": 0.20, "layer": "enc.0"})
    bus.emit(EventKind.DRIFT_CRITICAL.value, {"score": 0.42, "layer": "enc.2"})

    bus.emit(EventKind.PRECISION_TRANSITION.value, {"from": "FP32", "to": "FP16"})

    bus.emit(EventKind.OOD_DETECTED.value, {"count": 5})
    bus.emit(EventKind.ADAPTATION_STEP.value, {"count": 1, "delta": -0.02})

    print(f"  ✅ Event counts:")
    print(f"      Drift events:      {alerts['drift']}")
    print(f"      Precision events:  {alerts['precision']}")
    print(f"      Adaptation events: {alerts['adaptation']}")
    print(f"      Total history:     {len(bus.history())}")

    bus.print_stats()

    assert alerts['drift'] == 2
    assert alerts['precision'] == 1
    assert alerts['adaptation'] == 2
    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 13: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_events.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 13 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Basic Pub/Sub", verify_basic_pubsub()))
    results.append(("Wildcards", verify_wildcards()))
    results.append(("Priority Filtering", verify_priority()))
    results.append(("Error Isolation", verify_robustness()))
    results.append(("History", verify_history()))
    results.append(("Integration Flow", verify_integration()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 13 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 13 COMPLETE — Backend Engine Done!")
        print("\n  ┌─────────────────────────────────────────────────┐")
        print("  │  🎨 NEXT: DASHBOARD REBUILD (Phases 14–21)      │")
        print("  │  This is where the visual magic happens.        │")
        print("  │                                                  │")
        print("  │  • Phase 14: Theme Engine & Layout System       │")
        print("  │  • Phase 15: 3D Latent Space Explorer           │")
        print("  │  • Phase 16: Multi-Layer Drift Heatmap          │")
        print("  │  • Phase 17: Hardware Telemetry Dashboard       │")
        print("  │  • Phase 18: Anomaly & OOD Console              │")
        print("  │  • Phase 19: Model Comparison & A/B Workspace   │")
        print("  │  • Phase 20: Waveform Oscilloscope              │")
        print("  │  • Phase 21: System Topology Visualizer         │")
        print("  └─────────────────────────────────────────────────┘\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)