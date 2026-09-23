"""
Thread-safe Event Bus with wildcard subscriptions and bounded history.

Usage:
    bus = EventBus()

    def on_drift(event):
        print("Drift alert:", event.payload)

    bus.subscribe(EventKind.DRIFT_CRITICAL, on_drift)
    bus.subscribe("drift.*", on_drift)       # wildcard prefix match
    bus.subscribe("*", on_drift)             # catch-all

    bus.publish(make_event(EventKind.DRIFT_CRITICAL, {"layer": "enc.0"}))
"""

import time
import threading
import fnmatch
from typing import Callable, Dict, List, Optional, Any
from collections import deque, defaultdict

from core.events.event import Event, EventKind, EventPriority
from core.logger import get_logger

logger = get_logger("neuralstream.events")

EventListener = Callable[[Event], None]


class EventBus:
    """
    In-process publish/subscribe event bus.

    Features:
        - Thread-safe subscription and publication
        - Exact-match and wildcard (`*`, `prefix.*`) topic matching
        - Bounded history ring for dashboard timelines
        - Per-subscriber error isolation (a throwing handler doesn't crash the bus)
        - Optional priority filter (only forward events at or above a threshold)
    """

    def __init__(
        self,
        history_size: int = 500,
        default_priority_filter: EventPriority = EventPriority.DEBUG
    ):
        self.history_size = history_size
        self.default_priority_filter = default_priority_filter

        # topic string → list of (listener, priority_filter)
        self._subscribers: Dict[str, List[tuple]] = defaultdict(list)
        self._lock = threading.RLock()

        # history ring
        self._history: deque = deque(maxlen=history_size)

        # counters
        self._published: int = 0
        self._delivered: int = 0
        self._handler_errors: int = 0

    # ---- Subscription ----

    def subscribe(
        self,
        topic: str,
        listener: EventListener,
        priority_filter: Optional[EventPriority] = None
    ) -> Callable[[], None]:
        """
        Register an event listener.

        Args:
            topic: Topic pattern (`"drift.critical"`, `"drift.*"`, or `"*"`)
            listener: Callable accepting an `Event`
            priority_filter: Minimum priority for delivery. If None, uses
                             bus default.

        Returns:
            Unsubscribe function (callable, no args) for convenience.
        """
        if priority_filter is None:
            priority_filter = self.default_priority_filter

        entry = (listener, priority_filter)
        with self._lock:
            self._subscribers[topic].append(entry)

        def _unsubscribe():
            self.unsubscribe(topic, listener)

        return _unsubscribe

    def unsubscribe(self, topic: str, listener: EventListener) -> bool:
        """Remove a specific listener from a specific topic. Returns True if found."""
        with self._lock:
            if topic not in self._subscribers:
                return False
            before = len(self._subscribers[topic])
            self._subscribers[topic] = [
                (l, pf) for (l, pf) in self._subscribers[topic] if l != listener
            ]
            after = len(self._subscribers[topic])
            if after == 0:
                del self._subscribers[topic]
            return after < before

    def unsubscribe_all(self, listener: Optional[EventListener] = None) -> int:
        """
        Remove listener(s). If listener is None, clears *all* subscribers.
        Returns number of entries removed.
        """
        removed = 0
        with self._lock:
            if listener is None:
                removed = sum(len(v) for v in self._subscribers.values())
                self._subscribers.clear()
            else:
                for topic in list(self._subscribers.keys()):
                    before = len(self._subscribers[topic])
                    self._subscribers[topic] = [
                        (l, pf) for (l, pf) in self._subscribers[topic] if l != listener
                    ]
                    removed += before - len(self._subscribers[topic])
                    if not self._subscribers[topic]:
                        del self._subscribers[topic]
        return removed

    # ---- Publishing ----

    def publish(self, event: Event) -> int:
        """
        Dispatch an event to all matching subscribers.

        Returns number of successful deliveries.
        """
        # Normalize kind to string (allow EventKind enum input)
        kind = event.kind
        if isinstance(kind, EventKind):
            kind = kind.value
            event.kind = kind

        with self._lock:
            self._history.append(event)
            self._published += 1

            # Collect matching subscribers
            matches: List[tuple] = []
            for pattern, subs in self._subscribers.items():
                if self._topic_matches(pattern, kind):
                    matches.extend(subs)

        delivered = 0
        errors = 0
        for listener, pf in matches:
            if not self._priority_passes(event.priority, pf):
                continue
            try:
                listener(event)
                delivered += 1
            except Exception as e:
                errors += 1
                logger.warning(f"Event handler failed for {kind}: {e}")

        with self._lock:
            self._delivered += delivered
            self._handler_errors += errors

        return delivered

    def emit(
        self,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.INFO,
        source: Optional[str] = None
    ) -> int:
        """Convenience wrapper: build Event and publish."""
        event = Event(
            kind=kind if isinstance(kind, str) else str(kind),
            priority=priority,
            payload=payload or {},
            source=source,
        )
        return self.publish(event)

    # ---- Introspection ----

    @staticmethod
    def _topic_matches(pattern: str, topic: str) -> bool:
        """Topic pattern match: exact, glob (`drift.*`), or catch-all (`*`)."""
        if pattern == "*":
            return True
        if pattern == topic:
            return True
        if '*' in pattern:
            return fnmatch.fnmatchcase(topic, pattern)
        return False

    @staticmethod
    def _priority_passes(event_p: EventPriority, min_p: EventPriority) -> bool:
        order = {
            EventPriority.DEBUG: 0,
            EventPriority.INFO: 1,
            EventPriority.WARNING: 2,
            EventPriority.CRITICAL: 3,
        }
        return order[event_p] >= order[min_p]

    def history(self, limit: Optional[int] = None, topic_prefix: Optional[str] = None) -> List[Event]:
        """
        Return recent events from the history ring.
        Optionally filter by topic prefix.
        """
        with self._lock:
            events = list(self._history)

        if topic_prefix is not None:
            events = [e for e in events if e.kind.startswith(topic_prefix)]

        if limit is not None:
            events = events[-limit:]
        return events

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def subscribers(self) -> Dict[str, int]:
        """Map of topic pattern → number of subscribers."""
        with self._lock:
            return {t: len(subs) for t, subs in self._subscribers.items()}

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'published': self._published,
                'delivered': self._delivered,
                'handler_errors': self._handler_errors,
                'history_size': len(self._history),
                'topic_patterns': len(self._subscribers),
                'total_subscribers': sum(len(v) for v in self._subscribers.values()),
            }

    def print_stats(self) -> None:
        s = self.stats()
        print("=" * 60)
        print("  EVENT BUS STATUS")
        print("=" * 60)
        print(f"  Published:        {s['published']}")
        print(f"  Delivered:        {s['delivered']}")
        print(f"  Handler errors:   {s['handler_errors']}")
        print(f"  History size:     {s['history_size']}")
        print(f"  Topic patterns:   {s['topic_patterns']}")
        print(f"  Total listeners:  {s['total_subscribers']}")
        print("=" * 60)


# ---- Default bus ----

_DEFAULT_BUS: Optional[EventBus] = None


def get_default_bus() -> EventBus:
    """Module-level default event bus (lazy-init)."""
    global _DEFAULT_BUS
    if _DEFAULT_BUS is None:
        _DEFAULT_BUS = EventBus()
    return _DEFAULT_BUS