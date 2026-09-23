"""
Unit Tests for core.workers
"""

import time
import pytest


class TestTask:
    def test_task_creation(self):
        from core.workers.task import Task
        t = Task(fn=lambda: 42, kind="test")
        assert t.state.value == "pending"
        assert t.kind == "test"
        assert len(t.task_id) == 12

    def test_task_priority_ordering(self):
        from core.workers.task import Task
        low = Task(fn=lambda: None, priority=0)
        high = Task(fn=lambda: None, priority=10)
        assert high < low  # higher priority sorts first

    def test_task_elapsed(self):
        from core.workers.task import Task
        t = Task(fn=lambda: None)
        t.started_at = time.time() - 0.1
        t.finished_at = time.time()
        assert t.elapsed_ms > 50


class TestTaskResult:
    def test_ok_true_when_completed(self):
        from core.workers.task import TaskResult, TaskState
        r = TaskResult(task_id="x", state=TaskState.COMPLETED, value=42)
        assert r.ok is True
        assert r.unwrap() == 42

    def test_unwrap_raises_on_failure(self):
        from core.workers.task import TaskResult, TaskState
        r = TaskResult(task_id="x", state=TaskState.FAILED, error="boom")
        assert r.ok is False
        with pytest.raises(RuntimeError):
            r.unwrap()


class TestWorkerPool:
    def test_start_stop(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=2)
        pool.start()
        assert pool._started is True
        pool.stop()
        assert pool._started is False

    def test_submit_and_wait(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=2)
        try:
            tid = pool.submit_fn(lambda x: x * 2, 21)
            result = pool.wait_for(tid, timeout_s=2.0)
            assert result is not None
            assert result.ok
            assert result.value == 42
        finally:
            pool.stop()

    def test_failure_recorded(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=1)
        try:
            def boom():
                raise ValueError("intentional")
            tid = pool.submit_fn(boom)
            result = pool.wait_for(tid, timeout_s=2.0)
            assert result is not None
            assert not result.ok
            assert "intentional" in result.error
        finally:
            pool.stop()

    def test_retry_succeeds(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=1)
        try:
            state = {'fail_count': 0}

            def flaky():
                state['fail_count'] += 1
                if state['fail_count'] < 3:
                    raise RuntimeError("not yet")
                return "ok"

            tid = pool.submit_fn(flaky, max_retries=5)
            result = pool.wait_for(tid, timeout_s=3.0)
            assert result is not None
            assert result.ok
            assert result.value == "ok"
            assert result.attempts >= 3
        finally:
            pool.stop()

    def test_timeout(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=1, default_timeout_s=0.05)
        try:
            def slow():
                time.sleep(0.5)
                return "never"
            tid = pool.submit_fn(slow, timeout_s=0.05)
            result = pool.wait_for(tid, timeout_s=2.0)
            assert result is not None
            assert result.state.value == "timeout"
        finally:
            pool.stop()

    def test_priority_order(self):
        from core.workers.pool import WorkerPool
        # Single worker forces serial execution
        pool = WorkerPool(num_workers=1)
        order = []
        try:
            # Block the worker with a slow task so subsequent tasks queue
            pool.submit_fn(lambda: time.sleep(0.15), priority=100)
            time.sleep(0.02)  # let it start

            pool.submit_fn(lambda: order.append('low'), priority=0)
            pool.submit_fn(lambda: order.append('high'), priority=10)
            pool.submit_fn(lambda: order.append('mid'), priority=5)

            time.sleep(0.6)
            # High should have run before mid before low
            assert order == ['high', 'mid', 'low'], f"Got: {order}"
        finally:
            pool.stop()

    def test_wait_all(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=3)
        try:
            ids = [pool.submit_fn(lambda i=i: i * i) for i in range(5)]
            results = pool.wait_all(ids, timeout_s=2.0)
            assert len(results) == 5
            values = sorted([r.value for r in results if r.ok])
            assert values == [0, 1, 4, 9, 16]
        finally:
            pool.stop()

    def test_on_result_callback(self):
        from core.workers.pool import WorkerPool
        collected = []
        pool = WorkerPool(num_workers=1, on_result=lambda r: collected.append(r))
        try:
            tid = pool.submit_fn(lambda: "hello")
            pool.wait_for(tid, timeout_s=2.0)
            time.sleep(0.05)
            assert len(collected) >= 1
            assert collected[0].value == "hello"
        finally:
            pool.stop()

    def test_status_and_stats(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=2)
        try:
            pool.submit_fn(lambda: 1)
            pool.submit_fn(lambda: 2)
            time.sleep(0.3)
            status = pool.status()
            assert status['num_workers'] == 2
            assert 'stats' in status
            assert status['stats'].get('submitted', 0) >= 2
        finally:
            pool.stop()

    def test_autostart_on_submit(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=1)
        # Don't call .start() manually
        try:
            tid = pool.submit_fn(lambda: "auto")
            result = pool.wait_for(tid, timeout_s=2.0)
            assert result.ok
            assert result.value == "auto"
        finally:
            pool.stop()

    def test_drain_results(self):
        from core.workers.pool import WorkerPool
        pool = WorkerPool(num_workers=2)
        try:
            for i in range(5):
                pool.submit_fn(lambda i=i: i)
            time.sleep(0.5)
            drained = pool.drain_results()
            assert len(drained) >= 1
        finally:
            pool.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])