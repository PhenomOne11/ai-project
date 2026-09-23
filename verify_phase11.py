"""
Phase 11 Verification Script
"""

import sys
import time


def verify_task_lifecycle():
    print("=" * 60)
    print("  PHASE 11: Task Lifecycle")
    print("=" * 60)

    from core.workers.task import Task, TaskResult, TaskState

    t = Task(fn=lambda x: x ** 2, args=(5,), kind="square")
    print(f"  ✅ Task created: id={t.task_id}, kind={t.kind}")
    print(f"      State: {t.state.value}, Priority: {t.priority}")

    r = TaskResult(task_id=t.task_id, state=TaskState.COMPLETED, value=25)
    print(f"  ✅ Result: ok={r.ok}, value={r.unwrap()}")

    return True


def verify_pool_basic():
    print("\n" + "=" * 60)
    print("  PHASE 11: Worker Pool Basics")
    print("=" * 60)

    from core.workers.pool import WorkerPool

    pool = WorkerPool(num_workers=4)
    try:
        # Submit 20 tasks
        t0 = time.perf_counter()
        ids = [pool.submit_fn(lambda i=i: i * 10, priority=i % 3) for i in range(20)]
        results = pool.wait_all(ids, timeout_s=3.0)
        elapsed = (time.perf_counter() - t0) * 1000

        successful = [r for r in results if r.ok]
        values = sorted([r.value for r in successful])

        print(f"  ✅ Submitted 20 tasks, {len(successful)} succeeded in {elapsed:.1f} ms")
        print(f"      Values: {values[:8]}...")
        print(f"      Status: {pool.status()}")

        assert len(successful) == 20
    finally:
        pool.stop()

    return True


def verify_pool_retry():
    print("\n" + "=" * 60)
    print("  PHASE 11: Retry & Failure Handling")
    print("=" * 60)

    from core.workers.pool import WorkerPool

    pool = WorkerPool(num_workers=1)
    try:
        # Flaky task: fails twice, then succeeds
        state = {'n': 0}

        def flaky():
            state['n'] += 1
            if state['n'] < 3:
                raise RuntimeError(f"attempt {state['n']} failed")
            return f"success after {state['n']} attempts"

        tid = pool.submit_fn(flaky, max_retries=5)
        result = pool.wait_for(tid, timeout_s=3.0)

        print(f"  ✅ Flaky task: ok={result.ok}, attempts={result.attempts}")
        print(f"      Value: {result.value}")
        assert result.ok
        assert result.attempts == 3

        # Permanent failure
        def always_fail():
            raise ValueError("permanent")

        tid2 = pool.submit_fn(always_fail, max_retries=2)
        result2 = pool.wait_for(tid2, timeout_s=3.0)
        print(f"  ✅ Permanent failure: ok={result2.ok}, attempts={result2.attempts}")
        print(f"      Error: {result2.error[:60]}...(truncated)")
        assert not result2.ok
    finally:
        pool.stop()

    return True


def verify_pool_timeout():
    print("\n" + "=" * 60)
    print("  PHASE 11: Timeout Enforcement")
    print("=" * 60)

    from core.workers.pool import WorkerPool

    pool = WorkerPool(num_workers=1, default_timeout_s=0.05)
    try:
        def slow():
            time.sleep(0.5)
            return "never"

        tid = pool.submit_fn(slow, timeout_s=0.05)
        result = pool.wait_for(tid, timeout_s=2.0)

        print(f"  ✅ Timeout task: state={result.state.value}")
        print(f"      Elapsed: {result.elapsed_ms:.1f} ms (target < 100 ms)")
        assert result.state.value == "timeout"
    finally:
        pool.stop()

    return True


def verify_priority():
    print("\n" + "=" * 60)
    print("  PHASE 11: Priority Ordering")
    print("=" * 60)

    from core.workers.pool import WorkerPool

    pool = WorkerPool(num_workers=1)
    order = []
    try:
        # Block worker first
        pool.submit_fn(lambda: time.sleep(0.2), priority=100)
        time.sleep(0.02)

        pool.submit_fn(lambda: order.append('low'), priority=0)
        pool.submit_fn(lambda: order.append('high'), priority=10)
        pool.submit_fn(lambda: order.append('mid'), priority=5)

        time.sleep(0.6)
        print(f"  ✅ Execution order: {order}")
        print(f"      Expected order: ['high', 'mid', 'low']")
        assert order == ['high', 'mid', 'low']
    finally:
        pool.stop()

    return True


def run_pytest():
    print("\n" + "=" * 60)
    print("  PHASE 11: Running Unit Tests")
    print("=" * 60)

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/test_workers.py", "-v", "--tb=short"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 11 FULL VERIFICATION")
    print("🔬" * 30 + "\n")

    results = []
    results.append(("Task Lifecycle", verify_task_lifecycle()))
    results.append(("Pool Basics", verify_pool_basic()))
    results.append(("Retry & Failure", verify_pool_retry()))
    results.append(("Timeout", verify_pool_timeout()))
    results.append(("Priority Ordering", verify_priority()))
    results.append(("Unit Tests (pytest)", run_pytest()))

    print("\n" + "=" * 60)
    print("  PHASE 11 VERIFICATION SUMMARY")
    print("=" * 60)

    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Result: {total_pass}/{total} checks passed")

    if total_pass == total:
        print("\n  🎉 PHASE 11 COMPLETE — Ready for Phase 12!")
        print("  Next: Prometheus-style Metrics Collector (short!)\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding.\n")

    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)