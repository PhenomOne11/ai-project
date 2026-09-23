"""
Phase 1 Verification Script
Validates directory structure, imports, and configuration loading.
"""

import sys
import os
from pathlib import Path


def check_directories():
    """Verify all required directories exist."""
    required_dirs = [
        "core", "core/models", "core/drift", "core/ipc", "core/precision",
        "core/adaptation", "core/anomaly", "core/streams", "core/metrics",
        "core/events", "core/workers", "core/math_utils",
        "dashboard", "dashboard/theme", "dashboard/components",
        "dashboard/panels", "dashboard/layouts",
        "config", "config/profiles",
        "benchmarks", "tests", "tests/unit", "tests/integration",
        "assets", "logs", "data", "data/samples"
    ]
    
    print("=" * 60)
    print("  PHASE 1 VERIFICATION: Directory Structure")
    print("=" * 60)
    
    all_ok = True
    for d in required_dirs:
        exists = Path(d).is_dir()
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"  {status}  {d}/")
    
    return all_ok


def check_init_files():
    """Verify all __init__.py files exist."""
    init_dirs = [
        "core", "core/models", "core/drift", "core/ipc", "core/precision",
        "core/adaptation", "core/anomaly", "core/streams", "core/metrics",
        "core/events", "core/workers", "core/math_utils",
        "dashboard", "dashboard/theme", "dashboard/components",
        "dashboard/panels", "dashboard/layouts",
        "config", "config/profiles",
        "benchmarks", "tests", "tests/unit", "tests/integration"
    ]
    
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION: Package Init Files")
    print("=" * 60)
    
    all_ok = True
    for d in init_dirs:
        init_path = Path(d) / "__init__.py"
        exists = init_path.is_file()
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"  {status}  {init_path}")
    
    return all_ok


def check_core_files():
    """Verify all Phase 1 source files exist."""
    required_files = [
        "requirements.txt",
        "config/schema.py",
        "config/loader.py",
        "config/profiles/default.yaml",
        "config/profiles/high_performance.yaml",
        "core/logger.py",
        "core/exceptions.py",
        "core/constants.py",
    ]
    
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION: Source Files")
    print("=" * 60)
    
    all_ok = True
    for f in required_files:
        exists = Path(f).is_file()
        status = "✅" if exists else "❌"
        if not exists:
            all_ok = False
        print(f"  {status}  {f}")
    
    return all_ok


def check_imports():
    """Verify all modules can be imported."""
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION: Module Imports")
    print("=" * 60)
    
    modules = [
        ("config.schema", "SystemConfig"),
        ("config.loader", "load_config"),
        ("core.logger", "setup_logger"),
        ("core.exceptions", "NeuralStreamError"),
        ("core.constants", "PROJECT_NAME"),
    ]
    
    all_ok = True
    for module_name, attr_name in modules:
        try:
            mod = __import__(module_name, fromlist=[attr_name])
            attr = getattr(mod, attr_name)
            print(f"  ✅  {module_name}.{attr_name} → {type(attr).__name__}")
        except Exception as e:
            print(f"  ❌  {module_name}.{attr_name} → {e}")
            all_ok = False
    
    return all_ok


def check_config_loading():
    """Verify configuration system works end-to-end."""
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION: Configuration System")
    print("=" * 60)
    
    try:
        from config.loader import load_config
        
        # Test 1: Load default config
        cfg = load_config()
        print(f"  ✅  Default config loaded: {cfg.project_name} v{cfg.version}")
        print(f"      Model arch: {cfg.model.architecture}")
        print(f"      Latent dim: {cfg.model.latent_dim}")
        print(f"      Batch size: {cfg.stream.batch_size}")
        print(f"      Precision target: {cfg.precision.target_latency_ms} ms")
        
        # Test 2: Load with overrides
        cfg2 = load_config(overrides={"model": {"latent_dim": 64}, "debug": True})
        assert cfg2.model.latent_dim == 64
        assert cfg2.debug is True
        print(f"  ✅  Override config validated: latent_dim={cfg2.model.latent_dim}, debug={cfg2.debug}")
        
        # Test 3: Load high performance profile
        hp_path = "config/profiles/high_performance.yaml"
        if Path(hp_path).exists():
            cfg3 = load_config(profile_path=hp_path)
            print(f"  ✅  High-perf profile loaded: batch_size={cfg3.stream.batch_size}, latent_dim={cfg3.model.latent_dim}")
        
        return True
    except Exception as e:
        print(f"  ❌  Configuration loading failed: {e}")
        return False


def check_logger():
    """Verify logging system works."""
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION: Logging System")
    print("=" * 60)
    
    try:
        from core.logger import setup_logger, get_logger
        
        logger = setup_logger("test_phase1", level="DEBUG", enable_file=False)
        logger.info("Phase 1 verification logger operational")
        print(f"  ✅  Logger created: {logger.name}, handlers={len(logger.handlers)}")
        
        logger2 = get_logger("test_phase1")
        assert logger2 is logger
        print(f"  ✅  Logger retrieval consistent")
        
        return True
    except Exception as e:
        print(f"  ❌  Logger setup failed: {e}")
        return False


def check_exceptions():
    """Verify exception hierarchy."""
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION: Exception Hierarchy")
    print("=" * 60)
    
    try:
        from core.exceptions import (
            NeuralStreamError, CancelBatchException, DataCorruptionError,
            DriftDetectionError, IPCError, SharedMemoryError,
            CircuitBreakerTripped, EngineBootError
        )
        
        # Verify inheritance
        assert issubclass(CancelBatchException, NeuralStreamError)
        assert issubclass(SharedMemoryError, IPCError)
        assert issubclass(IPCError, NeuralStreamError)
        assert issubclass(CircuitBreakerTripped, NeuralStreamError)
        
        # Verify they can be raised and caught
        try:
            raise CancelBatchException("Test batch cancellation")
        except NeuralStreamError as e:
            pass
        
        print(f"  ✅  Exception hierarchy validated (8 classes checked)")
        print(f"      CancelBatchException → NeuralStreamError ✓")
        print(f"      SharedMemoryError → IPCError → NeuralStreamError ✓")
        
        return True
    except Exception as e:
        print(f"  ❌  Exception hierarchy failed: {e}")
        return False


def check_constants():
    """Verify constants module."""
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION: Constants Module")
    print("=" * 60)
    
    try:
        from core.constants import (
            PROJECT_NAME, VERSION, EngineState, PrecisionState,
            DriftSeverity, HealthStatus, DEFAULT_BATCH_SIZE,
            LATENCY_TARGET_NORMAL_MS, DRIFT_ALERT_THRESHOLD
        )
        
        print(f"  ✅  {PROJECT_NAME} v{VERSION} [{len(EngineState)} engine states]")
        print(f"      Precision states: {[s.value for s in PrecisionState]}")
        print(f"      Drift severity levels: {[s.value for s in DriftSeverity]}")
        print(f"      Health statuses: {[s.value for s in HealthStatus]}")
        print(f"      Defaults: batch={DEFAULT_BATCH_SIZE}, latency={LATENCY_TARGET_NORMAL_MS}ms, drift_alert={DRIFT_ALERT_THRESHOLD}")
        
        return True
    except Exception as e:
        print(f"  ❌  Constants module failed: {e}")
        return False


def main():
    print("\n" + "🔬" * 30)
    print("  NEURALSTREAM-CORE: PHASE 1 FULL VERIFICATION")
    print("🔬" * 30 + "\n")
    
    results = []
    results.append(("Directory Structure", check_directories()))
    results.append(("Package Init Files", check_init_files()))
    results.append(("Source Files", check_core_files()))
    results.append(("Module Imports", check_imports()))
    results.append(("Configuration System", check_config_loading()))
    results.append(("Logging System", check_logger()))
    results.append(("Exception Hierarchy", check_exceptions()))
    results.append(("Constants Module", check_constants()))
    
    print("\n" + "=" * 60)
    print("  PHASE 1 VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_pass = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")
    
    print(f"\n  Result: {total_pass}/{total} checks passed")
    
    if total_pass == total:
        print("\n  🎉 PHASE 1 COMPLETE — Ready for Phase 2!")
        print("  Next: Core Math Utilities & Tensor Operations Library\n")
    else:
        print("\n  ⚠️  Fix failing checks before proceeding to Phase 2.\n")
    
    return total_pass == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)