# NeuralStream-Core Configuration Profiles

Pre-curated YAML profiles for common deployment scenarios. Each
profile fully overrides the defaults defined in `config/schema.py`.

| Profile | Use Case | Latency Target | 3D Plots | Notes |
|---------|----------|---------------|----------|-------|
| **default.yaml** | Balanced baseline | 3.0 ms | ✓ | Best for demos and typical laptops |
| **high_performance.yaml** | Max throughput | 1.5 ms | ✓ | Uses more workers + larger batches |
| **edge.yaml** | Constrained devices | 5.0 ms | ✗ | Small inputs, single worker, relaxed targets |
| **ci.yaml** | Continuous integration | 10.0 ms | ✗ | Fixed seed, tiny batches, minimal logging |
| **debug.yaml** | Troubleshooting | 3.0 ms | ✓ | All drift methods, deep history, verbose logs |

## Usage

### From Python

```python
from config.loader import load_config

# Load a profile by name
cfg = load_config(profile="edge")

# Override specific values at runtime
cfg = load_config(profile="ci", overrides={"stream": {"batch_size": 2}})