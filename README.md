<div align="center">

# ⚡ NeuralStream-Core

**Enterprise-grade streaming autoencoder with dynamic precision scaling, Wasserstein drift radar, and a glassmorphic mission-control HUD.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## What Is NeuralStream-Core?

A complete, production-ready system for **streaming anomaly detection and adaptive inference** on edge hardware. It combines:

- **4 model architectures** (EdgeStreamingAE, ConvVAE, TransformerAE, UNetAE) plus attention modules and ensembles
- **Multi-method drift detection** using 7 statistical distances including Wasserstein-1
- **Dynamic precision scaling** (FP32 → BF16 → FP16 → INT8) with thermal awareness
- **EWC-protected online adaptation** that prevents catastrophic forgetting
- **Three-way anomaly scoring** (reconstruction + Isolation Forest + Mahalanobis)
- **Multi-modal ingestion** (images, audio, time-series, CSV, tensors)
- **Zero-copy IPC** via shared memory ring buffers
- **7-panel themed HUD** with real-time 3D PCA, drift heatmaps, telemetry gauges, waveform oscope, and topology graph

Everything is wrapped in a **unified CLI** (`neuralstream`) and **5 curated config profiles** so you can go from `pip install` to live dashboard in under a minute.

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd NeuralStream-Core
pip install -r requirements.txt
```

### 2. Optional: Install the CLI

```bash
pip install -e .
```

This gives you a global `neuralstream` command.

### 3. Verify Everything Works

```bash
python -m cli config validate       # all profiles valid
python -m cli bench --no-baseline   # benchmark every subsystem
python -m pytest tests/ -q          # ~200 unit + integration tests
```

### 4. Launch the Dashboard

**Windows:**
```cmd
run.bat
```

**Linux / macOS:**
```bash
./run.sh
```

Or directly:
```bash
python -m cli demo
```

Or:
```bash
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 5. Load Demo Data

In the sidebar, click **🌱 Load All Demo Data** — all 7 panels populate with illustrative data so you can see every visualization immediately.

---

## The 7 Dashboard Panels

| Panel | Purpose |
|-------|---------|
| 🌐 **Overview** | Cross-panel snapshot + topology + telemetry summary |
| 📡 **Latent Explorer** | 3D PCA projection of the 32-D bottleneck with time/cluster/novelty coloring |
| 🧠 **Drift Radar** | Layer × tick Wasserstein heatmap with severity bands and alert timeline |
| 📈 **Telemetry** | Real-time CPU/RAM/latency gauges + rolling chart + latency histogram |
| 🚨 **Anomaly Console** | Score timeline, component breakdown, OOD replay inspector, adaptation deltas |
| ⚖️ **Model Comparison** | A/B radar + latency distributions + parameter/memory/throughput leaderboards |
| 📻 **Oscilloscope** | Multi-channel waveform + FFT spectrum + spectrogram waterfall |
| 🗺️ **Topology** | Live node graph of the entire system with activity-driven glow |

---

## CLI Reference

```bash
# System + environment summary
neuralstream info --profile edge

# Configuration
neuralstream config show --profile high_performance
neuralstream config validate
neuralstream config dump --profile ci --format json > config.json

# Benchmark suite
neuralstream bench                              # compare to baseline
neuralstream bench --update-baseline            # refresh baseline
neuralstream bench --html report.html           # save HTML report
neuralstream bench --no-baseline --iterations 5 # quick check

# Run phase verifiers (from the historical build)
neuralstream verify 5                            # single phase
neuralstream verify 20-22                        # range
neuralstream verify all                          # every verifier

# Launch dashboard
neuralstream demo --profile debug --port 8502
```

---

## Configuration Profiles

| Profile | Input | Latent | Latency | 3D Plots | Workers | Use Case |
|---------|-------|--------|---------|----------|---------|----------|
| `default` | 64 | 32 | 3.0 ms | ✓ | 2+1+1 | Balanced baseline |
| `high_performance` | 64 | 64 | 1.5 ms | ✓ | 4+2+2 | Max throughput |
| `edge` | 32 | 16 | 5.0 ms | ✗ | 1+0+0 | Raspberry Pi / Jetson |
| `ci` | 32 | 16 | 10.0 ms | ✗ | 1+0+0 | CI/CD pipelines |
| `debug` | 64 | 32 | 3.0 ms | ✓ | 1+1+1 | Troubleshooting |

### Adding Your Own Profile

1. Copy any file from `config/profiles/`
2. Edit the values you want to override
3. Save as `config/profiles/my_name.yaml`
4. Use `--profile my_name` from the CLI

**Everything omitted falls back to schema defaults from `config/schema.py`.**

---

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   SOURCES    │ → │    ENGINE    │ → │    MODELS    │ → │   ANALYSIS   │
├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤
│ Synthetic    │   │ Bootstrap    │   │ EdgeAE       │   │ Drift Radar  │
│ .npy / .pt   │   │ Learner Loop │   │ ConvVAE      │   │ Precision    │
│ Images       │   │ LSUV Calib   │   │ U-Net AE     │   │ Anomaly      │
│ Audio        │   │              │   │ Transformer  │   │ Adaptation   │
│ CSV / Wav    │   │              │   │ Ensemble     │   │ Breaker      │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                │
                                                                ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  PANELS (7)  │ ← │     IPC      │ ← │    INFRA     │ ← │  ANALYSIS    │
├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤
│ Latent       │   │ Ring Buffer  │   │ Worker Pool  │   │ (continued)  │
│ Drift        │   │ Bus          │   │ Metrics      │   │              │
│ Telemetry    │   │ Channels     │   │ Event Bus    │   │              │
│ Anomaly      │   │              │   │              │   │              │
│ Compare      │   │              │   │              │   │              │
│ Oscilloscope │   │              │   │              │   │              │
│ Topology     │   │              │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### Module Layout

```
NeuralStream-Core/
├── app.py                       # Dashboard entry (Phase 25)
├── run.bat / run.sh             # One-click launchers
├── pyproject.toml               # Package metadata + CLI entry
├── requirements.txt
├── config/
│   ├── schema.py                # Pydantic v2 validation
│   ├── loader.py                # YAML loading + overrides
│   └── profiles/*.yaml          # 5 curated profiles
├── core/
│   ├── math_utils/              # Welford, PCA, distances, signals
│   ├── models/                  # 4 architectures + attention + ensemble
│   ├── drift/                   # Drift detector + radar
│   ├── ipc/                     # Shared memory ring buffer + bus
│   ├── precision/               # Thermal + adaptive precision v2
│   ├── adaptation/              # Fisher + priority replay + EWC
│   ├── anomaly/                 # Isolation + Mahalanobis + fusion
│   ├── streams/                 # 6 stream types + augment
│   ├── workers/                 # Priority pool with retry
│   ├── metrics/                 # Counter/Gauge/Histogram
│   └── events/                  # Wildcard pub/sub bus
├── dashboard/
│   ├── theme/                   # 4 palettes + CSS + components
│   ├── layouts/                 # Shell + grid helpers
│   └── panels/                  # 7 panels
├── benchmarks/                  # Suite + baseline + reporter
├── cli/                         # Unified command dispatcher
└── tests/
    ├── unit/                    # ~330 tests across 20+ modules
    └── integration/             # 72 cross-module tests
```

---

## Development

### Running Tests

```bash
# Everything
python -m pytest tests/ -v

# One layer at a time
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# With coverage
python -m pytest tests/ --cov=core --cov=dashboard --cov=cli
```

### Running Verifiers

Every historical build phase has a verifier that runs a self-contained check:

```bash
python verify_phase1.py
python verify_phase24.py

# Or via CLI
python -m cli verify all
python -m cli verify 5-10
```

### Benchmarks

```bash
# Compare to baseline
python -m benchmarks.run_suite

# Print to console + save HTML
python -m benchmarks.run_suite --html report.html

# Update baseline
python -m benchmarks.run_suite --update-baseline

# Via installed CLI
neuralstream bench --html report.html
```

---

## What this project demonstrates

- **Streaming ML engineering**: sub-3ms inference, closed-loop precision scaling
- **Optimal transport theory**: Wasserstein-1 for detecting subtle activation drift
- **Continual learning**: Fisher information + EWC to prevent catastrophic forgetting
- **Statistical ensembles**: three-way fusion for robust anomaly detection
- **Process-level IPC**: lock-free shared memory for decoupled async execution
- **Production dashboards**: 4 themes, 7 panels, real-time 3D + heatmaps + waterfalls
- **Clean architecture**: config schema in Pydantic, unified CLI, 5 deployment profiles
- **Regression-safe testing**: baseline tracking with tolerance bands and absolute floors

---

## License

MIT. See `LICENSE`.

---

<div align="center">

**Built with PyTorch · Streamlit · Plotly**

</div>