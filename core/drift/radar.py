"""
Production Drift Radar
High-level drift monitoring system that integrates with model hooks
to continuously track activation distribution shifts across all layers.

Features:
- Automatic reference capture during LSUV calibration
- Real-time per-layer drift scoring
- Rolling drift history with trend detection
- Severity-based alerting
- Heatmap data generation for dashboard visualization
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from core.drift.detector import DriftDetector, DriftReport, DriftSeverity
from core.constants import DriftSeverity as DS


class DriftRadar:
    """
    Production drift monitoring radar.

    Attaches to a model via forward hooks, captures reference distributions
    during calibration, and continuously monitors live activations for
    distributional shift.

    Args:
        model: The neural network model to monitor
        layer_names: Specific layers to monitor (None = all Conv2d/Linear)
        methods: Drift detection methods to use
        sample_size: Samples per distribution for distance computation
        alert_threshold: Warning threshold
        critical_threshold: Critical threshold
        history_length: Max drift history entries per layer
        evaluation_interval: Compute drift every N forward passes
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        layer_names: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        sample_size: int = 500,
        alert_threshold: float = 0.15,
        critical_threshold: float = 0.35,
        history_length: int = 200,
        evaluation_interval: int = 5
    ):
        self.detector = DriftDetector(
            methods=methods,
            alert_threshold=alert_threshold,
            critical_threshold=critical_threshold,
            sample_size=sample_size
        )

        self.layer_names = layer_names or []
        self.sample_size = sample_size
        self.history_length = history_length
        self.evaluation_interval = evaluation_interval

        # State
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.drift_history: Dict[str, List[Dict]] = defaultdict(list)
        self._hook_handles: List = []
        self._live_activations: Dict[str, np.ndarray] = {}
        self._tick_count: int = 0
        self._is_calibrated: bool = False
        self._model = model

        # Alert tracking
        self.alert_count: int = 0
        self.critical_count: int = 0
        self.last_alert_time: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def _discover_layers(self, model: nn.Module) -> List[str]:
        """Auto-discover monitorable layers if none specified."""
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name != '':
                layers.append(name)
        return layers

    def capture_reference(
        self,
        model: nn.Module,
        calibration_batch: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Capture baseline activation distributions from a calibration batch.
        Should be called after LSUV initialization.

        Args:
            model: Calibrated model
            calibration_batch: Representative input batch
            layer_names: Override layer names to capture

        Returns:
            Dictionary of {layer_name: reference_distribution}
        """
        self._model = model

        if layer_names is not None:
            self.layer_names = layer_names
        elif not self.layer_names:
            self.layer_names = self._discover_layers(model)

        # Register temporary hooks to capture activations
        captured = {}
        hooks = []

        for name, module in model.named_modules():
            if name not in self.layer_names:
                continue

            def make_hook(layer_name):
                def hook_fn(mod, inp, out):
                    flat = out.detach().cpu().numpy().flatten()
                    if len(flat) > self.sample_size:
                        flat = np.random.choice(flat, self.sample_size, replace=False)
                    captured[layer_name] = flat
                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(calibration_batch)

        # Clean up hooks
        for h in hooks:
            h.remove()

        # Store references
        self.reference_distributions = captured
        self._is_calibrated = True

        # Clear history
        self.drift_history = defaultdict(list)
        self.alert_count = 0
        self.critical_count = 0

        return captured

    def attach_hooks(self, model: nn.Module) -> None:
        """
        Attach persistent forward hooks for live monitoring.
        Call this once after calibration to enable continuous drift tracking.
        """
        self.remove_hooks()
        self._model = model

        for name, module in model.named_modules():
            if name not in self.layer_names:
                continue

            def make_hook(layer_name):
                def hook_fn(mod, inp, out):
                    flat = out.detach().cpu().numpy().flatten()
                    if len(flat) > self.sample_size:
                        flat = np.random.choice(flat, self.sample_size, replace=False)
                    self._live_activations[layer_name] = flat
                return hook_fn

            handle = module.register_forward_hook(make_hook(name))
            self._hook_handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all monitoring hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def evaluate(self, force: bool = False) -> Dict[str, DriftReport]:
        """
        Evaluate drift for all monitored layers.
        Only computes on evaluation_interval ticks unless forced.

        Returns:
            Dictionary of {layer_name: DriftReport}
        """
        self._tick_count += 1

        if not self._is_calibrated:
            return {}

        if not force and self._tick_count % self.evaluation_interval != 0:
            return {}

        if not self._live_activations:
            return {}

        reports = self.detector.compute_batch(
            self.reference_distributions,
            self._live_activations
        )

        # Record history and check alerts
        for layer_name, report in reports.items():
            entry = {
                'tick': self._tick_count,
                'composite_score': report.composite_score,
                'severity': report.severity.value,
                'timestamp': report.timestamp,
                'methods': {
                    m: r.score for m, r in report.results.items()
                }
            }
            self.drift_history[layer_name].append(entry)

            # Trim history
            if len(self.drift_history[layer_name]) > self.history_length:
                self.drift_history[layer_name] = \
                    self.drift_history[layer_name][-self.history_length:]

            # Alert tracking
            if report.is_alert:
                self.alert_count += 1
                self.last_alert_time = time.time()
                if report.severity == DS.EMERGENCY:
                    self.critical_count += 1

        return reports

    def evaluate_from_tensor(
        self,
        model: nn.Module,
        input_batch: torch.Tensor
    ) -> Dict[str, DriftReport]:
        """
        One-shot drift evaluation from an input batch.
        Runs a forward pass, captures activations, and computes drift.
        """
        if not self._is_calibrated:
            return {}

        # Temporarily attach hooks
        self.attach_hooks(model)

        model.eval()
        with torch.no_grad():
            _ = model(input_batch)

        # Evaluate
        reports = self.evaluate(force=True)

        # Clean up
        self.remove_hooks()

        return reports

    def get_heatmap_data(self) -> np.ndarray:
        """
        Generate 2D heatmap data for dashboard visualization.

        Returns:
            Array of shape (num_layers, history_length) with composite scores.
            Missing values are filled with 0.
        """
        if not self.drift_history:
            return np.zeros((1, 1))

        layers = sorted(self.drift_history.keys())
        max_len = max(len(h) for h in self.drift_history.values())

        heatmap = np.zeros((len(layers), max_len))

        for i, layer in enumerate(layers):
            history = self.drift_history[layer]
            for j, entry in enumerate(history):
                heatmap[i, j] = entry['composite_score']

        return heatmap

    def get_layer_names(self) -> List[str]:
        """Get sorted list of monitored layer names."""
        return sorted(self.drift_history.keys()) if self.drift_history else self.layer_names

    def get_trend(self, layer_name: str, window: int = 20) -> str:
        """
        Determine drift trend for a layer.

        Returns: 'rising', 'falling', 'stable'
        """
        if layer_name not in self.drift_history:
            return 'stable'

        history = self.drift_history[layer_name]
        if len(history) < window:
            return 'stable'

        recent = [h['composite_score'] for h in history[-window:]]
        first_half = np.mean(recent[:window // 2])
        second_half = np.mean(recent[window // 2:])

        diff = second_half - first_half
        if diff > 0.02:
            return 'rising'
        elif diff < -0.02:
            return 'falling'
        return 'stable'

    def get_summary(self) -> Dict[str, Any]:
        """Get overall drift monitoring summary."""
        summary = {
            'is_calibrated': self._is_calibrated,
            'num_layers': len(self.layer_names),
            'layers': self.layer_names,
            'tick_count': self._tick_count,
            'alert_count': self.alert_count,
            'critical_count': self.critical_count,
            'methods': self.detector.methods,
            'layer_status': {}
        }

        for layer in self.layer_names:
            if layer in self.drift_history and self.drift_history[layer]:
                latest = self.drift_history[layer][-1]
                summary['layer_status'][layer] = {
                    'latest_score': latest['composite_score'],
                    'severity': latest['severity'],
                    'trend': self.get_trend(layer),
                    'history_length': len(self.drift_history[layer])
                }
            else:
                summary['layer_status'][layer] = {
                    'latest_score': 0.0,
                    'severity': 'nominal',
                    'trend': 'stable',
                    'history_length': 0
                }

        return summary

    def reset(self) -> None:
        """Reset all state (references, history, hooks)."""
        self.remove_hooks()
        self.reference_distributions.clear()
        self.drift_history.clear()
        self._live_activations.clear()
        self._tick_count = 0
        self._is_calibrated = False
        self.alert_count = 0
        self.critical_count = 0