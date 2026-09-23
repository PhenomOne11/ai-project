"""
Multi-Method Drift Detector
Runs multiple statistical distance metrics simultaneously on activation
distributions to detect domain shift, concept drift, and data corruption.

Each method captures different aspects of distributional change:
- Wasserstein: Sensitive to location shifts (mean drift)
- MMD: Sensitive to shape and scale changes
- KL Divergence: Sensitive to tail behavior
- KS Test: Sensitive to CDF differences (non-parametric)
- Energy Distance: Sensitive to all distributional differences
- Hellinger: Bounded [0,1], good for thresholding
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from core.math_utils.distances import (
    wasserstein_1, wasserstein_2, mmd_rbf, kl_divergence,
    jensen_shannon, ks_statistic, energy_distance, hellinger_distance
)
from core.constants import DriftSeverity


@dataclass
class DriftResult:
    """Result of a single drift computation."""
    method: str
    score: float
    severity: DriftSeverity
    details: Dict = field(default_factory=dict)


@dataclass
class DriftReport:
    """Aggregated drift report across all methods for a single layer."""
    layer_name: str
    results: Dict[str, DriftResult]
    composite_score: float
    severity: DriftSeverity
    timestamp: float = 0.0

    @property
    def is_alert(self) -> bool:
        return self.severity in (DriftSeverity.CRITICAL, DriftSeverity.EMERGENCY)

    @property
    def is_warning(self) -> bool:
        return self.severity == DriftSeverity.WARNING

    @property
    def method_names(self) -> List[str]:
        return list(self.results.keys())

    def to_dict(self) -> Dict:
        return {
            'layer': self.layer_name,
            'composite_score': self.composite_score,
            'severity': self.severity.value,
            'is_alert': self.is_alert,
            'methods': {
                name: {'score': r.score, 'severity': r.severity.value}
                for name, r in self.results.items()
            }
        }


class DriftDetector:
    """
    Multi-method drift detector that computes multiple statistical
    distances between reference and live activation distributions.

    Args:
        methods: List of distance methods to compute
        alert_threshold: Score above which severity = WARNING
        critical_threshold: Score above which severity = CRITICAL
        emergency_threshold: Score above which severity = EMERGENCY
        sample_size: Max samples to use per computation (for speed)
    """

    # Map method names to callable functions
    METHOD_REGISTRY = {
        'wasserstein': wasserstein_1,
        'wasserstein_2': wasserstein_2,
        'mmd': mmd_rbf,
        'kl_divergence': kl_divergence,
        'jensen_shannon': jensen_shannon,
        'energy': energy_distance,
        'hellinger': hellinger_distance,
    }

    def __init__(
        self,
        methods: Optional[List[str]] = None,
        alert_threshold: float = 0.15,
        critical_threshold: float = 0.35,
        emergency_threshold: float = 0.70,
        sample_size: int = 500
    ):
        if methods is None:
            methods = ['wasserstein', 'mmd', 'kl_divergence']

        # Validate methods
        for m in methods:
            if m not in self.METHOD_REGISTRY:
                raise ValueError(
                    f"Unknown drift method '{m}'. "
                    f"Available: {list(self.METHOD_REGISTRY.keys())}"
                )

        self.methods = methods
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        self.emergency_threshold = emergency_threshold
        self.sample_size = sample_size

    def _classify_severity(self, score: float) -> DriftSeverity:
        """Classify drift score into severity level."""
        if score >= self.emergency_threshold:
            return DriftSeverity.EMERGENCY
        elif score >= self.critical_threshold:
            return DriftSeverity.CRITICAL
        elif score >= self.alert_threshold:
            return DriftSeverity.WARNING
        else:
            return DriftSeverity.NOMINAL

    def _prepare_samples(
        self,
        reference: np.ndarray,
        live: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample and clean arrays for distance computation."""
        ref = reference.flatten().astype(np.float64)
        liv = live.flatten().astype(np.float64)

        # Remove non-finite values
        ref = ref[np.isfinite(ref)]
        liv = liv[np.isfinite(liv)]

        # Subsample for speed
        if len(ref) > self.sample_size:
            ref = np.random.choice(ref, self.sample_size, replace=False)
        if len(liv) > self.sample_size:
            liv = np.random.choice(liv, self.sample_size, replace=False)

        return ref, liv

    def compute_single(
        self,
        method: str,
        reference: np.ndarray,
        live: np.ndarray
    ) -> DriftResult:
        """Compute drift using a single method."""
        ref, liv = self._prepare_samples(reference, live)

        if len(ref) == 0 or len(liv) == 0:
            return DriftResult(method=method, score=0.0, severity=DriftSeverity.NOMINAL)

        # KS test returns (statistic, p_value) tuple
        if method == 'ks_test':
            stat, pval = ks_statistic(ref, liv)
            score = stat
            details = {'p_value': pval}
        else:
            fn = self.METHOD_REGISTRY[method]
            score = float(fn(ref, liv))
            details = {}

        severity = self._classify_severity(score)

        return DriftResult(
            method=method,
            score=score,
            severity=severity,
            details=details
        )

    def compute(
        self,
        reference: np.ndarray,
        live: np.ndarray,
        layer_name: str = "unknown"
    ) -> DriftReport:
        """
        Compute drift across all configured methods.

        Args:
            reference: Reference distribution (from calibration)
            live: Current live activations
            layer_name: Name of the layer being monitored

        Returns:
            DriftReport with per-method scores and composite severity
        """
        import time

        results = {}
        scores = []

        for method in self.methods:
            result = self.compute_single(method, reference, live)
            results[method] = result
            scores.append(result.score)

        # Composite score = mean of all method scores (normalized)
        composite = float(np.mean(scores)) if scores else 0.0

        # Overall severity = worst severity across all methods
        severity_order = [
            DriftSeverity.NOMINAL,
            DriftSeverity.WARNING,
            DriftSeverity.CRITICAL,
            DriftSeverity.EMERGENCY
        ]
        worst_idx = 0
        for r in results.values():
            idx = severity_order.index(r.severity)
            worst_idx = max(worst_idx, idx)
        overall_severity = severity_order[worst_idx]

        return DriftReport(
            layer_name=layer_name,
            results=results,
            composite_score=composite,
            severity=overall_severity,
            timestamp=time.time()
        )

    def compute_batch(
        self,
        references: Dict[str, np.ndarray],
        live_activations: Dict[str, np.ndarray]
    ) -> Dict[str, DriftReport]:
        """
        Compute drift for multiple layers simultaneously.

        Args:
            references: {layer_name: reference_distribution}
            live_activations: {layer_name: current_activations}

        Returns:
            {layer_name: DriftReport}
        """
        reports = {}
        for layer_name in references:
            if layer_name in live_activations:
                reports[layer_name] = self.compute(
                    references[layer_name],
                    live_activations[layer_name],
                    layer_name=layer_name
                )
        return reports