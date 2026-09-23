"""
NeuralStream Dashboard Panels
"""

from dashboard.panels.latent_explorer import (
    LatentExplorerState, build_trajectory_figure, build_projection_panel,
    render as render_latent_explorer,
)
from dashboard.panels.drift_heatmap import (
    DriftHeatmapState, build_heatmap_figure, build_layer_trajectories,
    build_current_scores_bar, render as render_drift_heatmap,
)
from dashboard.panels.telemetry import (
    TelemetrySample, TelemetryState, build_multi_metric_chart,
    build_gauge, build_latency_histogram, render as render_telemetry,
)
from dashboard.panels.anomaly_console import (
    AnomalyRecord, OODEvent, AdaptationRecord, AnomalyConsoleState,
    build_score_timeline, build_component_breakdown, build_score_histogram,
    build_ood_buffer_view, build_adaptation_history, render as render_anomaly_console,
)
from dashboard.panels.model_comparison import (
    ModelProfile, ModelComparisonState,
    build_params_bar, build_latency_box, build_latency_percentile_grouped,
    build_recon_bar, build_radar, build_throughput_fps,
    render as render_model_comparison,
)
from dashboard.panels.oscilloscope import (
    OscilloscopeState, build_waveform_traces, build_fft_spectrum,
    build_waterfall, build_band_energy_bar, render as render_oscilloscope,
)
from dashboard.panels.topology import (
    TopologyNode, TopologyEdge, TopologyState,
    seed_default_topology, build_topology_figure,
    build_activity_timeline, build_category_activity_bar,
    render as render_topology,
)

__all__ = [
    # Latent
    'LatentExplorerState', 'build_trajectory_figure', 'build_projection_panel',
    'render_latent_explorer',
    # Drift
    'DriftHeatmapState', 'build_heatmap_figure', 'build_layer_trajectories',
    'build_current_scores_bar', 'render_drift_heatmap',
    # Telemetry
    'TelemetrySample', 'TelemetryState',
    'build_multi_metric_chart', 'build_gauge', 'build_latency_histogram',
    'render_telemetry',
    # Anomaly
    'AnomalyRecord', 'OODEvent', 'AdaptationRecord', 'AnomalyConsoleState',
    'build_score_timeline', 'build_component_breakdown', 'build_score_histogram',
    'build_ood_buffer_view', 'build_adaptation_history', 'render_anomaly_console',
    # Comparison
    'ModelProfile', 'ModelComparisonState',
    'build_params_bar', 'build_latency_box', 'build_latency_percentile_grouped',
    'build_recon_bar', 'build_radar', 'build_throughput_fps',
    'render_model_comparison',
    # Oscilloscope
    'OscilloscopeState', 'build_waveform_traces', 'build_fft_spectrum',
    'build_waterfall', 'build_band_energy_bar', 'render_oscilloscope',
    # Topology
    'TopologyNode', 'TopologyEdge', 'TopologyState',
    'seed_default_topology', 'build_topology_figure',
    'build_activity_timeline', 'build_category_activity_bar',
    'render_topology',
]