"""
Configuration Loader (Extended)

Handles YAML profile loading, deep merging, and runtime overrides,
with support for resolving profiles by short name (e.g. "edge") or
by explicit path.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from config.schema import SystemConfig


_PROFILES_DIR = Path(__file__).parent / "profiles"
_DEFAULT_CONFIG_PATH = _PROFILES_DIR / "default.yaml"
_ACTIVE_CONFIG: Optional[SystemConfig] = None


# =============================================================================
# YAML HELPERS
# =============================================================================

def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load a YAML file and return as dictionary."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data if data is not None else {}


def save_yaml(config: SystemConfig, filepath: str) -> None:
    """Save a SystemConfig to a YAML file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config.model_dump(mode='json'), f,
                  default_flow_style=False, sort_keys=False)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values in 'override' take precedence.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# =============================================================================
# PROFILE DISCOVERY
# =============================================================================

def list_profiles() -> List[str]:
    """Return the names of all available YAML profiles in the profiles dir."""
    if not _PROFILES_DIR.exists():
        return []
    return sorted([p.stem for p in _PROFILES_DIR.glob("*.yaml")])


def resolve_profile(profile: Optional[str]) -> Optional[Path]:
    """
    Resolve a profile name or path to an actual file path.

    Args:
        profile: Profile name ("edge", "ci") or filesystem path
                 (with or without .yaml extension).

    Returns:
        Resolved Path, or None if no profile was requested.

    Raises:
        FileNotFoundError if the profile cannot be located.
    """
    if profile is None:
        return None

    # Absolute or explicit path
    candidate = Path(profile)
    if candidate.is_file():
        return candidate

    # Path relative to profiles dir with extension
    candidate = _PROFILES_DIR / profile
    if candidate.is_file():
        return candidate

    # Path relative to profiles dir without extension
    candidate = _PROFILES_DIR / f"{profile}.yaml"
    if candidate.is_file():
        return candidate

    # Try relative to project root
    candidate = Path.cwd() / profile
    if candidate.is_file():
        return candidate

    candidate = Path.cwd() / f"{profile}.yaml"
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Profile '{profile}' not found. "
        f"Available: {list_profiles()}"
    )


# =============================================================================
# MAIN LOADER
# =============================================================================

def load_config(
    profile: Optional[str] = None,
    profile_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> SystemConfig:
    """
    Load system configuration with optional profile and runtime overrides.

    Priority order (highest wins):
        1. Runtime overrides dict
        2. Explicit profile (profile or profile_path)
        3. Default profile (config/profiles/default.yaml)
        4. Schema defaults

    Args:
        profile: Short profile name (e.g. "edge", "ci").
        profile_path: Explicit path to a YAML file (takes lower priority
                      than `profile` if both are provided).
        overrides: Dict of runtime overrides to merge on top.

    Returns:
        Validated SystemConfig instance.
    """
    global _ACTIVE_CONFIG

    base_data: Dict[str, Any] = {}

    # Load profile YAML
    resolved: Optional[Path] = None
    if profile is not None:
        resolved = resolve_profile(profile)
    elif profile_path is not None:
        resolved = resolve_profile(profile_path)
    elif _DEFAULT_CONFIG_PATH.exists():
        resolved = _DEFAULT_CONFIG_PATH

    if resolved is not None:
        try:
            base_data = load_yaml(str(resolved))
        except Exception:
            base_data = {}

    # Apply runtime overrides
    if overrides:
        base_data = deep_merge(base_data, overrides)

    # Validate
    if base_data:
        _ACTIVE_CONFIG = SystemConfig(**base_data)
    else:
        _ACTIVE_CONFIG = SystemConfig()

    return _ACTIVE_CONFIG


def get_active_config() -> SystemConfig:
    """Return the currently active system configuration (lazy init)."""
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        _ACTIVE_CONFIG = load_config()
    return _ACTIVE_CONFIG


def reset_config() -> None:
    """Reset the cached active configuration."""
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = None


# =============================================================================
# CONFIG SUMMARY
# =============================================================================

def summarize_config(cfg: SystemConfig) -> Dict[str, Any]:
    """
    Produce a compact summary of a SystemConfig for CLI display.
    """
    return {
        'project': {
            'name': cfg.project_name,
            'version': cfg.version,
            'debug': cfg.debug,
            'seed': cfg.seed,
            'log_level': cfg.log_level,
        },
        'model': {
            'architecture': cfg.model.architecture,
            'in_channels': cfg.model.in_channels,
            'input_size': cfg.model.input_size,
            'latent_dim': cfg.model.latent_dim,
        },
        'stream': {
            'format': cfg.stream.format,
            'batch_size': cfg.stream.batch_size,
            'shape': cfg.stream.shape,
        },
        'precision': {
            'target_latency_ms': cfg.precision.target_latency_ms,
            'window_size': cfg.precision.window_size,
            'cooldown_ticks': cfg.precision.cooldown_ticks,
        },
        'drift': {
            'methods': cfg.drift.methods,
            'sample_size': cfg.drift.sample_size,
            'evaluation_interval': cfg.drift.evaluation_interval,
            'layers': cfg.drift.layer_names,
        },
        'adaptation': {
            'error_threshold': cfg.adaptation.error_threshold,
            'buffer_capacity': cfg.adaptation.buffer_capacity,
            'ewc_enabled': cfg.adaptation.enable_ewc,
        },
        'dashboard': {
            'theme': cfg.dashboard.theme,
            'refresh_rate_ms': cfg.dashboard.refresh_rate_ms,
            'enable_3d_plots': cfg.dashboard.enable_3d_plots,
            'enable_heatmaps': cfg.dashboard.enable_heatmaps,
            'enable_oscilloscope': cfg.dashboard.enable_oscilloscope,
        },
        'workers': {
            'inference_workers': cfg.workers.num_inference_workers,
            'adaptation_workers': cfg.workers.num_adaptation_workers,
            'drift_workers': cfg.workers.num_drift_workers,
        },
    }