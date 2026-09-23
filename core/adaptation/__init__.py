"""
NeuralStream-Core Adaptation Package
"""

from core.adaptation.fisher import FisherInformation
from core.adaptation.replay import PriorityReplayBuffer
from core.adaptation.engine import (
    OnlineAdaptationEngineV2,
    ActiveOODAdaptationEngine,  # alias
)

__all__ = [
    'FisherInformation',
    'PriorityReplayBuffer',
    'OnlineAdaptationEngineV2',
    'ActiveOODAdaptationEngine',
]