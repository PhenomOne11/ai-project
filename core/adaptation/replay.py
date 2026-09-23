"""
Priority Replay Buffer for OOD samples.

Unlike FIFO queues, samples with the highest reconstruction error
are retained during overflow, ensuring the micro-adaptation pass
focuses on the hardest examples.
"""

import torch
import heapq
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass(order=True)
class _PrioritizedItem:
    priority: float  # Higher = more OOD
    seq: int
    sample: torch.Tensor = field(compare=False)


class PriorityReplayBuffer:
    """
    Fixed-capacity buffer that keeps the highest-error OOD samples.

    Args:
        capacity: Max number of samples retained
        mode: 'max-error' (keep highest loss) or 'min-error' or 'fifo'
    """

    def __init__(self, capacity: int = 64, mode: str = 'max-error'):
        self.capacity = capacity
        self.mode = mode
        self._buffer: List[_PrioritizedItem] = []
        self._seq = 0
        self.total_added = 0
        self.total_evicted = 0

    def add(self, sample: torch.Tensor, priority: float) -> None:
        """Add a sample with its OOD priority score."""
        self._seq += 1

        # For min-heap implementation, invert priority for max-error mode
        key = priority if self.mode == 'max-error' else -priority

        item = _PrioritizedItem(priority=key, seq=self._seq, sample=sample)

        if len(self._buffer) < self.capacity:
            heapq.heappush(self._buffer, item)
        else:
            # Replace the min if new item is better
            if item.priority > self._buffer[0].priority:
                heapq.heapreplace(self._buffer, item)
                self.total_evicted += 1

        self.total_added += 1

    def add_batch(self, batch: torch.Tensor, priorities: torch.Tensor) -> None:
        """Add multiple samples with individual priorities."""
        for i in range(batch.size(0)):
            self.add(batch[i:i+1].detach().cpu(), float(priorities[i].item()))

    def sample(self, n: int) -> Optional[torch.Tensor]:
        """
        Pop the N highest-priority samples and return as a batch.
        Remaining samples stay in the buffer.
        """
        if not self._buffer:
            return None

        # Pop top-N
        popped: List[_PrioritizedItem] = []
        n = min(n, len(self._buffer))
        for _ in range(n):
            popped.append(heapq.heappop(self._buffer))

        samples = torch.cat([item.sample for item in popped], dim=0)
        return samples

    def drain(self) -> Optional[torch.Tensor]:
        """Return all samples as a single batch and clear the buffer."""
        if not self._buffer:
            return None
        result = self.sample(len(self._buffer))
        return result

    def peek_top(self, n: int = 5) -> List[float]:
        """Return the top-N priorities without removing them."""
        # Copy for inspection
        top = heapq.nlargest(min(n, len(self._buffer)), self._buffer)
        return [item.priority for item in top]

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.capacity

    def clear(self) -> None:
        self._buffer.clear()

    def get_stats(self) -> dict:
        if not self._buffer:
            return {
                'size': 0, 'capacity': self.capacity,
                'total_added': self.total_added,
                'total_evicted': self.total_evicted,
                'min_priority': 0.0, 'max_priority': 0.0,
            }
        priorities = [item.priority for item in self._buffer]
        return {
            'size': len(self._buffer),
            'capacity': self.capacity,
            'total_added': self.total_added,
            'total_evicted': self.total_evicted,
            'min_priority': min(priorities),
            'max_priority': max(priorities),
        }