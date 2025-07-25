"""
Order-Flow Imbalance (OFI) computation helper for V1-α.

The class accumulates aggressive trade volumes (identified by side) and
returns a normalised OFI value over a rolling time window expressed in
seconds. It clamps the output to ±N standard deviations to avoid
extreme values.

See spec §3.3bis.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple
from time import time
import numpy as np
import math
from ..utils.config import mm_config

Trade = Tuple[float, float]  # (timestamp, signed_qty)  qty>0 buy, qty<0 sell

@dataclass
class OFICalculator:
    symbol: str
    window_seconds: float = field(default_factory=lambda: mm_config.ofi_window_seconds)
    clamp_std: float = field(default_factory=lambda: mm_config.ofi_clamp_std)

    def __post_init__(self):
        self._trades: Deque[Trade] = deque()
        self._mean: float = 0.0
        self._var: float = 0.0
        self._n: int = 0

    def _update_stats(self, value: float):
        """Incremental mean/variance (Welford) for robust std estimate."""
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._var += delta * delta2

    def register_trade(self, qty: float, side: str, ts: float | None = None):
        """Add a trade to the rolling window.

        Args:
            qty: executed quantity (base units)
            side: 'buy' if market buy (aggressive on ask), 'sell' otherwise
            ts: optional UNIX timestamp in seconds. Defaults to time.time().
        """
        if qty <= 0:
            return  # ignore empty trades
        signed_qty = qty if side.lower() == "buy" else -qty
        self._trades.append((ts or time(), signed_qty))

    def _prune(self, now: float):
        limit = now - self.window_seconds
        while self._trades and self._trades[0][0] < limit:
            self._trades.popleft()

    def current_ofi(self) -> float:
        """Return the normalised (z-scored & clamped) OFI over the window."""
        now = time()
        self._prune(now)
        
        # Debug: Log trade count occasionally
        if not hasattr(self, '_last_ofi_log') or now - self._last_ofi_log > 10:
            self._last_ofi_log = now
            print(f"DEBUG OFI {self.symbol}: {len(self._trades)} trades in {self.window_seconds}s window")
        
        if not self._trades:
            return 0.0
        volumes = np.fromiter((q for _, q in self._trades), dtype=float)
        raw = volumes.sum()
        norm_factor = volumes[np.nonzero(volumes)].sum()
        if norm_factor == 0:
            norm_factor = abs(volumes).sum()
        ofi = raw / max(1e-9, norm_factor)
        # Update running stats
        self._update_stats(ofi)
        std = math.sqrt(self._var / (self._n - 1)) if self._n > 1 else 1e-9
        if std <= 1e-9:
            return 0.0
        z = (ofi - self._mean) / std
        # Clamp
        z = max(-self.clamp_std, min(self.clamp_std, z))
        return z
