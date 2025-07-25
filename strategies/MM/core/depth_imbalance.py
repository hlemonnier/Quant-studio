"""
Depth Imbalance (DI) Calculator for V1.5

Implements the Depth Imbalance signal calculation according to V1.5 specification:
- DI_raw = (depth_bid - depth_ask) / (depth_bid + depth_ask)
- Aggregates L1-L5 depth data
- Applies Z-score normalization and ±3σ clamping
- Uses EMA smoothing with 3 observations

See spec §4.2 and §4.3.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional
from time import time
import numpy as np
import math
import logging
from .config import mm_config


@dataclass
class DepthSnapshot:
    """Represents a depth snapshot for DI calculation"""
    timestamp: float
    depth_bid_l1_l5: float  # Sum of bid sizes L1-L5
    depth_ask_l1_l5: float  # Sum of ask sizes L1-L5
    di_raw: float = field(init=False)
    
    def __post_init__(self):
        """Calculate raw DI on initialization"""
        total_depth = self.depth_bid_l1_l5 + self.depth_ask_l1_l5
        if total_depth > 0:
            self.di_raw = (self.depth_bid_l1_l5 - self.depth_ask_l1_l5) / total_depth
        else:
            self.di_raw = 0.0


class DepthImbalanceCalculator:
    """
    Calculates Depth Imbalance signal from L1-L5 order book data
    
    The DI signal measures asymmetry in order book depth between bid and ask sides.
    Positive DI indicates more bid-side liquidity (bullish pressure).
    Negative DI indicates more ask-side liquidity (bearish pressure).
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"DI-{symbol}")
        
        # Configuration
        self.ema_periods = 3  # EMA smoothing periods as per spec
        self.clamp_std = 3.0  # ±3σ clamping
        self.min_depth_threshold = 1e-8  # Minimum depth to avoid division by zero
        
        # Data storage
        self._snapshots: Deque[DepthSnapshot] = deque(maxlen=100)  # Keep last 100 snapshots
        
        # Statistics for Z-score normalization (Welford's algorithm)
        self._mean: float = 0.0
        self._var: float = 0.0
        self._n: int = 0
        
        # EMA state
        self._ema_di: Optional[float] = None
        self._alpha_ema: float = 2.0 / (self.ema_periods + 1)
        
        # Current state
        self._current_di_raw: float = 0.0
        self._current_di_normalized: float = 0.0
        self._current_di_filtered: float = 0.0
        
        self.logger.info(f"📊 DI Calculator initialized for {symbol}")
    
    def update_depth(self, depth_bid_l1_l5: float, depth_ask_l1_l5: float, 
                     timestamp: Optional[float] = None) -> float:
        """
        Update depth data and calculate current DI signal
        
        Args:
            depth_bid_l1_l5: Sum of bid sizes across L1-L5 levels
            depth_ask_l1_l5: Sum of ask sizes across L1-L5 levels  
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Current filtered DI signal (Z-scored, clamped, EMA smoothed)
        """
        if timestamp is None:
            timestamp = time()
        
        # Validate inputs
        if depth_bid_l1_l5 < 0 or depth_ask_l1_l5 < 0:
            self.logger.warning(f"⚠️  Negative depth values: bid={depth_bid_l1_l5}, ask={depth_ask_l1_l5}")
            return self._current_di_filtered
        
        # Create snapshot
        snapshot = DepthSnapshot(timestamp, depth_bid_l1_l5, depth_ask_l1_l5)
        self._snapshots.append(snapshot)
        
        # Debug: Log depth values occasionally
        if self._n % 50 == 0:  # Every 50 updates
            self.logger.info(
                f"DI Depth Debug: bid_depth={depth_bid_l1_l5:.4f}, "
                f"ask_depth={depth_ask_l1_l5:.4f}, "
                f"di_raw={snapshot.di_raw:.4f}"
            )
        
        # Update current raw DI
        self._current_di_raw = snapshot.di_raw
        
        # Update running statistics for Z-score normalization
        self._update_statistics(snapshot.di_raw)
        
        # Calculate normalized DI (Z-score)
        self._current_di_normalized = self._calculate_z_score(snapshot.di_raw)
        
        # Apply EMA smoothing
        self._current_di_filtered = self._apply_ema_smoothing(self._current_di_normalized)
        
        # Debug logging with more details
        std = math.sqrt(self._var / (self._n - 1)) if self._n > 1 else 0.0
        self.logger.debug(
            f"DI Update: raw={self._current_di_raw:.4f}, "
            f"norm={self._current_di_normalized:.4f}, "
            f"filtered={self._current_di_filtered:.4f}, "
            f"mean={self._mean:.4f}, std={std:.6f}, n={self._n}"
        )
        
        return self._current_di_filtered
    
    def _update_statistics(self, value: float):
        """Update running mean and variance using Welford's algorithm"""
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._var += delta * delta2
    
    def _calculate_z_score(self, value: float) -> float:
        """Calculate Z-score with clamping"""
        if self._n < 2:
            return 0.0  # Not enough data for meaningful Z-score
        
        # Calculate standard deviation
        std = math.sqrt(self._var / (self._n - 1))
        if std <= 1e-12:  # Reduced threshold to capture smaller variations
            return 0.0  # Avoid division by zero
        
        # Calculate Z-score
        z_score = (value - self._mean) / std
        
        # Apply ±3σ clamping
        z_score = max(-self.clamp_std, min(self.clamp_std, z_score))
        
        return z_score
    
    def _apply_ema_smoothing(self, value: float) -> float:
        """Apply EMA smoothing to the normalized DI signal"""
        if self._ema_di is None:
            # Initialize EMA with first value
            self._ema_di = value
        else:
            # Update EMA: EMA_new = α * value + (1-α) * EMA_old
            self._ema_di = self._alpha_ema * value + (1 - self._alpha_ema) * self._ema_di
        
        return self._ema_di
    
    def get_current_di(self) -> float:
        """Get the current filtered DI signal"""
        return self._current_di_filtered
    
    def get_current_di_raw(self) -> float:
        """Get the current raw DI signal (before normalization)"""
        return self._current_di_raw
    
    def get_signal_stats(self) -> dict:
        """Get statistics about the DI signal"""
        std = math.sqrt(self._var / (self._n - 1)) if self._n > 1 else 0.0
        
        return {
            'symbol': self.symbol,
            'current_di_raw': self._current_di_raw,
            'current_di_normalized': self._current_di_normalized,
            'current_di_filtered': self._current_di_filtered,
            'mean': self._mean,
            'std': std,
            'n_observations': self._n,
            'ema_alpha': self._alpha_ema,
            'snapshots_count': len(self._snapshots)
        }
    
    def reset_statistics(self):
        """Reset all statistics (useful for recalibration)"""
        self._mean = 0.0
        self._var = 0.0
        self._n = 0
        self._ema_di = None
        self._snapshots.clear()
        self.logger.info(f"📊 DI statistics reset for {self.symbol}")
    
    def test_correlation(self, price_changes: list, di_values: list) -> dict:
        """
        Test correlation between DI signal and subsequent price changes
        
        Args:
            price_changes: List of price changes (e.g., next tick direction)
            di_values: List of corresponding DI values
            
        Returns:
            Dictionary with correlation statistics
        """
        if len(price_changes) != len(di_values) or len(price_changes) < 10:
            return {'error': 'Insufficient or mismatched data'}
        
        # Calculate Pearson correlation
        correlation = np.corrcoef(di_values, price_changes)[0, 1]
        
        # Calculate hit ratio (sign prediction accuracy)
        correct_predictions = sum(
            1 for di, change in zip(di_values, price_changes)
            if (di > 0 and change > 0) or (di < 0 and change < 0) or (di == 0 and change == 0)
        )
        hit_ratio = correct_predictions / len(price_changes)
        
        return {
            'correlation': correlation,
            'hit_ratio': hit_ratio,
            'n_samples': len(price_changes),
            'mean_di': np.mean(di_values),
            'std_di': np.std(di_values),
            'mean_price_change': np.mean(price_changes),
            'std_price_change': np.std(price_changes)
        }


def test_di_calculator():
    """Test function for the DI calculator"""
    print("🧪 Testing Depth Imbalance Calculator...")
    
    calc = DepthImbalanceCalculator('BTCUSDT')
    
    # Simulate some depth updates
    test_data = [
        (100.0, 80.0),   # Bid-heavy (positive DI)
        (90.0, 110.0),   # Ask-heavy (negative DI)
        (100.0, 100.0),  # Balanced (zero DI)
        (120.0, 70.0),   # Strong bid-heavy
        (60.0, 140.0),   # Strong ask-heavy
    ]
    
    print("\n📊 DI Signal Evolution:")
    for i, (bid_depth, ask_depth) in enumerate(test_data):
        di_filtered = calc.update_depth(bid_depth, ask_depth)
        stats = calc.get_signal_stats()
        
        print(f"Update {i+1}: Bid={bid_depth:6.1f}, Ask={ask_depth:6.1f} → "
              f"DI_raw={stats['current_di_raw']:+.4f}, "
              f"DI_norm={stats['current_di_normalized']:+.4f}, "
              f"DI_filt={di_filtered:+.4f}")
    
    # Print final statistics
    final_stats = calc.get_signal_stats()
    print(f"\n📈 Final Statistics:")
    print(f"  Mean: {final_stats['mean']:.4f}")
    print(f"  Std:  {final_stats['std']:.4f}")
    print(f"  Observations: {final_stats['n_observations']}")
    print(f"  Current DI: {final_stats['current_di_filtered']:+.4f}")
    
    print("✅ DI Calculator test completed")


if __name__ == "__main__":
    test_di_calculator()
