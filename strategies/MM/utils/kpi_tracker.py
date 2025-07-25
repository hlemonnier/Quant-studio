"""
KPI Tracking System for V1 (§3.7)

Tracks all performance metrics specified in the spec:
- Spread captured (%)
- RMS inventory 
- Fill ratio
- Cancel ratio
- Latency metrics (P99)

Provides real-time monitoring and summary statistics.
"""

from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
from datetime import datetime, timedelta
import numpy as np
import time
import logging

from .config import mm_config


@dataclass
class FillRecord:
    """Record of a fill for KPI calculation"""
    timestamp: float
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    mid_price_at_fill: float
    spread_captured: float


@dataclass 
class LatencyRecord:
    """Record latency measurements"""
    timestamp: float
    metric_name: str  # 'quote_ack', 'loop_time', etc.
    value_ms: float


class KPITracker:
    """Tracks all V1 KPIs in real-time"""
    
    def __init__(self, symbol: str, window_minutes: int = 60):
        self.symbol = symbol
        self.window_seconds = window_minutes * 60
        self.logger = logging.getLogger(f"KPITracker-{symbol}")
        
        # Data storage with time-based windows
        self.fills: Deque[FillRecord] = deque()
        self.latencies: Dict[str, Deque[LatencyRecord]] = defaultdict(lambda: deque())
        self.inventory_history: Deque[tuple] = deque()  # (timestamp, inventory)
        
        # Counters
        self.total_quotes_sent = 0
        self.total_cancels = 0
        self.total_fills = 0
        
        # Running calculations
        self._last_cleanup = time.time()
        self.cleanup_interval = 300  # Clean old data every 5 minutes
        
        # Current market data for mark-to-market PnL
        self.current_mid_price = 0.0
        
    def record_fill(self, fill, mid_price: float, spread_captured: float):
        """Record a fill for KPI calculation"""
        fill_record = FillRecord(
            timestamp=time.time(),
            side=fill.side,
            price=fill.price,
            size=fill.size,
            mid_price_at_fill=mid_price,
            spread_captured=spread_captured
        )
        
        self.fills.append(fill_record)
        self.total_fills += 1
        
        # Cleanup old data periodically
        self._cleanup_old_data()
    
    def record_cancel(self, count: int = 1):
        """Record quote cancellations"""
        self.total_cancels += count
    
    def record_quotes_sent(self, count: int = 1):
        """Record quotes sent to market"""
        self.total_quotes_sent += count
    
    def record_latency(self, metric_name: str, value_ms: float):
        """Record latency measurement"""
        latency_record = LatencyRecord(
            timestamp=time.time(),
            metric_name=metric_name,
            value_ms=value_ms
        )
        
        self.latencies[metric_name].append(latency_record)
        
        # Keep only recent latency data
        self._cleanup_latencies(metric_name)
    
    def record_inventory(self, inventory: float):
        """Record inventory for RMS calculation"""
        self.inventory_history.append((time.time(), inventory))
        
        # Keep only recent inventory data
        while (self.inventory_history and 
               time.time() - self.inventory_history[0][0] > self.window_seconds):
            self.inventory_history.popleft()
    
    def update_mid_price(self, mid_price: float):
        """Update current mid price for mark-to-market PnL calculation"""
        self.current_mid_price = mid_price
    
    def get_spread_captured_pct(self) -> float:
        """Calculate spread captured percentage (§3.7)"""
        if not self.fills:
            return 0.0
        
        recent_fills = self._get_recent_fills()
        if not recent_fills:
            return 0.0
        
        total_spread_captured = sum(f.spread_captured for f in recent_fills)
        total_potential_spread = sum(abs(f.price - f.mid_price_at_fill) for f in recent_fills)
        
        if total_potential_spread == 0:
            return 0.0
        
        return (total_spread_captured / total_potential_spread) * 100
    
    def get_rms_inventory(self) -> float:
        """Calculate RMS inventory (§3.7)"""
        if len(self.inventory_history) < 2:
            return 0.0
        
        inventories = [inv for _, inv in self.inventory_history]
        return float(np.sqrt(np.mean(np.square(inventories))))
    
    def get_fill_ratio(self) -> float:
        """Calculate fill ratio (§3.7)"""
        if self.total_quotes_sent == 0:
            return 0.0
        
        return self.total_fills / self.total_quotes_sent
    
    def get_cancel_ratio(self) -> float:
        """Calculate cancel ratio (§3.7)"""
        if self.total_quotes_sent == 0:
            return 0.0
        
        return self.total_cancels / self.total_quotes_sent
    
    def get_average_latency(self, metric_name: str) -> float:
        """Get average latency for a metric"""
        if metric_name not in self.latencies or not self.latencies[metric_name]:
            return 0.0
        
        recent_latencies = [l.value_ms for l in self.latencies[metric_name]]
        return float(np.mean(recent_latencies))
    
    def get_p99_latency(self, metric_name: str) -> float:
        """Get P99 latency for a metric (§3.7)"""
        if metric_name not in self.latencies or not self.latencies[metric_name]:
            return 0.0
        
        recent_latencies = [l.value_ms for l in self.latencies[metric_name]]
        if len(recent_latencies) < 10:
            return float(np.max(recent_latencies)) if recent_latencies else 0.0
        
        return float(np.percentile(recent_latencies, 99))
    
    def get_total_pnl(self) -> float:
        """Calculate total PnL from fills using current mid price for mark-to-market"""
        if not self.fills:
            return 0.0
        
        recent_fills = self._get_recent_fills()
        pnl = 0.0
        
        # Use current mid price if available, otherwise fall back to fill-time mid
        mark_price = self.current_mid_price if self.current_mid_price > 0 else None
        
        for fill in recent_fills:
            if fill.side == 'bid':
                # Bought at fill.price, mark-to-market at current mid
                mid_to_use = mark_price if mark_price else fill.mid_price_at_fill
                pnl += (mid_to_use - fill.price) * fill.size
            else:  # ask
                # Sold at fill.price, mark-to-market at current mid  
                mid_to_use = mark_price if mark_price else fill.mid_price_at_fill
                pnl += (fill.price - mid_to_use) * fill.size
        
        return pnl
    
    def get_summary(self) -> Dict:
        """Get comprehensive KPI summary"""
        return {
            'spread_captured_pct': self.get_spread_captured_pct(),
            'rms_inventory': self.get_rms_inventory(),
            'fill_ratio': self.get_fill_ratio(),
            'cancel_ratio': self.get_cancel_ratio(),
            'total_pnl': self.get_total_pnl(),
            'avg_spread_captured': np.mean([f.spread_captured for f in self._get_recent_fills()]) if self._get_recent_fills() else 0.0,
            'total_quotes': self.total_quotes_sent,
            'total_fills': self.total_fills,
            'total_cancels': self.total_cancels,
            'latency_stats': {
                metric: {
                    'avg_ms': self.get_average_latency(metric),
                    'p99_ms': self.get_p99_latency(metric)
                } 
                for metric in self.latencies.keys()
            }
        }
    
    def check_performance_targets(self) -> Dict[str, bool]:
        """Check if we meet V1 performance targets (§3.7)"""
        summary = self.get_summary()
        
        targets = {
            'spread_captured_target': summary['spread_captured_pct'] >= mm_config.target_spread_capture_pct,
            'rms_inventory_target': summary['rms_inventory'] <= mm_config.target_rms_inventory_ratio,
            'fill_ratio_target': summary['fill_ratio'] >= (mm_config.target_fill_ratio_pct / 100.0),
            'cancel_ratio_target': summary['cancel_ratio'] <= (mm_config.target_cancel_ratio_pct / 100.0),
            'latency_target': self.get_p99_latency('quote_ack') <= mm_config.target_latency_p99_ms,
            'pnl_target': summary['total_pnl'] >= 0.0  # Positive PnL
        }
        
        return targets
    
    def print_performance_report(self):
        """Print detailed performance report"""
        summary = self.get_summary()
        targets = self.check_performance_targets()
        
        print(f"\n📊 {self.symbol} Performance Report (Last {self.window_seconds/60:.0f}min)")
        print("=" * 60)
        
        # Core KPIs (§3.7 V1-α targets)
        print(f"Spread Captured: {summary['spread_captured_pct']:.1f}% " +
              f"{'✅' if targets['spread_captured_target'] else '❌'} " +
              f"(Target: ≥{mm_config.target_spread_capture_pct:.0f}%)")
        print(f"RMS Inventory: {summary['rms_inventory']:.4f} " +
              f"{'✅' if targets['rms_inventory_target'] else '❌'} " +
              f"(Target: ≤{mm_config.target_rms_inventory_ratio:.1f})")
        print(f"Fill Ratio: {summary['fill_ratio']:.1%} " +
              f"{'✅' if targets['fill_ratio_target'] else '❌'} " +
              f"(Target: ≥{mm_config.target_fill_ratio_pct:.0f}%)")
        print(f"Cancel Ratio: {summary['cancel_ratio']:.1%} " +
              f"{'✅' if targets['cancel_ratio_target'] else '❌'} " +
              f"(Target: ≤{mm_config.target_cancel_ratio_pct:.0f}%)")
        
        # PnL
        pnl_emoji = "💰" if summary['total_pnl'] >= 0 else "📉"
        print(f"Total PnL: {pnl_emoji} ${summary['total_pnl']:+.2f} " +
              f"{'✅' if targets['pnl_target'] else '❌'}")
        
        # Volume stats
        print(f"\nVolume Stats:")
        print(f"  Total Quotes: {summary['total_quotes']}")
        print(f"  Total Fills: {summary['total_fills']}")
        print(f"  Total Cancels: {summary['total_cancels']}")
        
        # Latency stats
        if summary['latency_stats']:
            print(f"\nLatency Stats:")
            for metric, stats in summary['latency_stats'].items():
                p99_ok = "✅" if stats['p99_ms'] <= 300 else "❌"
                print(f"  {metric}: Avg {stats['avg_ms']:.1f}ms | P99 {stats['p99_ms']:.1f}ms {p99_ok}")
        
        # Overall assessment
        targets_met = sum(targets.values())
        total_targets = len(targets)
        overall_emoji = "🎯" if targets_met == total_targets else "⚠️" if targets_met >= total_targets * 0.7 else "🚨"
        
        print(f"\n{overall_emoji} Overall: {targets_met}/{total_targets} targets met")
        print("=" * 60)
    
    def _get_recent_fills(self, window_seconds: Optional[int] = None) -> List[FillRecord]:
        """Get fills within the specified time window"""
        if window_seconds is None:
            window_seconds = self.window_seconds
        
        cutoff_time = time.time() - window_seconds
        return [f for f in self.fills if f.timestamp >= cutoff_time]
    
    def _cleanup_old_data(self):
        """Remove old data outside the time window"""
        current_time = time.time()
        
        # Only cleanup every few minutes to avoid overhead
        if current_time - self._last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - self.window_seconds
        
        # Cleanup fills
        while self.fills and self.fills[0].timestamp < cutoff_time:
            self.fills.popleft()
        
        # Cleanup inventory history  
        while (self.inventory_history and 
               self.inventory_history[0][0] < cutoff_time):
            self.inventory_history.popleft()
        
        self._last_cleanup = current_time
    
    def _cleanup_latencies(self, metric_name: str, max_records: int = 1000):
        """Keep only recent latency records"""
        latency_deque = self.latencies[metric_name]
        
        # Remove old records
        cutoff_time = time.time() - 300  # Keep 5 minutes of latency data
        while latency_deque and latency_deque[0].timestamp < cutoff_time:
            latency_deque.popleft()
        
        # Limit total records
        while len(latency_deque) > max_records:
            latency_deque.popleft()


# Test function
if __name__ == "__main__":
    print("🧪 Testing KPI Tracker...")
    
    tracker = KPITracker('BTCUSDT')
    
    # Simulate some data
    from dataclasses import dataclass
    
    @dataclass
    class MockFill:
        side: str
        price: float
        size: float
    
    # Add some fills
    for i in range(20):
        fill = MockFill(side='bid' if i % 2 == 0 else 'ask', price=50000 + i*10, size=0.01)
        tracker.record_fill(fill, 50000, 5.0)
        tracker.record_inventory(i * 0.001)
    
    # Add latency data
    for i in range(50):
        tracker.record_latency('quote_ack', 30 + i)
    
    # Record quotes and cancels
    tracker.record_quotes_sent(100)
    tracker.record_cancel(15)
    
    # Print report
    tracker.print_performance_report()
