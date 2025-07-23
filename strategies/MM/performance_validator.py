"""
Performance Validator for V1-α (§3.7 & §3.8)

Validates that the trading strategy meets all performance targets
specified in the V1-α requirements document.
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np

from config import MMConfig
from kpi_tracker import KPITracker

mm_config = MMConfig()


class PerformanceValidator:
    """
    Validates performance against V1-α targets (§3.7)
    
    Targets from specification:
    - Spread capture ≥ 70%
    - RMS inventory ≤ 0.4 q_max
    - Fill ratio ≥ 5%
    - Cancel ratio ≤ 70%
    - Latency P99 ≤ 300ms
    - PnL > 0 (positive)
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"PerfValidator-{symbol}")
        
        # Validation history
        self.validation_history: List[Dict] = []
        self.last_validation_time = None
        
    def validate_performance(self, kpi_tracker: KPITracker) -> Dict:
        """
        Validate current performance against V1-α targets
        
        Returns:
            Dict with validation results and recommendations
        """
        summary = kpi_tracker.get_summary()
        targets = kpi_tracker.check_performance_targets()
        
        # Calculate overall compliance
        targets_met = sum(targets.values())
        total_targets = len(targets)
        compliance_pct = (targets_met / total_targets) * 100
        
        validation_result = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'compliance_pct': compliance_pct,
            'targets_met': targets_met,
            'total_targets': total_targets,
            'individual_targets': targets,
            'metrics': summary,
            'status': self._determine_status(compliance_pct),
            'recommendations': self._generate_recommendations(targets, summary)
        }
        
        # Store in history
        self.validation_history.append(validation_result)
        self.last_validation_time = datetime.now()
        
        # Log results
        self._log_validation_results(validation_result)
        
        return validation_result
    
    def _determine_status(self, compliance_pct: float) -> str:
        """Determine overall performance status"""
        if compliance_pct >= 100:
            return "EXCELLENT"
        elif compliance_pct >= 80:
            return "GOOD"
        elif compliance_pct >= 60:
            return "ACCEPTABLE"
        elif compliance_pct >= 40:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _generate_recommendations(self, targets: Dict[str, bool], 
                                summary: Dict) -> List[str]:
        """Generate actionable recommendations based on failed targets"""
        recommendations = []
        
        if not targets['spread_captured_target']:
            spread_pct = summary['spread_captured_pct']
            target_pct = mm_config.target_spread_capture_pct
            recommendations.append(
                f"🎯 Improve spread capture: {spread_pct:.1f}% < {target_pct:.0f}%. "
                f"Consider tighter quotes or better timing."
            )
        
        if not targets['rms_inventory_target']:
            rms_inv = summary['rms_inventory']
            target_rms = mm_config.target_rms_inventory_ratio
            recommendations.append(
                f"📦 Reduce inventory risk: RMS {rms_inv:.3f} > {target_rms:.1f}. "
                f"Increase inventory skew or reduce position sizes."
            )
        
        if not targets['fill_ratio_target']:
            fill_ratio = summary['fill_ratio'] * 100
            target_fill = mm_config.target_fill_ratio_pct
            recommendations.append(
                f"🎯 Improve fill ratio: {fill_ratio:.1f}% < {target_fill:.0f}%. "
                f"Consider more aggressive pricing or larger sizes."
            )
        
        if not targets['cancel_ratio_target']:
            cancel_ratio = summary['cancel_ratio'] * 100
            target_cancel = mm_config.target_cancel_ratio_pct
            recommendations.append(
                f"⏰ Reduce cancellations: {cancel_ratio:.1f}% > {target_cancel:.0f}%. "
                f"Improve quote stability or timing."
            )
        
        if not targets['latency_target']:
            recommendations.append(
                f"⚡ Reduce latency: P99 > {mm_config.target_latency_p99_ms:.0f}ms. "
                f"Optimize infrastructure or quote frequency."
            )
        
        if not targets['pnl_target']:
            pnl = summary['total_pnl']
            recommendations.append(
                f"💰 Improve profitability: PnL ${pnl:+.2f} ≤ 0. "
                f"Review spread settings and risk controls."
            )
        
        return recommendations
    
    def _log_validation_results(self, result: Dict):
        """Log validation results"""
        status = result['status']
        compliance = result['compliance_pct']
        targets_met = result['targets_met']
        total_targets = result['total_targets']
        
        if status in ['EXCELLENT', 'GOOD']:
            log_level = logging.INFO
            emoji = "✅"
        elif status == 'ACCEPTABLE':
            log_level = logging.WARNING
            emoji = "⚠️"
        else:
            log_level = logging.ERROR
            emoji = "❌"
        
        self.logger.log(
            log_level,
            f"{emoji} Performance {status}: {compliance:.0f}% "
            f"({targets_met}/{total_targets} targets met)"
        )
        
        # Log recommendations if any
        for rec in result['recommendations']:
            self.logger.warning(f"💡 {rec}")
    
    def get_validation_trend(self, hours: int = 24) -> Dict:
        """Get performance trend over specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_validations = [
            v for v in self.validation_history 
            if v['timestamp'] >= cutoff_time
        ]
        
        if not recent_validations:
            return {'trend': 'NO_DATA', 'validations': 0}
        
        # Calculate trend
        compliance_scores = [v['compliance_pct'] for v in recent_validations]
        
        if len(compliance_scores) < 2:
            trend = 'STABLE'
        else:
            # Simple linear trend
            x = np.arange(len(compliance_scores))
            slope = np.polyfit(x, compliance_scores, 1)[0]
            
            if slope > 5:
                trend = 'IMPROVING'
            elif slope < -5:
                trend = 'DECLINING'
            else:
                trend = 'STABLE'
        
        return {
            'trend': trend,
            'validations': len(recent_validations),
            'avg_compliance': np.mean(compliance_scores),
            'latest_compliance': compliance_scores[-1] if compliance_scores else 0,
            'slope': slope if len(compliance_scores) >= 2 else 0
        }
    
    def print_validation_report(self, kpi_tracker: KPITracker):
        """Print comprehensive validation report"""
        result = self.validate_performance(kpi_tracker)
        trend = self.get_validation_trend()
        
        print(f"\n🔍 V1-α Performance Validation Report")
        print("=" * 60)
        print(f"Symbol: {self.symbol}")
        print(f"Timestamp: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Status: {result['status']} ({result['compliance_pct']:.0f}%)")
        print(f"Targets Met: {result['targets_met']}/{result['total_targets']}")
        
        # Individual target status
        print(f"\n📊 Individual Target Status:")
        targets = result['individual_targets']
        metrics = result['metrics']
        
        status_emoji = lambda x: "✅" if x else "❌"
        
        print(f"  Spread Capture: {status_emoji(targets['spread_captured_target'])} "
              f"{metrics['spread_captured_pct']:.1f}% (≥{mm_config.target_spread_capture_pct:.0f}%)")
        print(f"  RMS Inventory:  {status_emoji(targets['rms_inventory_target'])} "
              f"{metrics['rms_inventory']:.3f} (≤{mm_config.target_rms_inventory_ratio:.1f})")
        print(f"  Fill Ratio:     {status_emoji(targets['fill_ratio_target'])} "
              f"{metrics['fill_ratio']:.1%} (≥{mm_config.target_fill_ratio_pct:.0f}%)")
        print(f"  Cancel Ratio:   {status_emoji(targets['cancel_ratio_target'])} "
              f"{metrics['cancel_ratio']:.1%} (≤{mm_config.target_cancel_ratio_pct:.0f}%)")
        print(f"  Latency P99:    {status_emoji(targets['latency_target'])} "
              f"(≤{mm_config.target_latency_p99_ms:.0f}ms)")
        print(f"  PnL Positive:   {status_emoji(targets['pnl_target'])} "
              f"${metrics['total_pnl']:+.2f}")
        
        # Trend analysis
        print(f"\n📈 Performance Trend (24h): {trend['trend']}")
        if trend['validations'] > 1:
            print(f"  Average Compliance: {trend['avg_compliance']:.0f}%")
            print(f"  Trend Slope: {trend['slope']:+.1f}%/validation")
        
        # Recommendations
        if result['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"  {i}. {rec}")
        else:
            print(f"\n🎉 All targets met! Strategy performing excellently.")
        
        print("=" * 60)


def quick_validation_test():
    """Quick test of the validation system"""
    from kpi_tracker import KPITracker
    
    # Create test tracker with some data
    tracker = KPITracker('BTCUSDT')
    validator = PerformanceValidator('BTCUSDT')
    
    # Simulate some performance data
    # (In real usage, this would come from actual trading)
    
    print("🧪 Testing Performance Validator...")
    validator.print_validation_report(tracker)


if __name__ == "__main__":
    quick_validation_test()
