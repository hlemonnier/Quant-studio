"""
Parameter Calibration Module (§3.4)

Implements the parameter calibration methodology described in the spec:
1. Market impact parameter k via log-linear regression P(fill) vs spread
2. Risk aversion γ via grid-search maximizing Sharpe under RMS inventory constraint  
3. Time horizon T based on target liquidation delay

This addresses the missing calibration algorithms from §3.4.
"""

import sys
import pathlib
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List, Optional
import logging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Enable imports when running as script
# We only need to go two levels up to reach the project root
# …/strategies/MM/parameter_calibration.py → …/strategies → …/ (repo root)
repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Use absolute import when run directly
if __name__ == "__main__":
    from strategies.MM.config import mm_config
else:
    from .config import mm_config


class ParameterCalibrator:
    """Calibrates Avellaneda-Stoikov parameters from historical data"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"ParamCalibrator-{symbol}")
        
    def calibrate_market_impact_k(self, historical_data: pd.DataFrame) -> float:
        """
        Calibrate k parameter via log-linear regression (§3.4)
        
        Method: P(fill) vs spread distance
        Model: log(P_fill) = log(A) - k * δ
        Where δ is spread distance in ticks
        
        Args:
            historical_data: DataFrame with columns ['spread_ticks', 'fill_probability']
        
        Returns:
            Calibrated k parameter
        """
        self.logger.info("🔧 Calibrating market impact parameter k...")
        
        if len(historical_data) < 20:
            self.logger.warning("Insufficient data for k calibration, using default")
            return mm_config.k
        
        # Prepare regression data
        spreads = historical_data['spread_ticks'].values
        fill_probs = np.clip(historical_data['fill_probability'].values, 1e-6, 1-1e-6)
        
        # Log-linear regression: ln(P) = ln(A) - k*δ
        X = spreads.reshape(-1, 1)
        y = np.log(fill_probs)
        
        try:
            reg = LinearRegression().fit(X, y)
            k_estimate = -reg.coef_[0]  # Negative slope = k
            r2 = r2_score(y, reg.predict(X))
            
            # Validate results
            if k_estimate <= 0 or r2 < 0.3:
                self.logger.warning(f"Poor k calibration (k={k_estimate:.3f}, R²={r2:.3f}), using default")
                return mm_config.k
            
            # Apply reasonable bounds
            k_calibrated = np.clip(k_estimate, 0.1, 5.0)
            
            self.logger.info(f"✅ k calibrated: {k_calibrated:.3f} (R²={r2:.3f})")
            return k_calibrated
            
        except Exception as e:
            self.logger.error(f"k calibration failed: {e}")
            return mm_config.k
    
    def calibrate_risk_aversion_gamma(self, 
                                    historical_returns: pd.DataFrame,
                                    target_inventory_rms: float = 0.3) -> float:
        """
        Calibrate γ via grid-search maximizing Sharpe under inventory constraint (§3.4)
        
        Args:
            historical_returns: DataFrame with columns ['timestamp', 'pnl', 'inventory']
            target_inventory_rms: Maximum acceptable RMS inventory
            
        Returns:
            Calibrated γ parameter
        """
        self.logger.info("🔧 Calibrating risk aversion parameter γ...")
        
        if len(historical_returns) < 100:
            self.logger.warning("Insufficient data for γ calibration, using default")
            return mm_config.gamma
        
        # Grid search range for γ
        gamma_range = np.logspace(-5, -1, 20)  # 1e-5 to 1e-1
        best_gamma = mm_config.gamma
        best_score = -np.inf
        
        results = []
        
        for gamma in gamma_range:
            try:
                # Calculate performance metrics for this γ
                metrics = self._evaluate_gamma_performance(historical_returns, gamma)
                
                # Score function: Sharpe ratio if inventory constraint is met
                if metrics['rms_inventory'] <= target_inventory_rms:
                    score = metrics['sharpe_ratio']
                else:
                    # Penalize excessive inventory
                    penalty = (metrics['rms_inventory'] - target_inventory_rms) / target_inventory_rms
                    score = metrics['sharpe_ratio'] - penalty * 2.0
                
                results.append({
                    'gamma': gamma,
                    'sharpe': metrics['sharpe_ratio'], 
                    'rms_inventory': metrics['rms_inventory'],
                    'total_pnl': metrics['total_pnl'],
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_gamma = gamma
                    
            except Exception as e:
                self.logger.warning(f"γ={gamma:.5f} evaluation failed: {e}")
                continue
        
        # Log results
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            self.logger.info(f"✅ γ calibrated: {best_gamma:.5f}")
            self.logger.info(f"   Best Sharpe: {results_df.loc[results_df['score'].idxmax(), 'sharpe']:.3f}")
            self.logger.info(f"   RMS Inventory: {results_df.loc[results_df['score'].idxmax(), 'rms_inventory']:.4f}")
        else:
            self.logger.warning("γ calibration failed, using default")
            
        return best_gamma
    
    def _evaluate_gamma_performance(self, historical_data: pd.DataFrame, gamma: float) -> Dict:
        """Evaluate performance metrics for a given γ value"""
        
        # Simple simulation: assume γ affects inventory management directly
        # In reality, this would re-run the full A&S strategy with different γ
        
        inventories = historical_data['inventory'].values
        pnl_values = historical_data['pnl'].values
        
        # Apply γ-based inventory scaling (simplified)
        scaled_inventories = inventories / (1 + gamma * 1000)  # Higher γ → smaller positions
        
        # Calculate metrics
        rms_inventory = np.sqrt(np.mean(scaled_inventories ** 2))
        total_pnl = np.sum(pnl_values) / (1 + gamma * 100)  # Higher γ → lower PnL
        pnl_volatility = np.std(pnl_values) / (1 + gamma * 50)  # Higher γ → lower vol
        
        sharpe_ratio = total_pnl / max(pnl_volatility, 1e-6) if pnl_volatility > 0 else 0
        
        return {
            'rms_inventory': rms_inventory,
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'pnl_volatility': pnl_volatility
        }
    
    def calibrate_time_horizon_T(self, 
                                typical_position_size: float,
                                daily_volume: float) -> float:
        """
        Calibrate T based on liquidation time analysis (§3.4)
        
        Args:
            typical_position_size: Average position size to liquidate
            daily_volume: Average daily trading volume
            
        Returns:
            Calibrated T in fraction of day (e.g., 120s = 120/86400)
        """
        self.logger.info("🔧 Calibrating time horizon T...")
        
        if daily_volume <= 0:
            self.logger.warning("Invalid volume data, using default T")
            return mm_config.T
        
        # Estimate liquidation time based on position size vs market volume
        # Rule of thumb: don't trade more than 1% of daily volume per interval
        safe_trade_rate = daily_volume * 0.01  # 1% of daily volume
        
        if safe_trade_rate <= 0:
            liquidation_time_seconds = 300  # 5 minutes default
        else:
            liquidation_time_seconds = (typical_position_size / safe_trade_rate) * 86400  # Convert to seconds
        
        # Apply reasonable bounds (60s to 1800s)
        liquidation_time_seconds = np.clip(liquidation_time_seconds, 60, 1800)
        
        # Convert to fraction of day
        T_calibrated = liquidation_time_seconds / 86400
        
        self.logger.info(f"✅ T calibrated: {T_calibrated:.6f} ({liquidation_time_seconds:.0f}s)")
        
        return T_calibrated
    
    def run_full_calibration(self, 
                           market_data: pd.DataFrame,
                           trading_history: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Run complete parameter calibration workflow (§3.4)
        
        Args:
            market_data: Historical market data for k calibration
            trading_history: Historical trading results for γ calibration
            
        Returns:
            Dictionary of calibrated parameters
        """
        self.logger.info(f"🚀 Starting full parameter calibration for {self.symbol}")
        
        calibrated_params = {}
        
        # 1. Calibrate k from market microstructure
        if 'spread_ticks' in market_data.columns and 'fill_probability' in market_data.columns:
            calibrated_params['k'] = self.calibrate_market_impact_k(market_data)
        else:
            self.logger.warning("Market data missing spread/fill columns, skipping k calibration")
            calibrated_params['k'] = mm_config.k
        
        # 2. Calibrate γ from trading performance
        if trading_history is not None and len(trading_history) > 0:
            calibrated_params['gamma'] = self.calibrate_risk_aversion_gamma(trading_history)
        else:
            self.logger.warning("No trading history provided, skipping γ calibration") 
            calibrated_params['gamma'] = mm_config.gamma
        
        # 3. Calibrate T from volume analysis
        if 'volume' in market_data.columns:
            daily_vol = market_data['volume'].sum()
            typical_pos = calibrated_params['gamma'] * 10  # Rough estimate
            calibrated_params['T'] = self.calibrate_time_horizon_T(typical_pos, daily_vol)
        else:
            calibrated_params['T'] = mm_config.T
        
        # 4. Apply calibrated parameters to config
        self._apply_calibrated_parameters(calibrated_params)
        
        self.logger.info("✅ Full calibration completed")
        return calibrated_params
    
    def _apply_calibrated_parameters(self, params: Dict[str, float]):
        """Apply calibrated parameters to global config"""
        for param_name, value in params.items():
            if hasattr(mm_config, param_name):
                setattr(mm_config, param_name, value)
                self.logger.info(f"Applied {param_name} = {value}")


def generate_synthetic_calibration_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data for testing calibration (demo purposes)"""
    
    # Market microstructure data
    spreads = np.linspace(0.5, 5.0, 50)  # 0.5 to 5 ticks
    fill_probs = 0.8 * np.exp(-0.4 * spreads) + np.random.normal(0, 0.05, len(spreads))
    fill_probs = np.clip(fill_probs, 0.01, 0.99)
    
    market_data = pd.DataFrame({
        'spread_ticks': spreads,
        'fill_probability': fill_probs,
        'volume': np.random.exponential(100, len(spreads))
    })
    
    # Trading performance data
    timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')
    inventories = np.cumsum(np.random.normal(0, 0.01, 1000))  # Random walk inventory
    pnl = np.cumsum(np.random.normal(0.001, 0.02, 1000))  # Slightly positive drift
    
    trading_history = pd.DataFrame({
        'timestamp': timestamps,
        'inventory': inventories,
        'pnl': pnl
    })
    
    return market_data, trading_history


# Demo/test function
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing Parameter Calibration...")
    
    calibrator = ParameterCalibrator('BTCUSDT')
    
    # Generate synthetic data
    market_data, trading_history = generate_synthetic_calibration_data()
    
    # Run calibration
    results = calibrator.run_full_calibration(market_data, trading_history)
    
    print("\n📊 Calibration Results:")
    for param, value in results.items():
        print(f"  {param}: {value:.6f}")
    
    print("\n✅ Calibration test completed!")
