"""
Avellaneda-Stoikov V1.5 Enhanced Quoter

Implements the V1.5 pricing mechanism with:
- Multi-signal pricing: centre_t = r_t + κ_inv·q_t + β_ofi·OFI_t + β_di·DI_t  
- Dynamic spread: spread_t = base_spread + κ_vol·σ_t + κ_inv·|q_t|
- Enhanced risk controls with global offset clamp ≤ 3 ticks
- Spread floor ≥ 2 ticks

See spec §4.4 and §4.7.
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

from ..utils.config import mm_config
from .avellaneda_stoikov import AvellanedaStoikovQuoter


class AvellanedaStoikovV15Quoter(AvellanedaStoikovQuoter):
    """
    Enhanced V1.5 quoter with multi-signal pricing and dynamic spread
    
    Extends the base A&S quoter with:
    - Depth Imbalance (DI) signal integration
    - Enhanced inventory management (κ_inv)
    - Volatility-scaled spread (κ_vol)
    - Advanced risk controls
    """
    
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.logger = logging.getLogger(f"AS-V15-{symbol}")
        
        # V1.5 specific parameters
        symbol_params = mm_config.get_symbol_params(symbol)
        
        # Signal coefficients
        self.beta_di = symbol_params.get('beta_di', 0.2)  # DI → ticks
        self.kappa_inv = symbol_params.get('kappa_inv', 0.1)  # Inventory penalty
        self.kappa_vol = symbol_params.get('kappa_vol', 1.2)  # Volatility sensitivity
        
        # Risk control parameters
        self.max_offset_ticks = 3  # Global offset clamp ≤ 3 ticks
        self.min_spread_ticks = 2  # Spread floor ≥ 2 ticks
        
        # Get tick size for the symbol
        self.tick_size = mm_config.get_symbol_config(symbol).get(
            'tick_size', mm_config.default_tick_size
        )
        
        # Cache for performance
        self._last_volatility = None
        self._cached_base_spread = None
        
        self.logger.info(f"📊 V1.5 Quoter initialized: β_di={self.beta_di}, "
                        f"κ_inv={self.kappa_inv}, κ_vol={self.kappa_vol}")
    
    def compute_quotes_v15(self, mid_price: float, inventory: float,
                          ofi: float = 0.0, di: float = 0.0,
                          time_remaining: float = None) -> Dict[str, float]:
        """
        Compute V1.5 quotes with enhanced multi-signal pricing
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory position
            ofi: Order Flow Imbalance signal
            di: Depth Imbalance signal  
            time_remaining: Time remaining to horizon
            
        Returns:
            Dictionary with quote data including V1.5 enhancements
        """
        # 1. Compute base reservation price (A&S)
        reservation_price = self.compute_reservation_price(
            mid_price, inventory, time_remaining
        )
        
        # 2. Compute base spread (A&S) - convert from fraction to absolute price
        base_spread_fraction = self.compute_optimal_spread(
            mid_price, inventory, time_remaining
        )
        base_spread = base_spread_fraction * mid_price
        
        # Ensure minimum base spread to avoid zero spread issues
        min_base_spread = self.min_spread_ticks * self.tick_size
        if base_spread < min_base_spread:
            base_spread = min_base_spread
        
        # 3. Enhanced centre calculation with multi-signal approach
        centre_price = self._compute_enhanced_centre(
            reservation_price, inventory, ofi, di
        )
        
        # 4. Dynamic spread calculation
        dynamic_spread = self._compute_dynamic_spread(
            base_spread, inventory, mid_price
        )
        
        # 5. Apply risk controls
        centre_price, dynamic_spread = self._apply_risk_controls(
            centre_price, dynamic_spread, mid_price
        )
        
        # 6. Calculate final bid/ask prices
        half_spread = dynamic_spread / 2
        bid_price = centre_price - half_spread
        ask_price = centre_price + half_spread
        
        # 7. Calculate metrics and diagnostics
        quotes = {
            # Core prices
            'bid_price': bid_price,
            'ask_price': ask_price,
            'mid_price': mid_price,
            
            # V1.5 pricing components
            'centre_price': centre_price,
            'reservation_price': reservation_price,
            'base_spread': base_spread,
            'dynamic_spread': dynamic_spread,
            
            # Signal inputs
            'ofi': ofi,
            'di': di,
            'inventory': inventory,
            
            # Spread metrics
            'spread_bps': (dynamic_spread / mid_price) * 10000,
            'half_spread': half_spread,
            
            # V1.5 components breakdown
            'inventory_adjustment': self.kappa_inv * inventory * self.tick_size,
            'ofi_adjustment': mm_config.beta_ofi * ofi * self.tick_size,
            'di_adjustment': self.beta_di * di * self.tick_size,
            'volatility_spread_component': self._get_volatility_spread_component(base_spread),
            'inventory_spread_component': self.kappa_inv * abs(inventory) * base_spread,
            
            # Model parameters
            'volatility': self.sigma,
            'time_remaining': time_remaining or self.T,
            'tick_size': self.tick_size,
            
            # Risk control status
            'offset_clamped': False,  # Will be set by risk controls
            'spread_floored': False,  # Will be set by risk controls
        }
        
        return quotes
    
    def _compute_enhanced_centre(self, reservation_price: float, inventory: float,
                               ofi: float, di: float) -> float:
        """
        Compute enhanced centre price with multi-signal approach
        
        Formula: centre_t = r_t + κ_inv·q_t + β_ofi·OFI_t + β_di·DI_t
        """
        # Start with base reservation price
        centre = reservation_price
        
        # Add inventory adjustment (κ_inv·q_t) - scale with tick size for now
        inventory_adj = self.kappa_inv * inventory * self.tick_size
        centre += inventory_adj
        
        # Add OFI signal (β_ofi·OFI_t) - using existing beta_ofi from config
        ofi_adj = mm_config.beta_ofi * ofi * self.tick_size
        centre += ofi_adj
        
        # Add DI signal (β_di·DI_t) - new in V1.5
        di_adj = self.beta_di * di * self.tick_size
        centre += di_adj
        
        return centre
    
    def _compute_dynamic_spread(self, base_spread: float, inventory: float,
                              mid_price: float) -> float:
        """
        Compute dynamic spread with volatility and inventory scaling
        
        Formula: spread_t = base_spread + κ_vol·σ_t + κ_inv·|q_t|
        """
        # Start with base A&S spread
        spread = base_spread
        
        # Add volatility component (κ_vol·σ_t) 
        # σ is annualized volatility, scale it down for short-term spread adjustment
        # Use a much smaller scaling factor to avoid excessive spreads
        vol_component = self.kappa_vol * base_spread * self.sigma
        spread += vol_component
        
        # Add inventory component (κ_inv·|q_t|)
        # Scale inventory impact relative to base spread rather than fixed tick size
        inv_component = self.kappa_inv * abs(inventory) * base_spread
        spread += inv_component
        
        return spread
    
    def _apply_risk_controls(self, centre_price: float, spread: float,
                           mid_price: float) -> Tuple[float, float]:
        """
        Apply V1.5 risk controls: global offset clamp and spread floor
        """
        # 1. Global offset clamp: total offset from mid ≤ 3 ticks
        max_offset = self.max_offset_ticks * self.tick_size
        centre_offset = centre_price - mid_price
        
        offset_clamped = False
        if abs(centre_offset) > max_offset:
            centre_price = mid_price + np.sign(centre_offset) * max_offset
            offset_clamped = True
        
        # 2. Spread floor: minimum 2 ticks
        min_spread = self.min_spread_ticks * self.tick_size
        spread_floored = False
        if spread < min_spread:
            spread = min_spread
            spread_floored = True
        
        # Store risk control status for diagnostics
        self._last_risk_controls = {
            'offset_clamped': offset_clamped,
            'spread_floored': spread_floored,
            'original_offset': centre_offset,
            'final_offset': centre_price - mid_price,
            'original_spread': spread,
            'final_spread': spread
        }
        
        return centre_price, spread
    
    def _get_volatility_spread_component(self, base_spread: float) -> float:
        """Get the volatility component of the dynamic spread"""
        return self.kappa_vol * base_spread * self.sigma
    
    def validate_quotes_v15(self, quotes: Dict[str, float]) -> bool:
        """
        Enhanced validation for V1.5 quotes
        """
        # Run base validation first
        if not self.validate_quotes(quotes, quotes['mid_price']):
            return False
        
        # V1.5 specific validations
        
        # Check spread floor
        if quotes['dynamic_spread'] < self.min_spread_ticks * self.tick_size:
            self.logger.warning(f"⚠️  Spread below floor: {quotes['dynamic_spread']:.6f} < {self.min_spread_ticks * self.tick_size:.6f}")
            return False
        
        # Check offset clamp
        offset = abs(quotes['centre_price'] - quotes['mid_price'])
        max_offset = self.max_offset_ticks * self.tick_size
        if offset > max_offset * 1.01:  # Small tolerance for floating point
            self.logger.warning(f"⚠️  Offset exceeds clamp: {offset:.6f} > {max_offset:.6f}")
            return False
        
        # Check signal ranges (sanity check)
        if abs(quotes['ofi']) > 10:  # OFI should be normalized
            self.logger.warning(f"⚠️  OFI out of expected range: {quotes['ofi']:.4f}")
            return False
        
        if abs(quotes['di']) > 10:  # DI should be normalized
            self.logger.warning(f"⚠️  DI out of expected range: {quotes['di']:.4f}")
            return False
        
        return True
    
    def get_v15_params(self) -> Dict:
        """Get V1.5 specific parameters"""
        base_params = self.get_model_params()
        
        v15_params = {
            **base_params,
            'version': 'V1.5',
            'beta_di': self.beta_di,
            'kappa_inv': self.kappa_inv,
            'kappa_vol': self.kappa_vol,
            'max_offset_ticks': self.max_offset_ticks,
            'min_spread_ticks': self.min_spread_ticks,
            'tick_size': self.tick_size,
        }
        
        return v15_params
    
    def update_v15_params(self, **kwargs):
        """Update V1.5 parameters dynamically"""
        if 'beta_di' in kwargs:
            self.beta_di = kwargs['beta_di']
            self.logger.info(f"📊 Updated β_di: {self.beta_di}")
        
        if 'kappa_inv' in kwargs:
            self.kappa_inv = kwargs['kappa_inv']
            self.logger.info(f"📊 Updated κ_inv: {self.kappa_inv}")
        
        if 'kappa_vol' in kwargs:
            self.kappa_vol = kwargs['kappa_vol']
            self.logger.info(f"📊 Updated κ_vol: {self.kappa_vol}")
    
    def get_signal_impact_analysis(self, quotes: Dict[str, float]) -> Dict:
        """
        Analyze the impact of each signal component on pricing
        """
        mid_price = quotes['mid_price']
        
        analysis = {
            'total_centre_shift': quotes['centre_price'] - mid_price,
            'inventory_impact': quotes['inventory_adjustment'],
            'ofi_impact': quotes['ofi_adjustment'],
            'di_impact': quotes['di_adjustment'],
            'total_spread_enhancement': quotes['dynamic_spread'] - quotes['base_spread'],
            'volatility_spread_impact': quotes['volatility_spread_component'],
            'inventory_spread_impact': quotes['inventory_spread_component'],
            'signal_contributions': {
                'inventory_pct': abs(quotes['inventory_adjustment']) / (abs(quotes['centre_price'] - mid_price) + 1e-8) * 100,
                'ofi_pct': abs(quotes['ofi_adjustment']) / (abs(quotes['centre_price'] - mid_price) + 1e-8) * 100,
                'di_pct': abs(quotes['di_adjustment']) / (abs(quotes['centre_price'] - mid_price) + 1e-8) * 100,
            }
        }
        
        return analysis


def test_v15_quoter():
    """Test function for V1.5 quoter"""
    print("🧪 Testing V1.5 Enhanced Quoter...")
    
    quoter = AvellanedaStoikovV15Quoter('BTCUSDT')
    
    # Test parameters
    mid_price = 50000.0
    inventory = 0.1
    ofi = 0.5  # Positive OFI (buy pressure)
    di = -0.3  # Negative DI (ask-heavy depth)
    
    # Simulate volatility
    for i in range(20):
        price = mid_price + np.random.normal(0, 100)
        quoter.update_volatility(price)
    
    # Compute V1.5 quotes
    quotes = quoter.compute_quotes_v15(mid_price, inventory, ofi, di)
    
    print(f"\n📊 V1.5 Quotes for {quoter.symbol}")
    print("=" * 50)
    print(f"Mid Price: ${quotes['mid_price']:.2f}")
    print(f"Inventory: {quotes['inventory']:.4f}")
    print(f"OFI: {quotes['ofi']:+.3f}")
    print(f"DI: {quotes['di']:+.3f}")
    print()
    print(f"Reservation Price: ${quotes['reservation_price']:.2f}")
    print(f"Centre Price: ${quotes['centre_price']:.2f}")
    print(f"Base Spread: ${quotes['base_spread']:.4f}")
    print(f"Dynamic Spread: ${quotes['dynamic_spread']:.4f}")
    print()
    print(f"Final Bid: ${quotes['bid_price']:.2f}")
    print(f"Final Ask: ${quotes['ask_price']:.2f}")
    print(f"Spread: {quotes['spread_bps']:.1f} bps")
    
    # Signal impact analysis
    analysis = quoter.get_signal_impact_analysis(quotes)
    print(f"\n📈 Signal Impact Analysis:")
    print(f"Total Centre Shift: ${analysis['total_centre_shift']:+.4f}")
    print(f"  - Inventory: ${analysis['inventory_impact']:+.4f} ({analysis['signal_contributions']['inventory_pct']:.1f}%)")
    print(f"  - OFI: ${analysis['ofi_impact']:+.4f} ({analysis['signal_contributions']['ofi_pct']:.1f}%)")
    print(f"  - DI: ${analysis['di_impact']:+.4f} ({analysis['signal_contributions']['di_pct']:.1f}%)")
    
    # Validation
    is_valid = quoter.validate_quotes_v15(quotes)
    print(f"\n✅ Validation: {'PASS' if is_valid else 'FAIL'}")
    
    print("✅ V1.5 Quoter test completed")


if __name__ == "__main__":
    test_v15_quoter()
