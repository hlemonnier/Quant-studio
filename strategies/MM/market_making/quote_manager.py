"""
Quote Manager with Ageing System for V1.5

Implements quote lifecycle management with:
- 750ms timeout for quote ageing
- Signal-based refresh when OFI/DI direction changes
- Coordinated quote cancel/replace logic

See spec §4.5.
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple
from enum import Enum
import logging

from ..utils.config import mm_config


class QuoteState(Enum):
    """Quote lifecycle states"""
    PENDING = "pending"      # Quote sent, waiting for ACK
    ACTIVE = "active"        # Quote confirmed active in market
    CANCELLED = "cancelled"  # Quote cancelled
    FILLED = "filled"        # Quote executed
    EXPIRED = "expired"      # Quote aged out


@dataclass
class ManagedQuote:
    """Represents a quote under management"""
    quote_id: str
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    timestamp: float
    state: QuoteState = QuoteState.PENDING
    
    # Signal state when quote was created (for change detection)
    ofi_sign: int = 0  # -1, 0, +1
    di_sign: int = 0   # -1, 0, +1
    
    # Execution tracking
    filled_size: float = 0.0
    remaining_size: float = field(init=False)
    
    def __post_init__(self):
        self.remaining_size = self.size
    
    @property
    def age_ms(self) -> float:
        """Get quote age in milliseconds"""
        return (time.time() - self.timestamp) * 1000
    
    @property
    def is_aged_out(self) -> bool:
        """Check if quote has exceeded age limit"""
        return self.age_ms > mm_config.quote_ageing_ms
    
    def update_fill(self, filled_qty: float):
        """Update quote with partial or full fill"""
        self.filled_size += filled_qty
        self.remaining_size = max(0, self.size - self.filled_size)
        
        if self.remaining_size <= 1e-8:  # Fully filled
            self.state = QuoteState.FILLED


class QuoteManager:
    """
    Manages quote lifecycle with ageing and signal-based refresh
    
    Key responsibilities:
    - Track active quotes and their ages
    - Detect when quotes need refresh (age or signal change)
    - Coordinate cancel/replace operations
    - Maintain quote state consistency
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"QuoteManager-{symbol}")
        
        # Configuration
        self.max_age_ms = getattr(mm_config, 'quote_ageing_ms', 750)  # 750ms default
        
        # Active quotes tracking
        self.active_quotes: Dict[str, ManagedQuote] = {}
        self.quotes_by_side: Dict[str, Set[str]] = {'bid': set(), 'ask': set()}
        
        # Signal state tracking for change detection
        self.last_ofi_sign: int = 0
        self.last_di_sign: int = 0
        
        # Performance metrics
        self.total_quotes_managed = 0
        self.quotes_aged_out = 0
        self.quotes_signal_refreshed = 0
        self.quotes_filled = 0
        self.quotes_cancelled = 0
        
        # Async coordination
        self._refresh_lock = asyncio.Lock()
        
        self.logger.info(f"📋 Quote Manager initialized for {symbol} (max_age={self.max_age_ms}ms)")
    
    def add_quote(self, quote_id: str, side: str, price: float, size: float,
                  ofi: float = 0.0, di: float = 0.0) -> ManagedQuote:
        """
        Add a new quote to management
        
        Args:
            quote_id: Unique quote identifier
            side: 'bid' or 'ask'
            price: Quote price
            size: Quote size
            ofi: Current OFI signal value
            di: Current DI signal value
            
        Returns:
            ManagedQuote object
        """
        # Create managed quote with signal state
        quote = ManagedQuote(
            quote_id=quote_id,
            side=side,
            price=price,
            size=size,
            timestamp=time.time(),
            ofi_sign=self._get_signal_sign(ofi),
            di_sign=self._get_signal_sign(di)
        )
        
        # Add to tracking structures
        self.active_quotes[quote_id] = quote
        self.quotes_by_side[side].add(quote_id)
        
        # Update signal state
        self.last_ofi_sign = quote.ofi_sign
        self.last_di_sign = quote.di_sign
        
        self.total_quotes_managed += 1
        
        self.logger.debug(f"📋 Added quote {quote_id}: {side} ${price:.2f} x {size:.4f}")
        
        return quote
    
    def remove_quote(self, quote_id: str, reason: str = "unknown"):
        """Remove a quote from management"""
        if quote_id not in self.active_quotes:
            return
        
        quote = self.active_quotes[quote_id]
        
        # Remove from tracking structures
        del self.active_quotes[quote_id]
        self.quotes_by_side[quote.side].discard(quote_id)
        
        # Update metrics based on reason
        if reason == "aged_out":
            self.quotes_aged_out += 1
        elif reason == "signal_change":
            self.quotes_signal_refreshed += 1
        elif reason == "filled":
            self.quotes_filled += 1
        elif reason == "cancelled":
            self.quotes_cancelled += 1
        
        self.logger.debug(f"📋 Removed quote {quote_id}: {reason}")
    
    def update_quote_state(self, quote_id: str, new_state: QuoteState):
        """Update quote state"""
        if quote_id in self.active_quotes:
            self.active_quotes[quote_id].state = new_state
    
    def update_quote_fill(self, quote_id: str, filled_qty: float):
        """Update quote with fill information"""
        if quote_id in self.active_quotes:
            quote = self.active_quotes[quote_id]
            quote.update_fill(filled_qty)
            
            if quote.state == QuoteState.FILLED:
                self.remove_quote(quote_id, "filled")
    
    def check_refresh_needed(self, current_ofi: float, current_di: float) -> Tuple[bool, str]:
        """
        Check if quotes need refresh due to age or signal changes
        
        Returns:
            Tuple of (needs_refresh, reason)
        """
        # Check for aged out quotes
        aged_quotes = [
            quote_id for quote_id, quote in self.active_quotes.items()
            if quote.is_aged_out and quote.state == QuoteState.ACTIVE
        ]
        
        if aged_quotes:
            return True, f"aged_out ({len(aged_quotes)} quotes)"
        
        # Check for signal direction changes
        current_ofi_sign = self._get_signal_sign(current_ofi)
        current_di_sign = self._get_signal_sign(current_di)
        
        ofi_changed = (current_ofi_sign != self.last_ofi_sign and 
                      self.last_ofi_sign != 0 and current_ofi_sign != 0)
        di_changed = (current_di_sign != self.last_di_sign and 
                     self.last_di_sign != 0 and current_di_sign != 0)
        
        if ofi_changed or di_changed:
            changes = []
            if ofi_changed:
                changes.append(f"OFI: {self.last_ofi_sign:+d}→{current_ofi_sign:+d}")
            if di_changed:
                changes.append(f"DI: {self.last_di_sign:+d}→{current_di_sign:+d}")
            
            return True, f"signal_change ({', '.join(changes)})"
        
        return False, ""
    
    async def refresh_quotes_if_needed(self, current_ofi: float, current_di: float,
                                     refresh_callback) -> bool:
        """
        Refresh quotes if needed, using provided callback
        
        Args:
            current_ofi: Current OFI signal
            current_di: Current DI signal  
            refresh_callback: Async function to call for quote refresh
            
        Returns:
            True if refresh was performed
        """
        async with self._refresh_lock:
            needs_refresh, reason = self.check_refresh_needed(current_ofi, current_di)
            
            if not needs_refresh:
                return False
            
            self.logger.info(f"🔄 Refreshing quotes: {reason}")
            
            # Get quotes to cancel
            quotes_to_cancel = list(self.active_quotes.keys())
            
            # Cancel existing quotes
            for quote_id in quotes_to_cancel:
                self.remove_quote(quote_id, reason.split()[0])  # Extract main reason
            
            # Update signal state
            self.last_ofi_sign = self._get_signal_sign(current_ofi)
            self.last_di_sign = self._get_signal_sign(current_di)
            
            # Call refresh callback to place new quotes
            try:
                await refresh_callback()
                return True
            except Exception as e:
                self.logger.error(f"❌ Quote refresh callback failed: {e}")
                return False
    
    def get_active_quotes_by_side(self, side: str) -> Dict[str, ManagedQuote]:
        """Get all active quotes for a specific side"""
        return {
            quote_id: quote for quote_id, quote in self.active_quotes.items()
            if quote.side == side and quote.state == QuoteState.ACTIVE
        }
    
    def get_quote_ages(self) -> Dict[str, float]:
        """Get ages of all active quotes in milliseconds"""
        return {
            quote_id: quote.age_ms 
            for quote_id, quote in self.active_quotes.items()
        }
    
    def get_management_stats(self) -> Dict:
        """Get quote management statistics"""
        active_ages = [quote.age_ms for quote in self.active_quotes.values()]
        
        return {
            'symbol': self.symbol,
            'active_quotes': len(self.active_quotes),
            'active_bids': len(self.quotes_by_side['bid']),
            'active_asks': len(self.quotes_by_side['ask']),
            'total_managed': self.total_quotes_managed,
            'aged_out': self.quotes_aged_out,
            'signal_refreshed': self.quotes_signal_refreshed,
            'filled': self.quotes_filled,
            'cancelled': self.quotes_cancelled,
            'avg_age_ms': sum(active_ages) / len(active_ages) if active_ages else 0,
            'max_age_ms': max(active_ages) if active_ages else 0,
            'last_ofi_sign': self.last_ofi_sign,
            'last_di_sign': self.last_di_sign,
            'max_age_limit_ms': self.max_age_ms,
        }
    
    def _get_signal_sign(self, signal_value: float) -> int:
        """Convert signal value to sign (-1, 0, +1)"""
        if signal_value > 0.1:  # Threshold to avoid noise
            return 1
        elif signal_value < -0.1:
            return -1
        else:
            return 0
    
    def cleanup_stale_quotes(self):
        """Clean up quotes that are in inconsistent states"""
        stale_quotes = []
        
        for quote_id, quote in self.active_quotes.items():
            # Remove quotes that have been pending too long
            if quote.state == QuoteState.PENDING and quote.age_ms > 5000:  # 5 seconds
                stale_quotes.append((quote_id, "stale_pending"))
            
            # Remove quotes that should be filled but aren't marked as such
            elif quote.remaining_size <= 1e-8 and quote.state != QuoteState.FILLED:
                stale_quotes.append((quote_id, "stale_filled"))
        
        for quote_id, reason in stale_quotes:
            self.remove_quote(quote_id, reason)
        
        if stale_quotes:
            self.logger.info(f"🧹 Cleaned up {len(stale_quotes)} stale quotes")


def test_quote_manager():
    """Test function for quote manager"""
    print("🧪 Testing Quote Manager...")
    
    manager = QuoteManager('BTCUSDT')
    
    # Add some test quotes
    quote1 = manager.add_quote("q1", "bid", 50000.0, 0.01, ofi=0.5, di=-0.2)
    quote2 = manager.add_quote("q2", "ask", 50010.0, 0.01, ofi=0.5, di=-0.2)
    
    print(f"\n📋 Added quotes:")
    stats = manager.get_management_stats()
    print(f"Active quotes: {stats['active_quotes']} (bids: {stats['active_bids']}, asks: {stats['active_asks']})")
    
    # Test age checking
    time.sleep(0.1)  # Wait a bit
    ages = manager.get_quote_ages()
    print(f"\nQuote ages: {[(qid, f'{age:.1f}ms') for qid, age in ages.items()]}")
    
    # Test signal change detection
    needs_refresh, reason = manager.check_refresh_needed(current_ofi=-0.5, current_di=0.3)
    print(f"\nSignal change check: needs_refresh={needs_refresh}, reason='{reason}'")
    
    # Test fill update
    manager.update_quote_fill("q1", 0.005)  # Partial fill
    manager.update_quote_fill("q2", 0.01)   # Full fill
    
    final_stats = manager.get_management_stats()
    print(f"\nFinal stats:")
    print(f"  Active: {final_stats['active_quotes']}")
    print(f"  Filled: {final_stats['filled']}")
    print(f"  Total managed: {final_stats['total_managed']}")
    
    print("✅ Quote Manager test completed")


if __name__ == "__main__":
    test_quote_manager()
