"""
Configuration pour le Market Making V1
"""

import os
from typing import List, Dict, Any

class MMConfig:
    """Configuration centralisée pour le market making"""
    
    def __init__(self):
        # API Binance
        self.api_key = os.getenv('BINANCE_API_KEY', '')
        self.api_secret = os.getenv('BINANCE_API_SECRET', '')
        
        # Symboles à trader
        self.symbols: List[str] = ['BTCUSDT', 'ETHUSDT']
        # Symbole principal utilisé si l’application n’en sélectionne qu’un
        self.default_symbol: str = 'BTCUSDT'
        
        # WebSocket Settings
        self.ws_depth_level = 20  # depth20
        self.ws_update_speed = '100ms'  # 100ms
        self.reconnect_attempts = 5
        
        # Avellaneda-Stoikov Parameters
        self.gamma = 0.1  # Risk aversion - À définir avec le boss
        self.sigma = 0.02  # Volatility estimate (initial)
        self.T = 1.0  # Time horizon (1 day normalized)
        self.k = 1.5  # Market impact parameter
        
        # Inventory Control
        self.inventory_target = 0.0  # Target inventory
        self.max_inventory = 1.0  # Maximum inventory (À définir: N_max)
        self.inventory_threshold = 0.5  # Inventory threshold (À définir: N★)
        self.skew_factor = 2.0  # Facteur de skew en fonction de l'inventaire
        
        # Position Sizing
        self.base_quote_size = 0.01  # Taille de base pour les quotes
        self.min_order_size = 0.001  # Taille minimum d'ordre
        self.max_order_size = 0.1  # Taille maximum d'ordre
        
        # Risk Management
        self.max_spread_bps = 200  # Spread maximum en basis points
        self.min_spread_bps = 5  # Spread minimum en basis points (plus réaliste)
        # Stop-loss global (en % du capital) – plus large pour laisser respirer le MM
        self.stop_loss_pct = 50.0
        # Limite de perte quotidienne spécifique au market making
        self.daily_loss_limit_pct = 20.0  # Arrêt de la journée si pertes > 20 %
        
        # Data Storage
        self.data_dir = 'data/mm_data'
        self.parquet_compression = 'snappy'
        self.save_interval_seconds = 30  # Sauvegarde toutes les 30 secondes
        
        # Backtesting
        self.backtest_latency_ms = 0  # Latence 0 pour la V1
        self.backtest_commission = 0.001  # Commission 0.1%
        
    def validate_config(self, require_api_keys: bool = False) -> bool:
        """Valide la configuration
        
        Args:
            require_api_keys: Si True, exige les API keys (pour trading live)
                             Si False, permet le mode observation (données publiques)
        """
        if require_api_keys and (not self.api_key or not self.api_secret):
            print("⚠️  API keys manquantes pour le trading live")
            print("💡 Configurez: export BINANCE_API_KEY='...' && export BINANCE_API_SECRET='...'")
            return False
            
        if not require_api_keys and (not self.api_key or not self.api_secret):
            print("ℹ️  Mode observation: WebSocket sans API keys (données publiques)")
            
        if self.gamma <= 0:
            print("⚠️  Risk aversion (gamma) doit être > 0")
            return False
            
        if self.max_inventory <= self.inventory_threshold:
            print("⚠️  max_inventory doit être > inventory_threshold")
            return False
            
        return True
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Configuration spécifique par symbole"""
        # Configuration par défaut, peut être surchargée par symbole
        return {
            'tick_size': 0.01 if 'USDT' in symbol else 0.00000001,
            'min_notional': 10.0 if 'USDT' in symbol else 0.001,
            'lot_size': 0.00001 if 'BTC' in symbol else 0.001,
        }
    
    def print_config(self):
        """Affiche la configuration actuelle"""
        print("\n🔧 Configuration Market Making V1")
        print("=" * 40)
        print(f"Symboles: {', '.join(self.symbols)}")
        print(f"Risk Aversion (γ): {self.gamma}")
        print(f"Inventory Max: {self.max_inventory}")
        print(f"Inventory Threshold (N★): {self.inventory_threshold}")
        print(f"Spread range: {self.min_spread_bps}-{self.max_spread_bps} bps")
        print(f"Base quote size: {self.base_quote_size}")
        print(f"Default symbol: {self.default_symbol}")
        print(f"Daily loss limit: {self.daily_loss_limit_pct}%")
        print("=" * 40)

# Instance globale de configuration
mm_config = MMConfig() 
