"""
Configuration pour le Market Making V1
"""

import os
import json
import pathlib
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
        
        # Real-time constraints (§3.2 V1-α)
        self.max_mid_price_latency_ms = 50  # ≤50ms pour mid-price
        self.max_quote_latency_ms = 300     # ≤300ms pour quotes (kill-switch)
        
        # Avellaneda-Stoikov Parameters
        self.gamma = 0.1  # Risk aversion - À définir avec le boss
        # Volatility estimate (initial). 5 % is more realistic for intra-day crypto
        # and avoids systematic “volatility spike” pauses.
        self.sigma = 0.05
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
        self.max_spread_bps = 500  # Spread maximum en basis points (increased for V1.5)
        self.min_spread_bps = 5  # Spread minimum en basis points (plus réaliste)
        # Stop-loss global (en % du capital) – plus large pour laisser respirer le MM
        self.stop_loss_pct = 50.0
        # Limite de perte quotidienne spécifique au market making
        self.daily_loss_limit_pct = 20.0  # Arrêt de la journée si pertes > 20 %

        # Volatility guardrail
        # Trading pauses if realised volatility > max_volatility_threshold
        # Crypto can easily reach 10 % intraday, so we allow up to 15 %.
        self.max_volatility_threshold = 0.15  # 15 %
        
        # ------------------------------------------------------------------
        # Order-Flow Imbalance (OFI) parameters  (§3.3bis)
        # ------------------------------------------------------------------
        # Coefficient β : nb de ticks de déplacement du centre par unité d’OFI
        self.beta_ofi = 0.30
        # Fenêtre (s) pour le calcul de l’OFI
        self.ofi_window_seconds = 5.0  # Increased from 1.0s to capture more fills in crypto
        # Clamp du z-score de l’OFI pour éviter les outliers
        self.ofi_clamp_std = 3.0
        # Tick-size par défaut (fallback si lookup symbole indisponible)
        self.default_tick_size = 0.01

        # ------------------------------------------------------------------
        # V1.5 Enhanced Parameters (§4.6)
        # ------------------------------------------------------------------
        # Depth Imbalance (DI) coefficient: DI → ticks
        self.beta_di = 0.2  # tick/unit (initial value from spec)
        
        # Inventory penalty coefficients
        self.kappa_inv = 0.01  # tick/lot (for both centre and spread) - reduced to avoid excessive impact
        
        # Volatility sensitivity for dynamic spread (reduced from 1.2 to avoid excessive spreads)
        self.kappa_vol = 0.2  # multiplier for volatility component (scaled for base spread)
        
        # Quote ageing parameters
        self.quote_ageing_ms = 750  # 750ms timeout as per spec
        
        # Enhanced risk controls (§4.7)
        self.max_offset_ticks = 3  # Global offset clamp ≤ 3 ticks
        self.min_spread_ticks = 2  # Spread floor ≥ 2 ticks
        
        # V1.5 Performance targets (§4.8)
        self.target_pnl_improvement_pct = 30.0    # +30% PnL/trade vs V1-α
        self.target_spread_capture_v15_pct = 78.0  # ≥78% spread capture (+8pts)
        self.target_rms_inventory_v15_ratio = 0.35  # ≤0.35 q_max RMS inventory
        self.target_hit_ratio_ofi_di_pct = 60.0    # ≥60% hit ratio OFI+DI
        
        # Version control
        self.strategy_version = "V1-α"  # Default version, can be overridden

        # Data Storage
        self.data_dir = 'data/mm_data'
        self.parquet_compression = 'snappy'
        self.save_interval_seconds = 30  # Sauvegarde toutes les 30 secondes
        
        # Performance targets (§3.7 V1-α)
        self.target_spread_capture_pct = 70.0    # ≥70% spread capture
        self.target_rms_inventory_ratio = 0.4    # ≤0.4 q_max RMS inventory
        self.target_fill_ratio_pct = 5.0         # ≥5% fill ratio
        self.target_cancel_ratio_pct = 70.0      # ≤70% cancel ratio
        self.target_latency_p99_ms = 300.0       # ≤300ms P99 latency
        
        # Backtesting
        self.backtest_latency_ms = 0  # Latence 0 pour la V1
        self.backtest_commission = 0.001  # Commission 0.1%

        # --------------------------------------------------------------
        # Source de données (WebSocket vs DB vs CSV) pour backtest/replay
        # --------------------------------------------------------------
        # Contrôlé via variables d'environnement (voir .env.sample)
        self.data_source = os.getenv('MM_DATA_SOURCE', 'websocket').lower()

        # Paramètres DB Replay (schéma top-of-book par défaut)
        self.db_uri = os.getenv('MM_DB_URI', '')
        self.db_table = os.getenv('MM_DB_TABLE', 'depth_messages')
        # Format des données en DB: 'json' (payload WS brut) ou 'top_of_book'
        self.db_format = os.getenv('MM_DB_FORMAT', 'top_of_book').lower()
        # Colonnes DB pour le format top_of_book
        self.db_symbol_col = os.getenv('MM_DB_SYMBOL_COL', 'symbol')
        self.db_time_col = os.getenv('MM_DB_TIME_COL', 'event_time')
        self.db_bid_col = os.getenv('MM_DB_BID_COL', 'best_bid')
        self.db_ask_col = os.getenv('MM_DB_ASK_COL', 'best_ask')
        # Colonnes quantités (optionnelles)
        self.db_bid_qty_col = os.getenv('MM_DB_BID_QTY_COL', '')
        self.db_ask_qty_col = os.getenv('MM_DB_ASK_QTY_COL', '')
        self.db_qty_col = os.getenv('MM_DB_QTY_COL', '')  # fallback si une seule col
        # Filtres optionnels
        self.db_feed_col = os.getenv('MM_DB_FEED_COL', '')
        self.db_feed_value = os.getenv('MM_DB_FEED', '')
        self.db_exchange_col = os.getenv('MM_DB_EXCHANGE_COL', '')
        self.db_exchange_value = os.getenv('MM_DB_EXCHANGE', '')
        # Fenêtre temporelle (epoch ms)
        self.db_start_ts = self._get_int_env('MM_DB_START_TS_MS')
        self.db_end_ts = self._get_int_env('MM_DB_END_TS_MS')
        # Facteur d'accélération du replay (1.0 = temps réel)
        self.replay_speed = float(os.getenv('MM_REPLAY_SPEED', '1.0'))

        # Paramètres CSV Replay (schéma top-of-book)
        self.csv_path = os.getenv('MM_CSV_PATH', '')
        self.csv_time_col = os.getenv('MM_CSV_TIME_COL', 'timestamp')
        self.csv_symbol_col = os.getenv('MM_CSV_SYMBOL_COL', 'symbol')
        self.csv_bid_col = os.getenv('MM_CSV_BID_COL', 'best_bid')
        self.csv_ask_col = os.getenv('MM_CSV_ASK_COL', 'best_ask')
        self.csv_bid_qty_col = os.getenv('MM_CSV_BID_QTY_COL', '')
        self.csv_ask_qty_col = os.getenv('MM_CSV_ASK_QTY_COL', '')
        self.csv_qty_col = os.getenv('MM_CSV_QTY_COL', '')
        self.csv_feed_col = os.getenv('MM_CSV_FEED_COL', '')
        self.csv_feed_value = os.getenv('MM_CSV_FEED', '')
        self.csv_exchange_col = os.getenv('MM_CSV_EXCHANGE_COL', '')
        self.csv_exchange_value = os.getenv('MM_CSV_EXCHANGE', '')
        # Durée du backtest (secondes) quand on utilise le moteur temps réel en replay
        self.backtest_duration_seconds = self._get_int_env('MM_BACKTEST_DURATION_SECONDS') or 60

        # --------------------------------------------------------------
        # Chargement éventuel des paramètres calibrés par symbole
        # --------------------------------------------------------------
        self.symbol_params: Dict[str, Dict[str, Any]] = {}
        calib_path = pathlib.Path(__file__).resolve().parents[2] / 'parameters' / 'calibrated_mm.json'
        if calib_path.exists():
            try:
                with open(calib_path, 'r') as f:
                    self.symbol_params = json.load(f)
                print(f"ℹ️  Paramètres calibrés chargés pour {len(self.symbol_params)} symbole(s) depuis {calib_path}")
            except Exception as e:
                print(f"⚠️  Impossible de charger les paramètres calibrés: {e}")
                self.symbol_params = {}

    # ------------------------------------------------------------------
    # Helpers pour récupérer les paramètres spécifiques à un symbole
    # ------------------------------------------------------------------
    def get_symbol_params(self, symbol: str) -> Dict[str, Any]:
        """Retourne un dict avec tous les paramètres pour le symbole (avec fallback défaut)"""
        params = self.symbol_params.get(symbol, {})
        return {
            # Base A&S parameters
            'gamma': params.get('gamma', self.gamma),
            'k': params.get('k', self.k),
            'T': params.get('T', self.T),
            
            # V1.5 parameters
            'beta_di': params.get('beta_di', self.beta_di),
            'kappa_inv': params.get('kappa_inv', self.kappa_inv),
            'kappa_vol': params.get('kappa_vol', self.kappa_vol),
        }
    
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
        print(f"Max volatility allowed: {self.max_volatility_threshold:.2%}")
        print(f"OFI β: {self.beta_ofi} | Window: {self.ofi_window_seconds}s | Clamp: ±{self.ofi_clamp_std}σ")
        print("=" * 40)

        # Affichage source de données
        print(f"Data source: {self.data_source}")
        if self.data_source == 'db':
            print(f"  DB: {self.db_uri} | table: {self.db_table} | format: {self.db_format}")
        elif self.data_source == 'csv':
            print(f"  CSV: {self.csv_path}")

    # -------------------------------
    # Helpers internes
    # -------------------------------
    def _get_int_env(self, key: str):
        val = os.getenv(key, '')
        try:
            return int(val) if val else None
        except Exception:
            return None

# Instance globale de configuration
mm_config = MMConfig()
