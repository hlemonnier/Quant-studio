import asyncio
import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

from .config import mm_config
from .ws_data_capture import BinanceWSCapture, test_connection
from .local_book import LocalBook, MultiBookManager
from .avellaneda_stoikov import AvellanedaStoikovQuoter, quick_quote_test
from .inventory_control import InventoryController, test_inventory_control
from .backtest import MMBacktester, quick_backtest_demo

class MMOrchestrator:
    """Orchestrateur principal du Market Making"""
    
    def __init__(self):
        self.is_running = False
        self.setup_logging()
        
        # Composants principaux
        self.book_manager = None
        self.ws_capture = None
        self.quoters = {}
        self.inventory_controllers = {}
        
    def setup_logging(self):
        """Configure le logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'mm_strategy_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
        self.logger = logging.getLogger("MMOrchestrator")
    
    def initialize_components(self):
        """Initialise tous les composants"""
        self.logger.info("🔧 Initialisation des composants MM...")
        
        # Vérifier la configuration (mode observation = pas besoin d'API keys)
        if not mm_config.validate_config(require_api_keys=False):
            self.logger.error("❌ Configuration invalide")
            return False
        
        mm_config.print_config()
        
        # Initialiser les order books
        self.book_manager = MultiBookManager(mm_config.symbols)
        
        # Initialiser les quoters pour chaque symbole
        for symbol in mm_config.symbols:
            self.quoters[symbol] = AvellanedaStoikovQuoter(symbol)
            self.inventory_controllers[symbol] = InventoryController(symbol)
        
        # Initialiser la capture WebSocket avec callback
        self.ws_capture = BinanceWSCapture(
            mm_config.symbols,
            on_data_callback=self.on_market_data
        )
        
        self.logger.info("✅ Composants initialisés")
        return True
    
    def on_market_data(self, market_data: dict):
        """Callback appelé à chaque update de marché"""
        try:
            symbol = market_data.get('symbol')
            if not symbol or symbol not in mm_config.symbols:
                return
            
            # ✅ NOUVEAU: Mettre à jour le book local avec les données WebSocket
            if self.book_manager and symbol in self.book_manager.books:
                local_book = self.book_manager.books[symbol]
                
                # Pour les données depth20 (snapshots complets), on met à jour directement le book
                bids_data = market_data.get('bids', [])
                asks_data = market_data.get('asks', [])
                
                if bids_data and asks_data:
                    # Mise à jour directe du book avec le snapshot depth20
                    local_book.bids = {float(bid[0]): float(bid[1]) for bid in bids_data}
                    local_book.asks = {float(ask[0]): float(ask[1]) for ask in asks_data}
                    local_book.last_update_id = market_data.get('last_update_id', local_book.last_update_id + 1)
                    local_book.is_synchronized = True
                    local_book.update_count += 1
                    
                    # Log de synchronisation réussie
                    if local_book.update_count % 100 == 0:
                        self.logger.debug(f"📖 {symbol} book mis à jour: {local_book.update_count} updates")
            
            # Utiliser les données du book local synchronisé au lieu des données brutes
            if self.book_manager and symbol in self.book_manager.books:
                local_book = self.book_manager.books[symbol]
                if local_book.is_synchronized:
                    # Utiliser les données du book local
                    mid_price = local_book.get_mid_price()
                    spread_bps = local_book.get_spread_bps()
                    imbalance = local_book.calculate_imbalance(5)
                    
                    # Log périodique des stats du book (toutes les 30 secondes)
                    if hasattr(self, '_last_book_log'):
                        if (datetime.now() - self._last_book_log).total_seconds() > 30:
                            self._log_book_stats(symbol, local_book)
                            self._last_book_log = datetime.now()
                    else:
                        self._last_book_log = datetime.now()
                else:
                    # Fallback sur les données brutes si le book n'est pas sync
                    mid_price = market_data.get('mid_price')
            else:
                # Fallback sur les données brutes
                mid_price = market_data.get('mid_price')
            
            # Mettre à jour la volatilité
            if mid_price and symbol in self.quoters:
                self.quoters[symbol].update_volatility(mid_price)
            
            # Calculer et logger les quotes optimaux
            if symbol in self.quoters and symbol in self.inventory_controllers:
                inventory = self.inventory_controllers[symbol].current_inventory
                quotes = self.quoters[symbol].compute_quotes(mid_price, inventory)
                skewed_quotes = self.inventory_controllers[symbol].apply_skew_to_quotes(quotes)
                
                # Log périodique des quotes (toutes les 10 secondes)
                if hasattr(self, '_last_quote_log'):
                    if (datetime.now() - self._last_quote_log).total_seconds() > 10:
                        self._log_quotes(symbol, skewed_quotes)
                        self._last_quote_log = datetime.now()
                else:
                    self._last_quote_log = datetime.now()
        
        except Exception as e:
            self.logger.error(f"❌ Erreur traitement market data: {e}")
    
    def _log_quotes(self, symbol: str, quotes: dict):
        """Log les quotes actuels"""
        bid = quotes['bid_price']
        ask = quotes['ask_price']
        spread_bps = quotes['spread_bps']
        inventory = quotes.get('current_inventory', 0)
        skew = quotes.get('inventory_skew', 0)
        
        self.logger.info(
            f"📊 {symbol}: Bid=${bid:.2f} Ask=${ask:.2f} "
            f"Spread={spread_bps:.1f}bps Inv={inventory:+.4f} Skew={skew:+.3f}"
        )
    
    def _log_book_stats(self, symbol: str, book):
        """Log les statistiques du book local"""
        stats = book.get_book_stats()
        self.logger.info(
            f"📖 {symbol} Book: Mid=${stats['mid_price']:.2f} "
            f"Spread={stats['spread_bps']:.1f}bps "
            f"Imbalance={stats['imbalance_5']:+.3f} "
            f"Updates={stats['update_count']} "
            f"Levels={stats['bid_levels']}/{stats['ask_levels']}"
        )
    
    async def run_live_strategy(self):
        """Lance la stratégie live"""
        self.logger.info("🚀 Démarrage stratégie MM live...")
        
        if not self.initialize_components():
            return False
        
        # Synchroniser les order books avec snapshots REST initiaux
        self.logger.info("📖 Synchronisation initiale des order books...")
        sync_results = self.book_manager.sync_all_books()
        for symbol, success in sync_results.items():
            if success:
                self.logger.info(f"✅ {symbol} book synchronisé (snapshot REST initial)")
            else:
                self.logger.error(f"❌ {symbol} book sync failed - continuera avec les données WebSocket uniquement")
        
        # Démarrer la capture WebSocket
        self.is_running = True
        
        try:
            await self.ws_capture.start_capture()
        except KeyboardInterrupt:
            self.logger.info("🛑 Arrêt demandé par l'utilisateur")
        finally:
            self.stop_strategy()
    
    def stop_strategy(self):
        """Arrête la stratégie"""
        self.logger.info("🛑 Arrêt de la stratégie MM...")
        self.is_running = False
        
        if self.ws_capture:
            self.ws_capture.stop_capture()
        
        # Afficher les stats finales
        for symbol in mm_config.symbols:
            if symbol in self.inventory_controllers:
                self.inventory_controllers[symbol].print_inventory_summary()
    
    def run_backtest(self, symbol: str, date: str, data_path: str = None):
        """Lance un backtest"""
        self.logger.info(f"📊 Backtest {symbol} pour {date}")
        
        backtester = MMBacktester(symbol, date)
        
        if data_path:
            success = backtester.load_market_data(data_path)
        else:
            success = backtester.load_market_data()
        
        if not success:
            self.logger.error("❌ Impossible de charger les données")
            return None
        
        results = backtester.run_backtest()
        backtester.save_results()
        
        return results

def setup_signal_handlers(orchestrator):
    """Configure les gestionnaires de signaux"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Signal {signum} reçu, arrêt en cours...")
        orchestrator.stop_strategy()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Market Making V1 - Enigma Quant")
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande test
    test_parser = subparsers.add_parser('test', help='Tests des composants')
    test_parser.add_argument('--component', choices=['ws', 'book', 'quotes', 'inventory', 'backtest'], 
                           default='all', help='Composant à tester')
    
    # Commande live
    live_parser = subparsers.add_parser('live', help='Lancer la stratégie live')
    
    # Commande backtest
    backtest_parser = subparsers.add_parser('backtest', help='Lancer un backtest')
    backtest_parser.add_argument('--symbol', default='BTCUSDT', help='Symbole à backtester')
    backtest_parser.add_argument('--date', default='2024-01-01', help='Date du backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--data', help='Chemin vers les données (optionnel)')
    
    # Commande config
    config_parser = subparsers.add_parser('config', help='Afficher la configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    orchestrator = MMOrchestrator()
    
    if args.command == 'config':
        mm_config.print_config()
        print(f"\nValidation observation: {'✅ OK' if mm_config.validate_config(require_api_keys=False) else '❌ Erreurs'}")
        print(f"Validation trading: {'✅ OK' if mm_config.validate_config(require_api_keys=True) else '❌ Erreurs'}")
    
    elif args.command == 'test':
        print("🧪 Tests des composants MM V1...")
        
        if args.component in ['all', 'ws']:
            print("\n1️⃣ Test WebSocket:")
            asyncio.run(test_connection())
        
        if args.component in ['all', 'book']:
            print("\n2️⃣ Test LocalBook:")
            book = LocalBook('BTCUSDT')
            if book.get_snapshot():
                book.print_book_summary(5)
        
        if args.component in ['all', 'quotes']:
            print("\n3️⃣ Test Avellaneda-Stoikov:")
            quick_quote_test()
        
        if args.component in ['all', 'inventory']:
            print("\n4️⃣ Test Inventory Control:")
            test_inventory_control()
        
        if args.component in ['all', 'backtest']:
            print("\n5️⃣ Test Backtesting:")
            quick_backtest_demo()
    
    elif args.command == 'live':
        setup_signal_handlers(orchestrator)
        asyncio.run(orchestrator.run_live_strategy())
    
    elif args.command == 'backtest':
        results = orchestrator.run_backtest(args.symbol, args.date, args.data)
        if results:
            print(f"\n✅ Backtest terminé - PnL: ${results['final_pnl']:+.2f}")

if __name__ == "__main__":
    main() 