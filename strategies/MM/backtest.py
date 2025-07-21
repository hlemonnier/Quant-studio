"""
Backtesting MM - Étape 5 de la roadmap MM V1

Objectif: Rejouer une journée, log PnL, Δinventory
Livrable: notebook Jupyter compatible, métriques de performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from .config import mm_config
from .local_book import LocalBook
from .avellaneda_stoikov import AvellanedaStoikovQuoter
from .inventory_control import InventoryController

class MMBacktester:
    """Backtesteur pour stratégie Market Making avec latence 0"""
    
    def __init__(self, symbol: str, start_date: str, end_date: str = None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date or start_date
        
        self.logger = logging.getLogger(f"MMBacktest-{symbol}")
        
        # Composants de la stratégie
        self.book = LocalBook(symbol)
        self.quoter = AvellanedaStoikovQuoter(symbol)
        self.inventory = InventoryController(symbol)
        
        # Données de backtesting
        self.market_data = []
        self.trades = []
        self.quotes_history = []
        self.pnl_history = []
        
        # Métriques de performance
        self.total_pnl = 0.0
        self.total_volume = 0.0
        self.total_trades = 0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        # Paramètres de simulation
        self.commission_rate = mm_config.backtest_commission
        self.latency_ms = mm_config.backtest_latency_ms
        
    def load_market_data(self, data_path: str = None) -> bool:
        """Charge les données de marché pour le backtest"""
        try:
            if data_path:
                data_file = Path(data_path)
            else:
                # Chercher les fichiers parquet pour la date
                data_dir = Path(mm_config.data_dir)
                pattern = f"depth_data_{self.start_date.replace('-', '')}*.parquet"
                files = list(data_dir.glob(pattern))
                
                if not files:
                    self.logger.error(f"❌ Aucune donnée trouvée pour {self.start_date}")
                    return False
                
                data_file = files[0]  # Prendre le premier fichier trouvé
            
            self.logger.info(f"📂 Chargement des données: {data_file}")
            df = pd.read_parquet(data_file)
            
            # Filtrer par symbole
            df = df[df['symbol'] == self.symbol].copy()
            
            if len(df) == 0:
                self.logger.error(f"❌ Aucune donnée pour {self.symbol}")
                return False
            
            # Trier par timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.market_data = df.to_dict('records')
            
            self.logger.info(f"✅ {len(self.market_data)} observations chargées")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur chargement données: {e}")
            return False
    
    def simulate_order_execution(self, quote_data: Dict, market_data: Dict) -> List[Dict]:
        """
        Simule l'exécution des ordres avec latence 0
        
        Returns:
            Liste des trades exécutés
        """
        executed_trades = []
        
        # Extraire les données du book
        market_bids = market_data.get('bids', [])
        market_asks = market_data.get('asks', [])
        
        if not market_bids or not market_asks:
            return executed_trades
        
        # Prix du marché
        best_market_bid = market_bids[0][0]
        best_market_ask = market_asks[0][0]
        
        # Nos quotes
        our_bid = quote_data['bid_price']
        our_ask = quote_data['ask_price']
        quote_size = quote_data.get('quote_size', mm_config.base_quote_size)
        
        # Vérifier exécution côté bid (quelqu'un vend au marché)
        if our_bid >= best_market_ask:
            # Notre bid est frappé
            executed_trades.append({
                'side': 'buy',
                'price': our_bid,
                'quantity': quote_size,
                'timestamp': market_data['timestamp'],
                'type': 'market_sell_hit_our_bid'
            })
        
        # Vérifier exécution côté ask (quelqu'un achète au marché)
        if our_ask <= best_market_bid:
            # Notre ask est frappé
            executed_trades.append({
                'side': 'sell',
                'price': our_ask,
                'quantity': quote_size,
                'timestamp': market_data['timestamp'],
                'type': 'market_buy_hit_our_ask'
            })
        
        return executed_trades
    
    def calculate_pnl(self, trade: Dict) -> float:
        """Calcule le PnL d'un trade"""
        quantity = trade['quantity']
        price = trade['price']
        side = trade['side']
        
        # PnL brut
        if side == 'buy':
            gross_pnl = 0  # Pas de PnL immédiat à l'achat
        else:  # sell
            gross_pnl = 0  # Sera calculé lors de la valorisation
        
        # Commission
        commission = quantity * price * self.commission_rate
        
        return gross_pnl - commission
    
    def run_backtest(self) -> Dict:
        """Exécute le backtest complet"""
        if not self.market_data:
            self.logger.error("❌ Pas de données de marché chargées")
            return {}
        
        self.logger.info(f"🚀 Démarrage backtest {self.symbol} - {len(self.market_data)} points")
        
        # Réinitialiser les métriques
        self.trades.clear()
        self.quotes_history.clear()
        self.pnl_history.clear()
        self.total_pnl = 0.0
        
        start_time = datetime.now()
        
        for i, market_tick in enumerate(self.market_data):
            try:
                # Mettre à jour le book local (simulation)
                mid_price = market_tick.get('mid_price')
                if not mid_price:
                    continue
                
                # Mettre à jour la volatilité
                self.quoter.update_volatility(mid_price)
                
                # Calculer les quotes optimaux
                current_inventory = self.inventory.current_inventory
                quotes = self.quoter.compute_quotes(mid_price, current_inventory)
                
                # Appliquer le skew d'inventaire
                skewed_quotes = self.inventory.apply_skew_to_quotes(quotes)
                
                # Calculer la taille des ordres
                bid_size = self.inventory.calculate_optimal_size('bid', skewed_quotes['bid_price'])
                ask_size = self.inventory.calculate_optimal_size('ask', skewed_quotes['ask_price'])
                
                # Ajouter les tailles aux quotes
                skewed_quotes['bid_size'] = bid_size
                skewed_quotes['ask_size'] = ask_size
                skewed_quotes['quote_size'] = (bid_size + ask_size) / 2
                
                # Valider les quotes
                if not self.quoter.validate_quotes(skewed_quotes, mid_price):
                    continue
                
                # Vérifier si le trading doit être suspendu
                should_pause, reason = self.inventory.should_pause_trading()
                if should_pause:
                    self.logger.warning(f"⏸️  Trading suspendu: {reason}")
                    continue
                
                # Simuler l'exécution des ordres
                executed_trades = self.simulate_order_execution(skewed_quotes, market_tick)
                
                # Traiter les trades exécutés
                for trade in executed_trades:
                    # Calculer le PnL
                    trade_pnl = self.calculate_pnl(trade)
                    trade['pnl'] = trade_pnl
                    
                    # Mettre à jour l'inventaire
                    trade_qty = trade['quantity'] if trade['side'] == 'buy' else -trade['quantity']
                    self.inventory.update_inventory(trade_qty, trade['price'])
                    
                    # Ajouter à l'historique
                    self.trades.append(trade)
                    self.total_pnl += trade_pnl
                
                # Enregistrer les quotes et PnL
                quote_record = skewed_quotes.copy()
                quote_record['timestamp'] = market_tick['timestamp']
                quote_record['unrealized_pnl'] = self._calculate_unrealized_pnl(mid_price)
                quote_record['total_pnl'] = self.total_pnl + quote_record['unrealized_pnl']
                
                self.quotes_history.append(quote_record)
                self.pnl_history.append({
                    'timestamp': market_tick['timestamp'],
                    'realized_pnl': self.total_pnl,
                    'unrealized_pnl': quote_record['unrealized_pnl'],
                    'total_pnl': quote_record['total_pnl'],
                    'inventory': current_inventory
                })
                
                # Progress log
                if i % 1000 == 0 and i > 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    progress = i / len(self.market_data) * 100
                    self.logger.info(f"📊 Progress: {progress:.1f}% | PnL: ${self.total_pnl:.2f} | Trades: {len(self.trades)} | {elapsed:.1f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Erreur tick {i}: {e}")
                continue
        
        # Calculer les métriques finales
        end_time = datetime.now()
        backtest_duration = (end_time - start_time).total_seconds()
        
        results = self._calculate_performance_metrics()
        results['backtest_duration_seconds'] = backtest_duration
        
        self.logger.info(f"✅ Backtest terminé en {backtest_duration:.1f}s")
        self._print_results(results)
        
        return results
    
    def _calculate_unrealized_pnl(self, current_mid_price: float) -> float:
        """Calcule le PnL non réalisé basé sur l'inventaire et le prix actuel"""
        # PnL non réalisé = inventaire * prix_actuel (simplifié)
        # Dans la réalité, il faudrait tracker le prix moyen d'achat
        return self.inventory.current_inventory * current_mid_price
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calcule les métriques de performance du backtest"""
        if not self.pnl_history:
            return {}
        
        pnl_df = pd.DataFrame(self.pnl_history)
        pnl_series = pnl_df['total_pnl']
        
        # Métriques de base
        final_pnl = pnl_series.iloc[-1] if len(pnl_series) > 0 else 0
        max_pnl = pnl_series.max()
        min_pnl = pnl_series.min()
        
        # Drawdown
        running_max = pnl_series.expanding().max()
        drawdown = pnl_series - running_max
        max_drawdown = drawdown.min()
        
        # Volatilité et Sharpe
        pnl_returns = pnl_series.pct_change().dropna()
        volatility = pnl_returns.std() * np.sqrt(86400)  # Annualisé (points par jour)
        sharpe_ratio = (pnl_returns.mean() / pnl_returns.std()) * np.sqrt(86400) if pnl_returns.std() > 0 else 0
        
        # Métriques de trading
        total_trades = len(self.trades)
        if total_trades > 0:
            avg_trade_pnl = final_pnl / total_trades
            win_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            win_rate = len(win_trades) / total_trades
        else:
            avg_trade_pnl = 0
            win_rate = 0
        
        return {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'final_pnl': final_pnl,
            'max_pnl': max_pnl,
            'min_pnl': min_pnl,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': total_trades,
            'avg_trade_pnl': avg_trade_pnl,
            'win_rate': win_rate,
            'final_inventory': self.inventory.current_inventory,
            'inventory_std': pnl_df['inventory'].std(),
            'data_points': len(self.market_data),
        }
    
    def _print_results(self, results: Dict):
        """Affiche les résultats du backtest"""
        print(f"\n📊 Résultats Backtest {results['symbol']}")
        print("=" * 50)
        print(f"Période: {results['start_date']}")
        print(f"Points de données: {results['data_points']:,}")
        print(f"Durée: {results.get('backtest_duration_seconds', 0):.1f}s")
        print()
        print("💰 Performance:")
        print(f"  PnL Final: ${results['final_pnl']:+.2f}")
        print(f"  PnL Max: ${results['max_pnl']:+.2f}")
        print(f"  PnL Min: ${results['min_pnl']:+.2f}")
        print(f"  Max Drawdown: ${results['max_drawdown']:+.2f}")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print()
        print("📈 Trading:")
        print(f"  Nombre de trades: {results['total_trades']}")
        print(f"  PnL moyen/trade: ${results['avg_trade_pnl']:+.4f}")
        print(f"  Taux de réussite: {results['win_rate']:.1%}")
        print()
        print("📦 Inventaire:")
        print(f"  Inventaire final: {results['final_inventory']:+.4f}")
        print(f"  Écart-type inventaire: {results['inventory_std']:.4f}")
        print("=" * 50)
    
    def save_results(self, output_path: str = None):
        """Sauvegarde les résultats du backtest"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"backtest_results_{self.symbol}_{timestamp}.json"
        
        results = {
            'metadata': {
                'symbol': self.symbol,
                'start_date': self.start_date,
                'backtest_time': datetime.now().isoformat(),
                'config': {
                    'gamma': self.quoter.gamma,
                    'sigma': self.quoter.sigma,
                    'max_inventory': self.inventory.max_inventory,
                    'commission_rate': self.commission_rate,
                }
            },
            'performance': self._calculate_performance_metrics(),
            'trades': self.trades,
            'pnl_history': self.pnl_history[-1000:],  # Derniers 1000 points
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Résultats sauvegardés: {output_path}")

# Fonction utilitaire pour test rapide
def quick_backtest_demo():
    """Demo rapide du backtesting"""
    print("🧪 Demo Backtest MM...")
    
    # Créer des données de test synthétiques
    np.random.seed(42)
    n_points = 1000
    base_price = 50000
    
    synthetic_data = []
    for i in range(n_points):
        price = base_price + np.cumsum(np.random.normal(0, 10))[0]
        synthetic_data.append({
            'timestamp': datetime.now() + timedelta(seconds=i*0.1),
            'symbol': 'BTCUSDT',
            'mid_price': price,
            'bids': [[price - 5, 1.0]],
            'asks': [[price + 5, 1.0]],
            'spread': 10,
            'spread_bps': 20,
        })
    
    # Créer le backtester
    backtester = MMBacktester('BTCUSDT', '2024-01-01')
    backtester.market_data = synthetic_data
    
    # Lancer le backtest
    results = backtester.run_backtest()
    
    return results

if __name__ == "__main__":
    quick_backtest_demo() 