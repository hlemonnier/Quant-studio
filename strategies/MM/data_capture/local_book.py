"""
Local Order Book - Étape 2 de la roadmap MM V1

Objectif: Rejouer snapshot REST + diff stream pour obtenir un L2 propre
Livrable: classe LocalBook (bid/ask arrays + checksum)
"""

import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import logging
from datetime import datetime
import hashlib
from ..utils.config import mm_config

class LocalBook:
    """Order Book local avec synchronisation REST + WebSocket"""
    
    def __init__(self, symbol: str, depth: int = 1000):
        self.symbol = symbol.upper()
        self.depth = depth
        self.bids = {}  # {price: quantity}
        self.asks = {}  # {price: quantity}
        self.last_update_id = 0
        self.first_update_id = 0
        self.is_synchronized = False
        self.last_checksum = None
        
        # Métriques et diagnostics
        self.update_count = 0
        self.last_sync_time = None
        self.sync_errors = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"LocalBook-{symbol}")
        
        # URLs Binance
        self.rest_base_url = "https://api.binance.com"
        
    def get_snapshot(self) -> bool:
        """Récupère un snapshot complet via REST API"""
        try:
            url = f"{self.rest_base_url}/api/v3/depth"
            params = {
                'symbol': self.symbol,
                'limit': self.depth
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # Initialiser le book
            self.bids = {float(price): float(qty) for price, qty in data['bids']}
            self.asks = {float(price): float(qty) for price, qty in data['asks']}
            self.last_update_id = data['lastUpdateId']
            self.first_update_id = data['lastUpdateId']
            
            self.is_synchronized = True
            self.last_sync_time = datetime.now()
            
            self.logger.info(f"✅ Snapshot récupéré: {len(self.bids)} bids, {len(self.asks)} asks, updateId={self.last_update_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur récupération snapshot: {e}")
            self.sync_errors += 1
            return False

    
    def apply_diff(self, diff_data: dict) -> bool:
        """Applique un diff WebSocket au book local"""
        try:
            # Vérifier la séquence
            first_update_id = diff_data.get('U')
            final_update_id = diff_data.get('u')
            
            # Log des premières applications pour diagnostic
            if self.update_count < 10:
                self.logger.info(f"🔄 Applying diff #{self.update_count+1} for {self.symbol}: U={first_update_id}, u={final_update_id}")
                self.logger.info(f"📊 Diff data keys: {list(diff_data.keys())}")
                if 'b' in diff_data:
                    self.logger.info(f"📈 Bids updates: {len(diff_data['b'])} entries")
                    if diff_data['b']:
                        self.logger.info(f"📈 First bid update: {diff_data['b'][0]}")
                if 'a' in diff_data:
                    self.logger.info(f"📉 Asks updates: {len(diff_data['a'])} entries")
                    if diff_data['a']:
                        self.logger.info(f"📉 First ask update: {diff_data['a'][0]}")
            
            if not self.is_synchronized:
                self.logger.warning("⚠️  Book non synchronisé, ignoré diff")
                return False
            
            # Vérifier la continuité des updates
            if first_update_id <= self.last_update_id + 1 <= final_update_id:
                # Update valide
                pass
            elif first_update_id == self.last_update_id + 1:
                # Update suivant direct
                pass
            else:
                self.logger.warning(f"⚠️  Gap détecté: attendu {self.last_update_id + 1}, reçu {first_update_id}-{final_update_id}")
                self.logger.info("🔄 Tentative de resynchronisation automatique...")
                
                # Essayer de resynchroniser automatiquement
                if self.get_snapshot():
                    self.logger.info("✅ Resynchronisation réussie, réessai de l'update")
                    # Réessayer l'update après resynchronisation
                    return self.apply_diff(diff_data)
                else:
                    self.logger.error("❌ Échec de la resynchronisation")
                    self.is_synchronized = False
                    return False
            
            # Appliquer les modifications bids
            for price_str, qty_str in diff_data.get('b', []):
                price = float(price_str)
                qty = float(qty_str)
                
                if qty == 0:
                    # Supprimer le niveau
                    self.bids.pop(price, None)
                else:
                    # Mettre à jour/ajouter le niveau
                    self.bids[price] = qty
            
            # Appliquer les modifications asks
            for price_str, qty_str in diff_data.get('a', []):
                price = float(price_str)
                qty = float(qty_str)
                
                if qty == 0:
                    # Supprimer le niveau
                    self.asks.pop(price, None)
                else:
                    # Mettre à jour/ajouter le niveau
                    self.asks[price] = qty
            
            # Mettre à jour l'ID
            self.last_update_id = final_update_id
            self.update_count += 1
            
            # Nettoyer le book (garder seulement les meilleurs niveaux)
            self._trim_book()
            
            # Log après application pour vérifier les changements
            if self.update_count <= 10:
                best_bid = max(self.bids.keys()) if self.bids else 0
                best_ask = min(self.asks.keys()) if self.asks else 0
                self.logger.info(f"✅ Update #{self.update_count} applied: Best bid={best_bid:.2f}, Best ask={best_ask:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erreur application diff: {e}")
            self.is_synchronized = False
            return False
    
    def _trim_book(self):
        """Limite le book aux meilleurs niveaux pour économiser la mémoire"""
        if len(self.bids) > self.depth:
            # Garder les meilleurs bids (prix les plus élevés)
            sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
            self.bids = dict(sorted_bids[:self.depth])
        
        if len(self.asks) > self.depth:
            # Garder les meilleurs asks (prix les plus bas)
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
            self.asks = dict(sorted_asks[:self.depth])
    
    def get_best_bid_ask(self) -> Tuple[Optional[float], Optional[float]]:
        """Retourne les meilleurs bid et ask"""
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        return best_bid, best_ask
    
    def get_mid_price(self) -> Optional[float]:
        """Calcule le prix moyen"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Calcule le spread en valeur absolue"""
        best_bid, best_ask = self.get_best_bid_ask()
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def get_spread_bps(self) -> Optional[float]:
        """Calcule le spread en basis points"""
        spread = self.get_spread()
        mid_price = self.get_mid_price()
        if spread and mid_price:
            return (spread / mid_price) * 10000
        return None
    
    def get_book_arrays(self, levels: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les arrays numpy des bids/asks triés"""
        # Bids triés par prix décroissant
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
        bid_array = np.array(sorted_bids[:levels])
        
        # Asks triés par prix croissant
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
        ask_array = np.array(sorted_asks[:levels])
        
        return bid_array, ask_array
    
    def calculate_imbalance(self, levels: int = 5) -> float:
        """Calcule l'imbalance du book sur N niveaux"""
        bid_array, ask_array = self.get_book_arrays(levels)
        
        if len(bid_array) == 0 or len(ask_array) == 0:
            return 0.0
        
        bid_volume = np.sum(bid_array[:, 1]) if len(bid_array) > 0 else 0
        ask_volume = np.sum(ask_array[:, 1]) if len(ask_array) > 0 else 0
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume
    
    def get_depth_l1_l5_bid(self) -> float:
        """
        Get aggregated bid depth for L1-L5 levels (V1.5 requirement)
        
        Returns:
            Sum of bid quantities across top 5 levels
        """
        if not self.bids:
            return 0.0
        
        # Get top 5 bid levels (highest prices)
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:5]
        return sum(qty for _, qty in sorted_bids)
    
    def get_depth_l1_l5_ask(self) -> float:
        """
        Get aggregated ask depth for L1-L5 levels (V1.5 requirement)
        
        Returns:
            Sum of ask quantities across top 5 levels
        """
        if not self.asks:
            return 0.0
        
        # Get top 5 ask levels (lowest prices)
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:5]
        return sum(qty for _, qty in sorted_asks)
    
    def get_depth_l1_l5_both(self) -> Tuple[float, float]:
        """
        Get aggregated depth for both bid and ask L1-L5 levels
        
        Returns:
            Tuple of (bid_depth_l1_l5, ask_depth_l1_l5)
        """
        bid_depth = self.get_depth_l1_l5_bid()
        ask_depth = self.get_depth_l1_l5_ask()
        
        # Debug: Log occasionally to check if values change
        import time
        if not hasattr(self, '_last_depth_log') or time.time() - self._last_depth_log > 10:
            self._last_depth_log = time.time()
            # Show top 3 levels for debugging
            top_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:3]
            top_asks = sorted(self.asks.items(), key=lambda x: x[0])[:3]
            print(f"DEBUG LocalBook {self.symbol}: bid_depth={bid_depth:.4f}, ask_depth={ask_depth:.4f}")
            print(f"  Top bids: {top_bids}")
            print(f"  Top asks: {top_asks}")
        
        return bid_depth, ask_depth
    
    def calculate_checksum(self) -> str:
        """Calcule un checksum du book pour validation"""
        # Concaténer les 10 meilleurs niveaux de chaque côté
        bid_array, ask_array = self.get_book_arrays(10)
        
        checksum_string = ""
        
        # Ajouter les bids
        for price, qty in bid_array:
            checksum_string += f"{price:.8f}:{qty:.8f}:"
        
        # Ajouter les asks
        for price, qty in ask_array:
            checksum_string += f"{price:.8f}:{qty:.8f}:"
        
        return hashlib.md5(checksum_string.encode()).hexdigest()
    
    def get_book_stats(self) -> Dict:
        """Retourne des statistiques du book"""
        bid_array, ask_array = self.get_book_arrays()
        
        stats = {
            'symbol': self.symbol,
            'is_synchronized': self.is_synchronized,
            'last_update_id': self.last_update_id,
            'update_count': self.update_count,
            'sync_errors': self.sync_errors,
            'last_sync_time': self.last_sync_time,
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'best_bid': max(self.bids.keys()) if self.bids else None,
            'best_ask': min(self.asks.keys()) if self.asks else None,
            'mid_price': self.get_mid_price(),
            'spread': self.get_spread(),
            'spread_bps': self.get_spread_bps(),
            'imbalance_5': self.calculate_imbalance(5),
            'imbalance_10': self.calculate_imbalance(10),
            'checksum': self.calculate_checksum(),
        }
        
        return stats
    
    def print_book_summary(self, levels: int = 10):
        """Affiche un résumé du book"""
        if not self.is_synchronized:
            print(f"❌ {self.symbol} - Book non synchronisé")
            return
        
        bid_array, ask_array = self.get_book_arrays(levels)
        stats = self.get_book_stats()
        
        print(f"\n📖 Order Book {self.symbol}")
        print("=" * 50)
        print(f"Updates: {self.update_count} | Sync: {self.is_synchronized}")
        print(f"Mid: ${stats['mid_price']:.2f} | Spread: {stats['spread_bps']:.1f}bps")
        print(f"Imbalance 5: {stats['imbalance_5']:.3f}")
        print()
        
        print("ASKS (ascending)")
        for price, qty in reversed(ask_array):
            print(f"  ${price:.2f} : {qty:.6f}")
        
        print(f"  --- SPREAD: ${stats['spread']:.4f} ---")
        
        print("BIDS (descending)")
        for price, qty in bid_array:
            print(f"  ${price:.2f} : {qty:.6f}")
        
        print("=" * 50)
    
    def resync_if_needed(self) -> bool:
        """Re-synchronise le book si nécessaire"""
        if not self.is_synchronized:
            self.logger.info("🔄 Re-synchronisation du book...")
            return self.get_snapshot()
        return True
    
    def apply_ws_update(self, symbol: str, diff_data: dict) -> bool:
        """Applique une mise à jour WebSocket (wrapper pour apply_diff)"""
        # Vérifier que le symbole correspond
        if symbol.upper() != self.symbol:
            self.logger.warning(f"⚠️ Symbol mismatch: expected {self.symbol}, got {symbol}")
            return False
        
        # Appliquer la mise à jour
        success = self.apply_diff(diff_data)
        
        if success:
            # Log périodique des mises à jour
            if self.update_count % 100 == 0:
                self.logger.debug(f"✅ {self.symbol}: {self.update_count} WebSocket updates applied")
        
        return success

class MultiBookManager:
    """Gestionnaire de plusieurs order books"""
    
    def __init__(self, symbols: List[str]):
        self.books = {symbol: LocalBook(symbol) for symbol in symbols}
        self.logger = logging.getLogger("MultiBookManager")
    
    def sync_all_books(self) -> Dict[str, bool]:
        """Synchronise tous les books"""
        results = {}
        for symbol, book in self.books.items():
            results[symbol] = book.get_snapshot()
        return results
    
    def apply_ws_update(self, symbol: str, diff_data: dict) -> bool:
        """Applique une mise à jour WebSocket"""
        if symbol in self.books:
            return self.books[symbol].apply_diff(diff_data)
        return False
    
    def get_all_mid_prices(self) -> Dict[str, float]:
        """Retourne tous les prix moyens"""
        return {symbol: book.get_mid_price() 
                for symbol, book in self.books.items() 
                if book.is_synchronized}
    
    def print_all_summaries(self):
        """Affiche le résumé de tous les books"""
        for symbol, book in self.books.items():
            book.print_book_summary(5)

# Test du module
if __name__ == "__main__":
    # Test basique
    print("🧪 Test LocalBook...")
    
    book = LocalBook('BTCUSDT')
    
    if book.get_snapshot():
        print("✅ Snapshot OK")
        book.print_book_summary()
        
        stats = book.get_book_stats()
        print(f"\n📊 Stats: Mid=${stats['mid_price']:.2f}, Spread={stats['spread_bps']:.1f}bps")
    else:
        print("❌ Erreur snapshot")
