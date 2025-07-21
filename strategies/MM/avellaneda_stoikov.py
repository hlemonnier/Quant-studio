"""
Avellaneda-Stoikov Quoting - Étape 3 de la roadmap MM V1

Objectif: Prix = Avellaneda-Stoikov (spread optimal)
Livrable: fonction compute_quotes()

Référence: "High-frequency trading in a limit order book" - Avellaneda & Stoikov (2008)
GitHub: fedecaccia/avellaneda-stoikov
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import math
from datetime import datetime, timedelta
import logging
from .config import mm_config

class AvellanedaStoikovQuoter:
    """
    Calculateur de quotes optimal selon le modèle Avellaneda-Stoikov
    
    Le modèle optimise le trade-off entre:
    - Profit par trade (spread large)
    - Probabilité d'exécution (spread étroit)
    - Contrôle du risque d'inventaire
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"AS-{symbol}")
        
        # Paramètres du modèle (depuis config)
        self.gamma = mm_config.gamma  # Risk aversion
        self.sigma = mm_config.sigma  # Volatility estimate
        self.T = mm_config.T  # Time horizon
        self.k = mm_config.k  # Market impact parameter
        
        # Historique pour estimation de volatilité
        self.price_history = []
        self.volatility_window = 100  # Nombre d'observations pour estimer σ
        
        # Cache pour optimisation
        self._cached_reservation_price = None
        self._cached_optimal_spread = None
        self._cache_timestamp = None
        self._cache_inventory = None
        
    def update_volatility(self, new_price: float):
        """Met à jour l'estimation de volatilité"""
        self.price_history.append({
            'price': new_price,
            'timestamp': datetime.now()
        })
        
        # Garder seulement les dernières observations
        if len(self.price_history) > self.volatility_window:
            self.price_history = self.price_history[-self.volatility_window:]
        
        # Recalculer la volatilité si on a assez de données
        if len(self.price_history) >= 10:
            self._estimate_volatility()
    
    def _estimate_volatility(self):
        """Estime la volatilité à partir de l'historique des prix"""
        if len(self.price_history) < 10:
            return
        
        prices = [p['price'] for p in self.price_history]
        log_returns = np.diff(np.log(prices))
        
        # Volatilité annualisée (en supposant des updates toutes les 100ms)
        # 1 jour = 86400 secondes, donc 864000 updates de 100ms
        periods_per_day = 864000
        volatility_daily = np.std(log_returns) * np.sqrt(periods_per_day)
        
        # Mise à jour avec lissage exponentiel
        alpha = 0.1  # Facteur de lissage
        self.sigma = alpha * volatility_daily + (1 - alpha) * self.sigma
        
        self.logger.debug(f"Volatilité mise à jour: {self.sigma:.4f}")
    
    def compute_reservation_price(self, mid_price: float, inventory: float, 
                                 time_remaining: float = None) -> float:
        """
        Calcule le prix de réservation r selon Avellaneda-Stoikov
        
        r = S - q * γ * σ² * (T - t)
        
        où:
        - S = prix mid actuel
        - q = inventaire actuel
        - γ = aversion au risque
        - σ = volatilité
        - (T - t) = temps restant jusqu'à l'horizon
        """
        if time_remaining is None:
            time_remaining = self.T  # Utiliser l'horizon complet par défaut
        
        inventory_adjustment = inventory * self.gamma * (self.sigma ** 2) * time_remaining
        reservation_price = mid_price - inventory_adjustment
        
        return reservation_price
    
    def compute_optimal_spread(self, inventory: float, time_remaining: float = None) -> float:
        """
        Calcule le spread optimal δ selon Avellaneda-Stoikov
        
        δ = γ * σ² * (T - t) + (2/γ) * ln(1 + γ/k)
        
        Le premier terme gère le risque d'inventaire
        Le second terme gère l'impact de marché
        """
        if time_remaining is None:
            time_remaining = self.T
        
        # Terme de risque d'inventaire
        risk_term = self.gamma * (self.sigma ** 2) * time_remaining
        
        # Terme d'impact de marché
        impact_term = (2 / self.gamma) * math.log(1 + self.gamma / self.k)
        
        optimal_spread = risk_term + impact_term
        
        # Contraintes min/max
        min_spread = mm_config.min_spread_bps / 10000  # Convertir bps en fraction
        max_spread = mm_config.max_spread_bps / 10000
        
        optimal_spread = max(min_spread, min(max_spread, optimal_spread))
        
        return optimal_spread
    
    def compute_quotes(self, mid_price: float, inventory: float, 
                      time_remaining: float = None) -> Dict[str, float]:
        """
        Calcule les prix bid et ask optimaux
        
        bid = r - δ/2
        ask = r + δ/2
        
        où r = prix de réservation, δ = spread optimal
        """
        # Utiliser le cache si les paramètres n'ont pas changé
        cache_key = (mid_price, inventory, time_remaining)
        if (self._cached_reservation_price is not None and 
            self._cache_inventory == inventory and
            abs(self._cached_reservation_price - mid_price) < 0.01):
            reservation_price = self._cached_reservation_price
            optimal_spread = self._cached_optimal_spread
        else:
            # Recalculer
            reservation_price = self.compute_reservation_price(mid_price, inventory, time_remaining)
            optimal_spread = self.compute_optimal_spread(inventory, time_remaining)
            
            # Mettre en cache
            self._cached_reservation_price = reservation_price
            self._cached_optimal_spread = optimal_spread
            self._cache_inventory = inventory
            self._cache_timestamp = datetime.now()
        
        # Calculer bid et ask
        half_spread = optimal_spread / 2
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread
        
        quotes = {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'reservation_price': reservation_price,
            'optimal_spread': optimal_spread,
            'spread_bps': (optimal_spread / mid_price) * 10000,
            'mid_price': mid_price,
            'inventory': inventory,
            'volatility': self.sigma,
            'time_remaining': time_remaining or self.T
        }
        
        return quotes
    
    def adjust_for_market_conditions(self, quotes: Dict[str, float], 
                                   book_imbalance: float,
                                   recent_volatility: float = None) -> Dict[str, float]:
        """
        Ajuste les quotes en fonction des conditions de marché
        
        - Imbalance du book: ajuste les prix vers le côté avec plus de liquidité
        - Volatilité récente: ajuste le spread
        """
        adjusted_quotes = quotes.copy()
        
        # Ajustement pour imbalance du book
        if abs(book_imbalance) > 0.1:  # Seuil d'imbalance significatif
            imbalance_adjustment = book_imbalance * 0.1 * quotes['optimal_spread']
            adjusted_quotes['bid_price'] += imbalance_adjustment
            adjusted_quotes['ask_price'] += imbalance_adjustment
            
        # Ajustement pour volatilité récente
        if recent_volatility and recent_volatility > self.sigma * 1.5:
            # Volatilité élevée: élargir le spread
            vol_multiplier = min(2.0, recent_volatility / self.sigma)
            current_spread = quotes['optimal_spread']
            new_spread = current_spread * vol_multiplier
            
            # Recalculer bid/ask avec le nouveau spread
            half_spread = new_spread / 2
            adjusted_quotes['bid_price'] = quotes['reservation_price'] - half_spread
            adjusted_quotes['ask_price'] = quotes['reservation_price'] + half_spread
            adjusted_quotes['optimal_spread'] = new_spread
            adjusted_quotes['spread_bps'] = (new_spread / quotes['mid_price']) * 10000
        
        return adjusted_quotes
    
    def validate_quotes(self, quotes: Dict[str, float], current_mid: float) -> bool:
        """Valide que les quotes sont raisonnables"""
        
        # Vérifier que bid < ask
        if quotes['bid_price'] >= quotes['ask_price']:
            self.logger.warning(f"⚠️  Bid >= Ask: {quotes['bid_price']:.2f} >= {quotes['ask_price']:.2f}")
            return False
        
        # Vérifier que les prix ne sont pas trop éloignés du mid
        max_deviation = current_mid * 0.1  # 10% max
        if (abs(quotes['bid_price'] - current_mid) > max_deviation or
            abs(quotes['ask_price'] - current_mid) > max_deviation):
            self.logger.warning(f"⚠️  Prix trop éloignés du mid: bid={quotes['bid_price']:.2f}, ask={quotes['ask_price']:.2f}, mid={current_mid:.2f}")
            return False
        
        # Vérifier le spread
        spread_bps = quotes['spread_bps']
        if spread_bps < mm_config.min_spread_bps or spread_bps > mm_config.max_spread_bps:
            self.logger.warning(f"⚠️  Spread hors limites: {spread_bps:.1f}bps")
            return False
        
        return True
    
    def get_model_params(self) -> Dict:
        """Retourne les paramètres actuels du modèle"""
        return {
            'symbol': self.symbol,
            'gamma': self.gamma,
            'sigma': self.sigma,
            'T': self.T,
            'k': self.k,
            'volatility_window': self.volatility_window,
            'price_history_size': len(self.price_history)
        }

# Fonction utilitaire pour test rapide
def quick_quote_test():
    """Test rapide du module de quoting"""
    print("🧪 Test Avellaneda-Stoikov...")
    
    quoter = AvellanedaStoikovQuoter('BTCUSDT')
    
    # Simuler quelques prix pour estimer la volatilité
    base_price = 50000
    for i in range(50):
        price = base_price + np.random.normal(0, 100)  # Volatilité simulée
        quoter.update_volatility(price)
    
    # Test de quoting
    mid_price = 50000
    inventory = 0.1  # Long de 0.1 BTC
    
    quotes = quoter.compute_quotes(mid_price, inventory)
    
    print(f"\n📊 Quotes pour {quoter.symbol}")
    print(f"Mid Price: ${quotes['mid_price']:.2f}")
    print(f"Inventory: {quotes['inventory']:.4f}")
    print(f"Reservation Price: ${quotes['reservation_price']:.2f}")
    print(f"Bid: ${quotes['bid_price']:.2f}")
    print(f"Ask: ${quotes['ask_price']:.2f}")
    print(f"Spread: ${quotes['optimal_spread']:.4f} ({quotes['spread_bps']:.1f} bps)")
    print(f"Volatility: {quotes['volatility']:.4f}")
    
    # Test avec différents inventaires
    print("\n📈 Impact de l'inventaire:")
    for inv in [-0.5, -0.1, 0, 0.1, 0.5]:
        q = quoter.compute_quotes(mid_price, inv)
        print(f"Inventory {inv:+.1f}: Bid=${q['bid_price']:.2f}, Ask=${q['ask_price']:.2f}, Reservation=${q['reservation_price']:.2f}")

if __name__ == "__main__":
    quick_quote_test() 