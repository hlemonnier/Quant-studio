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
from .ofi import OFICalculator   # Helper for users – quoter only consumes OFI value

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
        
        # Paramètres du modèle (spécifiques au symbole si disponibles)
        symbol_params = mm_config.get_symbol_params(symbol)
        self.gamma = symbol_params['gamma']  # Risk aversion
        self.T = symbol_params['T']  # Time horizon
        self.k = symbol_params['k']  # Market impact parameter

        # Estimation de la volatilité partagée (toujours globale)
        self.sigma = mm_config.sigma  # Volatility estimate
        
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
        
        # Recalculer la volatilité dès qu'on a au moins 2 observations (V1-α)
        if len(self.price_history) >= 2:
            self._estimate_volatility()
    
    def _estimate_volatility(self):
        """
        Estime la volatilité avec EWMA sur exactement 100 observations (§3.2 V1-α)
        Conforme au cahier des charges : "EWMA 100 observations ajustée every tick"
        """
        if len(self.price_history) < 2:
            return
        
        prices = [p['price'] for p in self.price_history]
        log_returns = np.diff(np.log(prices))
        
        # EWMA avec exactement les observations disponibles (max 100)
        # Facteur de décroissance pour EWMA : α = 2/(N+1) où N=100
        alpha_ewma = 2.0 / (100 + 1)  # ≈ 0.0198
        
        if len(log_returns) == 1:
            # Premier return : initialiser avec la variance simple
            ewma_variance = log_returns[0] ** 2
        else:
            # Calcul EWMA récursif sur les returns
            ewma_variance = 0.0
            weights_sum = 0.0
            
            for i, ret in enumerate(reversed(log_returns)):
                weight = (1 - alpha_ewma) ** i
                ewma_variance += weight * (ret ** 2)
                weights_sum += weight
                
            ewma_variance /= weights_sum
        
        # Volatilité annualisée (updates toutes les 100ms selon §3.2)
        # 1 jour = 86400 secondes = 864000 updates de 100ms
        periods_per_day = 864000
        volatility_daily = np.sqrt(ewma_variance * periods_per_day)
        
        # Mise à jour directe (pas de lissage supplémentaire car EWMA déjà lissé)
        self.sigma = volatility_daily
        
        self.logger.debug(f"📊 Volatilité EWMA-100: {self.sigma:.4f} (sur {len(log_returns)} obs)")
    
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
    
    def compute_optimal_spread(
        self,
        mid_price: float,
        inventory: float = 0.0,
        time_remaining: float = None
    ) -> float:
        """
        Calcule le **spread total coté** (2 × δ*) selon la solution fermée
        d'Avellaneda-Stoikov (§3.3 du cahier des charges) :
            δ* = (1/γ) · ln(1 + γ/k)      ← demi-spread théorique
            spread_total = 2 · δ*
        
        Cette formule **ne dépend pas** de l'inventaire ni du temps restant
        pour la V1 (horizon maintenu court dans la config). Les anciennes
        versions dépendantes de σ et (T-t) faisaient parfois tomber le spread
        à 0 bps, d'où la régression observée.
        """
        # 1️⃣ Calcule le spread **en dollars** (formule A&S)
        delta_star = (1 / self.gamma) * math.log(1 + self.gamma / self.k)  # demi-spread $
        spread_units = 2 * delta_star                                       # spread total $

        # 2️⃣ Normalise en fraction du prix pour travailler en bps correctement
        spread_fraction = spread_units / mid_price                          # ex: 0.0002
        spread_bps = spread_fraction * 10000                                # ex: 2.0 bps

        # 3️⃣ Applique limites mini/maxi en **bps**
        spread_bps = max(mm_config.min_spread_bps,
                         min(mm_config.max_spread_bps, spread_bps))

        # 4️⃣ Retourne la fraction finale
        return spread_bps / 10000
    
    def compute_quotes(self, mid_price: float, inventory: float,
                       time_remaining: float = None,
                       ofi: float = 0.0) -> Dict[str, float]:
        """
        Calcule les prix bid et ask optimaux
        
        bid = r - δ/2
        ask = r + δ/2
        
        où r = prix de réservation, δ = spread optimal
        """
        # Utiliser le cache si les paramètres n'ont pas changé
        cache_key = (mid_price, inventory, time_remaining, ofi)
        if (self._cached_reservation_price is not None and 
            self._cache_inventory == inventory and
            abs(self._cached_reservation_price - mid_price) < 0.01):
            reservation_price = self._cached_reservation_price
            optimal_spread = self._cached_optimal_spread
        else:
            # Recalculer
            reservation_price = self.compute_reservation_price(mid_price, inventory, time_remaining)
            # ⬇️  Nouvelle signature : le premier argument est désormais mid_price
            optimal_spread = self.compute_optimal_spread(mid_price,
                                                         inventory,
                                                         time_remaining)
            
            # Mettre en cache
            self._cached_reservation_price = reservation_price
            self._cached_optimal_spread = optimal_spread
            self._cache_inventory = inventory
            self._cache_timestamp = datetime.now()
        
        # Calculer bid et ask
        half_spread = optimal_spread / 2
        
        # ------------------------------------------------------------------
        # Centre-shift basé sur l'Order-Flow Imbalance  (§3.3bis)
        # ------------------------------------------------------------------
        tick_size = mm_config.get_symbol_config(self.symbol).get(
            "tick_size", mm_config.default_tick_size
        )
        max_shift = tick_size                  # ±1 tick
        center_shift = np.clip(mm_config.beta_ofi * ofi * tick_size,
                               -max_shift, max_shift)

        reservation_price_shifted = reservation_price + center_shift

        bid_price = reservation_price_shifted - half_spread
        ask_price = reservation_price_shifted + half_spread
        
        # Calculer le spread en bps
        # optimal_spread est déjà une FRACTION (ex: 0.0005 pour 5 bps)
        # Il suffit donc de multiplier par 10 000 pour obtenir les bps.
        spread_bps = optimal_spread * 10000
        
        quotes = {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'reservation_price': reservation_price,
            'reservation_price_shifted': reservation_price_shifted,
            'optimal_spread': optimal_spread,
            'spread_bps': spread_bps,
            'center_shift': center_shift,
            'ofi': ofi,
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
        
        # Vérifier le spread en bps (pas en valeur absolue)
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
    print(f"OFI: {quotes['ofi']:.4f}")
    print(f"Center Shift: {quotes['center_shift']:.6f}")
    
    # Test avec différents inventaires
    print("\n📈 Impact de l'inventaire:")
    for inv in [-0.5, -0.1, 0, 0.1, 0.5]:
        q = quoter.compute_quotes(mid_price, inv)
        print(f"Inventory {inv:+.1f}: Bid=${q['bid_price']:.2f}, Ask=${q['ask_price']:.2f}, Reservation=${q['reservation_price']:.2f}")

if __name__ == "__main__":
    quick_quote_test()
