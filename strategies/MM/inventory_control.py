"""
Inventory Control - Étape 4 de la roadmap MM V1

Objectif: Skew dès que l'inventaire dépasse les seuils
Livrable: fonction de contrôle d'inventaire avec skew
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
from .config import mm_config

class InventoryController:
    """Contrôleur d'inventaire avec skew automatique"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"InventoryControl-{symbol}")
        
        # Inventaire actuel
        self.current_inventory = 0.0
        self.target_inventory = mm_config.inventory_target
        
        # Seuils et paramètres
        self.max_inventory = mm_config.max_inventory  # N_max
        self.inventory_threshold = mm_config.inventory_threshold  # N★
        self.skew_factor = mm_config.skew_factor
        
        # Historique pour analyse
        self.inventory_history = []
        self.pnl_history = []
        
        # Métriques de risque
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
    def update_inventory(self, trade_quantity: float, trade_price: float):
        """Met à jour l'inventaire après un trade"""
        # Mettre à jour l'inventaire
        self.current_inventory += trade_quantity
        
        # Calculer le PnL (simplifié)
        pnl_change = -trade_quantity * trade_price  # Coût du trade
        self.total_pnl += pnl_change
        
        # Enregistrer dans l'historique
        self.inventory_history.append({
            'timestamp': datetime.now(),
            'inventory': self.current_inventory,
            'trade_qty': trade_quantity,
            'trade_price': trade_price,
            'pnl': self.total_pnl
        })
        
        # Log si inventaire important
        if abs(self.current_inventory) > self.inventory_threshold:
            self.logger.info(f"⚠️  Inventaire élevé: {self.current_inventory:.4f}")
        
        # Vérifier les limites
        self._check_inventory_limits()
    
    def _check_inventory_limits(self):
        """Vérifie si l'inventaire dépasse les limites"""
        if abs(self.current_inventory) > self.max_inventory:
            self.logger.warning(f"🚨 LIMITE INVENTAIRE DÉPASSÉE: {self.current_inventory:.4f} > {self.max_inventory}")
            return False
        return True
    
    def calculate_inventory_skew(self) -> float:
        """
        Calcule le skew à appliquer aux quotes en fonction de l'inventaire
        
        Retourne un facteur de skew:
        - Positif: favorise la vente (décale ask vers le bas, bid vers le bas)
        - Négatif: favorise l'achat (décale bid vers le haut, ask vers le haut)
        """
        if abs(self.current_inventory) <= self.inventory_threshold:
            return 0.0  # Pas de skew sous le seuil
        
        # Calculer l'excès d'inventaire
        excess_inventory = self.current_inventory - self.target_inventory
        
        # Normaliser par rapport aux seuils
        if excess_inventory > 0:
            # Trop long: favoriser la vente
            skew_intensity = min(1.0, excess_inventory / self.max_inventory)
        else:
            # Trop court: favoriser l'achat
            skew_intensity = max(-1.0, excess_inventory / self.max_inventory)
        
        # Appliquer le facteur de skew
        skew = skew_intensity * self.skew_factor
        
        return skew
    
    def apply_skew_to_quotes(self, quotes: Dict[str, float]) -> Dict[str, float]:
        """Applique le skew d'inventaire aux quotes"""
        skew = self.calculate_inventory_skew()
        
        if abs(skew) < 0.01:  # Pas de skew significatif
            return quotes
        
        skewed_quotes = quotes.copy()
        
        # Calculer l'ajustement de prix (en fraction du spread)
        mid_price = quotes['mid_price']
        spread = quotes['optimal_spread']
        
        # Ajustement proportionnel au spread
        price_adjustment = skew * spread * 0.1  # 10% du spread par unité de skew
        
        # Appliquer le skew
        skewed_quotes['bid_price'] = quotes['bid_price'] + price_adjustment
        skewed_quotes['ask_price'] = quotes['ask_price'] + price_adjustment
        
        # Mettre à jour les métriques
        skewed_quotes['inventory_skew'] = skew
        skewed_quotes['price_adjustment'] = price_adjustment
        skewed_quotes['current_inventory'] = self.current_inventory
        
        # Log si skew important
        if abs(skew) > 0.5:
            direction = "SELL" if skew > 0 else "BUY"
            self.logger.info(f"📊 Skew {direction}: {skew:.3f}, Ajustement: {price_adjustment:.4f}")
        
        return skewed_quotes
    
    def calculate_optimal_size(self, side: str, current_price: float) -> float:
        """
        Calcule la taille optimale d'ordre en fonction de l'inventaire
        
        Args:
            side: 'bid' ou 'ask'
            current_price: prix actuel pour l'ordre
        """
        base_size = mm_config.base_quote_size
        
        # Ajuster la taille en fonction de l'inventaire
        inventory_ratio = abs(self.current_inventory) / self.max_inventory
        
        if side == 'bid':
            # Ordres d'achat: réduire si déjà long
            if self.current_inventory > self.inventory_threshold:
                size_multiplier = 1.0 - inventory_ratio
            else:
                size_multiplier = 1.0 + inventory_ratio * 0.5  # Augmenter si court
        else:  # ask
            # Ordres de vente: réduire si déjà court
            if self.current_inventory < -self.inventory_threshold:
                size_multiplier = 1.0 - inventory_ratio
            else:
                size_multiplier = 1.0 + inventory_ratio * 0.5  # Augmenter si long
        
        # Calculer la taille finale
        optimal_size = base_size * size_multiplier
        
        # Appliquer les contraintes
        optimal_size = max(mm_config.min_order_size, 
                          min(mm_config.max_order_size, optimal_size))
        
        return optimal_size
    
    def should_pause_trading(self) -> Tuple[bool, str]:
        """
        Détermine si le trading doit être suspendu
        
        Returns:
            (should_pause, reason)
        """
        # Vérifier l'inventaire maximum
        if abs(self.current_inventory) >= self.max_inventory:
            return True, f"Inventaire limite atteinte: {self.current_inventory:.4f}"
        
        # Vérifier le PnL (stop loss)
        if hasattr(mm_config, 'stop_loss_pct'):
            max_loss = mm_config.stop_loss_pct / 100.0
            if self.total_pnl < -max_loss:
                return True, f"Stop loss déclenché: PnL={self.total_pnl:.4f}"
        
        return False, ""
    
    def get_inventory_stats(self) -> Dict:
        """Retourne les statistiques d'inventaire"""
        # Calculer les métriques sur l'historique récent
        if len(self.inventory_history) > 0:
            recent_history = self.inventory_history[-100:]  # 100 dernières observations
            inventories = [h['inventory'] for h in recent_history]
            
            inventory_mean = np.mean(inventories)
            inventory_std = np.std(inventories)
            inventory_min = min(inventories)
            inventory_max = max(inventories)
        else:
            inventory_mean = inventory_std = inventory_min = inventory_max = 0.0
        
        stats = {
            'current_inventory': self.current_inventory,
            'target_inventory': self.target_inventory,
            'inventory_threshold': self.inventory_threshold,
            'max_inventory': self.max_inventory,
            'current_skew': self.calculate_inventory_skew(),
            'total_pnl': self.total_pnl,
            'inventory_mean_100': inventory_mean,
            'inventory_std_100': inventory_std,
            'inventory_range': (inventory_min, inventory_max),
            'trades_count': len(self.inventory_history),
            'inventory_utilization': abs(self.current_inventory) / self.max_inventory,
        }
        
        return stats
    
    def print_inventory_summary(self):
        """Affiche un résumé de l'inventaire"""
        stats = self.get_inventory_stats()
        skew = stats['current_skew']
        
        print(f"\n📦 Inventaire {self.symbol}")
        print("=" * 40)
        print(f"Inventaire actuel: {stats['current_inventory']:+.4f}")
        print(f"Cible: {stats['target_inventory']:.4f}")
        print(f"Seuil (N★): ±{stats['inventory_threshold']:.4f}")
        print(f"Maximum (N_max): ±{stats['max_inventory']:.4f}")
        print(f"Utilisation: {stats['inventory_utilization']:.1%}")
        print(f"Skew actuel: {skew:+.3f}")
        print(f"PnL total: ${stats['total_pnl']:+.2f}")
        print(f"Nombre de trades: {stats['trades_count']}")
        
        # Indicateur visuel du skew
        if abs(skew) > 0.1:
            direction = "📈 FAVORISE VENTE" if skew > 0 else "📉 FAVORISE ACHAT"
            print(f"Direction: {direction}")
        else:
            print("Direction: ⚖️  NEUTRE")
        
        print("=" * 40)
    
    def reset_daily_stats(self):
        """Remet à zéro les stats quotidiennes"""
        self.daily_pnl = 0.0
        # Garder seulement l'historique récent pour économiser la mémoire
        if len(self.inventory_history) > 1000:
            self.inventory_history = self.inventory_history[-500:]

# Fonction utilitaire pour test
def test_inventory_control():
    """Test du contrôle d'inventaire"""
    print("🧪 Test Inventory Control...")
    
    controller = InventoryController('BTCUSDT')
    
    # Simuler quelques trades
    trades = [
        (0.05, 50000),   # Achat
        (0.03, 50100),   # Achat
        (-0.02, 50200),  # Vente partielle
        (0.04, 49900),   # Achat
    ]
    
    print("\n📊 Simulation de trades:")
    for qty, price in trades:
        controller.update_inventory(qty, price)
        skew = controller.calculate_inventory_skew()
        print(f"Trade: {qty:+.3f} @ ${price} | Inventaire: {controller.current_inventory:+.4f} | Skew: {skew:+.3f}")
    
    controller.print_inventory_summary()
    
    # Test de quotes avec skew
    sample_quotes = {
        'bid_price': 49950,
        'ask_price': 50050,
        'mid_price': 50000,
        'optimal_spread': 100,
    }
    
    skewed_quotes = controller.apply_skew_to_quotes(sample_quotes)
    
    print(f"\n📈 Impact du skew sur les quotes:")
    print(f"Bid original: ${sample_quotes['bid_price']:.2f} → Skewed: ${skewed_quotes['bid_price']:.2f}")
    print(f"Ask original: ${sample_quotes['ask_price']:.2f} → Skewed: ${skewed_quotes['ask_price']:.2f}")
    print(f"Ajustement: {skewed_quotes.get('price_adjustment', 0):.4f}")

if __name__ == "__main__":
    test_inventory_control() 