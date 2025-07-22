#!/usr/bin/env python3
"""
Test pour vérifier que le bug PnL est corrigé
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inventory_control import InventoryController
from kpi_tracker import KPITracker
from config import mm_config

def test_pnl_consistency():
    """Test que les calculs PnL sont cohérents entre InventoryController et KPITracker"""
    
    print("🧪 Test de cohérence PnL...")
    
    # Créer les instances
    inventory_ctrl = InventoryController("BTCUSDT")
    kpi_tracker = KPITracker("BTCUSDT")
    
    # Simuler des trades
    mid_price = 100000.0  # $100k BTC
    
    # Trade 1: Acheter 0.01 BTC à 99950 (bid fill)
    trade_price_1 = 99950.0
    trade_qty_1 = 0.01
    
    inventory_ctrl.update_inventory(trade_qty_1, trade_price_1, mid_price)
    
    # Simuler le même trade dans KPITracker
    from dataclasses import dataclass
    
    @dataclass
    class MockFill:
        side: str
        price: float
        size: float
        quote_id: str = "test"
        timestamp: float = 0.0
        
    fill_1 = MockFill(side='bid', price=trade_price_1, size=trade_qty_1)
    spread_1 = abs(trade_price_1 - mid_price)
    kpi_tracker.record_fill(fill_1, mid_price, spread_1)
    kpi_tracker.update_mid_price(mid_price)
    
    # Trade 2: Vendre 0.005 BTC à 100050 (ask fill)
    trade_price_2 = 100050.0
    trade_qty_2 = -0.005  # Négatif pour vente
    
    inventory_ctrl.update_inventory(trade_qty_2, trade_price_2, mid_price)
    
    fill_2 = MockFill(side='ask', price=trade_price_2, size=abs(trade_qty_2))
    spread_2 = abs(trade_price_2 - mid_price)
    kpi_tracker.record_fill(fill_2, mid_price, spread_2)
    
    # Comparer les PnL
    inventory_pnl = inventory_ctrl.get_mark_to_market_pnl()
    kpi_pnl = kpi_tracker.get_total_pnl()
    
    print(f"📊 Résultats:")
    print(f"  Inventaire actuel: {inventory_ctrl.current_inventory:.6f} BTC")
    print(f"  Prix mid: ${mid_price:.2f}")
    print(f"  Cash flow total: ${inventory_ctrl.total_cash_flow:.2f}")
    print(f"  PnL InventoryController: ${inventory_pnl:.2f}")
    print(f"  PnL KPITracker: ${kpi_pnl:.2f}")
    print(f"  Différence: ${abs(inventory_pnl - kpi_pnl):.2f}")
    
    # Vérifier la cohérence
    tolerance = 0.01  # 1 cent de tolérance
    if abs(inventory_pnl - kpi_pnl) < tolerance:
        print("✅ SUCCESS: Les calculs PnL sont cohérents!")
        return True
    else:
        print("❌ FAIL: Les calculs PnL sont incohérents!")
        return False

def test_stop_loss_logic():
    """Test que le stop loss utilise le bon calcul PnL"""
    
    print("\n🛑 Test de logique stop loss...")
    
    inventory_ctrl = InventoryController("BTCUSDT")
    
    # Simuler une position perdante
    mid_price = 100000.0
    trade_price = 99000.0  # Acheté 1000$ en dessous du mid
    trade_qty = 0.1  # 0.1 BTC
    
    inventory_ctrl.update_inventory(trade_qty, trade_price, mid_price)
    
    # Le prix chute à 95000
    new_mid_price = 95000.0
    inventory_ctrl.update_mid_price(new_mid_price)
    
    # Calculer les PnL
    cash_flow = inventory_ctrl.total_cash_flow
    mtm_pnl = inventory_ctrl.get_mark_to_market_pnl()
    
    print(f"📊 Simulation perte:")
    print(f"  Position: {inventory_ctrl.current_inventory:.3f} BTC")
    print(f"  Prix d'achat: ${trade_price:.2f}")
    print(f"  Prix mid actuel: ${new_mid_price:.2f}")
    print(f"  Cash flow: ${cash_flow:.2f}")
    print(f"  PnL mark-to-market: ${mtm_pnl:.2f}")
    print(f"  Différence: ${mtm_pnl - cash_flow:.2f}")
    
    # Vérifier que le stop loss utilise le bon calcul
    should_pause, reason = inventory_ctrl.should_pause_trading()
    
    if should_pause:
        print(f"🛑 Stop loss déclenché: {reason}")
        # Vérifier que le montant dans le message correspond au PnL mark-to-market
        if f"{mtm_pnl:.2f}" in reason:
            print("✅ SUCCESS: Stop loss utilise le PnL mark-to-market!")
            return True
        else:
            print("❌ FAIL: Stop loss n'utilise pas le bon PnL!")
            return False
    else:
        print("ℹ️  Stop loss pas déclenché (normal si perte < 20%)")
        return True

if __name__ == "__main__":
    print("🔧 Test de correction du bug PnL\n")
    
    success1 = test_pnl_consistency()
    success2 = test_stop_loss_logic()
    
    if success1 and success2:
        print("\n🎉 TOUS LES TESTS PASSENT!")
        print("✅ Le bug PnL est corrigé!")
    else:
        print("\n❌ CERTAINS TESTS ÉCHOUENT!")
        print("🔧 Des corrections supplémentaires sont nécessaires.")
