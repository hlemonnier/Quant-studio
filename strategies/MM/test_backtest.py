#!/usr/bin/env python3
"""
Test simple du backtest V1-α
"""

import asyncio
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from backtest_v1_alpha import MarketDataSimulator, BacktestEngine

async def test_simple():
    """Test simple et rapide"""
    print("🧪 Test simple du backtest V1-α")
    
    # 1. Test génération de données (1 jour seulement)
    print("📊 Génération de données de test...")
    simulator = MarketDataSimulator('BTCUSDT', 50000.0)
    
    # Générer seulement 1 heure de données pour le test
    regime_schedule = {0: 'normal'}  # 1 jour seulement
    market_data = simulator.generate_7day_data(regime_schedule)
    
    # Réduire à 1 heure pour le test
    one_hour_data = market_data.head(36000)  # 36000 ticks = 1 heure
    print(f"✅ Généré {len(one_hour_data)} ticks (1 heure)")
    
    # 2. Test du moteur de backtest
    print("🔄 Test du moteur de backtest...")
    engine = BacktestEngine('BTCUSDT', 100000.0)
    
    # Exécuter le backtest sur 1 heure seulement
    results = await engine.run_7day_backtest(one_hour_data, save_results=False)
    
    # 3. Afficher les résultats
    print("\n📊 Résultats du test:")
    performance = results['performance']
    kpi = performance['kpi_summary']
    
    print(f"  Trades: {len(results['metadata']['total_trades'])}")
    print(f"  PnL: ${kpi['total_pnl']:+.2f}")
    print(f"  Fill Ratio: {kpi['fill_ratio']:.2%}")
    print(f"  Spread Captured: {kpi['spread_captured_pct']:.1f}%")
    
    print("✅ Test terminé avec succès!")
    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(test_simple())
        print("\n🎉 Test réussi!")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
