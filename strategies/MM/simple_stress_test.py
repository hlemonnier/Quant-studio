#!/usr/bin/env python3
"""
Tests de stress simplifiés pour V1-α
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)

from simple_backtest import simple_backtest

def generate_flash_crash_data(duration_minutes: int = 15) -> pd.DataFrame:
    """Génère un scénario de flash crash"""
    print(f"📉 Génération d'un flash crash de {duration_minutes} minutes...")
    
    ticks_per_minute = 600
    total_ticks = duration_minutes * ticks_per_minute
    
    data = []
    base_price = 50000.0
    current_price = base_price
    base_time = datetime.now(timezone.utc)
    
    # Phase 1: Crash rapide (premiers 20% du temps)
    crash_ticks = int(total_ticks * 0.2)
    crash_magnitude = -0.20  # -20% crash
    
    for i in range(crash_ticks):
        # Crash accéléré
        progress = (i + 1) / crash_ticks
        price_drop = crash_magnitude * progress * progress  # Quadratique
        current_price = base_price * (1 + price_drop)
        
        # Spread très large pendant le crash
        spread_multiplier = 1 + 10 * progress  # Jusqu'à 11x le spread normal
        spread = current_price * 0.0005 * spread_multiplier
        
        tick_time = base_time + timedelta(milliseconds=i * 100)
        
        data.append({
            'timestamp': tick_time,
            'mid_price': current_price,
            'bid_price': current_price - spread / 2,
            'ask_price': current_price + spread / 2,
            'spread': spread,
            'spread_bps': (spread / current_price) * 10000,
            'volume': np.random.lognormal(3.0, 1.5),  # Volume élevé
            'regime': 'flash_crash',
            'day': 1
        })
    
    # Phase 2: Stabilisation (80% restant)
    stable_ticks = total_ticks - crash_ticks
    crash_price = current_price
    
    for i in range(stable_ticks):
        # Petites oscillations autour du niveau de crash
        noise = np.random.normal(0, 0.001)
        current_price = crash_price * (1 + noise)
        
        # Spread qui se normalise progressivement
        progress = i / stable_ticks
        spread_multiplier = 11 - 10 * progress  # De 11x vers 1x
        spread = current_price * 0.0005 * spread_multiplier
        
        tick_idx = crash_ticks + i
        tick_time = base_time + timedelta(milliseconds=tick_idx * 100)
        
        data.append({
            'timestamp': tick_time,
            'mid_price': current_price,
            'bid_price': current_price - spread / 2,
            'ask_price': current_price + spread / 2,
            'spread': spread,
            'spread_bps': (spread / current_price) * 10000,
            'volume': np.random.lognormal(2.5, 1.0),  # Volume élevé mais décroissant
            'regime': 'crash_recovery',
            'day': 1
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Généré {len(df)} ticks de flash crash")
    return df

def generate_volatility_spike_data(duration_minutes: int = 20) -> pd.DataFrame:
    """Génère un scénario de spike de volatilité"""
    print(f"⚡ Génération d'un spike de volatilité de {duration_minutes} minutes...")
    
    ticks_per_minute = 600
    total_ticks = duration_minutes * ticks_per_minute
    
    data = []
    base_price = 50000.0
    current_price = base_price
    base_time = datetime.now(timezone.utc)
    
    # Volatilité très élevée
    dt = 1.0 / (365 * 24 * 60 * 60 * 10)  # 100ms en fraction d'année
    high_vol = 2.0  # 200% volatilité annuelle (vs 60% normal)
    vol_per_tick = high_vol * np.sqrt(dt)
    
    for i in range(total_ticks):
        # Mouvement de prix très volatil
        price_change = np.random.normal(0, vol_per_tick)
        current_price *= (1 + price_change)
        
        # Spread qui s'élargit avec la volatilité
        spread_multiplier = 1 + 2.0  # 3x le spread normal
        spread = current_price * 0.0005 * spread_multiplier
        
        tick_time = base_time + timedelta(milliseconds=i * 100)
        
        data.append({
            'timestamp': tick_time,
            'mid_price': current_price,
            'bid_price': current_price - spread / 2,
            'ask_price': current_price + spread / 2,
            'spread': spread,
            'spread_bps': (spread / current_price) * 10000,
            'volume': np.random.lognormal(1.5, 1.0),  # Volume plus faible
            'regime': 'volatility_spike',
            'day': 1
        })
        
        # Progress
        if i % 5000 == 0 and i > 0:
            progress = (i / total_ticks) * 100
            print(f"  Progress: {progress:.1f}%")
    
    df = pd.DataFrame(data)
    print(f"✅ Généré {len(df)} ticks de volatilité élevée")
    return df

def generate_trending_data(duration_minutes: int = 30, trend_pct: float = 10.0) -> pd.DataFrame:
    """Génère un scénario de marché en tendance"""
    print(f"📈 Génération d'une tendance {trend_pct:+.1f}% sur {duration_minutes} minutes...")
    
    ticks_per_minute = 600
    total_ticks = duration_minutes * ticks_per_minute
    
    data = []
    base_price = 50000.0
    current_price = base_price
    base_time = datetime.now(timezone.utc)
    
    # Drift constant pour la tendance
    total_return = trend_pct / 100.0
    drift_per_tick = total_return / total_ticks
    
    for i in range(total_ticks):
        # Tendance + bruit
        noise = np.random.normal(0, 0.0005)
        price_change = drift_per_tick + noise
        current_price *= (1 + price_change)
        
        # Spread normal
        spread = current_price * 0.0005
        
        tick_time = base_time + timedelta(milliseconds=i * 100)
        
        data.append({
            'timestamp': tick_time,
            'mid_price': current_price,
            'bid_price': current_price - spread / 2,
            'ask_price': current_price + spread / 2,
            'spread': spread,
            'spread_bps': (spread / current_price) * 10000,
            'volume': np.random.lognormal(2.2, 1.0),  # Volume élevé en tendance
            'regime': 'trending_up' if trend_pct > 0 else 'trending_down',
            'day': 1
        })
        
        # Progress
        if i % 5000 == 0 and i > 0:
            progress = (i / total_ticks) * 100
            print(f"  Progress: {progress:.1f}%")
    
    df = pd.DataFrame(data)
    direction = "haussière" if trend_pct > 0 else "baissière"
    print(f"✅ Généré {len(df)} ticks de tendance {direction}")
    return df

async def run_stress_scenario(scenario_name: str, market_data: pd.DataFrame) -> dict:
    """Exécute un scénario de stress"""
    print(f"\n🧪 Test de stress: {scenario_name}")
    print("-" * 40)
    
    # Exécuter le backtest
    results = await simple_backtest(market_data)
    
    # Analyser les résultats
    performance = results['performance']
    kpi = performance['kpi_summary']
    
    # Critères de réussite pour les stress tests
    total_return = performance['total_return_pct']
    max_loss_threshold = -10.0  # Pas plus de -10% de perte
    
    # Évaluation
    passed = total_return > max_loss_threshold
    
    print(f"\n📊 Résultats {scenario_name}:")
    print(f"  Trades: {results['metadata']['total_trades']}")
    print(f"  PnL: ${kpi['total_pnl']:+.2f}")
    print(f"  Return: {total_return:+.2f}%")
    print(f"  Fill Ratio: {kpi['fill_ratio']:.2%}")
    print(f"  Inventory Final: {performance['final_inventory']:.4f}")
    
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"  Status: {status}")
    
    if not passed:
        print(f"  ⚠️ Raison: Perte excessive ({total_return:.2f}% > {max_loss_threshold}%)")
    
    return {
        'scenario': scenario_name,
        'passed': passed,
        'total_return': total_return,
        'total_pnl': kpi['total_pnl'],
        'trades': results['metadata']['total_trades'],
        'fill_ratio': kpi['fill_ratio']
    }

async def main():
    """Fonction principale des tests de stress"""
    print("🧪 Tests de Stress V1-α Simplifiés")
    print("=" * 50)
    
    stress_results = []
    
    try:
        # Test 1: Flash Crash
        print("\n1️⃣ Test Flash Crash")
        crash_data = generate_flash_crash_data(duration_minutes=15)
        crash_result = await run_stress_scenario("Flash Crash", crash_data)
        stress_results.append(crash_result)
        
        # Test 2: Spike de Volatilité
        print("\n2️⃣ Test Spike de Volatilité")
        vol_data = generate_volatility_spike_data(duration_minutes=20)
        vol_result = await run_stress_scenario("Volatility Spike", vol_data)
        stress_results.append(vol_result)
        
        # Test 3: Tendance Haussière
        print("\n3️⃣ Test Tendance Haussière")
        trend_up_data = generate_trending_data(duration_minutes=30, trend_pct=15.0)
        trend_up_result = await run_stress_scenario("Trending Up", trend_up_data)
        stress_results.append(trend_up_result)
        
        # Test 4: Tendance Baissière
        print("\n4️⃣ Test Tendance Baissière")
        trend_down_data = generate_trending_data(duration_minutes=30, trend_pct=-15.0)
        trend_down_result = await run_stress_scenario("Trending Down", trend_down_data)
        stress_results.append(trend_down_result)
        
        # Résumé final
        print("\n" + "=" * 60)
        print("🎯 RÉSUMÉ DES TESTS DE STRESS")
        print("=" * 60)
        
        passed_tests = sum(1 for r in stress_results if r['passed'])
        total_tests = len(stress_results)
        pass_rate = (passed_tests / total_tests) * 100
        
        print(f"Tests réussis: {passed_tests}/{total_tests} ({pass_rate:.0f}%)")
        
        total_pnl = sum(r['total_pnl'] for r in stress_results)
        print(f"PnL total: ${total_pnl:+.2f}")
        
        print(f"\nDétail par test:")
        for result in stress_results:
            status = "✅" if result['passed'] else "❌"
            print(f"  {status} {result['scenario']}: {result['total_return']:+.2f}% (${result['total_pnl']:+.2f})")
        
        # Recommandation
        if pass_rate >= 75:
            print(f"\n🎉 EXCELLENT: L'algorithme résiste bien aux conditions de stress!")
        elif pass_rate >= 50:
            print(f"\n⚠️ ACCEPTABLE: Quelques améliorations nécessaires pour les cas extrêmes.")
        else:
            print(f"\n❌ PROBLÉMATIQUE: L'algorithme nécessite des améliorations importantes.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Erreur dans les tests de stress: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
