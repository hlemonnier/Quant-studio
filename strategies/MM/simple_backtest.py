#!/usr/bin/env python3
"""
Backtest V1-α simplifié et rapide
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)

from config import MMConfig
from avellaneda_stoikov import AvellanedaStoikovQuoter
from kpi_tracker import KPITracker
from performance_validator import PerformanceValidator
from inventory_control import InventoryController

mm_config = MMConfig()

def generate_simple_data(duration_minutes: int = 60) -> pd.DataFrame:
    """Génère des données de marché simples pour test rapide"""
    print(f"📊 Génération de {duration_minutes} minutes de données...")
    
    ticks_per_minute = 600  # 100ms ticks
    total_ticks = duration_minutes * ticks_per_minute
    
    data = []
    base_price = 50000.0
    current_price = base_price
    base_time = datetime.now(timezone.utc)
    
    # Paramètres de simulation simple
    dt = 1.0 / (365 * 24 * 60 * 60 * 10)  # 100ms en fraction d'année
    vol = 0.6  # 60% volatilité annuelle
    vol_per_tick = vol * np.sqrt(dt)
    
    for i in range(total_ticks):
        # Mouvement de prix simple (random walk)
        price_change = np.random.normal(0, vol_per_tick)
        current_price *= (1 + price_change)
        
        # Spread simple
        spread = current_price * 0.0005  # 5 bps
        bid_price = current_price - spread / 2
        ask_price = current_price + spread / 2
        
        # Volume simple
        volume = np.random.lognormal(2.0, 1.0)
        
        tick_time = base_time + timedelta(milliseconds=i * 100)
        
        data.append({
            'timestamp': tick_time,
            'mid_price': current_price,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'spread': spread,
            'spread_bps': (spread / current_price) * 10000,
            'volume': volume,
            'regime': 'normal',
            'day': 1
        })
        
        # Progress
        if i % 10000 == 0 and i > 0:
            progress = (i / total_ticks) * 100
            print(f"  Progress: {progress:.1f}%")
    
    df = pd.DataFrame(data)
    print(f"✅ Généré {len(df)} ticks")
    return df

async def simple_backtest(market_data: pd.DataFrame) -> dict:
    """Backtest simplifié"""
    print("🔄 Démarrage du backtest...")
    
    symbol = 'BTCUSDT'
    initial_capital = 100000.0
    
    # Initialiser les composants
    quoter = AvellanedaStoikovQuoter(symbol)
    kpi_tracker = KPITracker(symbol)
    validator = PerformanceValidator(symbol)
    inventory_ctrl = InventoryController(symbol)
    
    # État du backtest
    trade_count = 0
    quote_count = 0
    
    print(f"📊 Traitement de {len(market_data)} ticks...")
    
    for idx, row in market_data.iterrows():
        current_time = row['timestamp']
        mid_price = row['mid_price']
        
        # Mettre à jour la volatilité
        quoter.update_volatility(mid_price)
        kpi_tracker.update_mid_price(mid_price)
        inventory_ctrl.update_mid_price(mid_price)
        
        # Calculer les quotes
        inventory = inventory_ctrl.current_inventory
        quotes = quoter.compute_quotes(
            mid_price=mid_price,
            inventory=inventory,
            time_remaining=None,
            ofi=0.0
        )
        
        if quotes and quoter.validate_quotes(quotes, mid_price):
            quote_count += 1
            # Enregistrer les quotes envoyées (bid + ask = 2 quotes)
            kpi_tracker.record_quotes_sent(2)
            
            # Taille par défaut pour les trades
            default_size = 0.01  # 0.01 BTC
            
            # Simuler des fills simples (probabilité réaliste pour test)
            fill_prob = 0.005  # 0.5% de chance de fill par tick
            
            # Fill bid
            if np.random.random() < fill_prob:
                trade_size = default_size
                fill = {
                    'timestamp': current_time,
                    'side': 'bid',
                    'price': quotes['bid_price'],
                    'size': trade_size,
                    'trade_size': trade_size,
                    'mid_price_at_fill': mid_price
                }
                
                inventory_ctrl.update_inventory(trade_size, quotes['bid_price'], mid_price)
                spread_captured = abs(quotes['bid_price'] - mid_price)
                
                # Créer un objet simple pour record_fill
                class SimpleFill:
                    def __init__(self, side, price, size):
                        self.side = side
                        self.price = price
                        self.size = size
                
                simple_fill = SimpleFill(fill['side'], fill['price'], fill['size'])
                kpi_tracker.record_fill(simple_fill, mid_price, spread_captured)
                trade_count += 1
            
            # Fill ask
            if np.random.random() < fill_prob:
                trade_size = -default_size
                fill = {
                    'timestamp': current_time,
                    'side': 'ask',
                    'price': quotes['ask_price'],
                    'size': default_size,
                    'trade_size': trade_size,
                    'mid_price_at_fill': mid_price
                }
                
                inventory_ctrl.update_inventory(trade_size, quotes['ask_price'], mid_price)
                spread_captured = abs(quotes['ask_price'] - mid_price)
                
                # Créer un objet simple pour record_fill
                class SimpleFill:
                    def __init__(self, side, price, size):
                        self.side = side
                        self.price = price
                        self.size = size
                
                simple_fill = SimpleFill(fill['side'], fill['price'], fill['size'])
                kpi_tracker.record_fill(simple_fill, mid_price, spread_captured)
                trade_count += 1
        
        # Progress
        if idx % 10000 == 0 and idx > 0:
            progress = (idx / len(market_data)) * 100
            print(f"  Progress: {progress:.1f}% - Trades: {trade_count}, Quotes: {quote_count}")
    
    # Résultats
    kpi_summary = kpi_tracker.get_summary()
    validation_result = validator.validate_performance(kpi_tracker)
    
    results = {
        'metadata': {
            'symbol': symbol,
            'total_ticks': len(market_data),
            'total_trades': trade_count,
            'total_quotes': quote_count,
            'initial_capital': initial_capital
        },
        'performance': {
            'kpi_summary': kpi_summary,
            'validation': validation_result,
            'final_inventory': inventory_ctrl.current_inventory,
            'final_capital': initial_capital + kpi_summary['total_pnl'],  # Capital + PnL
            'total_return_pct': (kpi_summary['total_pnl'] / initial_capital) * 100
        }
    }
    
    print("✅ Backtest terminé!")
    return results

def print_results(results: dict):
    """Affiche les résultats"""
    print(f"\n🎯 Résultats du Backtest V1-α")
    print("=" * 50)
    
    metadata = results['metadata']
    performance = results['performance']
    kpi = performance['kpi_summary']
    
    print(f"Symbol: {metadata['symbol']}")
    print(f"Ticks traités: {metadata['total_ticks']:,}")
    print(f"Quotes générées: {metadata['total_quotes']:,}")
    print(f"Trades exécutés: {metadata['total_trades']:,}")
    
    print(f"\n📊 KPIs:")
    print(f"  Fill Ratio: {kpi['fill_ratio']:.2%}")
    print(f"  Cancel Ratio: {kpi['cancel_ratio']:.2%}")
    print(f"  Spread Captured: {kpi['spread_captured_pct']:.1f}%")
    print(f"  RMS Inventory: {kpi['rms_inventory']:.4f}")
    print(f"  Total PnL: ${kpi['total_pnl']:+.2f}")
    print(f"  Total Return: {performance['total_return_pct']:+.2f}%")
    
    # Validation
    validation = performance['validation']
    print(f"\n✅ Validation V1-α: {validation['status']} ({validation['compliance_pct']:.0f}%)")
    print(f"  Targets atteints: {validation['targets_met']}/{validation['total_targets']}")
    
    print("=" * 50)

async def main():
    """Fonction principale"""
    print("🚀 Backtest V1-α Simplifié")
    print("=" * 40)
    
    try:
        # 1. Générer des données (30 minutes pour test rapide)
        market_data = generate_simple_data(duration_minutes=30)
        
        # 2. Exécuter le backtest
        results = await simple_backtest(market_data)
        
        # 3. Afficher les résultats
        print_results(results)
        
        print("\n🎉 Test terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def run_simple_backtest():
    """Interface compatible avec main.py pour le mode backtest"""
    try:
        # Générer des données (30 minutes pour test rapide)
        market_data = generate_simple_data(duration_minutes=30)
        
        # Exécuter le backtest de manière synchrone
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(simple_backtest(market_data))
        loop.close()
        
        # Adapter le format pour main.py
        kpi = results['performance']['kpi_summary']
        validation = results['performance']['validation']
        overall_pass = validation['status'] in ['EXCELLENT', 'GOOD']
        
        return {
            'total_pnl': kpi['total_pnl'],
            'final_inventory': results['performance']['final_inventory'],
            'validation': {
                'overall_pass': overall_pass
            }
        }
        
    except Exception as e:
        print(f"❌ Erreur backtest: {e}")
        return {
            'total_pnl': 0.0,
            'final_inventory': 0.0,
            'validation': {'overall_pass': False}
        }

if __name__ == "__main__":
    asyncio.run(main())
