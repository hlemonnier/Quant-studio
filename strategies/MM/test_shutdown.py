#!/usr/bin/env python3
"""
Test script pour vérifier l'arrêt propre du trading engine
"""

import asyncio
import signal
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.MM.trading_engine import TradingEngine
from strategies.MM.utils.config import mm_config

async def test_graceful_shutdown():
    """Test l'arrêt propre du trading engine"""
    print("🧪 Testing graceful shutdown...")
    
    # Créer un trading engine
    engine = TradingEngine(
        symbol="BTCUSDT",
        version="V1.5"
    )
    
    # Fonction pour gérer Ctrl+C
    def signal_handler(signum, frame):
        print("\n🛑 Received Ctrl+C, stopping gracefully...")
        asyncio.create_task(engine.stop())
    
    # Installer le gestionnaire de signal
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Démarrer le trading engine
        print("🚀 Starting trading engine...")
        task = asyncio.create_task(engine.run_trading_loop())
        
        # Attendre 3 secondes puis arrêter
        await asyncio.sleep(3)
        print("⏰ 3 seconds elapsed, stopping engine...")
        await engine.stop()
        
        # Attendre que la tâche se termine
        try:
            await task
        except asyncio.CancelledError:
            print("✅ Task cancelled cleanly")
        
        print("✅ Graceful shutdown test completed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Starting graceful shutdown test...")
    asyncio.run(test_graceful_shutdown())
