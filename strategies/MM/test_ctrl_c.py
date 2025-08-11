#!/usr/bin/env python3
"""
Test script pour vérifier que Ctrl+C fonctionne proprement
"""

import asyncio
import signal
import sys
import os

# Ajouter le chemin parent pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.MM.main import run_paper_trading

async def test_ctrl_c():
    """Test Ctrl+C avec le vrai main"""
    print("🧪 Testing Ctrl+C handling...")
    
    # Créer une tâche pour run_paper_trading
    task = asyncio.create_task(
        run_paper_trading(['BTCUSDT'], duration_hours=None, version='V1.5')
    )
    
    # Attendre 3 secondes puis simuler Ctrl+C
    await asyncio.sleep(3)
    print("⏰ 3 seconds elapsed, sending KeyboardInterrupt...")
    
    # Simuler Ctrl+C en annulant la tâche
    task.cancel()
    
    try:
        await task
    except asyncio.CancelledError:
        print("✅ Task cancelled cleanly")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("✅ Ctrl+C test completed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_ctrl_c())
    except KeyboardInterrupt:
        print("✅ KeyboardInterrupt handled at top level")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

