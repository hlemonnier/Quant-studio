"""
WebSocket Manager - Unified WebSocket handling for Market Making

This module consolidates all WebSocket functionality:
1. BinanceDepthStreamCapture: Real-time depth diffs for trading
2. WSLocalBookIntegration: Connects WebSocket to LocalBook
3. TradingEngineWSIntegration: Main integration for TradingEngine

Replaces: ws_data_capture.py + ws_integration.py
"""

import asyncio
import logging
import json
import websockets
from typing import Dict, Optional, Callable
from .local_book import LocalBook


class BinanceDepthStreamCapture:
    """WebSocket capture spécialisé pour les diffs depth incrémentaux"""
    
    def __init__(self, symbols: list, on_data_callback: Optional[Callable] = None):
        self.symbols = [s.lower() for s in symbols]
        self.on_data_callback = on_data_callback
        self.ws_url = "wss://stream.binance.com:9443/ws/"
        self.logger = logging.getLogger(f"DepthStream-{'-'.join(symbols)}")
        self.is_running = False
        
    def create_stream_url(self) -> str:
        """Crée l'URL pour les streams depth (diffs incrémentaux)"""
        # Utiliser depth@100ms pour les diffs incrémentaux au lieu de depth20@100ms
        streams = [f"{symbol}@depth@100ms" for symbol in self.symbols]
        if len(streams) == 1:
            return f"{self.ws_url}{streams[0]}"
        else:
            # Multi-stream
            stream_names = "/".join(streams)
            return f"{self.ws_url}{stream_names}"
    
    async def start_capture(self):
        """Démarre la capture WebSocket"""
        url = self.create_stream_url()
        self.logger.info(f"🚀 Connecting to {url}")
        
        try:
            self.is_running = True
            async with websockets.connect(url) as websocket:
                self.logger.info("✅ WebSocket connected for depth diffs")
                
                async for message in websocket:
                    if not self.is_running:
                        break
                        
                    try:
                        data = json.loads(message)
                        
                        # Traiter les données
                        if self.on_data_callback:
                            await self.on_data_callback(data)
                            
                    except json.JSONDecodeError as e:
                        self.logger.error(f"❌ JSON decode error: {e}")
                    except Exception as e:
                        self.logger.error(f"❌ Error processing message: {e}")
                        
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("🛑 WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"❌ WebSocket error: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Arrête la capture"""
        self.is_running = False


class WSLocalBookIntegration:
    """Intègre WebSocket capture avec LocalBook pour mises à jour temps réel"""
    
    def __init__(self, symbols: list, local_books: Dict[str, LocalBook]):
        """
        symbols: Liste des symboles à suivre
        local_books: Dict {symbol: LocalBook} des books locaux à mettre à jour
        """
        self.symbols = [s.upper() for s in symbols]
        self.local_books = local_books
        self.logger = logging.getLogger(f"WSIntegration-{'-'.join(self.symbols)}")
        
        # Créer le capture WebSocket avec callback
        self.ws_capture = BinanceDepthStreamCapture(
            symbols, 
            on_data_callback=self.on_websocket_data
        )
        
    async def on_websocket_data(self, data: dict):
        """Callback appelé quand des données WebSocket arrivent"""
        try:
            # Extraire le symbole et les données
            if 'stream' in data:
                # Format multi-stream
                stream_name = data['stream']
                symbol = stream_name.split('@')[0].upper()
                ws_data = data['data']
            else:
                # Format single stream - déduire le symbole du premier élément
                if len(self.symbols) == 1:
                    symbol = self.symbols[0]
                    ws_data = data
                else:
                    self.logger.warning("⚠️ Cannot determine symbol for multi-stream data")
                    return
            
            # Convertir les données au format attendu par LocalBook.apply_diff()
            diff_data = self.convert_ws_to_diff_format(ws_data)
            
            # Appliquer au LocalBook correspondant
            if symbol in self.local_books:
                success = self.local_books[symbol].apply_diff(diff_data)
                if not success:
                    self.logger.warning(f"⚠️ Failed to apply diff for {symbol}")
            else:
                self.logger.warning(f"⚠️ No LocalBook found for symbol {symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Error processing WebSocket data: {e}")
    
    async def start_integration(self):
        """Démarre l'intégration WebSocket"""
        self.logger.info(f"🚀 Starting WebSocket integration for {self.symbols}...")
        await self.ws_capture.start_capture()
    
    def stop_integration(self):
        """Arrête l'intégration WebSocket"""
        self.logger.info("🛑 Stopping WebSocket integration...")
        self.ws_capture.stop()
    
    def convert_ws_to_diff_format(self, ws_data: dict) -> dict:
        """Convertit les données WebSocket au format diff attendu par LocalBook"""
        try:
            # Les données WebSocket depth@100ms ont déjà le bon format:
            # {
            #   "e": "depthUpdate",
            #   "E": 1672515782136,
            #   "s": "BTCUSDT", 
            #   "U": 157,
            #   "u": 160,
            #   "b": [["0.0024", "10"]],
            #   "a": [["0.0026", "0"]]
            # }
            
            # Le format est déjà compatible avec LocalBook.apply_diff()
            return ws_data
            
        except Exception as e:
            self.logger.error(f"❌ Error converting WebSocket data: {e}")
            return {}


class TradingEngineWSIntegration:
    """Intégration WebSocket principale pour le TradingEngine"""
    
    def __init__(self, trading_engine):
        """
        trading_engine: Instance du TradingEngine à intégrer
        """
        self.trading_engine = trading_engine
        self.symbol = trading_engine.symbol
        self.logger = logging.getLogger(f"WSIntegration-{self.symbol}")
        
        # Créer l'intégration LocalBook
        local_books = {self.symbol: trading_engine.local_book}
        self.integration = WSLocalBookIntegration(
            symbols=[self.symbol],
            local_books=local_books
        )
        
        self.logger.info(f"🔌 WebSocket integration initialized for {self.symbol}")
    
    async def start_integration(self):
        """Démarre l'intégration WebSocket"""
        self.logger.info(f"🚀 Starting WebSocket integration for {self.symbol}...")
        
        # Créer une tâche pour l'intégration WebSocket
        self.ws_task = asyncio.create_task(
            self.integration.start_integration()
        )
        
        self.logger.info(f"✅ WebSocket integration active for {self.symbol}")
    
    async def stop_integration(self):
        """Arrête l'intégration WebSocket"""
        self.logger.info("🛑 Stopping WebSocket integration...")
        
        # Arrêter l'intégration
        self.integration.stop_integration()
        
        # Annuler la tâche si elle existe et attendre qu'elle se termine proprement
        if hasattr(self, 'ws_task') and not self.ws_task.done():
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                # C'est normal, la tâche a été annulée
                pass
            except Exception as e:
                self.logger.warning(f"⚠️ Error during task cleanup: {e}")
        
        self.logger.info(f"🛑 WebSocket integration stopped for {self.symbol}")


# Test standalone
if __name__ == "__main__":
    import asyncio
    
    async def test_websocket():
        """Test simple du WebSocket"""
        
        def on_data(data):
            print(f"📊 Received: {data}")
        
        capture = BinanceDepthStreamCapture(['BTCUSDT'], on_data)
        
        try:
            await capture.start_capture()
        except KeyboardInterrupt:
            print("🛑 Stopping...")
            capture.stop()
    
    print("🧪 Testing WebSocket Manager...")
    asyncio.run(test_websocket())
