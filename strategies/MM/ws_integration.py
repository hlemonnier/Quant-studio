"""
WebSocket Integration - Connecte les données WebSocket au LocalBook

Ce module intègre le BinanceWSCapture au TradingEngine pour que le LocalBook
reçoive les mises à jour en temps réel et que le DI fonctionne correctement.
"""

import asyncio
import logging
import json
import websockets
from typing import Dict, Optional, Callable
from .ws_data_capture import BinanceWSCapture
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
        combined_streams = '/'.join(streams)
        return f"{self.ws_url}{combined_streams}"
    
    async def start_capture(self):
        """Démarre la capture WebSocket"""
        self.is_running = True
        url = self.create_stream_url()
        self.logger.info(f"🚀 Connecting to {url}")
        
        try:
            async with websockets.connect(url) as websocket:
                self.logger.info("✅ WebSocket connected for depth diffs")
                
                async for message in websocket:
                    if not self.is_running:
                        break
                        
                    try:
                        data = json.loads(message)
                        
                        # Traiter les données et appeler le callback
                        if self.on_data_callback:
                            self.on_data_callback(data)
                            
                    except Exception as e:
                        self.logger.error(f"❌ Error processing message: {e}")
                        
        except Exception as e:
            self.logger.error(f"❌ WebSocket connection error: {e}")
        finally:
            self.is_running = False
            self.logger.info("🛑 WebSocket connection closed")
    
    async def stop_capture(self):
        """Arrête la capture"""
        self.is_running = False


class WSLocalBookIntegration:
    """Intègre WebSocket capture avec LocalBook pour mises à jour temps réel"""
    
    def __init__(self, symbols: list, local_books: Dict[str, LocalBook]):
        """
        Args:
            symbols: Liste des symboles à surveiller
            local_books: Dict {symbol: LocalBook} des books locaux à mettre à jour
        """
        self.symbols = symbols
        self.local_books = local_books
        self.logger = logging.getLogger(f"WSIntegration-{'-'.join(symbols)}")
        
        # Statistiques de mise à jour
        self.update_counts = {symbol: 0 for symbol in symbols}
        self.error_counts = {symbol: 0 for symbol in symbols}
        
        # Créer le WebSocket capture spécialisé pour les diffs depth
        self.ws_capture = BinanceDepthStreamCapture(
            symbols=symbols,
            on_data_callback=self._on_depth_update
        )
        
        self.logger.info(f"🔌 WebSocket integration initialized for {symbols}")
    
    def _on_depth_update(self, depth_data: dict):
        """Callback appelé à chaque mise à jour depth WebSocket"""
        try:
            symbol = depth_data.get('symbol')
            if not symbol or symbol not in self.local_books:
                return
            
            # Convertir les données au format attendu par LocalBook.apply_diff()
            diff_data = self._convert_to_diff_format(depth_data)
            
            # Appliquer la mise à jour au local book
            success = self.local_books[symbol].apply_ws_update(symbol, diff_data)
            
            if success:
                self.update_counts[symbol] += 1
                
                # Log périodique des mises à jour
                if self.update_counts[symbol] % 100 == 0:
                    self.logger.debug(f"✅ {symbol}: {self.update_counts[symbol]} updates applied")
            else:
                self.error_counts[symbol] += 1
                if self.error_counts[symbol] % 10 == 0:
                    self.logger.warning(f"⚠️ {symbol}: {self.error_counts[symbol]} update errors")
                    
        except Exception as e:
            self.logger.error(f"❌ Error processing depth update: {e}")
    
    def _convert_to_diff_format(self, depth_data: dict) -> dict:
        """Convertit les données WebSocket au format diff attendu par LocalBook"""
        # Les données depth@100ms de Binance arrivent déjà au bon format :
        # {
        #   "e": "depthUpdate",
        #   "E": 1672515782136,
        #   "s": "BNBBTC", 
        #   "U": 157,
        #   "u": 160,
        #   "b": [["0.0024", "10"]],
        #   "a": [["0.0026", "100"]]
        # }
        
        # Le format est déjà compatible avec LocalBook.apply_diff()
        return {
            'U': depth_data.get('U', 0),  # first_update_id
            'u': depth_data.get('u', 0),  # final_update_id  
            'b': depth_data.get('b', []), # bids updates (déjà en strings)
            'a': depth_data.get('a', []), # asks updates (déjà en strings)
            'E': depth_data.get('E', 0),  # event time
        }
    
    async def start(self):
        """Démarre la capture WebSocket et l'intégration"""
        self.logger.info("🚀 Starting WebSocket integration...")
        await self.ws_capture.start_capture()
    
    async def stop(self):
        """Arrête la capture WebSocket"""
        self.logger.info("🛑 Stopping WebSocket integration...")
        await self.ws_capture.stop_capture()
    
    def get_stats(self) -> Dict[str, dict]:
        """Retourne les statistiques de mise à jour"""
        return {
            symbol: {
                'updates': self.update_counts[symbol],
                'errors': self.error_counts[symbol],
                'success_rate': (
                    self.update_counts[symbol] / 
                    max(1, self.update_counts[symbol] + self.error_counts[symbol])
                ) * 100
            }
            for symbol in self.symbols
        }


class TradingEngineWSIntegration:
    """Intégration WebSocket spécifique pour TradingEngine"""
    
    def __init__(self, trading_engine):
        """
        Args:
            trading_engine: Instance de TradingEngine à intégrer
        """
        self.engine = trading_engine
        self.symbol = trading_engine.symbol
        self.logger = logging.getLogger(f"WSIntegration-{self.symbol}")
        
        # Créer l'intégration pour ce symbole
        self.integration = WSLocalBookIntegration(
            symbols=[self.symbol],
            local_books={self.symbol: self.engine.local_book}
        )
        
        self.logger.info(f"🔌 Trading engine WebSocket integration ready for {self.symbol}")
    
    async def start_integration(self):
        """Démarre l'intégration WebSocket pour ce trading engine"""
        self.logger.info(f"🚀 Starting WebSocket integration for {self.symbol}...")
        
        # Démarrer l'intégration en arrière-plan
        self.integration_task = asyncio.create_task(self.integration.start())
        
        # Attendre un peu pour que la connexion s'établisse
        await asyncio.sleep(2)
        
        self.logger.info(f"✅ WebSocket integration active for {self.symbol}")
    
    async def stop_integration(self):
        """Arrête l'intégration WebSocket"""
        if hasattr(self, 'integration_task'):
            self.integration_task.cancel()
            try:
                await self.integration_task
            except asyncio.CancelledError:
                pass
        
        await self.integration.stop()
        self.logger.info(f"🛑 WebSocket integration stopped for {self.symbol}")
    
    def get_integration_stats(self) -> dict:
        """Retourne les stats d'intégration pour ce symbole"""
        stats = self.integration.get_stats()
        return stats.get(self.symbol, {})
