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
                
                message_count = 0
                async for message in websocket:
                    if not self.is_running:
                        break
                    
                    message_count += 1
                    
                    # Log des premiers messages pour diagnostic
                    if message_count <= 5:
                        self.logger.info(f"📨 Received WebSocket message #{message_count}")
                        self.logger.info(f"📨 Message length: {len(message)} chars")
                        
                    try:
                        data = json.loads(message)
                        
                        # Log des premières données parsées
                        if message_count <= 3:
                            self.logger.info(f"📊 Parsed data keys: {list(data.keys())}")
                        
                        # Traiter les données pour extraire le symbole
                        processed_data = self._process_depth_data(data)
                        
                        # Log des données traitées
                        if message_count <= 3:
                            if processed_data:
                                self.logger.info(f"✅ Processed data for symbol: {processed_data.get('symbol')}")
                                self.logger.info(f"✅ Processed data keys: {list(processed_data.keys())}")
                            else:
                                self.logger.warning(f"⚠️ No processed data returned")
                        
                        # Appeler le callback avec les données traitées
                        if self.on_data_callback and processed_data:
                            if message_count <= 3:
                                self.logger.info(f"🔄 Calling callback with processed data")
                            self.on_data_callback(processed_data)
                        elif message_count <= 3:
                            self.logger.warning(f"⚠️ No callback or no processed data - callback: {self.on_data_callback is not None}, data: {processed_data is not None}")
                            
                    except Exception as e:
                        self.logger.error(f"❌ Error processing message: {e}")
                        if message_count <= 5:
                            self.logger.error(f"❌ Raw message: {message[:200]}...")
                        
        except Exception as e:
            self.logger.error(f"❌ WebSocket connection error: {e}")
        finally:
            self.is_running = False
            self.logger.info("🛑 WebSocket connection closed")
    
    def _process_depth_data(self, data: dict) -> Optional[dict]:
        """Traite les données WebSocket pour extraire le symbole"""
        try:
            # Format Binance depth update direct (le plus courant)
            if 'e' in data and data.get('e') == 'depthUpdate' and 's' in data:
                # Données directes de Binance avec symbole inclus
                symbol = data.get('s', '').upper()
                if symbol in [s.upper() for s in self.symbols]:
                    data['symbol'] = symbol
                    return data
                else:
                    self.logger.warning(f"⚠️ Received data for unknown symbol: {symbol}")
                    return None
            
            # Format stream combiné avec identifiant
            elif 'stream' in data and 'data' in data:
                stream_name = data.get('stream', '')
                symbol = stream_name.split('@')[0].upper()
                depth_data = data.get('data', {})
                
                # Ajouter le symbole aux données
                depth_data['symbol'] = symbol
                return depth_data
            
            # Format stream direct - identifier par prix (fallback)
            elif 'bids' in data and 'asks' in data:
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                
                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Identifier le symbole par range de prix
                    if mid_price > 10000:  # BTC range
                        symbol = 'BTCUSDT'
                    elif mid_price > 1000:  # ETH range  
                        symbol = 'ETHUSDT'
                    else:
                        # Fallback: utiliser le premier symbole configuré
                        symbol = self.symbols[0].upper() if self.symbols else 'UNKNOWN'
                    
                    # Ajouter le symbole aux données
                    data['symbol'] = symbol
                    return data
            
            # Format non reconnu
            else:
                self.logger.warning(f"⚠️ Unknown data format, keys: {list(data.keys())}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Error processing depth data: {e}")
            return None

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
        
        # Créer un logger de fichier pour les mises à jour WebSocket
        import os
        from datetime import datetime
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/websocket_updates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.file_logger = logging.getLogger(f"WSUpdates-{'-'.join(symbols)}")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.file_logger.addHandler(file_handler)
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.info(f"🚀 WebSocket updates logging started for {symbols}")
        
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
                self.logger.warning(f"⚠️ Received data for unknown symbol: {symbol}")
                return
            
            # Log des premières mises à jour pour diagnostic
            if self.update_counts[symbol] < 10:
                self.logger.info(f"🔄 Processing WebSocket update #{self.update_counts[symbol]+1} for {symbol}")
                self.logger.info(f"📊 Raw WebSocket data keys: {list(depth_data.keys())}")
                if 'bids' in depth_data and 'asks' in depth_data:
                    self.logger.info(f"📈 Bids count: {len(depth_data.get('bids', []))}, Asks count: {len(depth_data.get('asks', []))}")
                    if depth_data.get('bids'):
                        self.logger.info(f"📈 First bid: {depth_data['bids'][0]}")
                    if depth_data.get('asks'):
                        self.logger.info(f"📉 First ask: {depth_data['asks'][0]}")
            
            # Convertir les données au format attendu par LocalBook.apply_diff()
            diff_data = self._convert_to_diff_format(depth_data)
            
            # Log du format converti
            if self.update_counts[symbol] < 5:
                self.logger.info(f"🔄 Converted diff data keys: {list(diff_data.keys())}")
            
            # Logger dans le fichier pour diagnostic
            self.file_logger.info(f"SYMBOL: {symbol}")
            self.file_logger.info(f"RAW_DATA: {depth_data}")
            self.file_logger.info(f"CONVERTED_DATA: {diff_data}")
            
            # Appliquer la mise à jour au local book
            success = self.local_books[symbol].apply_ws_update(symbol, diff_data)
            
            # Logger le résultat
            self.file_logger.info(f"UPDATE_RESULT: {success}")
            if success:
                # Logger l'état du book après mise à jour
                book = self.local_books[symbol]
                best_bid = max(book.bids.keys()) if book.bids else 0
                best_ask = min(book.asks.keys()) if book.asks else 0
                self.file_logger.info(f"BOOK_STATE: Best bid={best_bid:.2f}, Best ask={best_ask:.2f}, Updates={book.update_count}")
            
            if success:
                self.update_counts[symbol] += 1
                
                # Log périodique des mises à jour
                if self.update_counts[symbol] % 100 == 0:
                    self.logger.info(f"✅ {symbol}: {self.update_counts[symbol]} updates applied successfully")
                elif self.update_counts[symbol] <= 5:
                    self.logger.info(f"✅ {symbol}: Update #{self.update_counts[symbol]} applied successfully")
            else:
                self.error_counts[symbol] += 1
                self.logger.warning(f"❌ {symbol}: Failed to apply update #{self.error_counts[symbol]}")
                if self.error_counts[symbol] % 10 == 0:
                    self.logger.warning(f"⚠️ {symbol}: {self.error_counts[symbol]} update errors total")
                    
        except Exception as e:
            self.logger.error(f"❌ Error processing depth update: {e}")
            self.file_logger.error(f"ERROR: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.file_logger.error(f"TRACEBACK: {traceback.format_exc()}")
    
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
