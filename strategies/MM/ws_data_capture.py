"""
WebSocket Data Capture - Module standalone pour capture de données

Module indépendant pour capturer les données depth20@100ms de Binance.
Utilisé pour la collecte de données historiques, pas intégré au trading engine principal.

Usage: python ws_data_capture.py (en tant que script standalone)
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Callable
import logging
from .config import mm_config

class BinanceWSCapture:
    """Capture WebSocket des données depth20 Binance"""
    
    def __init__(self, symbols: List[str], on_data_callback: Optional[Callable] = None):
        self.symbols = [s.lower() for s in symbols]
        self.on_data_callback = on_data_callback
        self.ws_url = "wss://stream.binance.com:9443/ws/"
        self.data_buffer = []
        self.last_save_time = time.time()
        self.is_running = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create data directory
        Path(mm_config.data_dir).mkdir(parents=True, exist_ok=True)
        
    def create_stream_params(self) -> List[str]:
        """Crée les paramètres de stream pour depth20@100ms"""
        streams = []
        for symbol in self.symbols:
            streams.append(f"{symbol}@depth20@100ms")
        return streams
    
    def create_ws_url(self) -> str:
        """Crée l'URL WebSocket avec tous les streams"""
        streams = self.create_stream_params()
        combined_streams = '/'.join(streams)
        return f"{self.ws_url}{combined_streams}"
    
    async def ping_pong_handler(self, websocket):
        """Gère le ping/pong pour maintenir la connexion"""
        try:
            while self.is_running:
                await asyncio.sleep(20)  # Ping toutes les 20 secondes
                if websocket.open:
                    await websocket.ping()
        except Exception as e:
            self.logger.error(f"Erreur ping/pong: {e}")
    
    def process_depth_data(self, data: dict) -> dict:
        """Traite les données de depth reçues"""
        # Gérer les deux formats: stream unique ou combiné
        if 'stream' in data and 'data' in data:
            # Format stream combiné
            stream_name = data.get('stream', '')
            symbol = stream_name.split('@')[0].upper()
            depth_data = data.get('data', {})
        else:
            # Format stream direct - déduire le symbole depuis le prix
            # Binance envoie les données sans identifiant de symbole pour les streams directs
            # On identifie le symbole par analyse du prix
            depth_data = data
            
            # Extraire le meilleur prix pour identification
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
                    symbol = 'UNKNOWN'
            else:
                symbol = 'UNKNOWN'
        
        # Vérifier que nous avons des données valides
        if not depth_data.get('bids') or not depth_data.get('asks'):
            return {'symbol': symbol, 'timestamp': datetime.now(timezone.utc), 'bids': [], 'asks': []}
        
        # Timestamp de réception pour mesurer la latence (§3.2 V1-α)
        reception_timestamp = datetime.now(timezone.utc)
        
        processed = {
            'timestamp': reception_timestamp,
            'symbol': symbol,
            'last_update_id': depth_data.get('lastUpdateId', 0),
            'bids': [[float(bid[0]), float(bid[1])] for bid in depth_data.get('bids', [])],
            'asks': [[float(ask[0]), float(ask[1])] for ask in depth_data.get('asks', [])],
            'event_time': depth_data.get('E', 0),  # Event time from exchange
            'reception_time_ms': reception_timestamp.timestamp() * 1000,  # Pour calcul latence
        }
        
        return processed
    
    def calculate_metrics(self, processed_data: dict) -> dict:
        """Calcule des métriques utiles pour le MM"""
        bids = processed_data.get('bids', [])
        asks = processed_data.get('asks', [])
        
        if len(bids) == 0 or len(asks) == 0:
            # Retourner des métriques par défaut
            processed_data.update({
                'mid_price': 0.0,
                'spread': 0.0,
                'spread_bps': 0.0,
                'bid_volume_5': 0.0,
                'ask_volume_5': 0.0,
                'bid_volume_total': 0.0,
                'ask_volume_total': 0.0,
                'imbalance': 0.0,
            })
            return processed_data
        
        # Convertir en numpy arrays
        bids_array = np.array(bids)
        asks_array = np.array(asks)
        
        # Bids sont triés par prix décroissant (le meilleur = le plus haut)
        # Asks sont triés par prix croissant (le meilleur = le plus bas)
        best_bid = bids_array[0][0] if len(bids_array) > 0 else 0
        best_ask = asks_array[0][0] if len(asks_array) > 0 else 0
        
        if best_bid == 0 or best_ask == 0:
            processed_data.update({
                'mid_price': 0.0,
                'spread': 0.0,
                'spread_bps': 0.0,
                'bid_volume_5': 0.0,
                'ask_volume_5': 0.0,
                'bid_volume_total': 0.0,
                'ask_volume_total': 0.0,
                'imbalance': 0.0,
            })
            return processed_data
        
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Calcul de la latence mid-price (§3.2 V1-α : ≤50ms)
        event_time_ms = processed_data.get('event_time', 0)
        reception_time_ms = processed_data.get('reception_time_ms', 0)
        mid_price_latency_ms = reception_time_ms - event_time_ms if event_time_ms > 0 else 0
        
        metrics = {
            'mid_price': mid_price,
            'spread': spread,
            'spread_bps': (spread / mid_price) * 10000 if mid_price > 0 else 0,
            'mid_price_latency_ms': mid_price_latency_ms,  # Latence mid-price
            'bid_volume_5': sum(bids_array[:5, 1]) if len(bids_array) >= 5 else sum(bids_array[:, 1]),
            'ask_volume_5': sum(asks_array[:5, 1]) if len(asks_array) >= 5 else sum(asks_array[:, 1]),
            'bid_volume_total': sum(bids_array[:, 1]),
            'ask_volume_total': sum(asks_array[:, 1]),
            'imbalance': (sum(bids_array[:, 1]) - sum(asks_array[:, 1])) / (sum(bids_array[:, 1]) + sum(asks_array[:, 1])) if (sum(bids_array[:, 1]) + sum(asks_array[:, 1])) > 0 else 0,
        }
        
        processed_data.update(metrics)
        return processed_data
    
    def save_to_parquet(self, force: bool = False):
        """Sauvegarde les données en parquet"""
        current_time = time.time()
        if not force and current_time - self.last_save_time < mm_config.save_interval_seconds:
            return
            
        if not self.data_buffer:
            return
            
        try:
            df = pd.DataFrame(self.data_buffer)
            
            # Créer le nom de fichier avec la date
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"depth_data_{date_str}_{int(current_time)}.parquet"
            filepath = Path(mm_config.data_dir) / filename
            
            df.to_parquet(filepath, compression=mm_config.parquet_compression)
            
            self.logger.info(f"✅ Sauvegardé {len(self.data_buffer)} records dans {filename}")
            
            # Vider le buffer et mettre à jour le timestamp
            self.data_buffer.clear()
            self.last_save_time = current_time
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde parquet: {e}")
    
    async def message_handler(self, websocket):
        """Gère les messages reçus du WebSocket"""
        message_count = 0
        async for message in websocket:
            try:
                data = json.loads(message)
                message_count += 1
                
                # Traiter les données
                processed = self.process_depth_data(data)
                processed_with_metrics = self.calculate_metrics(processed)
                
                # Ajouter au buffer
                self.data_buffer.append(processed_with_metrics)
                
                # Callback optionnel pour traitement en temps réel
                if self.on_data_callback:
                    self.on_data_callback(processed_with_metrics)
                
                # Sauvegarde périodique
                self.save_to_parquet()
                
                # Log périodique
                if len(self.data_buffer) % 100 == 0:
                    symbol = processed_with_metrics.get('symbol', 'Unknown')
                    mid_price = processed_with_metrics.get('mid_price', 0)
                    spread = processed_with_metrics.get('spread', 0)
                    spread_bps = processed_with_metrics.get('spread_bps', 0)
                    self.logger.info(f"📊 {symbol}: Mid=${mid_price:.2f} Spread=${spread:.3f} ({spread_bps:.2f}bps)")
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Erreur JSON: {e}")
            except Exception as e:
                self.logger.error(f"❌ Erreur traitement message: {e}")
    
    async def connect_and_capture(self):
        """Connexion principale et capture des données"""
        url = self.create_ws_url()
        self.logger.info(f"🔗 Connexion à: {url}")
        
        reconnect_count = 0
        
        while self.is_running and reconnect_count < mm_config.reconnect_attempts:
            try:
                async with websockets.connect(url) as websocket:
                    self.logger.info(f"✅ Connecté! Symboles: {', '.join(self.symbols)}")
                    reconnect_count = 0  # Reset counter on successful connection
                    
                    # Lancer ping/pong en parallèle
                    ping_task = asyncio.create_task(self.ping_pong_handler(websocket))
                    message_task = asyncio.create_task(self.message_handler(websocket))
                    
                    # Attendre que l'une des tâches se termine
                    await asyncio.gather(ping_task, message_task, return_exceptions=True)
                    
            except websockets.exceptions.ConnectionClosed:
                reconnect_count += 1
                self.logger.warning(f"🔄 Connexion fermée, tentative {reconnect_count}/{mm_config.reconnect_attempts}")
                await asyncio.sleep(2 ** reconnect_count)  # Backoff exponentiel
                
            except Exception as e:
                reconnect_count += 1
                self.logger.error(f"❌ Erreur connexion: {e}")
                await asyncio.sleep(5)
        
        if reconnect_count >= mm_config.reconnect_attempts:
            self.logger.error("❌ Trop de tentatives de reconnexion, arrêt")
    
    async def start_capture(self):
        """Démarre la capture"""
        self.is_running = True
        self.logger.info("🚀 Démarrage capture WebSocket...")
        await self.connect_and_capture()
    
    def stop_capture(self):
        """Arrête la capture"""
        self.is_running = False
        self.save_to_parquet(force=True)  # Sauvegarde finale
        self.logger.info("🛑 Arrêt capture WebSocket")

# Fonction utilitaire pour tester la connexion
async def test_connection():
    """Test minimal de connexion WebSocket"""
    print("🧪 Test de connexion WebSocket...")
    
    def on_data(data):
        symbol = data.get('symbol', 'Unknown')
        mid_price = data.get('mid_price', 0)
        spread = data.get('spread', 0)
        spread_bps = data.get('spread_bps', 0)
        print(f"📊 {symbol}: Mid=${mid_price:.2f}, Spread=${spread:.3f} ({spread_bps:.2f}bps)")
    
    capture = BinanceWSCapture(['BTCUSDT'], on_data_callback=on_data)
    
    try:
        # Test pendant 30 secondes
        task = asyncio.create_task(capture.start_capture())
        await asyncio.sleep(30)
        capture.stop_capture()
        task.cancel()
        print("✅ Test connexion terminé")
        
    except KeyboardInterrupt:
        print("🛑 Test interrompu")
        capture.stop_capture()

if __name__ == "__main__":
    # Test direct du module
    asyncio.run(test_connection())
