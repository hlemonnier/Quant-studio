from __future__ import annotations

"""Database replay integration for backtesting.

This module provides:
1. DBDepthStreamReplay – coroutine that replays historical depth update
   messages stored in a SQL table and feeds them to a LocalBook instance.
2. TradingEngineDBReplayIntegration – drop-in replacement for
   TradingEngineWSIntegration so the rest of the trading stack can work
   unchanged in *backtest* mode.

The implementation assumes a table structure with at least the columns:
    symbol (TEXT) – instrument symbol in upper-case, e.g. "BTCUSDT"
    event_time (INTEGER) – epoch milliseconds of the message (Binance field E)
    data (TEXT/JSON) – raw WebSocket JSON payload as string

Adjust the SQL query in _build_sql() if your schema differs.
"""

import asyncio
import json
import logging
from typing import Optional

try:
    from sqlalchemy import create_engine, text
except Exception:  # Optional dependency; only required for DB mode
    create_engine = None  # type: ignore
    text = None  # type: ignore
from datetime import datetime
from typing import Optional, Iterable

from .local_book import LocalBook

__all__ = [
    "DBDepthStreamReplay",
    "TradingEngineDBReplayIntegration",
    "CSVTopOfBookReplay",
    "TradingEngineCSVReplayIntegration",
]

logger = logging.getLogger(__name__)


class DBDepthStreamReplay:
    """Replay depth updates for a single symbol from a SQL table.

    Supports two formats:
    - json: a column with raw WS JSON payload (default)
    - top_of_book: columns with best_bid/best_ask and optional sizes
    """

    def __init__(
        self,
        symbol: str,
        local_book: LocalBook,
        db_uri: str,
        table: str = "depth_messages",
        data_format: str = "json",
        # Column names (for top_of_book format)
        symbol_col: str = "symbol",
        time_col: str = "event_time",
        bid_col: str = "best_bid",
        ask_col: str = "best_ask",
        bid_qty_col: str = "",
        ask_qty_col: str = "",
        qty_col: str = "",
        # Optional filters
        feed_col: str = "",
        feed_value: str = "",
        exchange_col: str = "",
        exchange_value: str = "",
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        speed: float = 1.0,
    ) -> None:
        self.symbol = symbol.upper()
        self._book = local_book
        if create_engine is None:
            raise RuntimeError("sqlalchemy est requis pour le mode DB. Installez-le ou utilisez MM_DATA_SOURCE=csv/websocket")
        self._engine = create_engine(db_uri)
        self._table = table
        self._start_ts = start_ts
        self._end_ts = end_ts
        self._speed = max(speed, 1e-6)  # prevent division by zero
        self._stop_event = asyncio.Event()
        self._log = logging.getLogger(f"DBReplay-{self.symbol}")
        self._format = (data_format or "json").lower()
        self._cols = {
            'symbol': symbol_col,
            'time': time_col,
            'bid': bid_col,
            'ask': ask_col,
            'bid_qty': bid_qty_col,
            'ask_qty': ask_qty_col,
            'qty': qty_col,
            'feed_col': feed_col,
            'feed_value': feed_value,
            'exchange_col': exchange_col,
            'exchange_value': exchange_value,
        }

    # ------------------------------------------------------------------
    # Public control helpers
    # ------------------------------------------------------------------
    async def start_replay(self) -> None:
        """Fetch rows ordered by event_time and feed LocalBook."""
        self._log.info(
            "🔄 Starting DB replay for %s from %s (speed ×%.2f)",
            self.symbol,
            self._table,
            self._speed,
        )

        stmt = self._build_sql()

        loop = asyncio.get_running_loop()
        # Use synchronous SQLAlchemy engine inside thread pool so we don't block
        rows = await loop.run_in_executor(
            None,
            lambda: list(self._fetch_rows(stmt)),
        )

        prev_ts: Optional[int] = None
        for row in rows:
            if self._stop_event.is_set():
                break

            if self._format == 'json':
                event_ts, raw_json = int(getattr(row, self._cols['time'])), getattr(row, 'data')  # type: ignore[attr-defined]
                diff_data = json.loads(raw_json)
            else:
                event_ts = int(getattr(row, self._cols['time']))
                bid = getattr(row, self._cols['bid'])
                ask = getattr(row, self._cols['ask'])
                bid_qty = getattr(row, self._cols['bid_qty']) if self._cols['bid_qty'] else None
                ask_qty = getattr(row, self._cols['ask_qty']) if self._cols['ask_qty'] else None
                qty = getattr(row, self._cols['qty']) if self._cols['qty'] else None

                bid_side = []
                ask_side = []
                try:
                    if bid is not None and bid != "":
                        q = bid_qty if bid_qty not in (None, "") else (qty if qty not in (None, "") else 0)
                        bid_side = [[str(float(bid)), str(float(q))]]
                    if ask is not None and ask != "":
                        q = ask_qty if ask_qty not in (None, "") else (qty if qty not in (None, "") else 0)
                        ask_side = [[str(float(ask)), str(float(q))]]
                except Exception:
                    bid_side, ask_side = [], []

                diff_data = {
                    'e': 'depthUpdate',
                    'E': event_ts,
                    's': self.symbol,
                    'U': 0,
                    'u': 0,
                    'b': bid_side,
                    'a': ask_side,
                }

            # Apply the diff to the local order-book representation
            try:
                self._book.apply_diff(diff_data)
            except Exception as exc:
                self._log.warning("⚠️ Failed to apply diff at %s: %s", event_ts, exc)

            if prev_ts is not None:
                await self._sleep((event_ts - prev_ts) / 1000.0)
            prev_ts = event_ts

        self._log.info("✅ DB replay finished for %s", self.symbol)

    def stop(self) -> None:
        """Request the replay coroutine to stop as soon as possible."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_sql(self):
        if self._format == 'json':
            time_col = 'event_time'
            sql = f"SELECT {time_col}, data FROM {self._table} WHERE {self._cols['symbol']} = :symbol"
        else:
            time_col = self._cols['time']
            cols = [time_col, self._cols['bid'], self._cols['ask']]
            if self._cols['bid_qty']:
                cols.append(self._cols['bid_qty'])
            if self._cols['ask_qty']:
                cols.append(self._cols['ask_qty'])
            if self._cols['qty']:
                cols.append(self._cols['qty'])
            select_cols = ", ".join(cols)
            sql = f"SELECT {select_cols} FROM {self._table} WHERE {self._cols['symbol']} = :symbol"

        params = {"symbol": self.symbol}
        # Optional filters
        if self._cols['feed_col'] and self._cols['feed_value']:
            sql += f" AND {self._cols['feed_col']} = :feed"
            params['feed'] = self._cols['feed_value']
        if self._cols['exchange_col'] and self._cols['exchange_value']:
            sql += f" AND {self._cols['exchange_col']} = :ex"
            params['ex'] = self._cols['exchange_value']

        if self._start_ts is not None:
            sql += f" AND {time_col} >= :start_ts"
            params["start_ts"] = self._start_ts
        if self._end_ts is not None:
            sql += f" AND {time_col} <= :end_ts"
            params["end_ts"] = self._end_ts
        sql += f" ORDER BY {time_col} ASC"
        return text(sql).bindparams(**params)

    # SQLAlchemy 1.x/2.x fetch compatibility
    def _fetch_rows(self, stmt) -> Iterable:
        try:
            # SA 2.0 style
            with self._engine.connect() as conn:
                result = conn.execution_options(stream_results=True).execute(stmt)
                return list(result)
        except Exception:
            # Fallback simple execute
            return list(self._engine.execute(stmt))

    async def _sleep(self, delay_s: float) -> None:
        # Apply speed factor (speed > 1 ⇒ faster)
        delay_s /= self._speed
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=max(delay_s, 0))
        except asyncio.TimeoutError:
            pass  # normal path – timer expired


class TradingEngineDBReplayIntegration:
    """Substitute for TradingEngineWSIntegration used in backtest mode."""

    def __init__(
        self,
        trading_engine,  # type: ignore[type-var]
        db_uri: str,
        table: str = "depth_messages",
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        speed: float = 1.0,
    ) -> None:
        self.trading_engine = trading_engine
        self.symbol = trading_engine.symbol
        self._log = logging.getLogger(f"DBIntegration-{self.symbol}")

        # Import config for columns/format
        from ..utils.config import mm_config

        self._replay = DBDepthStreamReplay(
            symbol=self.symbol,
            local_book=trading_engine.local_book,
            db_uri=db_uri,
            table=table,
            data_format=getattr(mm_config, 'db_format', 'json'),
            symbol_col=getattr(mm_config, 'db_symbol_col', 'symbol'),
            time_col=getattr(mm_config, 'db_time_col', 'event_time'),
            bid_col=getattr(mm_config, 'db_bid_col', 'best_bid'),
            ask_col=getattr(mm_config, 'db_ask_col', 'best_ask'),
            bid_qty_col=getattr(mm_config, 'db_bid_qty_col', ''),
            ask_qty_col=getattr(mm_config, 'db_ask_qty_col', ''),
            qty_col=getattr(mm_config, 'db_qty_col', ''),
            feed_col=getattr(mm_config, 'db_feed_col', ''),
            feed_value=getattr(mm_config, 'db_feed_value', ''),
            exchange_col=getattr(mm_config, 'db_exchange_col', ''),
            exchange_value=getattr(mm_config, 'db_exchange_value', ''),
            start_ts=start_ts,
            end_ts=end_ts,
            speed=speed,
        )
        self._task: Optional[asyncio.Task] = None

    async def start_integration(self):
        self._log.info("🚀 Starting DB replay integration for %s", self.symbol)
        self._task = asyncio.create_task(self._replay.start_replay())

    async def stop_integration(self):
        self._log.info("🛑 Stopping DB replay integration for %s", self.symbol)
        self._replay.stop()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # Provide an info helper similar to WSIntegration
    def get_integration_stats(self):
        return {
            "symbol": self.symbol,
            "active": self._task is not None and not self._task.done(),
            "source": "database",
            "updates": None,
            "errors": None,
            "success_rate": 0.0,
        }


class CSVTopOfBookReplay:
    """Replay top-of-book CSV as LocalBook diffs.

    Expected default column order (no header):
        timestamp, symbol, best_bid, best_ask, qty, feed, exchange

    Or provide header and set column names via mm_config.
    """

    def __init__(
        self,
        symbol: str,
        local_book: LocalBook,
        csv_path: str,
        time_col: str = 'timestamp',
        symbol_col: str = 'symbol',
        bid_col: str = 'best_bid',
        ask_col: str = 'best_ask',
        bid_qty_col: str = '',
        ask_qty_col: str = '',
        qty_col: str = '',
        feed_col: str = '',
        feed_value: str = '',
        exchange_col: str = '',
        exchange_value: str = '',
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        speed: float = 1.0,
    ) -> None:
        self.symbol = symbol.upper()
        self._book = local_book
        self._path = csv_path
        self._cols = {
            'time': time_col,
            'symbol': symbol_col,
            'bid': bid_col,
            'ask': ask_col,
            'bid_qty': bid_qty_col,
            'ask_qty': ask_qty_col,
            'qty': qty_col,
            'feed_col': feed_col,
            'feed_value': feed_value,
            'exchange_col': exchange_col,
            'exchange_value': exchange_value,
        }
        self._start_ts = start_ts
        self._end_ts = end_ts
        self._speed = max(speed, 1e-6)
        self._stop_event = asyncio.Event()
        self._log = logging.getLogger(f"CSVReplay-{self.symbol}")
        self._has_header = None  # type: Optional[bool]

    async def start_replay(self) -> None:
        self._log.info("🔄 Starting CSV replay for %s from %s (speed ×%.2f)", self.symbol, self._path, self._speed)

        prev_ts: Optional[int] = None
        with open(self._path, 'r') as f:
            first_line = f.readline().strip()
            # Detect header: if first token contains any letter except symbol names, assume header
            self._has_header = any(h in first_line.lower() for h in [
                'timestamp', 'time', 'best_bid', 'best_ask', 'symbol'
            ])
            if self._has_header:
                header = [h.strip() for h in first_line.split(',')]
                idx = {name: i for i, name in enumerate(header)}
            else:
                # no header: reset pointer to beginning for processing the first line
                f.seek(0)
                idx = {}

            for line in f:
                if self._stop_event.is_set():
                    break
                parts = [p.strip() for p in line.strip().split(',')]
                if not parts or len(parts) < 4:
                    continue

                # Extract fields either by header names or fixed positions
                if self._has_header:
                    try:
                        t_str = parts[idx.get(self._cols['time'], idx.get('timestamp'))]
                        sym = parts[idx.get(self._cols['symbol'], idx.get('symbol'))].upper()
                        bid_s = parts[idx.get(self._cols['bid'], idx.get('best_bid'))]
                        ask_s = parts[idx.get(self._cols['ask'], idx.get('best_ask'))]
                        bid_qty_s = parts[idx[self._cols['bid_qty']]] if self._cols['bid_qty'] and self._cols['bid_qty'] in idx else ''
                        ask_qty_s = parts[idx[self._cols['ask_qty']]] if self._cols['ask_qty'] and self._cols['ask_qty'] in idx else ''
                        qty_s = parts[idx[self._cols['qty']]] if self._cols['qty'] and self._cols['qty'] in idx else ''
                        if self._cols['feed_col'] and self._cols['feed_col'] in idx and self._cols['feed_value']:
                            feed_val = parts[idx[self._cols['feed_col']]]
                            if feed_val != self._cols['feed_value']:
                                continue
                        if self._cols['exchange_col'] and self._cols['exchange_col'] in idx and self._cols['exchange_value']:
                            ex_val = parts[idx[self._cols['exchange_col']]]
                            if ex_val != self._cols['exchange_value']:
                                continue
                    except Exception:
                        continue
                else:
                    # Fixed order: ts, symbol, best_bid, best_ask, qty, feed, exchange
                    t_str = parts[0]
                    sym = parts[1].upper()
                    bid_s = parts[2] if len(parts) > 2 else ''
                    ask_s = parts[3] if len(parts) > 3 else ''
                    qty_s = parts[4] if len(parts) > 4 else ''
                    bid_qty_s = qty_s
                    ask_qty_s = qty_s
                    # Optional filters on feed/exchange
                    if self._cols['feed_value'] and len(parts) > 5 and parts[5] != self._cols['feed_value']:
                        continue
                    if self._cols['exchange_value'] and len(parts) > 6 and parts[6] != self._cols['exchange_value']:
                        continue

                if sym != self.symbol:
                    continue

                # Parse timestamp → epoch ms
                try:
                    # Accept format like "2025-08-11 09:23:45.315000 +00:00"
                    dt = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S.%f %z')
                    event_ts = int(dt.timestamp() * 1000)
                except Exception:
                    # Try without timezone
                    try:
                        dt = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S.%f')
                        event_ts = int(dt.timestamp() * 1000)
                    except Exception:
                        continue

                if self._start_ts and event_ts < self._start_ts:
                    continue
                if self._end_ts and event_ts > self._end_ts:
                    break

                # Build diff
                bid_side = []
                ask_side = []
                try:
                    if bid_s:
                        q = bid_qty_s or qty_s or '0'
                        bid_side = [[str(float(bid_s)), str(float(q))]]
                    if ask_s:
                        q = ask_qty_s or qty_s or '0'
                        ask_side = [[str(float(ask_s)), str(float(q))]]
                except Exception:
                    pass

                diff_data = {
                    'e': 'depthUpdate',
                    'E': event_ts,
                    's': self.symbol,
                    'U': 0,
                    'u': 0,
                    'b': bid_side,
                    'a': ask_side,
                }

                try:
                    self._book.apply_diff(diff_data)
                except Exception as exc:
                    self._log.warning("⚠️ Failed to apply CSV diff at %s: %s", event_ts, exc)

                if prev_ts is not None:
                    await self._sleep((event_ts - prev_ts) / 1000.0)
                prev_ts = event_ts

        self._log.info("✅ CSV replay finished for %s", self.symbol)

    def stop(self) -> None:
        self._stop_event.set()

    async def _sleep(self, delay_s: float) -> None:
        delay_s /= self._speed
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=max(delay_s, 0))
        except asyncio.TimeoutError:
            pass


class TradingEngineCSVReplayIntegration:
    """CSV replay integration for TradingEngine."""

    def __init__(
        self,
        trading_engine,
        csv_path: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        speed: float = 1.0,
    ) -> None:
        self.trading_engine = trading_engine
        self.symbol = trading_engine.symbol
        self._log = logging.getLogger(f"CSVIntegration-{self.symbol}")

        from ..utils.config import mm_config

        self._replay = CSVTopOfBookReplay(
            symbol=self.symbol,
            local_book=trading_engine.local_book,
            csv_path=csv_path,
            time_col=getattr(mm_config, 'csv_time_col', 'timestamp'),
            symbol_col=getattr(mm_config, 'csv_symbol_col', 'symbol'),
            bid_col=getattr(mm_config, 'csv_bid_col', 'best_bid'),
            ask_col=getattr(mm_config, 'csv_ask_col', 'best_ask'),
            bid_qty_col=getattr(mm_config, 'csv_bid_qty_col', ''),
            ask_qty_col=getattr(mm_config, 'csv_ask_qty_col', ''),
            qty_col=getattr(mm_config, 'csv_qty_col', ''),
            feed_col=getattr(mm_config, 'csv_feed_col', ''),
            feed_value=getattr(mm_config, 'csv_feed_value', ''),
            exchange_col=getattr(mm_config, 'csv_exchange_col', ''),
            exchange_value=getattr(mm_config, 'csv_exchange_value', ''),
            start_ts=start_ts,
            end_ts=end_ts,
            speed=speed,
        )
        self._task: Optional[asyncio.Task] = None

    async def start_integration(self):
        self._log.info("🚀 Starting CSV replay integration for %s", self.symbol)
        self._task = asyncio.create_task(self._replay.start_replay())

    async def stop_integration(self):
        self._log.info("🛑 Stopping CSV replay integration for %s", self.symbol)
        self._replay.stop()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_integration_stats(self):
        return {
            'symbol': self.symbol,
            'active': self._task is not None and not self._task.done(),
            'source': 'csv',
            'updates': None,
            'errors': None,
            'success_rate': 0.0,
        }
