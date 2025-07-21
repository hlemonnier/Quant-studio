from __future__ import (absolute_import, division, print_function, unicode_literals)
import backtrader as bt
from datetime import datetime
import sys
import os

# Append project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from strategies.Signal.signal_generator import SignalGenerator
from strategies.Risk_Management.Order_Management.trailing_strategy import TrailingStrategy
from strategies.Risk_Management.Money_Management.position_sizing_strategy import FixedSize, RiskAdjustedSizer
from strategies.Signal.Indicators.build_in_indicators import EMABollingerBands, CustomROCR, KalmanFilterIndicator, SqrtWeightedMeanVolume



def custom_grid(first_price, last_order_down=0.5, last_order_up=1, down_grid_len=50, up_grid_len=50):
    """
    Function that creates two arrays: grid_buy and grid_sell, and their corresponding stop-loss levels.

    Args:
        first_price (float): Initial price.
        last_order_down (float, optional): Percentage of the last grid buy. Defaults to 0.5.
        last_order_up (float, optional): Percentage of the last grid sell. Defaults to 1.
        down_grid_len (int, optional): Initial length of grid buy. Defaults to 50.
        up_grid_len (int, optional): Initial length of grid sell. Defaults to 50.

    Returns:
        tuple: Two arrays of float for price (grid_buy, grid_sell) and stop-loss levels (stop_loss_buy, stop_loss_sell).
    """
    down_pct_unity = last_order_down / down_grid_len
    up_pct_unity = last_order_up / up_grid_len

    grid_sell = []
    grid_buy = []

    for i in range(down_grid_len):
        grid_buy.append(first_price - first_price * down_pct_unity * (i + 1))

    for i in range(up_grid_len):
        grid_sell.append(first_price + first_price * up_pct_unity * (i + 1))

    # Define stop-loss levels
    stop_loss_buy = grid_buy[-1] - first_price * down_pct_unity  # Lower than the lowest buy level
    stop_loss_sell = grid_sell[-1] + first_price * up_pct_unity  # Higher than the highest sell level

    return grid_buy, grid_sell, stop_loss_buy, stop_loss_sell



class Grid(bt.Strategy):
    """
    A grid trading strategy that dynamically adjusts to volatility and includes stop-loss levels.

    Attributes:
        log_entries (dict): Tracks log entries.
        transactions (dict): Tracks transactions.
    """
    params = dict(
        verbose=True,
        args = 'args', 
        last_order_down=0.1,
        last_order_up=0.25,
        down_grid_len=5
        ,
        up_grid_len=10,
        
        sizers_type='RiskAdjustedSizer',
        max_risk_per_trade=0.0001,
        stop_loss_deviation=1,
        max_authorized_position=0.05,
        atr_length=10,  # in bars

        brackets_validity=100,

        # Signal
        sma_period = 150,
        sma_slope_limit = 15,
        sma_slope_period = 10,

        adx_period =  14,
        adx_strength = 30, 
        ema_period1 = 27,

        use_stochastic_oscillator = True,
        stochastic_oscillator_period = 10,
        stochastic_oscillator_period_dfast = 2,    
        use_ema_bb = True,
        ema_bb_period = 25,
        bb_top_devfactor = 2,
        bb_bot_devfactor = 2,
        use_machine_learning = True,
        model_path = 'C:/Users/SolalDanan/Trading Signal/logistic_regression_model.joblib',
        use_rsi = True,
        rsi_period = 14,
        rsi_lower_threshold = 25,
        rsi_upper_threshold = 75,

        # Trailing Strategy
        use_trailing_strategy = True,
        trail_stop_pct = 1.3,   # Trailing stop at 0.5% above entry price
        trail_stop_abs = None,  # Alternatively, could set an absolute value
        secure_pct = 35,        # % of th eprofit to securize
        atr_multiplier = 3.6,


        )

    def log(self, txt, dt=None):
        """Logging function for this strategy."""
        dt = dt or self.datas[0].datetime.datetime(0)
        if self.p.verbose:
            print(f'{dt.isoformat()}, {txt}')
        self.log_entries['date'].append(dt.isoformat())
        self.log_entries['log'].append(txt)

    def set_grid(self, current_price):
        self.grid_buy, self.grid_sell, self.stop_loss_buy, self.stop_loss_sell = custom_grid(
            current_price, self.params.last_order_down, self.params.last_order_up,
            self.params.down_grid_len, self.params.up_grid_len
        )
        self.log(f'Set new grid with current price: {current_price:.2f}')



    def __init__(self):
        # Maps to translate order execution types for logging clarity
        self.exectype_map = {bt.Order.Market: 'Market', bt.Order.Limit: 'Limit', bt.Order.Stop: 'Stop'}
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=self.p.atr_length, plot=True)

        # Order management and control variables
        self.cum_comm = 0
        self.last_month = None
        self.buy_order = 0 
        self.sell_order = 0
        self.count_stop_loss_buy = 0 
        self.count_stop_loss_sell = 0

        # Logging and transaction tracking
        self.log_entries = {"date": [], "log": []}
        self.transactions = {"date": [], "transaction_id": [], "symbol": [], "quantity": [], "price": [], "nominal": [], "order_type": [], "trade_type": [], "reason": [], "entry_side": [], "broker_comm": []}

        self.grid_buy_to_insert = 0
        self.grid_sell_to_insert = 0
        self.grid_buy = []
        self.grid_sell = []
        self.stop_loss_buy = 0 
        self.stop_loss_sell = 0

        # Sizer
        if self.p.sizers_type == 'RiskAdjustedSizer':
            self.sizer = RiskAdjustedSizer(
                max_risk_per_trade=self.p.max_risk_per_trade,
                stop_loss_deviation=self.p.stop_loss_deviation,
                max_authorized_position=self.p.max_authorized_position,
                atr=self.atr
            )
        else:
            self.sizer = FixedSize(stake=self.p.sizers_stake)


        # Signal Indicators
        self.sma = bt.indicators.SMA(self.datas[1], period=self.p.sma_period)
        self.ema1 = bt.indicators.EMA(self.datas[0], period=self.p.ema_period1, plot=True)
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.datas[0], period=self.p.adx_period, plot=False)
        self.stochastic_oscillator = bt.indicators.StochasticFast(self.datas[0], period=self.p.stochastic_oscillator_period, period_dfast=self.p.stochastic_oscillator_period_dfast, plot=False)
        self.ema_bollinger_bands = EMABollingerBands(self.datas[0], period=self.p.ema_bb_period, top_devfactor=self.p.bb_top_devfactor, bot_devfactor=self.p.bb_bot_devfactor, plot=False)
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.p.rsi_period, plot=False)
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=self.p.atr_length, plot=True)


        # Define ML Features
        self.willr = bt.indicators.WilliamsR(self.datas[0], period=14,  plot=False)
        self.rocr = CustomROCR(self.datas[0], period=10, plot=False)
        self.rsi_ml = bt.indicators.RSI(self.datas[0], period=14, plot=False)
        self.atr_ml = bt.indicators.ATR(self.datas[0], period=14,  plot=False)
        self.mom = bt.indicators.Momentum(self.datas[0], period=10,  plot=False)
        self.macd = bt.indicators.MACD(self.datas[0],  plot=False)
        self.kalman = KalmanFilterIndicator(self.datas[0],  plot=False)
        self.sqrt_weighted_mean_volume_traded = SqrtWeightedMeanVolume(self.datas[0], period=4,  plot=False)
        self.price_close_d = self.datas[0].close * 2
        self.difference_mean = (self.datas[0].high - self.datas[0].low + self.datas[0].close - self.datas[0].open) / 2

        # Trading records and control variables
        self.cum_pnl_net = 0
        self.cum_comm = 0
        self.last_month = None
        self.winning_trade = 0
        self.losing_trade = 0

        # Order management
        self.order_refs = {}  # Dictionary to track pending orders by reference
        self.entry_order = None
        self.stop_loss_order = None
        self.take_profit_order = None

        # Logging and transaction tracking
        self.log_entries = {"date": [], "log": []}
        self.transactions = {"date": [], "transaction_id": [], "symbol": [], "quantity": [], "price": [], "nominal": [], "order_type": [], "trade_type": [], "reason": [], "entry_side": [], "broker_comm": []}


        # Signal
        self.signal_generator = SignalGenerator(

            data = self.datas,            
            logger = self.log,
            args = self.p.args,

            sma = self.sma,
            sma_slope_limit = self.p.sma_slope_limit,
            sma_slope_period = self.p.sma_slope_period,

            ema1 = self.ema1,
            adx = self.adx,
            adx_strength = self.p.adx_strength,
            use_stochastic_oscillator = self.p.use_stochastic_oscillator,
            stochastic_oscillator = self.stochastic_oscillator,
            use_ema_bb = self.p.use_ema_bb,
            ema_bollinger_bands = self.ema_bollinger_bands,
            use_machine_learning = self.p.use_machine_learning,
            model_path = self.p.model_path,
            use_rsi = self.p.use_rsi,
            rsi = self.rsi,
            rsi_lower_threshold = self.p.rsi_lower_threshold,
            rsi_upper_threshold = self.p.rsi_upper_threshold,


            price_close_d = self.price_close_d,
            willr = self.willr,
            rocr = self.rocr,
            rsi_ml = self.rsi_ml,
            atr = self.atr_ml,
            mom = self.mom,
            macd = self.macd,
            kalman = self.kalman,
            sqrt_weighted_mean_volume_traded = self.sqrt_weighted_mean_volume_traded,
            difference_mean = self.difference_mean
        )            

    def notify_cashvalue(self, cash, value):
        """Print the date, cash, and portfolio value at the beginning of each month."""
        current_date = self.datetime.date().strftime('%Y-%m-%d')
        current_month = self.datetime.date().month

        if self.last_month is None or self.last_month != current_month:
            self.last_month = current_month
            self.log(f'\nDate: {current_date}, Cash: {cash}, Portfolio Value: {value}, Cumulative Broker Commission: {self.cum_comm}')

    def notify_order(self, order):
        """Log order status updates."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            size = order.executed.size
            price = order.executed.price
            self.log(f'{("Buy" if order.isbuy() else "Sell")} Order Executed, Price: {price:.2f}, '
                     f'Size: {size:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, '
                     f'Total Position Size: {self.position.size:.2f}')
            self.cum_comm += order.executed.comm
            self.record_transaction(order)

            if order.isbuy():
                self.buy_order += 1
            else:
                self.sell_order += 1

        elif order.status in [order.Margin, order.Rejected, order.Canceled, order.Expired]:
            self.log(f'{("Buy" if order.isbuy() else "Sell")} Order {order.getstatusname()} (ref: {order.ref})')

    def record_transaction(self, order):
        """Record details of the executed order into the transaction dictionary."""
        dt = self.datas[0].datetime.datetime(0)
        self.transactions['date'].append(dt.isoformat())
        self.transactions['symbol'].append('BTC-USD')  # assuming BTC-USD, update as needed
        self.transactions['quantity'].append(order.executed.size)
        self.transactions['price'].append(order.executed.price)
        self.transactions['nominal'].append(order.executed.value)
        # Ensure correct access to nested info dictionary
        info = order.info.get('info', {})  # Adjust this line based on the actual structure
        self.transactions['order_type'].append(info.get('order_type', ''))
        self.transactions['trade_type'].append(info.get('trade_type', ''))
        self.transactions['reason'].append(info.get('reason', ''))
        self.transactions['entry_side'].append(info.get('entry_side', ''))
        
        self.transactions['broker_comm'].append(order.executed.comm)

    def nextstart(self):
        """Initializes the grid at the start of the strategy."""
        self.set_grid(self.datas[0].close[0])

    def next(self):
        """Core trading logic for each bar."""
        current_price = self.datas[0].close[0]
        size = 0 


        long_entry_signals, short_entry_signals = self.signal_generator.entry_signal_generator(position_size=self.position.size)
        long_close_signals, short_close_signals = self.signal_generator.close_signal_generator(position_size=self.position.size)


        # Execute sell orders
        while len(self.grid_sell) > 0 and current_price > self.grid_sell[0]:
            new_size = self.sizer.getsizing(
                comminfo=self.broker.getcommissioninfo(self.data),
                value=self.broker.getvalue(),
                data=self.data,
                isbuy=False,
                entry_price=current_price,
                grid_len=self.p.up_grid_len
            )
            size += new_size / len(self.grid_sell)
            self.log(f'Sell Order Created at {self.grid_sell[0]:.2f}, Size: {size:.2f}')
            del self.grid_sell[0]
            self.grid_buy_to_insert += 1
        self.sell(size=size, exectype=bt.Order.Market, info={'trade_type': 'open', 'reason': 'percuted grid', 'entry_side': 'sell', 'order_type': self.exectype_map[bt.Order.Market]})

        # Execute buy orders
        while len(self.grid_buy) > 0 and current_price < self.grid_buy[0]:
            new_size = self.sizer.getsizing(
                comminfo=self.broker.getcommissioninfo(self.data),
                value=self.broker.getvalue(),
                data=self.data,
                isbuy=True,
                entry_price=current_price,
                grid_len=self.p.down_grid_len
            )
            size += new_size / len(self.grid_buy)
            self.log(f'Buy Order Created at {self.grid_buy[0]:.2f}, Size: {size:.2f}')
            del self.grid_buy[0]
            self.grid_sell_to_insert += 1
        self.buy(size=size, exectype=bt.Order.Market, info={'trade_type': 'open', 'reason': 'percuted grid', 'entry_side': 'buy', 'order_type': self.exectype_map[bt.Order.Market]})


        # Insert new buy levels
        if self.grid_buy:
            if self.grid_buy_to_insert > 0:
                grid_buy_diff = (current_price - self.grid_buy[0]) / (self.grid_buy_to_insert + 1)
                for i in range(self.grid_buy_to_insert):
                    new_buy_level = self.grid_buy[0] + grid_buy_diff
                    self.grid_buy.insert(0, new_buy_level)
                    self.log(f'Inserted new buy grid at {new_buy_level:.2f}')
            self.grid_buy_to_insert = 0

        # Insert new sell levels
        if self.grid_sell:
            if self.grid_sell_to_insert > 0:
                grid_sell_diff = (self.grid_sell[0] - current_price) / (self.grid_sell_to_insert + 1)
                for i in range(self.grid_sell_to_insert):
                    new_sell_level = self.grid_sell[0] - grid_sell_diff
                    self.grid_sell.insert(0, new_sell_level)
                    self.log(f'Inserted new sell grid at {new_sell_level:.2f}')
            self.grid_sell_to_insert = 0


        # Check for stop-loss condition
        if current_price < self.stop_loss_buy or current_price > self.stop_loss_sell:
            self.log('Stop-loss triggered, resetting grid.')
            if current_price < self.stop_loss_buy:
                self.count_stop_loss_buy += 1
                self.close(exectype=bt.Order.Market, info={'trade_type': 'close', 'reason': 'stop loss', 'entry_side': 'sell', 'order_type': self.exectype_map[bt.Order.Market]})

            elif current_price > self.stop_loss_sell:
                self.count_stop_loss_sell += 1
                self.close(exectype=bt.Order.Market, info={'trade_type': 'close', 'reason': 'stop loss', 'entry_side': 'sell', 'order_type': self.exectype_map[bt.Order.Market]})
            self.set_grid(current_price)  # Reset grid with the current price



    def stop(self):

        # Log final portfolio status
        self.log(f'Cash: {self.broker.getcash()}, Portfolio Value: {self.broker.getvalue()}, Cumulative Broker Commission: {self.cum_comm}')
        self.log(f'Buy Orders: {self.buy_order}, Sell Orders: {self.sell_order}, Total Orders: {self.buy_order + self.sell_order}')
        self.log(f'Stop Loss Buy: {self.count_stop_loss_buy}, Stop Loss Sell: {self.count_stop_loss_sell}')
        
