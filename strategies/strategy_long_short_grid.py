from __future__ import (absolute_import, division, print_function, unicode_literals)
import backtrader as bt
from datetime import datetime, timedelta
import sys
import os

# Append project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from strategies.Risk_Management.Order_Management.trailing_strategy import TrailingStrategy
from strategies.Risk_Management.Money_Management.position_sizing_strategy import FixedSize, PercentSizer, RiskAdjustedSizer
from strategies.Risk_Management.Money_Management.position_management_strategy import PositionManagement
from strategies.Signal.Indicators.build_in_indicators import EMABollingerBands, CustomROCR, KalmanFilterIndicator, SqrtWeightedMeanVolume
from strategies.Signal.signal_generator import SignalGenerator


class LongShortGrid(bt.Strategy):
    '''
    Creation of a new strategy as a class that inherits from Backtrader Strategy.     
    The heart of the strategy is written in the 'next' function below. There is also 'log' function and 'notify' functions to keep track of every action.

    '''
    params = dict(
        verbose = True,
        brackets_validity=100,
        args = None,

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


        # Sizer Parameters
        sizers_type='RiskAdjustedSizer',  # Options: 'RiskAdjustedSizer', 'PercentSizer', 'FixedSize'
        sizers_percent=30,
        sizers_stake=3,
        max_risk_per_trade=0.001,
        stop_loss_deviation=1,
        max_authorized_position=0.2,
        atr_length=12,  # in bars


        # Trailing Strategy
        use_trailing_strategy = True,
        trail_stop_pct = 1.3,   # Trailing stop at 0.5% above entry price
        trail_stop_abs = None,  # Alternatively, could set an absolute value
        secure_pct = 35,        # % of th eprofit to securize
        atr_multiplier = 3.6,

        # Position Management
        # Signal 
        use_position_management = True,
        # Param
        atr_long_multiplier = 2,
        atr_short_multiplier = 1,
        scale_in_factor = 1/3, 
        scale_out_factor = 2/3, 
        position_scaling_hold = 4,
        max_position_count = 5,

        # Long Parameter
        long_exectype = bt.Order.Limit,    
        long_limit = 0.005,
        long_validity =10,
        long_away_from_price = 0.03,  # Entre 0.01 et 0.05 en general
        long_risk_reward_ratio = 2/3,
        long_stop_loss_exectype =  bt.Order.Stop,    
        long_take_profit_exectype = bt.Order.Limit,
        long_hold = 9,


        # Short Parameter
        short_exectype = bt.Order.Limit,
        short_limit=0.005,
        short_validity = 5,
        short_away_from_price = 0.02,
        short_risk_reward_ratio = 1/2,
        short_stop_loss_exectype = bt.Order.Stop,
        short_take_profit_exectype = bt.Order.Limit ,
        short_hold = 3,

        # Grid Trading Parameters
        grid_distance_factor=0.01,  # Distance between grid levels as a factor of price
        grid_levels=3,  # Number of levels in the grid
        grid_max_position_count=5,  # Maximum number of grid positions allowed

        # Switching Parameters (Decide between Normal and Grid)
        atr_threshold=200,  # Threshold to switch to/from grid trading
        adx_threshold=25,   # ADX threshold to decide between normal and grid strategies
    )


    def log(self, txt, dt=None):
        """
        Logging function for this strategy   

        Parameters : 

            txt: A string containing the log message.
            dt : An optional datetime object representing the date and time of the log entry. If not provided, the current date and time from the first data feed are used. 

        Return :
            This function tracks and records actions taken by the trading strategy, along with their timestamps.

        """
        dt = dt or datetime.combine(self.datas[0].datetime.date(0), self.datas[0].datetime.time(0))
        if self.p.verbose : 
            print(f'{dt.isoformat()}, {txt}') 
        self.log_entries['date'].append(dt.isoformat())
        self.log_entries['log'].append(txt)



    def __init__(self):
        """
        Parameters :
            self.dataclose : stores the closing prices of the first data feed for easy access.
            self.order : Stores the current order.
            self.cum_pnl_net : Tracks cumulative net profit and loss.
            self.cum_comm : Tracks cumulative commissions paid.
            self.winning_trade and self.losing_trade : Count the number of winning and losing trades, respectively.
            self.short_orefs and self.long_orefs : Lists to store references to short and long orders, respectively.
            self.last_month : Keeps track of the last month processed, useful for monthly operations.
            self.log_entries : Dictionary to store log entries with dates and messages - used for the report.
            self.transactions : Dictionary to keep detailed records of transactions - used to complete the report.
         
            Function used to initialize the different parameters and to choose the sizer.
        """
        # Maps to translate order execution types for logging clarity
        self.exectype_map = {bt.Order.Market: 'Market', bt.Order.Limit: 'Limit', bt.Order.Stop: 'Stop' }

        # Data references for ease of use
        # Short Term Data    self.datas[0].close          
        # Long Term Data    self.datas[1].close


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

        # Sizer
        if self.p.sizers_type == 'PercentSizer':
            self.sizer = PercentSizer(percents=self.p.sizers_percent)
        elif self.p.sizers_type == 'RiskAdjustedSizer':
            self.sizer = RiskAdjustedSizer(max_risk_per_trade=self.p.max_risk_per_trade,
                                           stop_loss_deviation=self.p.stop_loss_deviation,
                                           max_authorized_position=self.p.max_authorized_position,
                                           atr = self.atr
                                           )
        else:
            self.sizer = FixedSize(stake=self.p.sizers_stake)

        # Trailing
        if self.p.use_trailing_strategy:
            self.trailing_strategy = TrailingStrategy(
                data = self.datas,
                logger = self.log,
                trail_stop_pct = self.p.trail_stop_pct,
                trail_stop_abs = self.p.trail_stop_abs,
                secure_pct = self.p.secure_pct,
                atr_multiplier = self.p.atr_multiplier,
                long_risk_reward_ratio = self.p.long_risk_reward_ratio,
                short_risk_reward_ratio = self.p.short_risk_reward_ratio,
                atr=self.atr
                )
        self.securized = False


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
            
        # Position Management
        if self.p.use_position_management:
            self.position_management_strategy = PositionManagement(
                data=self.datas,            
                logger=self.log,
                atr=self.atr,
                atr_long_multiplier=self.p.atr_long_multiplier,
                atr_short_multiplier=self.p.atr_short_multiplier,
                signal_generator = self.signal_generator
            )
        self.bars_since_scaling = 0
        self.position_count = 0 
        self.cash_unavailable = 0
        self.count_hold_expiry = 0        
        self.count_close_signal = 0        


    def notify_cashvalue(self, cash, value):
        """
        Print the date, cash, and portfolio value at the beginning of each month.

        Parameters : 
            cash: The current amount of cash available.
            value: The current value of the portfolio.

        """
        current_date = self.datetime.date().strftime('%Y-%m-%d')
        current_month = self.datetime.date().month

        # Vérifier si le mois a changé depuis la dernière notification
        if self.last_month is None or self.last_month != current_month:
            self.last_month = current_month
            # Envoyer la notification au début de chaque mois
            self.log(f'\nDate: {current_date}, Cash: {cash}, Portfolio Value: {value}')


    def notify_order(self, order):
        """
        Function to notify which orders were executed, when and at what price.
        If the order is Completed, a log message is generated indicating the type of order (BUY/SELL), trade type, execution price, size, cost, and commission.
        The index (bar number) when the last order was executed is tracked for short and long entry orders. 
        The function notify if the order status is either Margin, Rejected, Canceled or Expired.
        All these logs are used also for the transaction report

        Parameter :
            order : The order object whose status is being notified.

        """

        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status == order.Completed:
            trade_type = order.info['info']['trade_type']
            size = order.executed.size
            price = order.executed.price
            high = self.datas[0].high[0]
            low =  self.datas[0].low[0]
            init_stop_loss = self.stop_loss_price
            init_take_profit = self.take_profit_price
            entry_side = order.info['info']['entry_side']

            self.log(f'{("Buy" if order.isbuy() else "Sell")} {order.info["info"]["trade_type"]} Order Executed, Price: {price:.2f}, '
                        f'Size: {size:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            
            # Fill the transaction dictionary for report
            self.record_transaction(order)

            if trade_type=='entry':
                if self.p.use_trailing_strategy:        
                    self.trailing_strategy.set_initial_levels(price, high, low, init_stop_loss, init_take_profit, entry_side)
                    self.securized = False

                if order.isbuy():
                    self.bar_executed_long = len(self)
                else:
                    self.bar_executed_short = len(self)
                self.bars_since_scaling = 0

                if self.p.use_position_management:        
                    self.position_management_strategy.set_initial_levels(price, entry_side, size)
                    self.position_count += 1
                    self.log(f"Position count {self.position_count} ")     

            elif trade_type=='close':
                if order.isbuy():                   
                    self.bar_executed_long = 0
                else:
                     self.bar_executed_short = 0
            
        elif order.status in [order.Margin]:
            order_type_info =  order.info['info']['trade_type']
            self.log(f'{("Buy" if order.isbuy() else "Sell")} {order_type_info} Order Margin (ref: {order.ref})')
            self.close_position(reason='margin call')
                                
        elif order.status in [order.Rejected]:
            order_type_info =  order.info['info']['trade_type']
            self.log(f'{("Buy" if order.isbuy() else "Sell")} {order_type_info} Order Rejected (ref: {order.ref})')

        elif order.status == order.Canceled:
            order_type_info =  order.info['info']['trade_type']
            self.log(f'{("Buy" if order.isbuy() else "Sell")} {order_type_info} Order Canceled (ref: {order.ref})')

        elif order.status == order.Expired:
            order_type_info =  order.info['info']['trade_type']
            self.log(f'{("Buy" if order.isbuy() else "Sell")} {order_type_info} Order Expired (ref: {order.ref})')

        if not order.alive() and order.ref in self.order_refs:
            del self.order_refs[order.ref]


    def record_transaction(self, order):
        """
        Record details of the executed order into the transaction dictionary.
        
        Parameter :
            order : The order object whose status is being notified.
        """
        dt = datetime.combine(self.datas[0].datetime.date(0), self.datas[0].datetime.time(0))
        self.transactions['date'].append(dt.isoformat())
        self.transactions['symbol'].append('BTC-USD')  # assuming BTC-USD, update as needed
        self.transactions['quantity'].append(order.executed.size)
        self.transactions['price'].append(order.executed.price)
        self.transactions['nominal'].append(order.executed.value)
        self.transactions['order_type'].append(order.info['info']['order_type'])
        self.transactions['trade_type'].append(order.info['info']['trade_type'])
        self.transactions['reason'].append(order.info['info']['reason'])
        self.transactions['entry_side'].append(order.info['info']['entry_side'])
        self.transactions['broker_comm'].append(order.executed.comm)


    def notify_trade(self, trade):
        """
        The function tracks and logs trade outcomes, updating cumulative performance metrics and providing detailed logs

        Parameters :
            trade: A trade object representing the trade whose status is being notified.

        """

        if trade.isclosed:
            self.position_count = 0 
            self.log(f'TRADE ID: {trade.ref:.2f}, LENGTH: {trade.barlen:.2f}, OPERATION: GROSS PnL: {trade.pnl:.2f}, NET PnL: {trade.pnlcomm:.2f}')
            self.cum_pnl_net += trade.pnlcomm
            self.cum_comm += trade.pnl-trade.pnlcomm
            if trade.pnlcomm < 0:
                self.losing_trade += 1
            else:
                self.winning_trade +=1
            self.log(f'CUMULATIVE PROFIT NET: {self.cum_pnl_net:.2f}, CUMULATIVE BROKER COMMISSION: {self.cum_comm:.2f}')


    def next(self):
        """
        The next method is called on each bar update. It checks for pending orders,
        evaluates positions, and executes trading orders based on defined strategy.

        Parameters :
            self : The strategy instance, giving access to instance variables and methods.

        """
        if self.p.use_trailing_strategy :
            self.trailing_strategy.update_high_low()
        current_price = self.datas[0].close[0]


        #self.log(current_price)
        # self.log(f"{self.order_refs}")
        if len(self.order_refs) > 3  or len(self.order_refs) == 1 :
            self.log(f"reference, {self.order_refs}")
    
        # Generate signals
        long_entry_signals, short_entry_signals = self.signal_generator.entry_signal_generator(position_size=self.position.size)
        long_close_signals, short_close_signals = self.signal_generator.close_signal_generator(position_size=self.position.size)

        # No pending orders
        if not self.order_refs:
            if  self.position.size == 0:
                # Long entry signal detected
                if long_entry_signals >= 1 :
                    self.log('Long entry signal detected')

                    # Check market conditions for strategy selection
                    if self.atr[0] > self.p.atr_threshold and self.adx[0] > self.p.adx_threshold:
                        # High ATR and ADX: Use normal strategy
                        self.log('Favorable condition to use normal strategy')
                        self.execute_normal_strategy(is_long=True, current_price=current_price)
                    else:
                        # Low ATR and ADX: Use grid trading strategy
                        self.log('Favorable condition to use grid trading strategy')
                        self.execute_grid_trading_strategy(is_long=True, current_price=current_price)

                # Short entry signal detected
                elif short_entry_signals >= 1 :
                    self.log('Short entry signal detected')

                    # Check market conditions for strategy selection
                    if self.atr[0] > self.p.atr_threshold and self.adx[0] > self.p.adx_threshold:
                        # High ATR and ADX: Use normal strategy
                        self.log('Favorable condition to use normal strategy')
                        self.execute_normal_strategy(is_long=False, current_price=current_price)
                    else:
                        # Low ATR and ADX: Use grid trading strategy
                        self.log('Favorable condition to use grid trading strategy')
                        self.execute_grid_trading_strategy(is_long=False, current_price=current_price)

            else: # Position is not nul
                self.log("Critical State: Open position exists but no related pending orders found. Check order management and execution.")

        # Pending orders exist
        else:
            # Manage open positions depending on whether we're in a normal or grid strategy
            if hasattr(self, 'grid_orders') and self.grid_orders:  # If grid orders exist, manage the grid strategy
                self.manage_grid_strategy(long_close_signals, short_close_signals)
            else:  # Otherwise, manage the normal strategy
                self.manage_normal_strategy(long_close_signals, short_close_signals)


 

    def execute_grid_trading_strategy(self, is_long, current_price):
        grid_distance = current_price * self.p.grid_distance_factor
        size = self.sizer.getsizing(comminfo=self.broker.getcommissioninfo(self.data), cash=self.broker.getcash(), data=self.data, isbuy=is_long, entry_price=current_price)

        grid_orders = []

        for i in range(1, self.p.grid_levels + 1):
            if is_long:
                buy_price = current_price - (i * grid_distance)
                grid_order = self.buy(price=buy_price, size=size, exectype=bt.Order.Limit, info={'trade_type': 'entry', 'reason': 'grid buy', 'entry_side': 'long', 'order_type': self.exectype_map[self.p.long_exectype]})
            else:
                sell_price = current_price + (i * grid_distance)
                grid_order = self.sell(price=sell_price, size=size, exectype=bt.Order.Limit, info={'trade_type': 'entry', 'reason': 'grid sell', 'entry_side': 'long', 'order_type': self.exectype_map[self.p.long_exectype]})
 
            grid_orders.append(grid_order)
            self.log(f'Grid Order placé: {"Buy" if is_long else "Sell"} à {buy_price if is_long else sell_price:.2f}')

        self.grid_orders = grid_orders

    def manage_grid_strategy(self, long_close_signals, short_close_signals):
        # If a new signal is detected, close all grid orders and the position
        if (self.position.size > 0.0 and long_close_signals >= 1) or (self.position.size < 0.0 and short_close_signals >= 1):
            self.log("New signal detected during grid trading, closing all grid orders and position.")
            self.close_position(reason='new signal')
            for order in self.grid_orders:
                if order.alive():
                    self.cancel(order)
            self.grid_orders = []
        else:
            # Continue managing grid orders
            self.manage_grid_orders()

    def manage_grid_orders(self):
        active_orders = []
        for order in self.grid_orders:
            if order.alive():
                active_orders.append(order)
            else:
                self.log(f'Grid Order exécuté: {order.info["trade_type"]} à {order.executed.price:.2f}')

        # Update active orders
        self.grid_orders = active_orders





    def execute_normal_strategy(self, is_long, current_price):
        # Exécuter la logique d'entrée classique
        if is_long:
            entry_valid = timedelta(hours=self.p.long_validity)
            brackets_valid = timedelta(hours=self.p.brackets_validity)

            if self.p.long_exectype == bt.Order.Limit:
                self.entry_price = self.datas[0].close[0] * (1.0 - self.p.long_limit)
            else:
                self.entry_price = self.datas[0].close[0]

            size = self.sizer.getsizing(comminfo=self.broker.getcommissioninfo(self.data), cash=self.broker.getcash(), data=self.data, isbuy=True, entry_price=self.entry_price)
            self.entry_order = self.buy(price=self.entry_price, size=size, exectype=self.p.long_exectype, transmit=False, valid=entry_valid, info={'trade_type': 'entry', 'reason': 'signal', 'entry_side': 'long', 'order_type': self.exectype_map[self.p.long_exectype]})
            self.log(f'Buy {self.exectype_map[self.p.long_exectype]} created at {self.entry_price:.2f}')
            self.order_refs[self.entry_order.ref] = self.entry_order

            # Backets Orders
            self.stop_loss_price = self.entry_price - self.p.long_risk_reward_ratio * self.p.long_away_from_price * self.entry_price
            self.take_profit_price = self.entry_price + self.p.long_away_from_price * self.entry_price

            # Long - Stop Loss Order
            self.stop_loss_order = self.sell(price=self.stop_loss_price, size=size, exectype=self.p.long_stop_loss_exectype, parent=self.entry_order, valid=brackets_valid, transmit=False, info={'trade_type': 'close', 'reason':'stop loss', 'entry_side':'long', 'order_type' :self.exectype_map[self.p.long_stop_loss_exectype] })
            self.log(f'Sell {self.exectype_map[self.p.long_stop_loss_exectype]} loss order created at {self.stop_loss_price:.2f}')
            self.order_refs[self.stop_loss_order.ref] = self.stop_loss_order

            # Long - Take Profit Order
            self.take_profit_order = self.sell(price=self.take_profit_price, size=size, exectype=self.p.long_take_profit_exectype, parent=self.entry_order, valid=brackets_valid, transmit=True, info={'trade_type': 'close', 'reason': 'take profit', 'entry_side':'long', 'order_type' : self.exectype_map[self.p.long_take_profit_exectype] })
            self.log(f'Sell {self.exectype_map[self.p.long_take_profit_exectype]} take profit order created at {self.take_profit_price:.2f}')
            self.order_refs[self.take_profit_order.ref] = self.take_profit_order
    

        else:
            # Logique d'entrée classique pour une position courte (similaire à la logique longue)
            entry_valid = timedelta(hours=self.p.short_validity)
            brackets_valid = timedelta(hours=self.p.brackets_validity) 
                
            if self.p.short_exectype == bt.Order.Limit:
                self.entry_price = current_price * (1.0 + self.p.short_limit)
            else : #bt.Order.Market
                self.entry_price = current_price

            size = self.sizer.getsizing(comminfo = self.broker.getcommissioninfo(self.data), cash=self.broker.getcash(), data = self.data, isbuy = False, entry_price = self.entry_price)
            self.entry_order = self.sell(price=self.entry_price, size=size, exectype=self.p.short_exectype, transmit=False, valid=entry_valid, info={'trade_type':'entry', 'reason':'signal', 'entry_side':'short', 'order_type':self.exectype_map[self.p.short_exectype]})
            self.log(f'Sell {self.exectype_map[self.p.short_exectype]} order created at {self.entry_price:.2f}')
            self.order_refs[self.entry_order.ref] = self.entry_order

            # Backets Orders
            self.stop_loss_price = self.entry_price + self.p.short_risk_reward_ratio * self.p.short_away_from_price * self.entry_price
            self.take_profit_price = self.entry_price - self.p.short_away_from_price * self.entry_price

            # Short - Stop Loss Order
            self.stop_loss_order = self.buy(price=self.stop_loss_price, size=size, exectype=self.p.short_stop_loss_exectype, parent=self.entry_order, valid=brackets_valid, transmit=False, info={'trade_type':'close', 'reason':'stop loss', 'entry_side':'short', 'order_type' : self.exectype_map[self.p.short_stop_loss_exectype]})
            self.log(f'Buy {self.exectype_map[self.p.short_stop_loss_exectype]} stop loss order created at {self.stop_loss_price:.2f}')
            self.order_refs[self.stop_loss_order.ref] = self.stop_loss_order

            # Short - Take Profit Order
            self.take_profit_order = self.buy(price=self.take_profit_price, size=size, exectype=self.p.short_take_profit_exectype, parent=self.entry_order, valid=brackets_valid, transmit=True, info={'trade_type': 'close', 'reason':'take profit', 'entry_side':'short', 'order_type' : self.exectype_map[self.p.short_take_profit_exectype] })
            self.log(f'Buy {self.exectype_map[self.p.short_take_profit_exectype]} take profit order created at {self.take_profit_price:.2f}')
            self.order_refs[self.take_profit_order.ref] = self.take_profit_order



    def manage_normal_strategy(self, long_close_signals, short_close_signals):
        if self.position :
            # Handle signal-based position closure (long & negative signal or short & positive signal -> close)
            if (self.position.size > 0.0 and long_close_signals >= 1)  or (self.position.size < 0.0 and short_close_signals >= 1): 
                self.close_position(reason='close signal')

            # Check for expiry-based position closure
            elif self.check_hold_period_expiry():
                self.close_position(reason='hold expiry')

            else:
                # Adjust position size based on scaling signals
                if self.p.use_position_management and  self.position_count < self.p.max_position_count :
                        self.bars_since_scaling += 1
                        if self.bars_since_scaling >= self.p.position_scaling_hold:
                            scale_in = self.position_management_strategy.should_scale_in()
                            scale_out = self.position_management_strategy.should_scale_out()
                            if scale_in or scale_out:
                                update_position = self.update_position(scale_in=scale_in , scale_out=scale_out)
                                if update_position:
                                    return

                # Secure position using trailing strategy
                if self.p.use_trailing_strategy :
                    if not self.securized:
                        secure_stop_loss, stop_loss_message = self.trailing_strategy.get_secure_stop_loss()
                        if secure_stop_loss != self.stop_loss_price and self.stop_loss_order.alive():
                            self.log(stop_loss_message)
                            self.update_brackets(new_stop_loss=secure_stop_loss)
                            self.securized = True
                            return
                    else:
                        new_stop_loss, new_take_profit = self.trailing_strategy.get_volatility_brackets()
                        self.update_brackets(new_stop_loss=new_stop_loss, new_take_profit=new_take_profit)
                        return

    def check_hold_period_expiry(self):
        """
        Check if the hold period has expired to close the position.
        """
        if self.position.size > 0:
            return (len(self) - self.bar_executed_long) >= self.p.long_hold
        elif self.position.size < 0:
            return (len(self) - self.bar_executed_short) >= self.p.short_hold
        return False


    def close_position(self, reason):
        """
        Close the entire position.
        """
        if self.position.size >= 0:
            self.close(exectype=bt.Order.Market, info={'trade_type': 'close', 'reason': reason, 'entry_side': 'long', 'order_type': 'Market'})
        elif self.position.size < 0:
            self.close(exectype=bt.Order.Market, info={'trade_type': 'close', 'reason': reason, 'entry_side': 'short', 'order_type': 'Market'})
        self.log(f'Closing position due to {reason}')
        if reason == 'hold expiry':
            self.count_hold_expiry +=1        
        elif reason == 'close signal':
            self.count_close_signal +=1        

        for ref, order in list(self.order_refs.items()):
            if order.alive():
                self.cancel(order)


    def update_brackets(self, new_stop_loss=None, new_take_profit=None):
        """
        Update the stop loss and take profit orders based on new levels and optionally adjust sizes.

        Args:
            new_stop_loss (float): New stop loss price, if applicable.
            new_take_profit (float): New take profit price, if applicable.
        """
        if not self.position:
            self.log('No active position to update brackets for.')
            return

        # Calculate new total size if adjusting size 
        size = abs(self.position.size)

        # Cancel existing orders if active
        if (self.stop_loss_order and self.stop_loss_order.alive()) or (self.take_profit_order and self.take_profit_order.alive()) :
            self.cancel(self.stop_loss_order)
            self.cancel(self.take_profit_order)

        # Update prices
        if new_stop_loss:
            self.stop_loss_price = new_stop_loss
        if new_take_profit:
            self.take_profit_price = new_take_profit

        # Create new stop loss and take profit orders
        if self.position.size > 0:
            self.stop_loss_order = self.sell(size=size, price=self.stop_loss_price, exectype=bt.Order.Stop, info={'trade_type':'close', 'reason':'stop loss', 'entry_side':'long', 'order_type': 'Stop'}) 
            self.take_profit_order = self.sell(size=size, price=self.take_profit_price, exectype=bt.Order.Limit, oco=self.stop_loss_order, info={'trade_type':'close', 'reason':'take profit', 'entry_side':'long', 'order_type': 'Limit'})
        
        elif self.position.size < 0:
            self.stop_loss_order = self.buy(size=size, price=self.stop_loss_price, exectype=bt.Order.Stop, info={'trade_type': 'close', 'reason':'stop loss', 'entry_side':'short',  'order_type': 'Stop'})
            self.take_profit_order = self.buy(size=size, price=self.take_profit_price, exectype=bt.Order.Limit, oco=self.stop_loss_order, info={'trade_type': 'close', 'reason':'take profit', 'entry_side':'short', 'order_type': 'Limit'})
        
        self.log(f'Updated brackets - New SL: {self.stop_loss_price:.2f}, New TP: {self.take_profit_price:.2f}, Size: {size:.2f}')
        
        self.order_refs[self.stop_loss_order.ref] = self.stop_loss_order
        self.order_refs[self.take_profit_order.ref] = self.take_profit_order



    
    def update_position(self, scale_in=False, scale_out=False):
        """
        Adjust the position size based on scaling signals and update brackets accordingly.

        Args:
            scale_in (bool): If True, scale into the position (increase size).
            scale_out (bool): If True, scale out of the position (decrease size).
        """

        if not self.position:
            self.log('No active position to adjust.')
            return
        
        update_position = False
        self.entry_price = self.datas[0].close[0]
        current_cash = self.broker.getcash()
        current_value = self.broker.getvalue()
        comm_info = self.broker.getcommissioninfo(self.data)
        position_size = abs(self.position.size)
        size = position_size / 4

        if scale_in:
            additional_size = size * (1 + self.p.scale_in_factor)
            additional_cost = additional_size * self.entry_price + comm_info.getcommission(price = self.entry_price, size=additional_size)
            if self.position.size > 0:
                cash_available_for_trading = current_cash * self.p.max_authorized_position
            elif self.position.size < 0:
                cash_available_for_trading = (current_value - (current_cash - current_value)) * self.p.max_authorized_position

            if cash_available_for_trading > additional_cost:
                new_size = position_size + additional_size
                if self.position.size > 0:
                    self.entry_order = self.buy(size=additional_size, price=self.entry_price, exectype=bt.Order.Market, transmit=False, info={'trade_type': 'entry', 'reason':'scale in', 'entry_side':'long', 'order_type': 'Market'})
                elif self.position.size < 0:
                    self.entry_order = self.sell(size=additional_size, price=self.entry_price, exectype=bt.Order.Market, transmit=False, info={'trade_type': 'entry', 'reason':'scale in', 'entry_side':'short', 'order_type': 'Market'})
                self.log(f'Position Scaled in by {additional_size:.2f} at price {self.entry_price:.2f}')
                self.order_refs[self.entry_order.ref] = self.entry_order
                update_position = True
            else:
                self.log(f'Unable to scale in due to insufficient cash: Required {additional_cost:.2f}, Available {cash_available_for_trading:.2f}')
                self.cash_unavailable += 1 
                return


        elif scale_out:
            reduction_size = size * (1 + self.p.scale_out_factor)
            new_size = position_size - reduction_size

            if new_size > 0:
                if self.position.size > 0:
                    self.entry_order = self.sell(size=reduction_size, price=self.entry_price, exectype=bt.Order.Market, transmit=False, info={'trade_type':'entry', 'reason':'scale out', 'entry_side':'long', 'order_type':'Market'})
                elif self.position.size < 0:
                    self.entry_order = self.buy(size=reduction_size, price=self.entry_price, exectype=bt.Order.Market, transmit=False, info={'trade_type':'entry', 'reason':'scale out', 'entry_side':'short', 'order_type':'Market'})
                self.log(f'Position Scaled out by {reduction_size:.2f} at price {self.entry_price:.2f}')
                self.order_refs[self.entry_order.ref] = self.entry_order
                update_position = True
            else:
                self.close_position(reason='scale out to zero')
                return


        if scale_in or scale_out :
            self.update_brackets_for_new_size(new_size=new_size)

        return update_position


    def update_brackets_for_new_size(self, new_size):
        """
        Update stop loss and take profit orders for the new size.
        """
        if not self.position:
            self.log('No active position to update brackets for.')
            return

        if self.stop_loss_order.alive() or self.take_profit_order.alive():
            self.cancel(self.stop_loss_order)
            self.cancel(self.take_profit_order)

        # Update brackets for new size
        if self.position.size > 0:
            self.stop_loss_order = self.sell(size=new_size, price=self.stop_loss_price, exectype=bt.Order.Stop, parent=self.entry_order, transmit=False, info={'trade_type': 'close', 'reason':'stop loss', 'entry_side':'long', 'order_type': 'Stop'})
            self.take_profit_order = self.sell(size=new_size, price=self.take_profit_price, exectype=bt.Order.Limit, parent=self.entry_order, transmit=True, info={'trade_type': 'close', 'reason':'take profit', 'entry_side':'long', 'order_type': 'Limit'})
        
        elif self.position.size < 0:
            self.stop_loss_order = self.buy(size=new_size, price=self.stop_loss_price, exectype=bt.Order.Stop, parent=self.entry_order, transmit=False, info={'trade_type': 'close', 'reason':'stop loss', 'entry_side':'short', 'order_type': 'Stop'})
            self.take_profit_order = self.buy(size=new_size, price=self.take_profit_price, exectype=bt.Order.Limit, parent=self.entry_order, transmit=True, info={'trade_type': 'close', 'reason':'take profit', 'entry_side':'short', 'order_type': 'Limit'})

        self.log(f'Updated brackets - SL: {self.stop_loss_price:.2f}, TP: {self.take_profit_price:.2f}, New Size: {new_size:.2f}')

        if new_size != 0:
            self.order_refs[self.stop_loss_order.ref] = self.stop_loss_order
            self.order_refs[self.take_profit_order.ref] = self.take_profit_order


    
    
    def stop(self):
        """
        The function logs the final cumulative net profit, cumulative broker commissions, number of winning trades, and number of losing trades at the end of the strategy run.

        Parameters : 
            self : The strategy instance, giving access to instance variables and methods.
        """
        sma_long_count, bb_long_count, stochastic_oscillator_long_count, sma_short_count, bb_short_count, stochastic_oscillator_short_count, sma_close_long_count, sma_close_short_count = self.signal_generator.get_signal_count()

        self.log(f'CUMULATIVE PROFIT NET: {self.cum_pnl_net:.2f}, CUMULATIVE BROKER COMMISSION: {self.cum_comm:.2f}')
        self.log(f'Winning Trade Count: {self.winning_trade}')
        self.log(f'Losing Trade Count: {self.losing_trade}')
        self.log(f'Hold expiry Close Count: {self.count_hold_expiry}')
        self.log(f'Signal Close Count: {self.count_close_signal}')

        self.log(f'SMA Long Signal Count: {sma_long_count}')
        self.log(f'SMA Short Signal Count: {sma_short_count}')
        self.log(f'EMA BB Long Signal Count: {bb_long_count}')
        self.log(f'EMA BB SHort Signal Count: {bb_short_count}')
        self.log(f'Oscillator Stochastic Long Signal Count: {stochastic_oscillator_long_count}')
        self.log(f'Oscillator Stochastic Signal Short Count: {stochastic_oscillator_short_count}')
        self.log(f'SMA Close Long Count: {sma_close_long_count}')        
        self.log(f'SMA Close Short Count: {sma_close_short_count}')
        self.log(f'Cash Unavailable Count: {self.cash_unavailable}')
