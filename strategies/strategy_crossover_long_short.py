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


class CrossOver_LongShort(bt.Strategy):
    '''
    Creation of a new strategy as a class that inherits from Backtrader Strategy. 
    This strategy CrossOver_Longshort is based on the crossing of two SMA (Simple Moving Average) curves wich are built on two different periods of time.
    When the signal becomes positive (short-term moving average crosses above long-term moving average), it checks if there is an existing position in the market and based on it, 
    it enters a long position. If the signal is negative, it enters a short position.
    
    The heart of the strategy is written in the 'next' function below. There is also 'log' function and 'notify' functions to keep track of every action.

    '''
    params = dict(
    verbose = True,
    sma=bt.ind.SMA,
    period1=9,
    period2=25,
    brackets_validity=100,
    

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
    rsi_period = 14,
    rsi_in_long_lower=50,
    rsi_in_long_upper=65,
    rsi_in_short_lower=35,
    rsi_in_short_upper=50,
    rsi_out_long=35,
    rsi_out_short=50,
    bbands_period = 20, 
    bbands_devfactor = 2,
    # Param
    atr_long_multiplier = 2,
    atr_short_multiplier = 1,
    scale_in_factor = 1/3, 
    scale_out_factor = 2/3, 
    position_scaling_hold = 4,
    max_position_count = 5,

    # Long Parameter
    long_exectype = bt.Order.Limit,    
    long_limit =0.005,
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
            sma1 and sma2 : Simple Moving Averages created with periods defined by self.p.period1 and self.p.period2.
            self.cross : Crossover indicator to detect when sma1 crosses sma2.
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

        # Simple Moving Averages for crossover strategy
        sma1 = self.p.sma(period=self.p.period1)
        sma2 = self.p.sma(period=self.p.period2)
        self.cross = bt.ind.CrossOver(sma1, sma2,  plot=False)

        # Volatility and momentum indicators
        self.atr = bt.indicators.AverageTrueRange(period=self.p.atr_length)
        self.bollinger_bands = bt.indicators.BollingerBands(period=self.p.bbands_period, devfactor=self.p.bbands_devfactor,  plot=False)
        self.rsi = bt.indicators.RSI_EMA(period=self.p.rsi_period,  plot=False)

        # Data references for ease of use
        self.data_close = self.datas[0].close
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low

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
                logger = self.log,
                trail_stop_pct=self.p.trail_stop_pct,
                trail_stop_abs=self.p.trail_stop_abs,
                secure_pct = self.p.secure_pct,
                atr_multiplier = self.p.atr_multiplier,
                long_risk_reward_ratio = self.p.long_risk_reward_ratio,
                short_risk_reward_ratio = self.p.short_risk_reward_ratio,
                atr=self.atr
                )
        self.securized = False
            
        # Position Management
        if self.p.use_position_management:
            self.position_management_strategy = PositionManagement(
                data=self.datas[0],
                logger=self.log,
                bollinger_bands=self.bollinger_bands,
                rsi=self.rsi,
                atr=self.atr,
                atr_long_multiplier=self.p.atr_long_multiplier,
                atr_short_multiplier=self.p.atr_short_multiplier,
                rsi_in_long_lower=self.p.rsi_in_long_lower,
                rsi_in_long_upper=self.p.rsi_in_long_upper,
                rsi_in_short_lower=self.p.rsi_in_short_lower,
                rsi_in_short_upper=self.p.rsi_in_short_upper,
                rsi_out_long=self.p.rsi_out_long,
                rsi_out_short=self.p.rsi_out_short
                )
        self.bars_since_scaling = 0
        self.position_count = 0 


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
            high = self.data_high[0]
            low = self.data_low[0]
            init_stop_loss = self.stop_loss_price
            init_take_profit = self.take_profit_price

            self.log(f'{("Buy" if order.isbuy() else "Sell")} {order.info["info"]["trade_type"]} Order Executed, Price: {price:.2f}, '
                        f'Size: {size:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            
            # Fill the transaction dictionary for report
            self.record_transaction(order)

            # Keep track of the index (bar number) when the last order was executed  
            if trade_type=='entry':
                entry_side = order.info['info']['entry_side']
                if order.isbuy():
                    self.bar_executed_long = len(self)
                else:
                    self.bar_executed_short = len(self)
                self.bars_since_scaling = 0

                if self.p.use_trailing_strategy:        
                    self.trailing_strategy.set_initial_levels(price, high, low, init_stop_loss, init_take_profit, entry_side)
                    self.securized = False

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
            self.trailing_strategy.update_high_low(self.data)
        current_price = self.data_close[0]

        if len(self.order_refs) > 3  or len(self.order_refs) == 1 :
            self.log(f"reference, {self.order_refs}")

        # NO PENDING ORDERS
        if not self.order_refs:

            # NO POSITION 
            if self.position.size == 0 :  

                # POSITIVE SIGNAL
                if self.cross > 0.0 :
                    entry_valid = timedelta(self.p.long_validity)
                    brackets_valid= timedelta(self.p.brackets_validity)
            
                    if self.p.long_exectype == bt.Order.Limit:
                        self.entry_price = current_price * (1.0 - self.p.long_limit)
                    else:
                        self.entry_price = current_price

                    size = self.sizer.getsizing(comminfo = self.broker.getcommissioninfo(self.data), cash=self.broker.getcash(), data = self.data, isbuy = True, entry_price = self.entry_price)
                    self.entry_order = self.buy(price=self.entry_price, size=size, exectype=self.p.long_exectype , transmit=False, valid=entry_valid, info={'trade_type': 'entry', 'reason':'signal', 'entry_side':'long', 'order_type':self.exectype_map[self.p.long_exectype] })
                    self.log(f'Buy {self.exectype_map[self.p.long_exectype] } created at {self.entry_price:.2f}')
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
            

                # NEGATIVE SIGNAL
                elif self.cross < 0.0:
                    entry_valid = timedelta(self.p.short_validity)
                    brackets_valid = timedelta(self.p.brackets_validity) 
                        
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

            else: # Position is not nul
                self.log("Critical State: Open position exists but no related pending orders found. Check order management and execution.")


        # PENDING ORDERS
        else:
            if self.position :
                # Handle signal-based position closure (long & negative signal or short & positive signal -> close)
                if (self.position.size > 0.0 and self.cross < 0.0)  or (self.position.size < 0.0 and self.cross > 0.0): 
                    self.close_position(reason='signal')

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
                                    self.update_position(scale_in=scale_in , scale_out=scale_out)
                                    return

                    # Secure position using trailing strategy
                    if self.p.use_trailing_strategy :
                        if not self.securized:
                            secure_stop_loss, stop_loss_message = self.trailing_strategy.get_secure_stop_loss(self.data)
                            if secure_stop_loss != self.stop_loss_price and self.stop_loss_order.alive():
                                self.log(stop_loss_message)
                                self.update_brackets(new_stop_loss=secure_stop_loss)
                                self.securized = True
                                return
                        else:
                            new_stop_loss, new_take_profit = self.trailing_strategy.get_volatility_brackets(self.data)
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
        #if reason != 'scale out to zero':
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
        
        self.entry_price = self.data_close[0]
        current_cash = self.broker.getcash()
        current_value = self.broker.getvalue()
        comm_info = self.broker.getcommissioninfo(self.data)
        new_size = abs(self.position.size)
        if self.position.size > 0:
            percent_size = current_cash / self.entry_price * (self.p.sizers_percent / 100)
        elif self.position.size < 0:
            percent_size = (current_value - (current_cash - current_value)) / self.entry_price * (self.p.sizers_percent / 100)

        if scale_in:
            additional_size = percent_size * (1 + self.p.scale_in_factor)
            additional_cost = additional_size * self.entry_price + comm_info.getcommission(price = self.entry_price, size=additional_size)
            if self.position.size > 0:
                cash_available_for_trading = current_cash * self.p.max_authorized_position
            elif self.position.size < 0:
                cash_available_for_trading = (current_value - (current_cash - current_value)) * self.p.max_authorized_position

            if cash_available_for_trading > additional_cost:
                new_size += additional_size
                if self.position.size > 0:
                    self.entry_order = self.buy(size=additional_size, price=self.entry_price, exectype=bt.Order.Market, transmit=False, info={'trade_type': 'entry', 'reason':'scale in', 'entry_side':'long', 'order_type': 'Market'})
                elif self.position.size < 0:
                    self.entry_order = self.sell(size=additional_size, price=self.entry_price, exectype=bt.Order.Market, transmit=False, info={'trade_type': 'entry', 'reason':'scale in', 'entry_side':'short', 'order_type': 'Market'})
                self.log(f'Position Scaled in by {additional_size:.2f} at price {self.entry_price:.2f}')
                self.order_refs[self.entry_order.ref] = self.entry_order
            else:
                self.log(f'Unable to scale in due to insufficient cash: Required {additional_cost:.2f}, Available {cash_available_for_trading:.2f}')
                return


        elif scale_out:
            reduction_size = percent_size * (1 + self.p.scale_out_factor)
            new_size -= reduction_size

            if new_size > 0:
                if self.position.size > 0:
                    self.entry_order = self.sell(size=reduction_size, price=self.entry_price, exectype=bt.Order.Market, transmit=False, info={'trade_type':'entry', 'reason':'scale out', 'entry_side':'long', 'order_type':'Market'})
                elif self.position.size < 0:
                    self.entry_order = self.buy(size=reduction_size, price=self.entry_price, exectype=bt.Order.Market, transmit=False, info={'trade_type':'entry', 'reason':'scale out', 'entry_side':'short', 'order_type':'Market'})
                self.log(f'Position Scaled out by {reduction_size:.2f} at price {self.entry_price:.2f}')
                self.order_refs[self.entry_order.ref] = self.entry_order
            else:
                self.close_position(reason='scale out to zero')
                return


        if scale_in or scale_out :
            self.update_brackets_for_new_size(new_size=new_size)


    def stop(self):
        """
        The function logs the final cumulative net profit, cumulative broker commissions, number of winning trades, and number of losing trades at the end of the strategy run.

        Parameters : 
            self : The strategy instance, giving access to instance variables and methods.
        """

        self.log(f'CUMULATIVE PROFIT NET: {self.cum_pnl_net:.2f}, CUMULATIVE BROKER COMMISSION: {self.cum_comm:.2f}')
        self.log(f'Winning Trade Count: {self.winning_trade}')
        self.log(f'Losing Trade Count: {self.losing_trade}')



# msg + video jon 

