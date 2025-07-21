class TrailingStrategy:
    """
    This class implement a trailing stop strategy to secure profits once a position reaches a specified threshold
    The stop loss is adjusted dynamically based on price movements, and the take profit can be optionally adjusted.
    """

    def __init__(self, data, logger, trail_stop_pct=None, trail_stop_abs=None, secure_pct=25, atr_multiplier=3.0, long_risk_reward_ratio=1/2, short_risk_reward_ratio=2/3, atr=None):
        """
        Args:
            trail_stop_pct (float, optional): Percentage to set the trailing stop from the entry price.
            trail_stop_abs (float, optional): Absolute dollar amount to set the trailing stop from the entry price.
            secure_pct (float): Percentage of gains to secure when the threshold is reached.
            atr_multiplier (float): Multiplier for ATR to determine volatility-based adjustments.
            long_risk_reward_ratio (float): Risk/reward ratio for long positions.
            short_risk_reward_ratio (float): Risk/reward ratio for short positions.
            atr (list): Average True Range values.
        """
        self.datas = data
        self.trail_stop_pct = trail_stop_pct
        self.trail_stop_abs = trail_stop_abs
        self.secure_pct = secure_pct
        self.stop_loss = None               # To track the stop loss
        self.take_profit = None
        self.entry_price = None             # Entry price of the trade
        self.entry_side = None              # Entry Side (long or short)
        self.highest_high = float('-inf')   # Initialize to a very low value
        self.lowest_low = float('inf')      # Initialize to a very high value
        self.atr_multiplier = atr_multiplier
        self.long_risk_reward_ratio = long_risk_reward_ratio
        self.short_risk_reward_ratio = short_risk_reward_ratio
        self.atr = atr
        self.log = logger


    def set_initial_levels(self, entry_price, high, low, init_stop_loss, init_take_profit, entry_side):
        """
        Set the initial trading levels including entry price, stop loss, and take profit.
        
        Args:
            entry_price (float): The entry price of the trade
            high (float): Initial high price
            low (float): Initial low price
            stop_loss (float): The initial stop loss level.
            take_profit (float): The initial take profit level.
            entry_side (string): Either 'long' or 'short'
        """
        self.entry_price = entry_price
        self.stop_loss = init_stop_loss       
        self.take_profit = init_take_profit  
        self.entry_side = entry_side
        self.highest_high = high    
        self.lowest_low = low     
        self.log(f"Initial Levels: Entry Price={entry_price:.2f}, Entry Side={entry_side}, Stop Loss={init_stop_loss:.2f}, Take Profit={init_take_profit:.2f}")


    def update_high_low(self):
        """
        Update the highest high and lowest low based on current high price and low price    
        """
        high = self.datas[0].high[0]    
        low = self.datas[0].low[0]

        self.highest_high = max(self.highest_high, high)
        self.lowest_low = min(self.lowest_low, low)


    def get_secure_stop_loss(self):
        """
        Calculate and adjust the trailing stop and optionally the take profit based on the current price.
        It staticly securise our position by adjusting the stop loss level and optionally adjust the take profit

        Returns:
        (float): New stop loss price
        """

        if self.entry_price is None:
            raise ValueError("Entry price not set. Call set_initial_levels first.")
        
        if self.trail_stop_pct is None and self.trail_stop_abs is None:
            raise ValueError("Trailing Stop amount not set. Set either trail_stop_pct or trail_stop_abs.")
        
        current_price = self.datas[0].close[0]
        stop_log_message = "No adjustment needed."  # Default message
        new_stop_loss = self.stop_loss  # Default to current stop loss if no new calculation is done
        
        # Check if current price is above the threshomd to strat trailing
        if self.entry_side == 'long':
            threshold_price = self.entry_price * (1 + self.trail_stop_pct / 100) if self.trail_stop_pct else self.entry_price + self.trail_stop_abs
            # Adjust the stop loss if the current price is above the threshold
            if current_price >= threshold_price:
                gain = current_price - self.entry_price
                new_stop_loss = self.entry_price + gain * self.secure_pct / 100
                self.stop_loss = max(self.stop_loss, new_stop_loss)
                stop_log_message = "Stop loss adjusted to securize long position."
            else:
                stop_log_message = "Current price for long position is below the threshold."

        elif self.entry_side == 'short':
            threshold_price = self.entry_price * (1 - self.trail_stop_pct / 100) if self.trail_stop_pct else self.entry_price - self.trail_stop_abs
            # Adjust the stop loss if the current price is below the threshold
            if current_price <= threshold_price:
                gain = self.entry_price - current_price
                new_stop_loss = self.entry_price - gain * self.secure_pct / 100
                self.stop_loss = min(self.stop_loss, new_stop_loss)
                stop_log_message = "Stop loss adjusted to securize short position."
            else:
                stop_log_message = "Current price for short position is above the threshold."

        return self.stop_loss, stop_log_message
    

    def get_volatility_brackets(self):
        """
        Dynamically adjust the stop loss and take profit based on volatility using ATR.
        However, it locks the profit to never go below a previous stop loss

        Returns:
            (float, float): New stop loss and new take profit levels adjusted for volatility.
        """
        stop_loss = self.stop_loss  # Because we use this function only after having used the first one but if not then change this

        current_price = self.datas[0].close[0]

        if self.entry_side == 'long':
            chandelier_stop = self.highest_high - self.atr[0] * self.atr_multiplier * self.long_risk_reward_ratio
            
            if chandelier_stop < current_price:
                stop_log_message = "Chandelier stop used for long stop loss."
                self.stop_loss = max(chandelier_stop, stop_loss)
            else: # Chandelier Stop was set higher the current price
                pass

            if self.stop_loss == stop_loss:
                stop_log_message = "Original stop loss retained."

            chandelier_tp = self.highest_high + self.atr[0] * self.atr_multiplier
            
            if chandelier_tp > current_price:
                tp_log_message = "Chandelier TP used for long take profit."
                self.take_profit = chandelier_tp
            else: # Chandelier TP was set lower the current price
                pass

 
        elif self.entry_side == 'short':
            chandelier_stop = self.lowest_low + self.atr[0] * self.atr_multiplier * self.short_risk_reward_ratio
            
            if chandelier_stop > current_price:
                stop_log_message = "Chandelier stop used for short stop loss."
                self.stop_loss = min(chandelier_stop, stop_loss)
            else: # Chandelier Stop was set lower than the current price
                pass

            if self.stop_loss == stop_loss:
                stop_log_message = "Original stop loss retained."

            chandelier_tp = self.lowest_low - self.atr[0] * self.atr_multiplier
            
            if chandelier_tp < current_price:
                tp_log_message = "Chandelier TP used for short take profit."
                self.take_profit = chandelier_tp
            else: # Chandelier TP was set higher than the current price
                pass

        self.log(stop_log_message)
        self.log(tp_log_message)

        return self.stop_loss, self.take_profit

