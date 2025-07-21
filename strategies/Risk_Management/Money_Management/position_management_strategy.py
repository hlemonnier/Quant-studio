class PositionManagement:
    """
    Manages scaling in and out of positions for a trading strategy based on volatility.
    """

    def __init__(self, data, logger, atr, atr_long_multiplier=1.5, atr_short_multiplier=1, signal_generator=None):
        """
        Initializes the position management system with necessary market data and indicators.

        Args:
            data (datafeed): The market data feed.
            position (tuple): Current position info, typically including size and other relevant details.
            logger (function): Logging function to output diagnostic messages.
            atr (ATR): Average True Range indicator instance for assessing market volatility.
            atr_long_multiplier (float): Multiplier to adjust ATR for setting dynamic thresholds for long entry.
            atr_short_multiplier (float): Multiplier to adjust ATR for setting dynamic thresholds for short entry.
        """
        self.datas = data
        self.log = logger
        self.atr = atr
        self.atr_long_multiplier = atr_long_multiplier
        self.atr_short_multiplier = atr_short_multiplier
        self.position_size = None
        self.entry_price = None
        self.tradentry_side_type = None
        self.signal_generator = signal_generator

    def set_initial_levels(self, entry_price, entry_side, position_size):
        """
        Set the initial trading levels including entry price.
        
        Args:
            entry_price (float): The entry price of the trade
            entry_side (string): Either 'long' or 'short'
            position_size (int): Size of the position.
        """
        self.entry_price = entry_price
        self.entry_side = entry_side
        self.position_size = position_size
        self.log(f"Initial levels set: Entry Price={entry_price:.2f}, Entry Side={entry_side}, Position Size={position_size:.2f}")
        

    def _atr_threshold(self, scale_type):
        """Calculates ATR thresholds for scaling in or out based on trade type."""
        atr = self.atr[0]
        if self.entry_side == 'long':
            return self.entry_price + atr * self.atr_long_multiplier if scale_type == 'in' else self.entry_price - atr * self.atr_long_multiplier
        elif self.entry_side == 'short':
            return self.entry_price - atr * self.atr_short_multiplier if scale_type == 'in' else self.entry_price + atr * self.atr_short_multiplier
        return self.entry_price


    def should_scale_in(self) -> bool:
        """
        Evaluates if conditions are favorable for scaling in the position.
        
        Returns:
            bool: True if conditions are met to increase the position size, False otherwise.
        """
        if self.entry_price is None or self.entry_side is None:
            self.log("Cannot evaluate scale-in conditions: Entry price or trade type not set")
            return False

        current_price = self.datas[0].close[0]
        atr_threshold = self._atr_threshold('in')
        long_scaling_signals, short_scaling_signals = self.signal_generator.scaling_signal_generator()


        if atr_threshold != self.entry_price :
            if long_scaling_signals != short_scaling_signals :            
                if (self.entry_side == 'long' and current_price > atr_threshold and long_scaling_signals > short_scaling_signals) or (self.entry_side == 'short' and current_price < atr_threshold and long_scaling_signals < short_scaling_signals):
                    self.log(f"Scaling In ({self.entry_side}): Conditions met at price {current_price:.2f}, ATR Threshold {atr_threshold:.2f}")
                    return True

        return False



    def should_scale_out(self) -> bool:
        """
        Evaluates if conditions are favorable for scaling out the position.
        
        Returns:
            bool: True if conditions are met to reduce the position size, False otherwise.
        """
        if self.entry_price is None or self.entry_side is None:
            self.log("Cannot evaluate scale-out conditions: Entry price or trade type not set")
            return False

        current_price = self.datas[0].close[0]
        atr_threshold = self._atr_threshold('out')
        long_scaling_signals, short_scaling_signals = self.signal_generator.scaling_signal_generator()

        if atr_threshold != self.entry_price :
            if long_scaling_signals != short_scaling_signals :
                if (self.entry_side == 'long' and current_price < atr_threshold and long_scaling_signals < short_scaling_signals) or (self.entry_side == 'short' and current_price > atr_threshold and long_scaling_signals > short_scaling_signals ):
                    self.log(f"Scaling Out ({self.entry_side}): Conditions met at price {current_price:.2f}, ATR Threshold {atr_threshold:.2f}")
                    return True
        return False
