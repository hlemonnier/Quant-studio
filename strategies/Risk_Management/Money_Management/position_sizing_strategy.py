'''
This module defines three different sizer classes for use with the Backtrader framework:
FixedSize, PercentSizer, and RiskAdjustedSizer.

Sizers are used our trading strategies to determine the size of each trade. They control how 
much capital to allocate for each position based on various criteria and strategies.

The FixedSize sizer allocates a fixed number of units per trade. 
The PercentSizer allocates a percentage of available cash for each trade. 
The RiskAdjustedSizer adjusts the size based on risk parameters and market volatility.
'''

from __future__ import (absolute_import, division, print_function, unicode_literals)
import backtrader as bt
from datetime import datetime, timedelta

# entry_price == data.close or entry_price in argument

class FixedSize(bt.Sizer):
    """
    This sizer returns a fixed size for any operation. The size can be controlled
    by the number of tranches used to scale into trades by specifying the 'tranches' parameter.

    Params:
        - stake (default: 1): Number of units to trade.
        - tranches (default: 1): Number of tranches to divide the stake into.
    """

    params = (('stake', 1),
              ('tranches', 1))

    def getsizing(self, comminfo, cash, data, isbuy, entry_price):
        """
        Returns the size of the position to be taken.

        Parameters:
            comminfo: Commission information.
            cash: Available cash.
            data: Data feed.
            isbuy: True if the order is a buy order.
            entry_price: The entry price for the order.

        Returns:
            int: The size of the position.
        """
        if self.p.tranches > 1:
            return abs(int(self.p.stake / self.p.tranches))
        else:
            return self.p.stake

    def setsizing(self, stake):
        """
        Sets the size of the stake.

        Parameters:
            stake: The new stake size.

        Returns:
            None
        """
        if self.p.tranches > 1:
            self.p.stake = abs(int(stake / self.p.tranches))
        else:
            self.p.stake = stake


class PercentSizer(bt.Sizer):
    """
    This sizer returns a percentage of available cash to be used for a position.

    Params:
        - percents (default: 20): Percentage of available cash to use.
        - retint (default: False): Return an integer size or a float value.
    """

    params = (('percents', 20),
              ('retint', False))

    def getsizing(self, comminfo, cash, data, isbuy, entry_price):
        """
        Returns the size of the position to be taken.

        Parameters:
            comminfo: Commission information.
            cash: Available cash.
            data: Data feed.
            isbuy: True if the order is a buy order.
            entry_price: The entry price for the order.

        Returns:
            int or float: The size of the position.
        """
        position = self.broker.getposition(data)
        if not position:
            size = cash / entry_price * (self.p.percents / 100)
        else:
            size = position.size

        if self.p.retint:
            size = int(size)

        return size


class RiskAdjustedSizer(bt.Sizer):
    """
    This sizer adjusts the size of the position based on the risk parameters and volatility.

    Params:
        - max_risk_per_trade (default: 0.001): Maximum risk per trade as a percentage of total capital.
        - stop_loss_deviation (default: 1): Multiplier for ATR to calculate the stop-loss distance.
        - max_authorized_position (default: 0.2): Maximum percentage of capital authorized for a single position.
        - atr_length (default: 12): Number of bars used to calculate the ATR for volatility assessment.
    """

    params = dict(
        max_risk_per_trade=0.001,
        stop_loss_deviation=1,
        max_authorized_position=0.2    
        )

    def __init__(self, atr):
        self.atr = atr

    def getsizing(self, comminfo, cash, data, isbuy, entry_price):
        """
        Calculates the position size based on provided entry price and strategy parameters.

        Parameters:
            comminfo: Commission information.
            cash: Available cash.
            data: Data feed.
            isbuy: True if the order is a buy order.
            entry_price: The entry price for the order.

        Returns:
            float: The size of the position.
        """
        position = self.broker.getposition(data)
        if not position:
            max_authorized_loss = self.p.max_risk_per_trade * cash
            stop_loss_price = (entry_price - self.p.stop_loss_deviation * self.atr[0]) if isbuy else (entry_price + self.p.stop_loss_deviation * self.atr[0])
            risk_per_unit = abs(entry_price - stop_loss_price)
            quantity_to_open = max_authorized_loss / risk_per_unit
            quantity_to_trade = min(quantity_to_open, cash * self.p.max_authorized_position / entry_price)
            size = quantity_to_trade
        else:
            size = position.size
            
        return size
    

class GridVolatilityAdjustedSizer(bt.Sizer):
    """
    This sizer adjusts the size of the position based on the risk parameters and volatility,
    tailored for a grid trading strategy.

    Params:
        - max_risk_per_trade (default: 0.001): Maximum risk per trade as a percentage of total capital.
        - stop_loss_deviation (default: 1): Multiplier for ATR to calculate the stop-loss distance.
        - max_authorized_position (default: 0.2): Maximum percentage of capital authorized for a single position.
        - atr_length (default: 12): Number of bars used to calculate the ATR for volatility assessment.
        - num_grid_levels (default: 10): Number of grid levels.
    """

    params = dict(
        max_risk_per_trade=0.001,
        stop_loss_deviation=1,
        max_authorized_position=0.2,
    )

    def __init__(self, atr):
        self.atr = atr

    def getsizing(self, comminfo, value, data, isbuy, entry_price, grid_len):
        """
        Calculates the position size based on provided entry price and strategy parameters,
        adjusted by volatility (ATR) and divided by the remaining grid levels.

        Parameters:
            comminfo: Commission information.
            cash: Available cash.
            data: Data feed.
            isbuy: True if the order is a buy order.
            entry_price: The entry price for the order.
            remaining_grid_levels: Number of remaining grid levels.

        Returns:
            float: The size of the position.
        """
        position = self.broker.getposition(data)
        max_authorized_loss = self.p.max_risk_per_trade * value
        stop_loss_price = (entry_price - self.p.stop_loss_deviation * self.atr[0]) if isbuy else (entry_price + self.p.stop_loss_deviation * self.atr[0])
        risk_per_unit = abs(entry_price - stop_loss_price)
        quantity_to_open = max_authorized_loss / risk_per_unit
        quantity_to_trade = min(quantity_to_open, value * self.p.max_authorized_position / entry_price)
        size = quantity_to_trade
        size = quantity_to_trade / grid_len 
        return size
