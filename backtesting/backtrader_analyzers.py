"""
SORTINO RATIO ANALYZER - NOT USED 

This module defines a custom analyzer for calculating the Sortino Ratio using the R package PerformanceAnalytics.
The Sortino Ratio is a risk-adjusted performance metric that evaluates the return of an investment relative to the
downside risk, which is measured as the deviation of negative asset returns from a specified minimum acceptable return (MAR).

Note: This analyzer is not currently used in the main backtesting framework.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import backtrader as bt
from pandas import Series
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

# Activate R's data conversion for numpy and pandas objects
numpy2ri.activate()
pandas2ri.activate()

# Import the PerformanceAnalytics package from R
pa = importr("PerformanceAnalytics")

class SortinoRatio(bt.Analyzer):
    """
    Computes the Sortino Ratio for the entire account using the strategy, based on the R package
    PerformanceAnalytics SortinoRatio function.

    Parameters:
        MAR (float): Minimum Acceptable Return (MAR). This must be in the same periodicity as the data.
    """
    params = {"MAR": 0}

    def __init__(self):
        self.acct_return = dict()
        self.acct_last = self.strategy.broker.get_value()
        self.sortinodict = dict()

    def next(self):
        """
        Computes the log returns of the account value at each step.
        """
        if len(self.data) > 1:
            curdate = self.strategy.datetime.date(0)
            self.acct_return[curdate] = np.log(self.strategy.broker.get_value()) - np.log(self.acct_last)
            self.acct_last = self.strategy.broker.get_value()

    def stop(self):
        """
        Finalizes the calculation of the Sortino Ratio at the end of the backtest.
        """
        srs = Series(self.acct_return)  # Convert the returns dictionary to a pandas Series
        srs.sort_index(inplace=True)    # Sort by date index
        self.sortinodict['sortinoratio'] = pa.SortinoRatio(srs, MAR=self.params.MAR)[0]  # Calculate Sortino Ratio
        del self.acct_return  # Clean up the returns dictionary

    def get_analysis(self):
        """
        Returns the calculated Sortino Ratio.

        Returns:
            dict: A dictionary containing the Sortino Ratio.
        """
        return self.sortinodict
