"""
This module defines custom observers for use with the Backtrader framework:
OrderObserver and DrawdownLength.

Observers in Backtrader are used to monitor and track certain aspects of the 
strategy's performance or the market data. They can plot or log information 
that helps in analyzing the strategy's behavior.

OrderObserver tracks the creation and expiration of orders and visualizes
them on the plot. DrawdownLength tracks the duration of drawdowns and the
maximum drawdown length observed during the strategy execution.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt

class OrderObserver(bt.observer.Observer):
    """
    Observer to track and plot order creation and expiration events.

    Lines:
        - created: Marks the creation of an order.
        - expired: Marks the expiration of an order.

    Plotinfo:
        - plot: Whether to plot this observer.
        - subplot: Whether to plot this as a subplot.
        - plotlinelabels: Whether to plot line labels.

    Plotlines:
        - created: Custom plotting parameters for created orders.
        - expired: Custom plotting parameters for expired orders.
    """

    lines = ('created', 'expired',)

    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)

    plotlines = dict(
        created=dict(marker='*', markersize=8.0, color='lime', fillstyle='full'),
        expired=dict(marker='D', markersize=8.0, color='red', fillstyle='full')
    )

    def next(self):
        """
        This method is called on each next bar to update the observer lines.
        """
        for order in self._owner._orderspending:
            if order.status in [bt.Order.Accepted, bt.Order.Submitted]:
                self.lines.created[0] = order.created.price
            elif order.status in [bt.Order.Expired]:
                self.lines.expired[0] = order.created.price


class DrawdownLength(bt.Observer):
    """
    Observer to track and plot the length of drawdowns and the maximum drawdown length.

    Lines:
        - L: Current drawdown length.
        - maxL: Maximum drawdown length observed.

    Plotinfo:
        - plot: Whether to plot this observer.
        - subplot: Whether to plot this as a subplot.
        - plotname: Name of the plot.

    Plotlines:
        - L: Custom plotting parameters for current drawdown length.
        - maxL: Custom plotting parameters for maximum drawdown length.
    """

    lines = ('L', 'maxL',)

    plotinfo = dict(plot=True, subplot=True, plotname='Drawdown Length')

    plotlines = dict(
        L=dict(ls='-', linewidth=1.0, color='blue'),
        maxL=dict(ls='--', linewidth=1.0, color='black')
    )

    def __init__(self):
        """
        Initialize the observer and add a DrawDown analyzer.
        """
        self._dd = self._owner._addanalyzer_slave(bt.analyzers.DrawDown)

    def next(self):
        """
        This method is called on each next bar to update the observer lines.
        """
        self.lines.L[0] = self._dd.rets.len
        self.lines.maxL[0] = self._dd.rets.max.len
