from __future__ import (absolute_import, division, print_function, unicode_literals)
import backtrader as bt
from backtrader.indicators import Indicator, ExponentialMovingAverage, StdDev
from pykalman import KalmanFilter
import numpy as np



class EMABollingerBands(Indicator):
    '''
    Defined by John Bollinger in the 80s. It measures volatility by defining
    upper and lower bands at distance x standard deviations

    Formula:
      - midband = ExponentialMovingAverage(close, period)
      - topband = midband + top_devfactor * StandardDeviation(data, period)
      - botband = midband - low_devfactor * StandardDeviation(data, period)
    '''
    alias = ('EMA_BBands',)

    lines = ('mid', 'top', 'bot',)
    params = (('period', 20), 
              ('top_devfactor', 2.0),               
              ('bot_devfactor', 2.0), 
              ('movav', ExponentialMovingAverage),)

    plotinfo = dict(subplot=False)
    plotlines = dict(
        mid=dict(ls='--'),
        top=dict(_samecolor=True),
        bot=dict(_samecolor=True),
    )
    
    def _plotlabel(self):
        plabels = [self.p.period, self.p.top_devfactor, self.p.bot_devfactor]
        plabels += [self.p.movav] * self.p.notdefault('movav')
        return plabels

    def __init__(self):
        self.lines.mid = ema = self.p.movav(self.data, period=self.p.period)
        top_stddev = self.p.top_devfactor * StdDev(self.data, ema, period=self.p.period, movav=self.p.movav)        
        bot_stddev = self.p.bot_devfactor * StdDev(self.data, ema, period=self.p.period, movav=self.p.movav)
        self.lines.top = ema + top_stddev
        self.lines.bot = ema - bot_stddev

        super(EMABollingerBands, self).__init__()


class CustomROCR(bt.Indicator):
    lines = ('rocr',)
    params = (('period', 10),)

    def __init__(self):
        self.addminperiod(self.params.period + 1)
    
    def next(self):
        past_price = self.data.close[-self.params.period]
        current_price = self.data.close[0]
        if past_price != 0:  # Prevent division by zero
            self.lines.rocr[0] = (current_price / past_price) * 100  - 100


class KalmanFilterIndicator(bt.Indicator):
    lines = ('filtered',)
    params = (('initial_state', 0), ('observation_covariance', 1.0), ('transition_covariance', 0.01),)
    
    def __init__(self):
        # Initialize the Kalman Filter
        self.kf = KalmanFilter(initial_state_mean=self.p.initial_state,
                               observation_covariance=self.p.observation_covariance,
                               transition_covariance=self.p.transition_covariance)
        # This variable will keep the latest state and covariance
        self.state_estimate = self.p.initial_state
        self.cov_estimate = 1.0  # Example initial covariance
        self.addminperiod(1)

    def next(self):
        current_price = self.data[0]  # Current closing price
        # Update the filter with the current price
        state_estimate, cov_estimate = self.kf.filter_update(
            filtered_state_mean=self.state_estimate,
            filtered_state_covariance=self.cov_estimate,
            observation=current_price
        )
        self.state_estimate = state_estimate
        self.cov_estimate = cov_estimate
        self.lines.filtered[0] = np.sqrt(state_estimate)  


class WeightedMeanVolume(bt.Indicator):
    lines = ('weighted_mean_volume',)
    params = (('period', 4),)

    def __init__(self):
        self.addminperiod(self.params.period)
        self.weights = np.linspace(1, 0, self.params.period)
        self.weights /= self.weights.sum()  # Normalize weights

    def next(self):
        volumes = np.array([self.data.volume[-i] for i in range(1, self.params.period + 1)])
        self.lines.weighted_mean_volume[0] = np.dot(volumes, self.weights)


class SqrtWeightedMeanVolume(bt.Indicator):
    lines = ('sqrt_weighted_mean_volume',)
    params = (('period', 4),)

    def __init__(self):
        self.addminperiod(self.params.period)
        self.wmv = WeightedMeanVolume(self.data, period=self.params.period)
    
    def next(self):
        if self.wmv[0] > 0:  # Ensure positive before applying sqrt
            self.lines.sqrt_weighted_mean_volume[0] = np.sqrt(self.wmv[0])
        else:
            self.lines.sqrt_weighted_mean_volume[0] = 0
