import pandas as pd
import numpy as np
import joblib
import sys
import os
'''
Idea :  - Give weights to some signals 
        - Ameliorer le signal avec stacking, date en cyclique par semaine, heure (Goal : 58%)
        - Need more signal on maybe shorter term data.
        - We can imporve all signals, espcially the scaling signals.
    
'''




class SignalGenerator:
    """
    This class generates some signals based  on current market condition and send them to the strategy.

    """
        
    def __init__(self, data, logger, args,
                adx, adx_strength, ema1,
                sma, sma_slope_limit, sma_slope_period, 
                use_ema_bb, ema_bollinger_bands, 
                use_rsi, rsi, rsi_lower_threshold, rsi_upper_threshold, 
                use_stochastic_oscillator, stochastic_oscillator, 
                use_machine_learning, model_path,
                price_close_d, willr, rocr, atr, rsi_ml, mom, macd, kalman, sqrt_weighted_mean_volume_traded, difference_mean):
        """
        Initializes the position management system with necessary market data and indicators.

        Args:
            data (datafeed): The market data feed.
            position (tuple): Current position info, typically including size and other relevant details.
            logger (function): Logging function to output diagnostic messages.
            ema_bollinger_bands (EMA_BollingerBands): Built in EMA Bollinger Bands indicator instance.
            rsi (RSI): Relative Strength Index indicator instance.
        """
        self.datas = data
        self.log = logger
        self.args = args

        self.long_entry_signals = 0
        self.short_entry_signals = 0
        self.long_close_signals = 0
        self.short_close_signals = 0

        # Indicator
        self.sma = sma
        self.sma_slope_limit = sma_slope_limit
        self.sma_slope_period = sma_slope_period

        self.ema1 = ema1
        self.adx = adx
        self.adx_strength = adx_strength

        self.use_stochastic_oscillator = use_stochastic_oscillator
        self.stochastic_oscillator = stochastic_oscillator        
        self.use_ema_bb = use_ema_bb
        self.ema_bollinger_bands = ema_bollinger_bands
        self.use_machine_learning = use_machine_learning
        self.model_path = model_path
        self.use_rsi = use_rsi
        self.rsi = rsi
        self.rsi_lower_threshold = rsi_lower_threshold,
        self.rsi_upper_threshold = rsi_upper_threshold,


        # Calculate the root directory of the project, going up three levels
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        # Append the project root directory to the system path
        sys.path.insert(0, project_root)
        
        # Construct the full path to the model file
        full_model_path = os.path.join(project_root, self.model_path)
        with open(full_model_path, 'rb') as f:
            self.model = joblib.load(f)

            # PATCH pour XGBClassifier (récupère le bon estimateur, corrige l’attribut manquant)
            if hasattr(self.model, "named_estimators_") and 'xgb' in self.model.named_estimators_:
                xgb = self.model.named_estimators_['xgb']
                if not hasattr(xgb, "use_label_encoder"):
                    xgb.use_label_encoder = False


        # ML Features
        self.price_close_d = price_close_d
        self.willr = willr
        self.rocr = rocr
        self.atr = atr
        self.rsi_ml = rsi_ml
        self.mom = mom
        self.macd = macd
        self.kalman = kalman
        self.sqrt_weighted_mean_volume_traded = sqrt_weighted_mean_volume_traded
        self.difference_mean = difference_mean
        self.feature_names = ['RSI', 'ATR', 'difference_mean', 'ROCR', 'WILLR']

        self.sma_long_count = 0 
        self.bb_long_count = 0 
        self.stochastic_oscillator_long_count = 0         
        self.sma_short_count = 0 
        self.bb_short_count = 0 
        self.stochastic_oscillator_short_count = 0 
        self.sma_close_short_count = 0     
        self.sma_close_long_count = 0 

    def get_features(self):
        features = [
            self.rsi_ml[0],
            self.atr[0],
            self.difference_mean[0],
            self.rocr[0],
            self.willr[0],
        ]
        return pd.DataFrame([features], columns=self.feature_names)
    


    def get_slope(self, ma, slope_period = 10):

        if len(ma) < slope_period:
            return 0 
        
        ma_values = [ma[-i-1] for i in range(slope_period)]

        if len(ma_values) < 2:
            return 0
        
        # Ensure all values are finite (not NaN or Inf)
        if not np.all(np.isfinite(ma_values)):
            return 0
        
        # Calculate the difference between the last and first value
        delta = ma_values[0] - ma_values[-1]
        # Normalize by the number of periods
        slope = delta / (len(ma_values) - 1)

        if slope is None:
            return 0
        
        return slope
    

    def entry_signal_generator(self, position_size):
        """
        Generates trading entry signals based on current conditions, specifically for pullbacks.
        
        Parameters :
            self : The strategy instance, giving access to instance variables and methods.
            position_size : The size of the current position.

        Returns: 
            int: Number of long entry signals.            
            int: Number of short entry signals.
        """
        self.long_entry_signals, self.short_entry_signals = 0, 0

        sma_slope = self.get_slope(self.sma, self.sma_slope_period)
        #self.log(sma_slope)

        y_pred = -1
        if self.use_machine_learning:
            current_datetime = self.datas[0].datetime.datetime(0)
            if current_datetime.minute == 0:  # Checks if it's the top of the hour
                X_new = self.get_features()
                y_pred = self.model.predict(X_new)            
                # self.log(f"Close Price : {self.datas[1].close[0]} - Prediction : {y_pred}")
        
        if self.use_stochastic_oscillator:
            # Lines of Stochastic Oscillator for easier access
            self.percK = self.stochastic_oscillator.percK
            self.percD = self.stochastic_oscillator.percD

        
        if position_size == 0:
            long_signals = 0
            short_signals = 0


            if self.use_ema_bb:
                if self.adx > self.adx_strength and sma_slope > self.sma_slope_limit :
                    # Close crosses the BB low band from below
                    if self.datas[0].close[0] > self.ema_bollinger_bands.lines.bot[0] and self.datas[0].close[-1] <= self.ema_bollinger_bands.lines.bot[-1]:
                        long_signals += 1
                        self.bb_long_count += 1          
        
                
            # General Uptrend and confirming trend strength    
            if self.adx > self.adx_strength and sma_slope > self.sma_slope_limit : 
                if self.datas[0].close[-1] >= self.ema1[-1] and self.datas[0].close[0] > self.ema1 :
                    if self.use_machine_learning : 
                        if y_pred == 1 :
                            long_signals += 1
                            self.sma_long_count += 1
                    else:
                        long_signals += 1
                        self.sma_long_count += 1
                        
            elif self.adx > self.adx_strength and sma_slope < -self.sma_slope_limit:
                if self.datas[0].close[-1] <= self.ema1[-1] and self.datas[0].close[0] < self.ema1 :
                    if self.use_machine_learning :
                        if y_pred == 0 :
                            short_signals += 1
                            self.sma_short_count += 1
                    else:
                        short_signals += 1
                        self.sma_short_count += 1

            if self.use_stochastic_oscillator :
                if self.datas[0].close[0] > self.sma :
                    # %K crosses %D from below in oversold region (<20)
                    if self.percK[-1] < self.percD[-1] and self.percK[0] > self.percD[0] and self.percK[0] < 15 :                        
                        if self.use_machine_learning : 
                            if y_pred == 1 :
                                long_signals += 1
                                self.stochastic_oscillator_long_count += 1
                        else:
                            long_signals += 1
                            self.stochastic_oscillator_long_count += 1

                if self.datas[0].close[0] < self.sma :
                    # %K crosses %D from above in overbought region (>80) AND  Close crosses the BB upper band from above
                    if self.percK[-1] > self.percD[-1] and self.percK[0] < self.percD[0] and self.percK[0] > 85 :
                        if self.use_machine_learning :
                            if y_pred == 0 :
                                short_signals += 1
                                self.stochastic_oscillator_short_count += 1

                        else:
                            short_signals += 1
                            self.stochastic_oscillator_short_count += 1

            # Technical Ensemble Strategy
            if (long_signals >= 2) or (short_signals >= 2) or (long_signals == short_signals == 1) :
                self.log(f'Here: Long Signals =  {long_signals}, Short Signals = {short_signals}')

            if long_signals > short_signals:
                # other logic to handle risk ( short signals then no entry for exemple)
                self.long_entry_signals = 1
            elif short_signals > long_signals:
                self.short_entry_signals = 1

        return self.long_entry_signals, self.short_entry_signals



    def close_signal_generator(self, position_size):
        """
        Generates trading close signals based on current conditions.

        Parameters :
            self : The strategy instance, giving access to instance variables and methods.
            position_size : The size of the current position.

        Returns: 
            int: Number of long close signals.            
            int: Number of short close signals.
        """
        self.long_close_signals, self.short_close_signals = 0, 0

        if position_size > 0:
            # General uptrend weakening
            if self.datas[0].close[0] < self.sma and self.datas[0].close[0] < self.ema1 :
                    self.long_close_signals += 1
                    self.sma_close_long_count += 1 
        
        elif position_size < 0:
            # General downtrend weakening
            if self.datas[0].close[0] > self.sma and self.datas[0].close[0] > self.ema1 :
                    self.short_close_signals += 1
                    self.sma_close_short_count += 1     


        return self.long_close_signals, self.short_close_signals
    


    def scaling_signal_generator(self):
        """
        Generates trading scaling signals based on current conditions, specifically for pullbacks.

        Returns: 
            int: Number of long scaling signals.            
            int: Number of short scaling signals.
        """
        self.long_scaling_signals, self.short_scaling_signals = 0, 0
        long_signals = 0
        short_signals = 0

        sma_slope = self.get_slope(self.sma, self.sma_slope_period)
          
        # General Uptrend and confirming trend strength    
        if self.adx > self.adx_strength and sma_slope > self.sma_slope_limit : 
            if self.datas[0].close[-1] >= self.ema1[-1] and self.datas[0].close[0] > self.ema1 :
                    long_signals += 1
                        
        elif self.adx > self.adx_strength and sma_slope < -self.sma_slope_limit:
            if self.datas[0].close[-1] <= self.ema1[-1] and self.datas[0].close[0] < self.ema1 :
                    short_signals += 1

        # Technical Ensemble Strategy
        if (long_signals >= 2) or (short_signals >= 2) or (long_signals == short_signals == 1) :
            self.log(f'Here: Long Signals =  {long_signals}, Short Signals = {short_signals}')

        if long_signals > short_signals:
            self.long_scaling_signals += 1
        elif short_signals > long_signals:
            self.short_scaling_signals += 1

        return self.long_scaling_signals, self.short_scaling_signals



    def get_signal_count(self):

        return (self.sma_long_count, self.bb_long_count, self.stochastic_oscillator_long_count, 
    self.sma_short_count, self.bb_short_count, self.stochastic_oscillator_short_count, 
    self.sma_close_long_count, self.sma_close_short_count)