import backtrader as bt

parameters_details = {
    "data_used": {
        "default": "Market Data",
        "type": str,
        "choices": ["Market Data", "Simulated Data"],
        "help": "Dataset used for backtesting"
    },   
    "timeframe_shortterm": {
        "default": "15MIN",
        "type": str,
        "help": "Short term Timeframe in [1MIN, 5MIN, 15MIN]"
    },    
    "timeframe_longterm": {
        "default": "1HRS",
        "type": str,
        "help": "Long term Timeframe in [1HRS, 4HRS]"
    },
    "fromdate": {
        "default": "2023-12-05",
        "type": str,
        "help": "Start date in YYYY-MM-DD format"
    },
    "todate": {
        "default": "2024-06-11",
        "type": str,
        "help": "End date in YYYY-MM-DD format"
    },

    # Broker & Commission Schemes
    "commission_percentage": {
        "default": 0.02,
        "type": float,
        "help": "Commission for each trade as a percentage of the trade nominal (for 3 bps we put 0.03)"
    },
    "start_cash": {
        "default": 1000000,
        "type": float,
        "help": "Initial cash for the backtest"
    },
    "slippage_percentage": {
        "default": 0.0001,
        "type": float,
        "help": "Slippage percentage (e.g., between 0.5 and 2 BPS). Slippage is applied to Market and Stop orders but not Limit"
    },

    # Observers
    "timeframe_time_return": {
        "default": bt.TimeFrame.NoTimeFrame,
        "type": object,  # Specifying 'object' because the value will be one of the `backtrader.TimeFrame` enum values
        "help": "TimeFrame to apply to the Observer",
        "choices": [ bt.TimeFrame.Minutes, bt.TimeFrame.Days, bt.TimeFrame.Weeks, bt.TimeFrame.Months, bt.TimeFrame.Years, bt.TimeFrame.NoTimeFrame]
    },

    # Analyzers
    "benchmark": {
        "default": "BTC-USD",
        "type": str,
        "help": "Benchmark for comparing strategy performance"
    },
    "riskfree_rate": {
        "default": 0.01,
        "type": float,
        "help": "Risk-free rate for computing ratios like Sharpe and Sortino"
    },
    "periods": {
        "default": 365,
        "type": int,
        "help": "Number of trading days in a year"
    },

    # Strategy
    "strategy_used": {
        "default": "LongShort",
        "type": str,
        "help": "Choose which strategy to apply",
        "choices": ['CrossOver_LongShort', 'BuyAndHold', 'LongShort', 'Grid', 'LongShortGrid'],
    },

    # Signal Parameters  
    "sma_period": {
        "default": 48,
        "type": int,
        "help": "Period for the long trend SMA (based on  long term data)"
    },   
    "sma_slope_limit": {
        "default": 0,
        "type": int,
        "help": "SLope need to be high (lower) than this limit to be considered as a trend"
    },    
    "sma_slope_period": {
        "default": 30,
        "type": int,
        "help": "Period used to compute the SMA Slope"
    },

    "adx_period": {
        "default": 25,
        "type": int,
        "help": "Period for the ADX"
    },        
    "adx_strength": {
        "default": 33,
        "type": int,
        "help": "Strength for the ADX"
    },  
    "ema_period1": {
        "default": 37,
        "type": int,
        "help": "Period for the EMA signal"
    },

    "use_stochastic_oscillator": {
        "default": True,
        "type": bool,
        "help": "Wether to use the Stochastic Oscillator"
    },    
    "stochastic_oscillator_period": {
        "default": 19,
        "type": int,
        "help": "Period used to compute the Stochastc Oscillator"
    },
    "stochastic_oscillator_period_dfast": {
        "default": 12,
        "type": int,
        "help": "Number of periods over which moving average is calculated."
    },

    "use_ema_bb": {
        "default": True,
        "type": bool,
        "help": "Wether to  use the Modified Bollinger Band with EMA"
    },    
    "bb_top_devfactor": {
        "default": 1.85,
        "type": float,
        "help": "Top Deviation Factor of the Modified BB"
    },
    "bb_bot_devfactor": {
        "default": 1.56,
        "type": float,
        "help": "Bottom Deviation Factor of the Modified BB"
    },
    "ema_bb_period": {
        "default": 19,
        "type": int,
        "help": "Period to compute Bollinger Band"
    },

    "use_machine_learning": {
        "default": True,
        "type": bool,
        "help": "Wether to  use the Machine Learning Prediction Signal (Need to train a model on 15 min data)"
    },       
    "model_path": {
        "default": "strategies/Signal/Machine_Learning/stacking_model.joblib",
        "type": str,
        "help": "Machine Learning Model PAth"
    },


    "use_rsi": {
        "default": True,
        "type": bool,
        "help": "Wether to  use the RSI"
    },       
    "rsi_period": {
        "default": 20,
        "type": int,
        "help": "Period to compute RSI"
    },
    "rsi_lower_threshold": {
        "default": 30,
        "type": int,
        "help": "Lower RSI threshold"
    },    
    "rsi_upper_threshold": {
        "default": 70,
        "type": int,
        "help": "Upper RSI threshold"
    },


    # Money management
    "sizers_type": {
        "default": "RiskAdjustedSizer",
        "type": str,
        "choices": ["PercentSizer", "FixedSize", "RiskAdjustedSizer"],
        "help": "Choose the sizer type"
    },
    "sizers_percent": {
        "default": 33,
        "type": float,
        "help": "Percentage of available cash to use for the PercentSizer"
    },
    "sizers_stake": {
        "default": 10,
        "type": int,
        "help": "Number of units to trade for the FixedSize sizer"
    },
    "max_risk_per_trade": {
        "default": 0.016,
        "type": float,
        "help": "Maximum risk per trade as a percentage of total capital"
    },
    "stop_loss_deviation": {
        "default": 4.75,
        "type": float,
        "help": "Multiplier for ATR to calculate the stop-loss distance"
    },
    "max_authorized_position": {
        "default": 0.4,
        "type": float,
        "help": "Maximum percentage of capital authorized for a single position (e.g., 0.3 for 30%)"
    },
    "atr_length": {
        "default": 23,
        "type": int,
        "help": "Number of bars used to calculate the ATR for volatility assessment"
    },

    # Order Management
    "brackets_validity": {
        "default": 100,
        "type": int,
        "help": "Secondary validity (bars) for the take profit and stop loss orders"
    },

    # LONG Order Management
    "long_exectype": {
        "default": bt.Order.Limit,
        "type": object,  # bt.Order type object
        "choices": [bt.Order.Limit, bt.Order.Market],
        "help": "Execution type for the GO LONG order"
    },
    "long_limit": {
        "default": 0.0038,
        "type": float,
        "help": "Limit for the GO LONG LIMIT order"
    },
    "long_validity": {
        "default": 5,
        "type": int,
        "help": "Validity (bars) for the GO LONG order"
    },
    "long_away_from_price": {
        "default": 0.033,
        "type": float,
        "help": "Distance from price for the stop loss and take profit of the GO LONG order (e.g., between 0.01 and 0.05)"
    },
    "long_risk_reward_ratio": {
        "default": 0.36,
        "type": float,
        "help": "LONG risk-reward ratio (e.g., 1/2 for a 1:2 ratio)"
    },
    "long_hold": {
        "default": 150,
        "type": int,
        "help": "Holding period (bars) for the LONG side"
    },
    "long_stop_loss_exectype": {
        "default": bt.Order.Stop,
        "type": object,  # bt.Order type object
        "choices": [bt.Order.Stop],
        "help": "Execution type for the LONG stop loss order"
    },
    "long_take_profit_exectype": {
        "default": bt.Order.Limit,
        "type": object,  # bt.Order type object
        "choices": [bt.Order.Limit, bt.Order.Stop],
        "help": "Execution type for the LONG take profit order"
    },

    # SHORT Order Management
    "short_exectype": {
        "default": bt.Order.Limit,
        "type": object,  # bt.Order type object
        "choices": [bt.Order.Limit, bt.Order.Market],
        "help": "Execution type for the GO SHORT order"
    },
    "short_limit": {
        "default": 0.004,
        "type": float,
        "help": "Limit for the GO SHORT LIMIT order"
    },
    "short_validity": {
        "default": 5,
        "type": int,
        "help": "Validity (bars) for the GO SHORT order"
    },
    "short_away_from_price": {
        "default": 0.014,
        "type": float,
        "help": "Distance from price for the stop loss and take profit of the GO SHORT order (e.g., between 0.01 and 0.05)"
    },
    "short_risk_reward_ratio": {
        "default": 0.34,
        "type": float,
        "help": "SHORT risk-reward ratio"
    },
    "short_hold": {
        "default": 145,
        "type": int,
        "help": "Holding period (bars) for the SHORT side"
    },
    "short_stop_loss_exectype": {
        "default": bt.Order.Stop,
        "type": object,  # bt.Order type object
        "choices": [bt.Order.Stop],
        "help": "Execution type for the SHORT stop loss order"
    },
    "short_take_profit_exectype": {
        "default": bt.Order.Limit,
        "type": object,  # bt.Order type object 
        "choices": [bt.Order.Limit, bt.Order.Stop],
        "help": "Execution type for the SHORT take profit order"
    },

    # Trailing Strategy - Order Management
    "use_trailing_strategy": {
        "default": True,
        "type": bool,
        "help": "Decide whether using the trailing strategy"
    },
    "trail_stop_pct": {
        "default": 1.5,
        "type": float,
        "help": "Stop trail percentage for the stop order in %"
    },
    "trail_stop_abs": {
        "default": None,
        "type": float,
        "help": "Stop trail absolute amount for the stop order"
    },
    "secure_pct": {
        "default": 30,
        "type": float,  
        "help": "Percentage of the gain profit made to secure"
    },
    "atr_multiplier": {
        "default": 3.5,
        "type": float,
        "help": "Multiplier of the ATR when computing the Chandelier Exit"
    },

    # Position Management Strategy
    "use_position_management": {
        "default": True,
        "type": bool,
        "help": "Decide whether using the position management strategy"
    },
    "atr_long_multiplier": {
        "default": 1.7,
        "type": float,
        "help": "Multiplier of the ATR when computing dynamic thresholds for scaling in/out for long side",
        "warning" : "The ATR Multiplier 2 must be inferior than the one concerning the trailing strategy"     
    },
    "atr_short_multiplier": {
        "default": 1.85,
        "type": float,
        "help": "Multiplier of the ATR when computing dynamic thresholds for scaling in/out for short side entry"
    },
    "scale_out_factor": {
        "default": 0.60,
        "type": float,
        "help": "Factor to determine the size to scale out"
    },
    "scale_in_factor": {
        "default": 0.561,
        "type": float,
        "help": "Factor to determine the size to scale in"
    },
    "position_scaling_hold": {
        "default": 4,
        "type": int,
        "help": "Number of bars before considering scaling in or out"
    },
    "max_position_count": {
        "default": 4,
        "type": int,
        "help": "Maximum number of reposition"
    },

    # Logs - Tearsheet - Report - Plot
    "verbose": {
        "default": True,
        "type": bool,
        "help": "Print logs during the strategy"
    },

    "print_logs": {
        "default": True,
        "type": bool,
        "help": "Print information, results, and analysis of the strategy used"
    },
    "print_tearsheet": {
        "default": True,
        "type": bool,
        "help": "Create the Quantstats HTML Tearsheet"
    },
    "plot": {
        "default": True,
        "type": bool,
        "help": "Plot results with optional kwargs in key=value format"
    },
    "write": {
        "default": True,
        "type": bool,
        "help": "Write the information to an Excel file"
    }
}