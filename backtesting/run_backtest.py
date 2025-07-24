"""
This script sets up the necessary environment for running backtests on trading strategies.
It includes database connection setup, data preprocessing, strategy execution, and result analysis with logs, plots and reports.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')
import psycopg2
import sys
import os
import backtrader as bt
import backtrader.analyzers as btanalyzers
import quantstats

# Append project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import configuration and strategies
#from data.database.config_database import load_config
from parameters.config_file import Config
from backtesting.backtrader_observers import OrderObserver
from strategies.strategies_model.strategy_crossover_long_short import CrossOver_LongShort
from strategies.strategies_model.strategy_buy_and_hold import BuyAndHold
from strategies.strategies_model.strategy_long_short import LongShort
from strategies.strategy_long_short_grid import LongShortGrid
from strategies.strategies_model.strategy_grid_trading import Grid
from backtesting.backtesting_reports_generator import ExcelReport

# def get_database_connection():
#     """
#     Établit une connexion à la base de données PostgreSQL en utilisant les détails de configuration.
#
#     Retourne:
#         conn (psycopg2.extensions.connection): L'objet de connexion à la base de données.
#     """
#     # Charger la configuration de connexion à la base de données (hôte, port, user, password, dbname, etc.)
#     config = load_config()
#
#     # Établir une connexion à PostgreSQL avec les paramètres récupérés
#     conn = psycopg2.connect(**config)
#
#     # Retourner l'objet de connexion pour interagir avec la base de données
#     return conn


def fetch_table_data(conn, table_name):
    """
    Fetches data from the specified PostgreSQL table and loads it into a pandas DataFrame.

    Parameters:
        conn (psycopg2.extensions.connection): The database connection object.
        table_name (str): The name of the table to fetch data from.

    Returns:
        df (pd.DataFrame): The DataFrame containing the table data.
    """
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, conn)
    return df



def get_table_name(period_id):
    """
    Constructs the table name based on the given period ID.

    Parameters:
        period_id (str): The period identifier.

    Returns:
        str: The constructed table name.
    """
    return f"ohlcv_btc_usd_spot_{period_id}"



def fill_gaps_with_forward_filling(df, expected_freq):
    """
    Fills gaps in the DataFrame using forward filling, based on the expected frequency.
    Ensures 'time_period_start' is in datetime format and 'id' is the index.

    Parameters:
        df (pd.DataFrame): The DataFrame to be forward filled.
        expected_freq (str): The frequency string indicating the expected interval between rows.

    Returns:
        pd.DataFrame: The interpolated DataFrame with gaps filled.
    """
    # Ensure 'time_period_start' is in datetime format
    df['time_period_start'] = pd.to_datetime(df['time_period_start'], utc=True)
    # Temporarily set 'time_period_start' as the index for interpolation
    df_temp = df.set_index('time_period_start')
    # Adjust frequency string to pandas-compatible format
    expected_freq = expected_freq.replace('HRS', 'h').replace('MIN', 'min')
    # Generate the complete DateTime index based on the expected frequency
    full_index = pd.date_range(start=df_temp.index.min(), end=df_temp.index.max(), freq=expected_freq)
    # Reindex the DataFrame to this full index, introducing NaNs for missing timestamps
    df_reindexed = df_temp.reindex(full_index)
    # Flag rows that are newly introduced by reindexing (contain NaNs)
    df_reindexed['is_interpolated'] = df_reindexed.isna().any(axis=1)
    # Perform linear interpolation
    df_forward_filled = df_reindexed.interpolate(method='ffill')
    # Move 'time_period_start' back to a column from the index
    df_forward_filled.reset_index(inplace=True)
    df_forward_filled.rename(columns={'index': 'time_period_start'}, inplace=True)

    return df_forward_filled


import pandas as pd
import datetime
import backtrader as bt


def get_data(data_used, from_date, to_date, time_frame):
    """
    Fetches and processes data for Backtrader backtest.

    Parameters:
        data_used (str): 'Market Data' or 'Simulated Data'
        from_date (str): The start date ('YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS')
        to_date (str): The end date (same format as from_date)
        time_frame (str): Time frame identifier (e.g. '5min', '1h', etc.)

    Returns:
        bt.feeds.PandasData: Backtrader data feed
    """

    # Helper to parse dates
    def parse_date(date_str):
        if date_str is None:
            return None
        date_format = '%Y-%m-%d'
        time_format = 'T%H:%M:%S'
        strp_format = date_format + (time_format if 'T' in date_str else '')
        return datetime.datetime.strptime(date_str, strp_format)

    from_date_dt = parse_date(from_date)
    to_date_dt = parse_date(to_date)

    if data_used == 'Market Data':
        # # Get table name based on the given time frame
        # table_name = get_table_name(time_frame)
        #
        # # Connect to the database and fetch table data
        # conn = get_database_connection()
        # df = fetch_table_data(conn, table_name)
        # df = fill_gaps_with_forward_filling(df, time_frame)
        #
        # data = bt.feeds.PandasData(
        #     dataname=df,
        #     fromdate=from_date_dt,
        #     todate=to_date_dt,
        #     tz='UTC',
        #     datetime=0,
        #     open=5,
        #     high=6,
        #     low=7,
        #     close=8,
        #     volume=9
        # )
        # <-- Ajoute ici ton code de récupération pour les données marché si besoin
        raise NotImplementedError("La récupération des données marché n'est pas implémentée dans ce template.")

    elif data_used == 'Simulated Data':
        csv_path = f"/home/hlemonnier/Bureau/Coding/enigma-quant/data/simulators/hybrid_model_ohlcv_btc_usd_15m.csv"
        df = pd.read_csv(csv_path)
        df = fill_gaps_with_forward_filling(df, time_frame)
        data = bt.feeds.PandasData(
            dataname=df,
            fromdate=from_date_dt,
            todate=to_date_dt,
            tz='UTC',
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5
        )
    else:
        raise ValueError("Invalid value for data_used. Must be 'Market Data' or 'Simulated Data'.")

    return data


def print_strategy_result(thestrat, returns, args):
    """
    Prints the results of the strategy.

    Parameters:
        thestrat (bt.Strategy): The backtrader strategy object.
        returns (pd.Series): The returns series.
        args (argparse.Namespace): Arguments containing various settings and parameters.

    Returns:
        None
    """
    print("\nStrategy Results:")
    print('Final Portfolio Value: {:.3f}'.format(thestrat.broker.getvalue()))
    print('Cash Available: {:.3f}'.format(thestrat.broker.getcash()))
    print('Cumulative Returns (%): {:.3f}'.format(100 * ((thestrat.broker.getvalue() - args.start_cash) / args.start_cash)))
    print('PnL Net: {:.3f}'.format(thestrat.broker.getvalue() - args.start_cash))
    print('Sharpe Ratio: {:.3f}'.format(quantstats.stats.sharpe(returns, periods=args.periods, rf=args.riskfree_rate)))
    print('Sortino Ratio: {:.3f}'.format(quantstats.stats.sortino(returns, periods=args.periods, rf=args.riskfree_rate)))
    print('Max Monetary DrawDown: {:.3f}'.format(thestrat.analyzers.mydrawdown.get_analysis()["max"]["moneydown"]))
    print('\n')



# run the strategy
def runstrat(args):
    """
    Runs the specified trading strategy with the given parameters and configurations.

    Parameters:
        args (argparse.Namespace): Arguments containing various settings and parameters for the backtest.

    Returns:
        None
    """
    # Create a cerebro entity
    cerebro = bt.Cerebro(stdstats=False)

    # Add data
    data_st = get_data(data_used=args.data_used, from_date=args.fromdate, to_date=args.todate, time_frame=args.timeframe_shortterm)
    cerebro.adddata(data_st)
    
    data_lt = get_data(data_used=args.data_used, from_date=args.fromdate, to_date=args.todate, time_frame=args.timeframe_longterm)
    data_lt.plotinfo.plot = False
    cerebro.adddata(data_lt)

    # Set initial cash
    cerebro.broker.setcash(args.start_cash)

    # Set commission scheme
    cerebro.broker.setcommission(commission=args.commission_percentage, margin=None, percabs=False)

    # Set slippage
    cerebro.broker.set_slippage_perc(perc=args.slippage_percentage, slip_match=True, slip_limit=True, slip_out=False)

    # Add observers
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.Trades)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(OrderObserver)
    # cerebro.addobserver(DrawdownLength)
    # cerebro.addobserver(bt.observers.Benchmark, data=data_st, timeframe=args.timeframe_time_return)

    # Add analyzers
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='mydrawdown')
    cerebro.addanalyzer(btanalyzers.SQN, _name='mySQN')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='mytradeanalyzer')
    cerebro.addanalyzer(btanalyzers.PyFolio, _name='pyfolio')

    # Add the strategy with parameters
    if args.strategy_used == "CrossOver_LongShort":
        cerebro.addstrategy(CrossOver_LongShort,
                            period1 = args.period1,
                            period2 = args.period2,
                            brackets_validity = args.brackets_validity,
                            verbose = args.verbose,


                            sizers_type = args.sizers_type,
                            sizers_percent = args.sizers_percent,
                            sizers_stake = args.sizers_stake,
                            max_risk_per_trade = args.max_risk_per_trade,
                            stop_loss_deviation = args.stop_loss_deviation,
                            max_authorized_position = args.max_authorized_position,
                            atr_length = args.atr_length,
                            
                            long_exectype = args.long_exectype,
                            long_hold = args.long_hold,
                            long_limit = args.long_limit,
                            long_validity = args.long_validity,
                            long_away_from_price = args.long_away_from_price,
                            long_risk_reward_ratio = args.long_risk_reward_ratio,
                            long_stop_loss_exectype = args.long_stop_loss_exectype,
                            long_take_profit_exectype = args.long_take_profit_exectype,
                            
                            short_exectype = args.short_exectype,
                            short_limit = args.short_limit,
                            short_validity = args.short_validity,
                            short_away_from_price = args.short_away_from_price,
                            short_risk_reward_ratio = args.short_risk_reward_ratio,
                            short_stop_loss_exectype = args.short_stop_loss_exectype,
                            short_take_profit_exectype = args.short_take_profit_exectype,
                            short_hold = args.short_hold,
                            
                            use_trailing_strategy = args.use_trailing_strategy,
                            trail_stop_pct = args.trail_stop_pct,
                            trail_stop_abs = args.trail_stop_abs,
                            secure_pct = args.secure_pct,
                            atr_multiplier = args.atr_multiplier,
                            
                            use_position_management = args.use_position_management,
                            rsi_period = args.rsi_period,
                            rsi_in_long_lower=args.rsi_in_long_lower,
                            rsi_in_long_upper=args.rsi_in_long_upper,
                            rsi_in_short_lower=args.rsi_in_short_lower,
                            rsi_in_short_upper=args.rsi_in_short_upper,
                            rsi_out_long=args.rsi_out_long,
                            rsi_out_short=args.rsi_out_short,
                            bbands_period = args.bbands_period,
                            bbands_devfactor = args.bbands_devfactor,
                            scale_in_factor = args.scale_in_factor,
                            scale_out_factor = args.scale_out_factor,
                            atr_long_multiplier = args.atr_long_multiplier,
                            atr_short_multiplier = args.atr_short_multiplier,
                            position_scaling_hold = args.position_scaling_hold
                            )
    elif args.strategy_used == "LongShort" :
        cerebro.addstrategy(LongShort,
                            
                            brackets_validity = args.brackets_validity,
                            verbose = args.verbose,
                            args = args,

                            sma_period = args.sma_period,
                            sma_slope_period = args.sma_slope_period,
                            sma_slope_limit = args.sma_slope_limit,
                            ema_period1 = args.ema_period1,
                            adx_period = args.adx_period,
                            adx_strength = args.adx_strength,
                            use_stochastic_oscillator = args.use_stochastic_oscillator,
                            stochastic_oscillator_period = args.stochastic_oscillator_period,
                            stochastic_oscillator_period_dfast = args.stochastic_oscillator_period_dfast,
                            use_ema_bb = args.use_ema_bb,
                            ema_bb_period = args.ema_bb_period,
                            bb_top_devfactor = args.bb_top_devfactor,
                            bb_bot_devfactor = args.bb_bot_devfactor,
                            use_machine_learning = args.use_machine_learning,
                            model_path = args.model_path,
                            use_rsi = args.use_rsi,
                            rsi_period = args.rsi_period,
                            rsi_lower_threshold = args.rsi_lower_threshold,
                            rsi_upper_threshold = args.rsi_upper_threshold,


                            sizers_type = args.sizers_type,
                            sizers_percent = args.sizers_percent,
                            sizers_stake = args.sizers_stake,
                            max_risk_per_trade = args.max_risk_per_trade,
                            stop_loss_deviation = args.stop_loss_deviation,
                            max_authorized_position = args.max_authorized_position,
                            atr_length = args.atr_length,
                            
                            long_exectype = args.long_exectype,
                            long_hold = args.long_hold,
                            long_limit = args.long_limit,
                            long_validity = args.long_validity,
                            long_away_from_price = args.long_away_from_price,
                            long_risk_reward_ratio = args.long_risk_reward_ratio,
                            long_stop_loss_exectype = args.long_stop_loss_exectype,
                            long_take_profit_exectype = args.long_take_profit_exectype,
                            
                            short_exectype = args.short_exectype,
                            short_limit = args.short_limit,
                            short_validity = args.short_validity,
                            short_away_from_price = args.short_away_from_price,
                            short_risk_reward_ratio = args.short_risk_reward_ratio,
                            short_stop_loss_exectype = args.short_stop_loss_exectype,
                            short_take_profit_exectype = args.short_take_profit_exectype,
                            short_hold = args.short_hold,
                            
                            use_trailing_strategy = args.use_trailing_strategy,
                            trail_stop_pct = args.trail_stop_pct,
                            trail_stop_abs = args.trail_stop_abs,
                            secure_pct = args.secure_pct,
                            atr_multiplier = args.atr_multiplier,
                            
                            use_position_management = args.use_position_management,
                            scale_in_factor = args.scale_in_factor,
                            scale_out_factor = args.scale_out_factor,
                            atr_long_multiplier = args.atr_long_multiplier,
                            atr_short_multiplier = args.atr_short_multiplier,
                            position_scaling_hold = args.position_scaling_hold
                            )
    elif args.strategy_used == "LongShortGrid" :
        cerebro.addstrategy(LongShortGrid,
                            
                            brackets_validity = args.brackets_validity,
                            verbose = args.verbose,
                            args = args,

                            sma_period = args.sma_period,
                            sma_slope_period = args.sma_slope_period,
                            sma_slope_limit = args.sma_slope_limit,
                            ema_period1 = args.ema_period1,
                            adx_period = args.adx_period,
                            adx_strength = args.adx_strength,
                            use_stochastic_oscillator = args.use_stochastic_oscillator,
                            stochastic_oscillator_period = args.stochastic_oscillator_period,
                            stochastic_oscillator_period_dfast = args.stochastic_oscillator_period_dfast,
                            use_ema_bb = args.use_ema_bb,
                            ema_bb_period = args.ema_bb_period,
                            bb_top_devfactor = args.bb_top_devfactor,
                            bb_bot_devfactor = args.bb_bot_devfactor,
                            use_machine_learning = args.use_machine_learning,
                            model_path = args.model_path,
                            use_rsi = args.use_rsi,
                            rsi_period = args.rsi_period,
                            rsi_lower_threshold = args.rsi_lower_threshold,
                            rsi_upper_threshold = args.rsi_upper_threshold,


                            sizers_type = args.sizers_type,
                            sizers_percent = args.sizers_percent,
                            sizers_stake = args.sizers_stake,
                            max_risk_per_trade = args.max_risk_per_trade,
                            stop_loss_deviation = args.stop_loss_deviation,
                            max_authorized_position = args.max_authorized_position,
                            atr_length = args.atr_length,
                            
                            long_exectype = args.long_exectype,
                            long_hold = args.long_hold,
                            long_limit = args.long_limit,
                            long_validity = args.long_validity,
                            long_away_from_price = args.long_away_from_price,
                            long_risk_reward_ratio = args.long_risk_reward_ratio,
                            long_stop_loss_exectype = args.long_stop_loss_exectype,
                            long_take_profit_exectype = args.long_take_profit_exectype,
                            
                            short_exectype = args.short_exectype,
                            short_limit = args.short_limit,
                            short_validity = args.short_validity,
                            short_away_from_price = args.short_away_from_price,
                            short_risk_reward_ratio = args.short_risk_reward_ratio,
                            short_stop_loss_exectype = args.short_stop_loss_exectype,
                            short_take_profit_exectype = args.short_take_profit_exectype,
                            short_hold = args.short_hold,
                            
                            use_trailing_strategy = args.use_trailing_strategy,
                            trail_stop_pct = args.trail_stop_pct,
                            trail_stop_abs = args.trail_stop_abs,
                            secure_pct = args.secure_pct,
                            atr_multiplier = args.atr_multiplier,
                            
                            use_position_management = args.use_position_management,
                            scale_in_factor = args.scale_in_factor,
                            scale_out_factor = args.scale_out_factor,
                            atr_long_multiplier = args.atr_long_multiplier,
                            atr_short_multiplier = args.atr_short_multiplier,
                            position_scaling_hold = args.position_scaling_hold
                            )
    elif args.strategy_used == "Grid" :
        cerebro.addstrategy(Grid,
                            args = args,
                            verbose = args.verbose,

                            max_risk_per_trade = args.max_risk_per_trade,
                            stop_loss_deviation = args.stop_loss_deviation,
                            max_authorized_position = args.max_authorized_position,
                            atr_length = args.atr_length,
                            )
    else:
        cerebro.addstrategy(BuyAndHold)

    # Run the strategy
    thestrats = cerebro.run()
    thestrat = thestrats[0]

    # Get PyFolio analyzer data
    pyfoliozer = thestrat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, _ = pyfoliozer.get_pf_items()
    returns.index = returns.index.tz_convert(None)
    trade_analysis = thestrat.analyzers.mytradeanalyzer.get_analysis()

    # Print strategy information if enabled
    if args.print_logs:
        print_strategy_result(thestrat, returns, args)
        # print_trade_analysis(trade_analysis)

    # Generate tearsheet if enabled
    if args.print_tearsheet:
        html_path = f"/home/hlemonnier/Bureau/Coding/enigma-quant/backtesting/tearsheet_{args.strategy_used}_btc_usd_spot.html"
        quantstats.reports.html(returns, periods_per_year=args.periods, benchmark=args.benchmark, rf=args.riskfree_rate,
                                output=html_path, title=f"{args.strategy_used} Tearsheet")
    
    
    # Write results to an Excel file if enabled
    if args.write:
        excel_path = f"/home/hlemonnier/Bureau/Coding/enigma-quant/backtesting/results_{args.strategy_used}_btc_usd_spot.xlsx"
        report = ExcelReport(args=args, thestrat=thestrat, analysis=trade_analysis, returns=returns, 
                            positions=positions, transactions=transactions, excel_path=excel_path)
        report.get_excel_report()
        
            
    # Plot the results if enabled
    if args.plot:
        cerebro.plot()

    return returns



if __name__ == '__main__':
    config = Config()
    args = config.parse_args()
    runstrat(args)