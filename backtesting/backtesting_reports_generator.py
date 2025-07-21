import pandas as pd
import quantstats

class ExcelReport:
    """
    Generates an Excel file with detailed analysis and results of the backtest strategy.

    Parameters:
        strat_used (str): Name of the strategy used.
        args (argparse.Namespace): Arguments containing various settings and parameters.
        thestrat (object): The backtrader strategy object.
        analysis (dict): Dictionary containing the results of the trade analysis.
        returns (pd.DataFrame): DataFrame containing the daily returns data.
        positions (pd.DataFrame): DataFrame containing the daily positions data.
        transactions (pd.DataFrame): DataFrame containing the transactions data.
        excel_path (str): Path to save the generated Excel file.

    """

    def __init__(self, args, thestrat, analysis, returns, positions, transactions, excel_path):
        self.args = args
        self.thestrat = thestrat
        self.analysis = analysis
        self.returns = returns
        self.positions = positions
        self.transactions = transactions
        self.excel_path = excel_path
        self.last_day_position_quantity = 0
        self.last_day_position_nominal = 0

    def normalize_dates(self, df):
        df['normalized_date'] = pd.to_datetime(df['date']).dt.normalize().dt.tz_localize(None)
        return df

    def format_dates_to_iso(self, df):
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%dT%H:%M:%S')
        return df
    
    def drop_normalize_dates(self, df):
        if 'normalized_date' in df.columns:
            df.drop('normalized_date', axis=1, inplace=True)
        return df


    def compile_strategy_info(self):
        """Compile basic information about the strategy into a DataFrame for reporting."""
       # Dynamically access all the arguments stored in self.args
        info_mapping = {arg: lambda arg=arg: getattr(self.args, arg) for arg in vars(self.args)}
        
        info_labels = list(info_mapping.keys())
        info_values = [calc() for calc in info_mapping.values()]

        info = {
            "info": info_labels,
            "value": info_values
        }

        return pd.DataFrame(info)



    def compile_strategy_results(self):
        """Compile key performance indicators from strategy results into a DataFrame."""
        kpi_calculations = {
            'Final Portfolio Value': lambda: self.thestrat.broker.getvalue(),
            'Cash Available': lambda: self.thestrat.broker.getcash(),
            'Cumulative Returns (%)': lambda: 100 * ((self.thestrat.broker.getvalue() - self.args.start_cash) / self.args.start_cash),
            'PnL Net': lambda: self.thestrat.broker.getvalue() - self.args.start_cash,
            'Sharpe Ratio': lambda: quantstats.stats.sharpe(self.returns, periods=self.args.periods, rf=self.args.riskfree_rate),
            'Sortino Ratio': lambda: quantstats.stats.sortino(self.returns, periods=self.args.periods, rf=self.args.riskfree_rate),
            'Volatility (%)': lambda: 100 * quantstats.stats.volatility(self.returns, periods=self.args.periods, annualize=False),
            'Daily Value-at-Risk (%)': lambda: 100 * quantstats.stats.value_at_risk(self.returns, sigma=1, confidence=0.95),
            'System Quality Number (SQN)': lambda: self.thestrat.analyzers.mySQN.get_analysis()['sqn'],
            'Max DrawDown Length (bars)': lambda: self.thestrat.analyzers.mydrawdown.get_analysis()['max']['len'],
            'Max Percentage DrawDown (%)': lambda: self.thestrat.analyzers.mydrawdown.get_analysis()['max']['drawdown'],
            'Max Monetary DrawDown': lambda: self.thestrat.analyzers.mydrawdown.get_analysis()['max']['moneydown'],
        }

        kpis = list(kpi_calculations.keys())
        kpi_values = [calc() for calc in kpi_calculations.values()]

        result = {
            "KPI": kpis,
            "value": kpi_values
        }

        return pd.DataFrame(result)



    def get_total_trade(self, prefix, len_analysis):
        return {
            f"{prefix}": lambda: len_analysis.get('total', '---'),
            f"{prefix} Open": lambda: len_analysis.get('open', '---'),
            f"{prefix} Closed ": lambda: len_analysis.get('closed', '---'),
        } 

    def get_streak_trade(self, prefix, len_analysis):
        return {
            f"{prefix} Current Winning": lambda: len_analysis.get('won', {}).get('current', '---'),
            f"{prefix} Longest Winning": lambda: len_analysis.get('won', {}).get('longest', '---'),
            f"{prefix} Current Losing": lambda: len_analysis.get('lost', {}).get('current', '---'),
            f"{prefix} Longest Losing": lambda: len_analysis.get('lost', {}).get('longest', '---'),
        } 

    def get_pnl_trade(self, prefix, len_analysis):
        return {
            f"{prefix} Gross Total": lambda: len_analysis.get('gross', {}).get('total', '---'),
            f"{prefix} Gross Average": lambda: len_analysis.get('gross', {}).get('average', '---'),
            f"{prefix} Net Total": lambda: len_analysis.get('net', {}).get('total', '---'),
            f"{prefix} Net Average": lambda: len_analysis.get('net', {}).get('average', '---'),
        }    

    def get_win_lost_trade(self, prefix, len_analysis):
        return {
            f"{prefix} Count": lambda: len_analysis.get('total', '---'),
            f"{prefix} Total": lambda: len_analysis.get('pnl', {}).get('total', '---'),
            f"{prefix} Average": lambda: len_analysis.get('pnl', {}).get('average', '---'),
            f"{prefix} Max": lambda: len_analysis.get('pnl', {}).get('max', '---')
        }

    def get_long_short_trade(self, prefix, len_analysis):
        return {
            f"{prefix} Total Trade": lambda: len_analysis.get('total', '---'),
            f"{prefix} Total": lambda: len_analysis.get('pnl', {}).get('total', '---'),
            f"{prefix} Average Profit": lambda: len_analysis.get('pnl', {}).get('average', '---'),
            f"{prefix} Won": lambda: len_analysis.get('won', '---'),
            f"{prefix} Lost": lambda: len_analysis.get('lost', '---')
        }

    def get_len_trade(self, prefix, len_analysis):
        return {
            f"{prefix} Total": lambda: len_analysis.get('total', '---'),
            f"{prefix} Average": lambda: len_analysis.get('average', '---'),
            f"{prefix} Max": lambda: len_analysis.get('max', '---'),
            f"{prefix} Min": lambda: len_analysis.get('min', '---')
        }
    
    def get_ratio(self, sufix):
        return {
            f"Trade Win-Loss {sufix}": lambda: self.safe_divide(self.analysis['won']['total'], self.analysis['lost']['total']),
            f"Monetary Win-Loss {sufix}": lambda: self.safe_divide(abs(self.analysis['won']['pnl']['total']), abs(self.analysis['lost']['pnl']['total'])),
            f"Average Win-Loss {sufix}": lambda: self.safe_divide(abs(self.analysis['won']['pnl']['average']), abs(self.analysis['lost']['pnl']['average'])),
            f"Winning {sufix}": lambda: self.safe_divide(self.analysis['won']['total'], self.analysis['total']['total']),
            f"Losing {sufix}": lambda: self.safe_divide(self.analysis['lost']['total'], self.analysis['total']['total'])
        }

    def safe_divide(self, numerator, denominator):
        return numerator / denominator if denominator != 0 else None

    
    def compile_trade_analysis(self):
        """Compile detailed trade analysis metrics into a DataFrame."""
        trade_values_mapping = {}

        # Adding pnl data
        trade_values_mapping.update(self.get_total_trade("Total Trade", self.analysis['total']))   
        # Adding pnl data
        trade_values_mapping.update(self.get_pnl_trade("PnL", self.analysis['pnl']))
        # Adding probability and ratio data
        trade_values_mapping.update(self.get_ratio("Ratio"))       
        # Adding streak data
        trade_values_mapping.update(self.get_streak_trade("Streak", self.analysis['streak']))
        # Adding long data
        trade_values_mapping.update(self.get_win_lost_trade("Winning Trade", self.analysis['won']))
        # Adding short data
        trade_values_mapping.update(self.get_win_lost_trade("Lost Trade", self.analysis['lost']))
        # Adding long data
        trade_values_mapping.update(self.get_long_short_trade("Long Trade", self.analysis['long']))
        # Adding short data
        trade_values_mapping.update(self.get_long_short_trade("Short Trade", self.analysis['short']))
        # Adding length data
        trade_values_mapping.update(self.get_len_trade("Trade Length", self.analysis['len']))
        trade_values_mapping.update(self.get_len_trade("Winning Trade Lengths", self.analysis['len']['won']))
        trade_values_mapping.update(self.get_len_trade("Losing Trade Lengths", self.analysis['len']['lost']))
        trade_values_mapping.update(self.get_len_trade("Long Trade Lengths", self.analysis['len']['long']))
        trade_values_mapping.update(self.get_len_trade("Winning Long Trade Lengths", self.analysis['len']['long']['won']))
        trade_values_mapping.update(self.get_len_trade("Losing Long Trade Lengths", self.analysis['len']['long']['lost']))
        trade_values_mapping.update(self.get_len_trade("Short Trade Lengths", self.analysis['len']['short']))
        trade_values_mapping.update(self.get_len_trade("Winning Short Trade Lengths", self.analysis['len']['short']['won']))
        trade_values_mapping.update(self.get_len_trade("Losing Short Trade Lengths", self.analysis['len']['short']['lost']))

        metrics = list(trade_values_mapping.keys())
        values = [calc() for calc in trade_values_mapping.values()]

        trade_analysis = {
            "KPI": metrics,
            "value": values
        }

        return pd.DataFrame(trade_analysis)
            

    def compile_log_entries(self):
        """Compile log entries related to strategy execution into a DataFrame."""
        log_data = pd.DataFrame({
            "date": self.thestrat.log_entries['date'],
            "log": self.thestrat.log_entries['log']
        })
        return log_data




    def compile_positions(self, threshold=1e-9):
        positions = self.positions.reset_index().rename(columns={'Datetime': 'date'})
        positions = self.normalize_dates(positions)
        transactions = self.normalize_dates(self.transactions.copy())

        # Initialize cumulative variables
        cumulative_quantity = 0
        cumulative_nominal = 0

        # Sort transactions by date to ensure chronological processing
        transactions = transactions.sort_values(by='normalized_date')

        # Prepare a list to hold the adjusted data
        adjusted_data_list = []

        # Iterate through each date in the positions DataFrame
        for position_date in positions['normalized_date']:
            # Filter transactions for the current date
            daily_transactions = transactions[transactions['normalized_date'] == position_date]

            # Process each transaction
            for _, transaction in daily_transactions.iterrows():
                cumulative_quantity += transaction['quantity']
                cumulative_nominal += transaction['nominal']

            # At the end of the day, determine what to store in the positions table
            if abs(cumulative_quantity) < threshold:
                # If the cumulative quantity is below the threshold, consider it as zero
                adjusted_data_list.append({
                    'normalized_date': position_date,
                    'quantity': 0,
                    'nominal': 0
                })
                cumulative_quantity = 0
                cumulative_nominal = 0
            else:
                # Otherwise, store the current cumulative values
                adjusted_data_list.append({
                    'normalized_date': position_date,
                    'quantity': cumulative_quantity,
                    'nominal': cumulative_nominal
                })

        # Convert the list to a DataFrame
        adjusted_data = pd.DataFrame(adjusted_data_list)

        # Merge the 'positions' with the adjusted quantities and nominal per day
        positions = positions.merge(adjusted_data, on='normalized_date', how='left')
        positions['quantity'].fillna(0, inplace=True)
        positions['nominal'].fillna(0, inplace=True)

        positions = positions[['date', 'cash', 'quantity', 'nominal']]

        return positions




    def compile_returns(self):
        # Prepare the returns DataFrame
        returns = self.returns.reset_index().rename(columns={'index': 'date'})
        returns = self.normalize_dates(returns)
        transactions = self.normalize_dates(self.transactions.copy())

        # Calculate the number of transactions per day
        transactions_count = transactions.groupby('normalized_date').size().reset_index(name='num_transactions')
        returns = returns.merge(transactions_count, on='normalized_date', how='left')
        returns['num_transactions'].fillna(0, inplace=True)

        # Initialize starting portfolio value with the start cash
        start_cash = self.args.start_cash
        returns['starting_portfolio_value'] = 0  # Initialize with 0, will set the first value in the loop
        returns['final_portfolio_value'] = 0
        returns['return_in_cash'] = 0

        # Iteratively calculate the final portfolio value and return in cash for each day
        current_portfolio_value = start_cash
        for i in range(len(returns)):
            returns.at[i, 'starting_portfolio_value'] = current_portfolio_value
            returns.at[i, 'final_portfolio_value'] = current_portfolio_value * (1 + returns.at[i, 'return'])
            returns.at[i, 'return_in_cash'] = returns.at[i, 'final_portfolio_value'] - current_portfolio_value
            current_portfolio_value = returns.at[i, 'final_portfolio_value']  # Update for the next day

        return returns



    def compile_transactions(self):
        # Create the DataFrame
        transactions = pd.DataFrame({
            "date": self.thestrat.transactions['date'],
            "symbol": self.thestrat.transactions['symbol'],
            "quantity": self.thestrat.transactions['quantity'],
            "price": self.thestrat.transactions['price'],
            "order_type": self.thestrat.transactions['order_type'],
            "trade_type": self.thestrat.transactions['trade_type'],
            "reason": self.thestrat.transactions['reason'],
            "entry_side": self.thestrat.transactions['entry_side'],
            "broker_comm": self.thestrat.transactions['broker_comm']
        })

        transactions['nominal'] = transactions['price'] * transactions['quantity']         # Adjust nominal value based on trade type
        transactions['date'] = pd.to_datetime(transactions['date']).dt.tz_localize(None)   # Remove timezone information

        transactions['transaction_id'] = range(1, len(transactions) + 1)
        transactions = transactions[["transaction_id", "date", "symbol", "quantity", "price", "nominal", "order_type", "trade_type", "reason", "entry_side", "broker_comm"]]

        transactions = self.normalize_dates(transactions)

        return transactions
    

    def preprocess_dataframes(self):
        self.transactions = self.format_dates_to_iso(self.transactions)
        self.positions = self.format_dates_to_iso(self.positions)
        self.returns = self.format_dates_to_iso(self.returns)

        self.transactions = self.drop_normalize_dates(self.transactions)
        self.positions = self.drop_normalize_dates(self.positions)
        self.returns = self.drop_normalize_dates(self.returns)

    def get_excel_report(self):
        """Generate an Excel report with compiled data and save it to the specified path."""

        strat_information = self.compile_strategy_info()
        strat_result = self.compile_strategy_results()
        trade_analysis = self.compile_trade_analysis()
        log_data = self.compile_log_entries()


        self.transactions = self.compile_transactions()
        self.positions = self.compile_positions()
        self.returns = self.compile_returns()

        self.preprocess_dataframes()

        with pd.ExcelWriter(self.excel_path, engine="xlsxwriter", mode='w') as writer:
            strat_information.to_excel(writer, sheet_name='Strategy Information', index=False)
            strat_result.to_excel(writer, sheet_name='Strategy Results', index=False)
            trade_analysis.to_excel(writer, sheet_name='Trade Analysis', index=False)
            log_data.to_excel(writer, sheet_name='Log Entries', index=False)
            self.returns.to_excel(writer, sheet_name='Returns', index=False)
            self.positions.to_excel(writer, sheet_name='Positions', index=False)            
            self.transactions.to_excel(writer, sheet_name='Transactions', index=False)
