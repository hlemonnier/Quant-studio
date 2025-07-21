#################################################################### DATA VALIDATION ##########################################################################################

'''      
Data validation is a crucial step in ensuring the quality and integrity of data before it's processed or analyzed. It involves verifying that the data meets specific criteria and standards.                                          
                                                                                                                                                                             
Rule Established: 
- Gather all dates : ensuring consecutive records span the entirety of your data range without missing periods. 
WE USE LINEAR INTERPOLATION TO FILL IN THE RESTING GAPS AND FLAG THEM  -----  Autre idée pour fill the gap : forward fill - more complex interpolation

- Ensure Sequential Time Intervals: Verify that all expected time intervals are respected by checking that the difference between 'time_period_start' and 'time_period_end' matches the expected interval length 
                                                                                                                                                           
- Check for Data Entry Error: Ensure that 'price_open', 'price_high', 'price_low', and 'price_close' are greater than zero, that 'volume_traded' & 'trades_count' is non-negative.

- Validate Price Fluctuations: Validate that 'price_low' <= 'price_open', 'price_high', and 'price_close' <= 'price_high'. This rule checks for logical consistency in price data.

- Check for Duplicate Entries: Ensure there are no duplicate entries for the same 'time_period_start'. Duplicate entries could distort analysis and lead to incorrect conclusions.

- Data Type and Precision Checks: Validate the data types and precision of numerical fields to ensure they match your model's requirements.              

Supprimer colonne : 'time_open' - 'time_close' - 'trades_count' -  'time_period_end'

'''

'''
Do we change directly the missing value on the dataset ? Maybe not because we will try different methods

'''


############################################################################# IMPORT PACKAGES ####################################################################################

import psycopg2
#from ohlcv.config import load_config
from config import load_config
import pandas as pd
import warnings

# Suppress specific Pandas warning
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')


####################################################################### IMPORT DATABASE AND PANDAS CONVERTION ###################################################################

def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        conn = psycopg2.connect(**config)
        print('Connected to the PostgreSQL server successfully.')
        return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print("Failed to connect to the database:", error)
        return None


# Import Database
def get_database_connection():
    config = load_config()
    conn = connect(config)
    return conn


def fetch_table_data(conn, table_name):
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, conn)
    return df


############################################################################ CONSECUTIVE DATES ######################################################################################


def validate_consecutive_dates(df):
    # Ensure DataFrame is sorted by 'time_period_start'
    df = df.sort_values('time_period_start').reset_index(drop=True)
    
    # Shift 'time_period_start' to create a column for comparison
    df['next_start'] = df['time_period_start'].shift(-1)
    
    # Compare 'time_period_end' with the next record's 'time_period_start'
    gaps = df[df['time_period_end'] != df['next_start']].index
    
    # Exclude the last row from gap check, as it has no next record
    if not gaps.empty and gaps[-1] == df.index[-1]:
        gaps = gaps[:-1]

    # Check if gaps exist and print relevant information
    if not gaps.empty:
        print(f"Found {len(gaps.to_list())} gaps")
        #print(f"Found gaps {len(gaps.to_list())} at indices: {gaps.tolist()}")
        '''    
        for index in gaps:
            # Safely check if the next index exists
            if index + 1 < len(df):
                print(f"Gap between {df.loc[index, 'time_period_end']} and {df.loc[index + 1, 'time_period_start']} at index {index}")
            else:
                # Handle or report the case where the next index doesn't exist if necessary
                print(f"End of data reached at index {index} with end time {df.loc[index, 'time_period_end']}.")
                '''
        
        return False
    else:
        print("No gaps found. Data is consecutive.")
        return True


############################################################################ PROPER INTERVAL ######################################################################################


def identify_mismatched_intervals(df, expected_interval):
    """
    Identify records where the interval between 'time_period_start' and 'time_period_end' does not match the expected length, indicating precisely where the issues occur.

    Parameters:
    - df: DataFrame containing OHLCV data.
    - expected_interval: String representing the expected interval length 
    """
    expected_interval = expected_interval.replace('HRS', 'h').replace('MIN', 'min') 

    # Ensure datetime format for 'time_period_start' and 'time_period_end'
    df['time_period_start'] = pd.to_datetime(df['time_period_start'], utc=True)
    df['time_period_end'] = pd.to_datetime(df['time_period_end'], utc=True)

    # Calculate the actual interval length for each record
    df['actual_interval_length'] = df['time_period_end'] - df['time_period_start']

    # Convert expected_interval to Timedelta for comparison
    expected_timedelta = pd.to_timedelta(expected_interval)

    # Identify records where the interval length does not match the expected length
    mismatched_intervals = df[df['actual_interval_length'] != expected_timedelta]

    if not mismatched_intervals.empty:
        print(f"Total mismatches: {len(mismatched_intervals)}")
        # Return DataFrame slice for mismatches for further analysis if needed
        return mismatched_intervals
    else:
        print("No mismatches found. All records match the expected interval length.")
        return pd.DataFrame()  # Return empty DataFrame if no mismatches




#########################################################################  SORT CHECK ######################################################################################



def check_sorted(df):
    """
    Checks if the DataFrame is correctly sorted by the 'time_exchange' column.
    If not sorted, prints the rows where the order breaks.

    Parameters:
    - df (pd.DataFrame): DataFrame containing trading data with a 'time_exchange' column.
    """
    
    try:
        # Compute differences between consecutive time_exchange values to find where it is not increasing
        sorted_mask = df['time_period_start'].diff().dropna() >= pd.Timedelta(0)
        
        if sorted_mask.all():
            print("The DataFrame is sorted.")
        else:
            # Find indexes where the sort order breaks
            unsorted_indexes = sorted_mask[~sorted_mask].index
            # Print the rows before and after the break in sort order for context
            for idx in unsorted_indexes:
                if idx > 0 and idx < len(df):
                    print("Unsorted rows:")
                    print(df.loc[idx-1:idx+1])
                else:
                    print("Unsorted at the beginning or end of the DataFrame")
                    
    except KeyError:
        print("Error: The DataFrame does not contain a 'time_exchange' column.")



############################################################################ ZERO/NEGATIVE VALUES  ######################################################################################


def check_for_negative_or_zero_values(df):
    """
    Check for negative or zero values in price columns and negative values in volume column.
    Parameters:
    - df: DataFrame containing OHLCV data.
    """
    price_columns = ['price_open', 'price_high', 'price_low', 'price_close']
    volume_column = 'volume_traded'
    count_column = "trades_count"
    errors_found = False

    for column in price_columns:
        if (df[column] <= 0).any():
            print(f"Negative or zero values found in {column}")
            errors_found = True

    if (df[volume_column] <= 0).any():
        print(f"Negative values found in {volume_column}")
        errors_found = True
    
    if( df[count_column] <= 0).any():
        print(f"Negative values found in {count_column}")
        errors_found = True

    if not errors_found:
        print("No negative or zero values found in price & trades_count columns and no negative values found in volume column.")


############################################################################ PRICE CONSISTENCY  ######################################################################################


def validate_price_fluctuations(df):
    """
    Validate that price data is logically consistent.
    Parameters:
    - df: DataFrame containing OHLCV data.
    """
    inconsistent_rows = df[~(
        (df['price_low'] <= df['price_open']) &
        (df['price_low'] <= df['price_high']) &
        (df['price_low'] <= df['price_close']) &
        (df['price_open'] <= df['price_high']) &
        (df['price_close'] <= df['price_high'])
    )]
    
    if not inconsistent_rows.empty:
        print(f"Found inconsistencies in {len(inconsistent_rows)} rows:")
        print(inconsistent_rows[['time_period_start', 'price_low', 'price_open', 'price_high', 'price_close']])
        return False
    else:
        print("All price fluctuations are logically consistent.")
        return True


############################################################################ DUPLICATE ENTRIES ######################################################################################


def check_for_duplicate_entries(df):
    """
    Check for duplicate entries based on 'time_period_start'.
    Parameters:
    - df: DataFrame containing OHLCV data.
    """

    if df['time_period_start'].duplicated().any():
        duplicate_count = df['time_period_start'].duplicated().sum()
        print(f"Found {duplicate_count} duplicate entries based on 'time_period_start'.")
        
        # Optional: Display the duplicate entries for review
        duplicates = df[df['time_period_start'].duplicated(keep=False)]  # keep=False marks all duplicates as True
        print("Duplicate entries:")
        print(duplicates[['time_period_start', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded']])
        return False
    else:
        print("No duplicate entries found based on 'time_period_start'.")
        return True



############################################################################ DATA TYPES CHECK ######################################################################################



def validate_data_types_and_precision(df, precision_requirements):
    """
    Validate the data types and precision of numerical fields.
    Parameters:
    - df: DataFrame containing the data.
    - precision_requirements: A dictionary specifying the expected data type and precision
      for each column. The keys are column names, and the values are tuples of (expected_dtype, precision).
      Precision is the number of decimal places for float columns, ignored for non-floats.
    """
    issues_found = False

    for column, (expected_dtype, precision) in precision_requirements.items():
        # Check data type
        if df[column].dtype != expected_dtype:
            print(f"Column '{column}' has incorrect data type '{df[column].dtype}', expected '{expected_dtype}'.")
            issues_found = True

        # For float columns, check precision
        if pd.api.types.is_float_dtype(df[column]) and precision is not None:
            if not all(df[column].dropna().map(lambda x: x == round(x, precision))):
                print(f"Column '{column}' has values exceeding the expected precision of {precision} decimal places.")
                issues_found = True

    if not issues_found:
        print("All columns have correct data types and precision.")
    else:
        print("Data type and precision issues found.")



############################################################################ FILL IN GAPS WITH INTERPOLATION ######################################################################################

# i think we should try other one also

def fill_gaps_with_interpolation(df, expected_freq):
    """
    Fill gaps in the DataFrame using linear interpolation, based on the expected frequency,
    and ensuring 'id' as the index with 'time_period_start' as a column.
    
    Parameters:
    - df: DataFrame to be interpolated.
    - expected_freq: The frequency string, indicating the expected interval between rows.
    """

    # Ensure 'time_period_start' is in datetime format
    df['time_period_start'] = pd.to_datetime(df['time_period_start'])
    
    # Temporarily set 'time_period_start' as the index for interpolation
    df_temp = df.set_index('time_period_start')
    
    # Generate the complete DateTime index based on the expected frequency
    expected_freq = expected_freq.replace('HRS', 'h').replace('MIN', 'min')
    full_index = pd.date_range(start=df_temp.index.min(), end=df_temp.index.max(), freq=expected_freq)
    
    # Reindex the DataFrame to this full index, introducing NaNs for missing timestamps
    df_reindexed = df_temp.reindex(full_index)

    # Before interpolation, flag rows that are newly introduced by reindexing
    # True for new rows (with NaN values), False for existing rows
    df_reindexed['is_interpolated'] = df_reindexed.isna().any(axis=1)
    
    # Perform linear interpolation
    df_interpolated = df_reindexed.interpolate(method='linear')
    
    # Move 'time_period_start' back to a column from the index
    df_interpolated.reset_index(inplace=True)
    df_interpolated.rename(columns={'index': 'time_period_start'}, inplace=True)


    return df_interpolated



############################################################################ Handle Duplicates ######################################################################################

def resolve_duplicates(df):
    """
    Resolve duplicates in the OHLCV data by retaining the row with the earliest 'time_open' and the latest 'time_close' for each 'time_period_start'.
    If 'time_open' or 'time_close' are tied, further aggregation will be applied.
    Parameters:
    - df: DataFrame containing OHLCV data.
    """
    # Sort to prepare for dropping duplicates: earliest time_open first, latest time_close last
    df = df.sort_values(by=['time_period_start', 'time_open', 'time_close'], ascending=[True, True, False])

    # Remove duplicates with conditions for ties on 'time_close'
    df = df.drop_duplicates(subset='time_period_start', keep='first')

    # Further aggregation to handle ties effectively (if needed)
    # Group by 'time_period_start' to ensure no duplicates and aggregate other fields
    df = df.groupby('time_period_start').agg({
        'time_open': 'min',  # Earliest open time
        'time_close': 'max',  # Latest close time
        'price_open': 'first',  # First opening price encountered
        'price_high': 'max',  # Highest price encountered
        'price_low': 'min',  # Lowest price encountered
        'price_close': 'last',  # Last closing price encountered
        'volume_traded': 'sum',  # Total volume traded
        'trades_count': 'sum'  # Total trades count
    }).reset_index()

    return df


############################################################################ MAIN ######################################################################################





precision_requirements = {
    'price_open': ('float64', 8),
    'price_high': ('float64', 8),
    'price_low': ('float64', 8),
    'price_close': ('float64', 8),
    'volume_traded': ('float64', None),  # Precision not checked for volume
}

columns_to_keep = ['time_period_start', 'price_open', 'price_high', 'price_low', 'price_close', 'volume_traded', 'is_interpolated']
timeframes = ['1MIN','5MIN','15MIN','30MIN','1HRS','4HRS']
# timeframes = ['5MIN']


if __name__ == '__main__':
    conn = get_database_connection()
    for timeframe in timeframes:
        table_name = f"ohlcv_btc_usd_spot_{timeframe}" 

        print(f"Validating data for table: {table_name}")
        df = fetch_table_data(conn, table_name)
        
        # Apply validation functions
        
        # Validate consecutive dates, mismatached intervals, duplicate entries
        validate_consecutive_dates(df)
        mismatches = identify_mismatched_intervals(df, expected_interval=timeframe)
        check_for_duplicate_entries(df)

        # Validate data consistency & zero
        check_for_negative_or_zero_values(df)
        validate_price_fluctuations(df)
        validate_data_types_and_precision(df, precision_requirements)


        # Check for sorted
        check_sorted(df)
        

        # After all validations, print the dimensions of the DataFrame
        print(f"Number of rows and columns in {table_name}: {df.shape}")
        print(f"Filling data for table: {table_name}")

        # Apply the function to fill gaps
        print("We fill the gap with linear interpolation")
        df = fill_gaps_with_interpolation(df, timeframe)
        validate_consecutive_dates(df)

        # Keep only intresting column and print the final dimension 
        df = df[columns_to_keep]
        print(f"Number of rows and columns in {table_name}: {df.shape}")
        missing_values_count = df.isna().sum() 

        #print(missing_values_count)
        #no missing values
        
        # Check for sorted
        check_sorted(df)

        #print(df.tail(15))
        print('----------------------------------------------------')

    

