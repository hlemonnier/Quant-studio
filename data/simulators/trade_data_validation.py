'''      
DATA VALIDATION                                       
                                                                                                                                                                             
Rule Established:
- Check for ZERO NEGATIVE Error: Ensure that 'price' and 'size' are positive.
- Validate Time Consistency: Validate that 'time_exchange' <= 'time_coinapi'
- Check for Duplicate Entries: Ensure there are no duplicate entries for the same 'uuid' as it represents the unique identifier for the trade.
- Data Type and Precision Checks: Validate the data types and precision of numerical fields to ensure they match your model's requirements. 
- Check that all rows have 'COINBASE_SPOT_BTC_USD' as symbol_id, 'SELL' or 'BUY' as taker_side, print the rows that haven't
- Sorted Dataframe = data frame
- Valider les jointures: A chaque début de mois, print la ligne avant(sauf pour le premier mois) et les 3 premiers row du mois 
- Checks if the DataFrame is correctly sorted by the 'time_exchange' column.
- Check the number of missing values 

Supprimer colonne : 'symbol_id' - 'uuid' - 'time_coinapi'


Table were not sorted, had time inconsistency and duplicates so we created function to solve these problems 
There are many very small inconsistencies, we cannot change something with that though. We will keep the 'time_exchange' column

'''



'''
How do we deal with the 2x 4HRS missing values on TO DO:  2023/06/04 : 20:02:16 and 2023/06/05 : 00:00:00    &&&   2023-06-08 : 20:15:35 and 2023-06-09 : 00:00:00
Try a scrpit to check for exactly these date


TO DO : 
Faire un code qui prend la database merged et SORT & REMOVE DUPLICATES direclty and delete columns on the db and no pandas


'''    
   



#############################################################################     IMPORT PACKAGES   ####################################################################################


from datetime import datetime,  timedelta
import psycopg2
#from ohlcv.config import load_config
from config_2 import load_config
import pandas as pd
import warnings
warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')


#######################################################################    IMPORT DATABASE AND PANDAS CONVERTION    ###################################################################


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
    query = f"""SELECT * FROM {table_name};  
            """
    df = pd.read_sql(query, conn)
    return df


############################################################################     ZERO/NEGATIVE VALUES    ######################################################################################


def check_for_negative_or_zero_values(df):
    """
    Check for negative or zero values in price & size columns
    Parameters:
    - df: DataFrame containing OHLCV data.
    """
    columns = ['price', 'size']
    errors_found = False

    for column in columns:
        if (df[column] <= 0).any():
            print(f"Negative or zero values found in {column}")
            errors_found = True

    if not errors_found:
        print("No negative or zero values found in price & size columns ")


############################################################################ PRICE CONSISTENCY  ######################################################################################

# There are many very small inconsistencies, we cannot change something with that though. We will keep the 'time_exchange' column

def validate_time_fluctuations(df):
    """
    Validate that time data is logically consistent.
    Parameters:
    - df: DataFrame containing OHLCV data.
    """
    inconsistent_rows = df[(df['time_exchange'] > df['time_coinapi']) ]
    
    if not inconsistent_rows.empty:
        print(f"Found inconsistencies in {len(inconsistent_rows)} rows:")
        #print(inconsistent_rows.head(10))
        return False
    else:
        print("All time are logically consistent.")
        return True


############################################################################ DUPLICATE ENTRIES ######################################################################################
 

def check_for_duplicate_entries(df):
    """
    Check for duplicate entries based on 'uuid' and print them.
    Parameters:
    - df (pd.DataFrame): DataFrame containing trades data.
    """
    
    duplicates = df[df['uuid'].duplicated(keep=False)]
    
    if not duplicates.empty:
        duplicate_count = duplicates.shape[0]
        print(f"Found {duplicate_count} duplicate entries based on 'uuid':")
        #print(duplicates.head(10))
        return False
    else:
        print("No duplicate entries found based on 'uuid'.")
        return True


def delete_duplicates(df):
    """
    Delete duplicate entries based on 'uuid', keeping the first occurrence and print the number of duplicates removed.
    Parameters:
    - df (pd.DataFrame): DataFrame containing trades data.
    """
    
    # Count original number of rows for reference
    original_count = len(df)
    
    # Remove duplicates, keeping the first occurrence
    df.drop_duplicates(subset='uuid', keep='first', inplace=True)
    
    # Calculate how many rows were removed
    removed_count = original_count - len(df)
    
    print(f"Removed {removed_count} duplicate entries based on 'uuid'.")
    return df


############################################################################ DATA TYPES CHECK ######################################################################################


def validate_data_types(df, precision_requirements):
    """
    Validate the data types and precision of numerical fields.
    Parameters:
    - df: DataFrame containing the data.
    - precision_requirements: A dictionary specifying the expected data type for each column.
      The keys are column names, and the values are expected_dtype.
    """
    issues_found = False

    for column, (expected_dtype) in precision_requirements.items():
        # Check data type
        if df[column].dtype != expected_dtype:
            print(f"Column '{column}' has incorrect data type '{df[column].dtype}', expected '{expected_dtype}'.")
            issues_found = True

    if not issues_found:
        print("All columns have correct data types.")
    else:
        print("Data type issues found.")


############################################################################  DIFFERENT VALUE  ######################################################################################


def validate_symbol_side(df):
    """
    Validate that all rows have 'COINBASE_SPOT_BTC_USD' as symbol_id and 'SELL' or 'BUY' as taker_side.
    Prints the rows that do not meet these criteria.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the trade data.
    """
    # Check for rows where 'symbol_id' is not 'COINBASE_SPOT_BTC_USD'
    incorrect_symbol = df['symbol_id'] != 'COINBASE_SPOT_BTC_USD' #Series where True if the symbol_id in that row is not equal to 'COINBASE_SPOT_BTC_USD'.
    
    # Check for rows where 'taker_side' is neither 'SELL' nor 'BUY'
    incorrect_taker_side = ~df['taker_side'].isin(['SELL', 'BUY'])
    
    # Combine the conditions to find rows with any issues
    issues = df[incorrect_symbol | incorrect_taker_side]   #New boolean Series where True for row if either incorrect_symbol is True or incorrect_taker_side is True
    
    # Check if there are any issues found and print them
    if not issues.empty:
        print(f"{len(issues)} issues found in the following rows ( c pas un pb :)) ")
        print(issues)
    else:
        print("No issues found. All rows meet the specified criteria.")
                                                                                     

#########################################################################  Monthly Checkpoints to verify the join procedure  ######################################################################################


# This one will be used after the reviewing the join procedure

def print_monthly_checkpoints(df):
    """
    Prints the row before the start of each new month (except for the first month)
    and the first three rows of each new month to ensure data continuity and order.

    Parameters:
    - df (pd.DataFrame): DataFrame containing trading data with a 'time_exchange' column.
    """
    # Ensure 'time_exchange' is in datetime format
    df['time_exchange'] = pd.to_datetime(df['time_exchange'])

    # Sort the DataFrame by 'time_exchange' to ensure correct order
    df.sort_values(by='time_exchange', inplace=True)

    # Group by year and month
    grouped = df.groupby([df['time_exchange'].dt.year, df['time_exchange'].dt.month])

    # Initialize a list to store the results
    results = []

    # Iterate over each group
    for (year, month), group in grouped:
        # Get the index of the first entry in the group
        first_entry_index = group.index.min()
        
        # Fetch the last row of the previous month if it's not the first group
        if first_entry_index != 0:
            previous_row = df.loc[first_entry_index - 1: first_entry_index - 1]
            results.append(previous_row)
        
        # Get the first three rows of the current month
        first_three_rows = group.head(3)
        results.append(first_three_rows)

    # Concatenate all results into a single DataFrame
    result_df = pd.concat(results, ignore_index=True)

    # Print the resulting DataFrame
    print(result_df)


#########################################################################  SORT CHECK ######################################################################################


def check_sorted(df):
    """
    Checks if the DataFrame is correctly sorted by the 'time_exchange' column and prints the result.

    Parameters:
    - df (pd.DataFrame): DataFrame containing trading data with a 'time_exchange' column.
    """
    
    try:
        # Check if 'time_exchange' is sorted in a monotonically increasing order
        is_sorted = df['time_exchange'].is_monotonic_increasing
        if is_sorted:
            print("The DataFrame is sorted.")
        else:
            print("The DataFrame is not sorted.")
    except KeyError:
        # Handle the case where 'time_exchange' column does not exist in the DataFrame
        print("Error: The DataFrame does not contain a 'time_exchange' column.")




def sort_frame(df):
    """
    Sorts thze DataFrame by the 'time_exchange' column
    Args:
        df (pd.DataFrame): DataFrame containing trading data with a 'time_exchange' column.

    Returns:
        df (pd.DataFrame): DataFrame sorted.   
         
    """
    return df.sort_values(by=['time_exchange'])


###################################################################################    MAIN    ############################################################################################


precision_requirements = {
    'symbol_id': 'object',
    'time_exchange': 'datetime64[ns, UTC]',
    'time_coinapi' : 'datetime64[ns, UTC]',
    'price': 'float64',
    'size': 'float64',
    'taker_side': 'object',
    }
columns_to_keep = ['time_exchange', 'price', 'size', 'taker_side']
tables = ['trades_mar_may', 'trades_june1', 'trades_june2', 'trades_jul1', 'trades_jul2', 'trades_aug_sept', 'trades_oct_nov', 'trades_dec_jan', 'trades_fev_apr', 'trades_from_april']
tables = ['trades_dec_jan']



if __name__ == '__main__':
    conn = get_database_connection()

    for table_name in tables:
        print(f"Validating data for table: {table_name}")
        df = fetch_table_data(conn, table_name)

        # Check for duplicate entries, sort, zero, missing values
        check_for_duplicate_entries(df)
        check_sorted(df)
        #check_for_negative_or_zero_values(df)
        # missing_values_count = df.isna().sum()
        # print(missing_values_count)

        # Validate time consistency, symbol consistency ,data types
        # validate_time_fluctuations(df)
        # validate_symbol_side(df)
        # validate_data_types(df, precision_requirements)  

        # Solvez the pb and double checking
        df = sort_frame(df)
        check_sorted(df)
        df =  delete_duplicates(df)
        check_for_duplicate_entries(df)
        print(df.shape)

        # Keep only intresting column and print the final dimension 
        print(df[columns_to_keep].head(3))
        print(df[columns_to_keep].tail(3))







