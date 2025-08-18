import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads OHLCVT data from a file (CSV or Excel).

    Args:
        filepath: The path to the data file.

    Returns:
        A pandas DataFrame with the preprocessed data.
    """
    # Check if the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: The file '{filepath}' was not found.")

    # Get the file extension
    _, file_extension = os.path.splitext(filepath)

    # Load the data based on the file extension
    if file_extension == '.csv':
        df = pd.read_csv(filepath)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: '{file_extension}'. Please use CSV or Excel.")

    # --- Data Preprocessing ---

    # 1. Rename columns to a standard format (lowercase)
    df.columns = [col.lower() for col in df.columns]

    # 2. Check for required columns
    required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input data must contain the following columns: {required_columns}")

    # 3. Convert 'time' column to datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # 4. Ensure data is sorted by time
    df.sort_index(inplace=True)

    # 5. Handle missing values (e.g., forward-fill)
    df.ffill(inplace=True)
    # also back-fill any remaining NaNs at the beginning of the series
    df.bfill(inplace=True)

    # 6. Ensure correct data types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])

    print("Data loaded and preprocessed successfully.")
    return df
