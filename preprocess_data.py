import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def load_raw_data():
    """Load all CSV files from data directory"""
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    data_dict = {}
    
    for file in data_files:
        symbol = file.replace('.csv', '').replace('_', '/')
        df = pd.read_csv(f'data/{file}', index_col='timestamp', parse_dates=True)
        data_dict[symbol] = df
    
    print(f"Loaded {len(data_dict)} pairs")
    return data_dict

def align_timestamps(data_dict):
    """Align all pairs to common timestamp index"""
    common_index = None
    for symbol, df in data_dict.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    aligned_data = {}
    for symbol, df in data_dict.items():
        aligned_df = df.reindex(common_index)
        aligned_data[symbol] = aligned_df
    
    print(f"Aligned to {len(common_index)} common timestamps")
    return aligned_data

def handle_missing_data(df):
    """Fill small gaps, drop large gaps"""
    # Forward fill for small gaps (up to 2 hours)
    df_filled = df.ffill(limit=2)
    
    # Drop any remaining NaN rows
    initial_len = len(df)
    df_clean = df_filled.dropna()
    dropped = initial_len - len(df_clean)
    
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing data")
    
    return df_clean

def calculate_returns(df):
    """Calculate log returns and simple returns"""
    df = df.copy()
    df['log_price'] = np.log(df['close'])
    df['log_return'] = df['log_price'].diff()
    df['simple_return'] = df['close'].pct_change()
    return df

def create_pair_dataframes(aligned_data):
    """Create combined dataframes for each possible pair"""
    symbols = list(aligned_data.keys())
    pairs = []
    
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            asset1, asset2 = symbols[i], symbols[j]
            
            df1 = aligned_data[asset1][['close']].copy()
            df2 = aligned_data[asset2][['close']].copy()
            
            df1.columns = [f'{asset1.split("/")[0]}_close']
            df2.columns = [f'{asset2.split("/")[0]}_close']
            
            pair_df = pd.concat([df1, df2], axis=1).dropna()
            pair_name = f"{asset1.split('/')[0]}-{asset2.split('/')[0]}"
            
            pairs.append((pair_name, asset1, asset2, pair_df))
    
    return pairs

def main():
    print("=== Data Preprocessing ===\n")
    
    # Load raw data
    raw_data = load_raw_data()
    
    # Align timestamps
    aligned_data = align_timestamps(raw_data)
    
    # Preprocess each series
    processed_data = {}
    for symbol, df in aligned_data.items():
        print(f"Processing {symbol}:")
        df_clean = handle_missing_data(df)
        df_returns = calculate_returns(df_clean)
        processed_data[symbol] = df_returns
    
    # Create pair combinations
    pairs = create_pair_dataframes(processed_data)
    print(f"\nCreated {len(pairs)} pair combinations")
    
    # Save processed data
    os.makedirs('processed_data', exist_ok=True)
    for symbol, df in processed_data.items():
        filename = f"processed_data/{symbol.replace('/', '_')}_processed.csv"
        df.to_csv(filename)
    
    # Save pair data
    os.makedirs('pairs', exist_ok=True)
    for pair_name, asset1, asset2, pair_df in pairs:
        filename = f"pairs/{pair_name}.csv"
        pair_df.to_csv(filename)
    
    print("✓ All data processed and saved")
    
    return processed_data, pairs

if __name__ == "__main__":
    processed_data, pairs = main()