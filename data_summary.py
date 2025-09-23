import pandas as pd
import numpy as np
import os

def analyze_data_quality():
    """Quick analysis of our dataset"""
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    
    print("=== DATA QUALITY REPORT ===")
    print(f"Found {len(data_files)} data files\n")
    
    for file in data_files:
        symbol = file.replace('.csv', '').replace('_', '/')
        df = pd.read_csv(f'data/{file}', index_col='timestamp', parse_dates=True)
        
        print(f"📊 {symbol}:")
        print(f"   Period: {df.index.min()} to {df.index.max()}")
        print(f"   Records: {len(df)}")
        print(f"   Price: ${df['close'].iloc[-1]:.2f} (Δ: {df['close'].pct_change().std():.2%} daily vol)")
        print(f"   Volume: {df['volume'].mean():.0f}/hour")
        print()

def check_pair_combinations():
    """Show how many pairs we'll test"""
    symbols = [f.replace('.csv', '').replace('_', '/') for f in os.listdir('data') if f.endswith('.csv')]
    n_pairs = len(symbols) * (len(symbols) - 1) // 2
    
    print(f"=== PAIR COMBINATIONS ===")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Total pairs to test: {n_pairs}")
    print(f"Expected pairs: {', '.join([f'{s1}-{s2}' for i, s1 in enumerate(symbols) for j, s2 in enumerate(symbols) if i < j])}")

if __name__ == "__main__":
    analyze_data_quality()
    check_pair_combinations()