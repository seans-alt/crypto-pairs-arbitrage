"""
Phase 1: Simple Data Collector
Purpose: Fetch basic OHLCV data from Kraken for 2-3 pairs
"""

import ccxt
import pandas as pd
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def initialize_exchange():
    # Initialize Kraken connection
    exchange = ccxt.kraken({
        'apiKey': os.getenv('KRAKEN_API_KEY', ''),  # Public endpoints work without keys initially
        'secret': os.getenv('KRAKEN_SECRET', ''),
        'enableRateLimit': True,  # Crucial: respect exchange rate limits
        'timeout': 30000,  # 30 second timeout
    })
    
    print("✓ Kraken exchange initialized")
    return exchange

def get_test_pairs():
    return ['BTC/USD', 'ETH/USD']

def fetch_ohlcv_safe(exchange, symbol, timeframe='1h', limit=100):
    #Safe data fetching with error handling
    try:
        print(f"Fetching {symbol} {timeframe} data...")
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['symbol'] = symbol
        
        print(f"✓ Retrieved {len(df)} candles for {symbol}")
        return df
        
    except Exception as e:
        print(f"✗ Error fetching {symbol}: {str(e)}")
        return None

def main():
    # Main execution function
    print("=== Crypto Pairs Data Collector ===\n")
    
    # Step 1: Initialize exchange
    exchange = initialize_exchange()
    
    # Step 2: Get our test pairs
    pairs = get_test_pairs()
    print(f"Testing with pairs: {pairs}\n")
    
    # Step 3: Fetch data for each pair
    all_data = {}
    
    for pair in pairs:
        # Fetch data with rate limiting
        data = fetch_ohlcv_safe(exchange, pair)
        
        if data is not None:
            all_data[pair] = data
            
        time.sleep(1)  # 1 second delay
    
    # Step 4: Basic validation
    print(f"\n=== Results ===")
    print(f"Successfully fetched data for {len(all_data)} out of {len(pairs)} pairs")
    
    for pair, df in all_data.items():
        print(f"{pair}: {len(df)} records, from {df.index.min()} to {df.index.max()}")
        
        # Show basic stats
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"  Volume avg: {df['volume'].mean():.2f} BTC/ETH\n")
    
    return all_data

if __name__ == "__main__":
    data = main()