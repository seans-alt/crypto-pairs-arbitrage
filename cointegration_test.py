import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from scipy.stats import pearsonr
import os

def engle_granger_test(series1, series2):
    """Engle-Granger cointegration test"""
    # Align series
    aligned_s1, aligned_s2 = series1.align(series2, join='inner')
    
    if len(aligned_s1) < 30:
        return None
    
    # Step 1: Regression
    X = np.column_stack([np.ones(len(aligned_s2)), aligned_s2.values])
    beta = np.linalg.lstsq(X, aligned_s1.values, rcond=None)[0]
    
    # Step 2: Stationarity test on residuals
    spread = aligned_s1 - beta[1] * aligned_s2
    adf_result = adfuller(spread.dropna())
    
    return {
        'hedge_ratio': beta[1],
        'pvalue': adf_result[1],
        'adf_statistic': adf_result[0],
        'spread_std': spread.std()
    }

def calculate_half_life(spread):
    """Calculate mean reversion half-life"""
    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag
    
    spread_ret = spread_ret.dropna()
    spread_lag = spread_lag.dropna()
    
    if len(spread_ret) < 2:
        return float('inf')
    
    beta = np.cov(spread_ret, spread_lag)[0, 1] / np.var(spread_lag)
    return -np.log(2) / beta if beta < 0 else float('inf')

def test_pair(pair_df, pair_name):
    """Comprehensive test for a single pair"""
    col1, col2 = pair_df.columns[0], pair_df.columns[1]
    series1, series2 = pair_df[col1], pair_df[col2]
    
    # Correlation
    corr, _ = pearsonr(series1, series2)
    
    # Cointegration test
    eg_result = engle_granger_test(series1, series2)
    
    if eg_result is None:
        return None
    
    # Calculate spread and half-life
    spread = series1 - eg_result['hedge_ratio'] * series2
    half_life = calculate_half_life(spread)
    
    return {
        'pair': pair_name,
        'correlation': corr,
        'coint_pvalue': eg_result['pvalue'],
        'hedge_ratio': eg_result['hedge_ratio'],
        'half_life_hours': half_life,
        'spread_std': eg_result['spread_std'],
        'is_cointegrated': eg_result['pvalue'] < 0.05,
        'data_points': len(series1)
    }

def main():
    print("=== Cointegration Testing ===\n")
    
    # Load pair data
    pair_files = [f for f in os.listdir('pairs') if f.endswith('.csv')]
    results = []
    
    for file in pair_files:
        pair_name = file.replace('.csv', '')
        pair_df = pd.read_csv(f'pairs/{file}', index_col='timestamp', parse_dates=True)
        
        result = test_pair(pair_df, pair_name)
        if result:
            results.append(result)
            status = "✓" if result['is_cointegrated'] else "✗"
            print(f"{status} {pair_name}: p={result['coint_pvalue']:.4f}, HL={result['half_life_hours']:.1f}h")
    
    # Create results dataframe
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('coint_pvalue')
        
        # Save results
        os.makedirs('results', exist_ok=True)
        results_df.to_csv('results/cointegration_results.csv', index=False)
        
        print(f"\n=== Top Cointegrated Pairs ===")
        top_pairs = results_df[results_df['is_cointegrated']].head(5)
        
        for _, row in top_pairs.iterrows():
            print(f"{row['pair']}: p-value={row['coint_pvalue']:.4f}, half-life={row['half_life_hours']:.1f}h")
    
    return results_df

if __name__ == "__main__":
    results = main()