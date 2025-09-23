import pandas as pd
import numpy as np
from backtester import PairsBacktester
from itertools import product

def optimize_parameters(pair_name, pair_df, hedge_ratio):
    """Find optimal z-score parameters for each pair"""
    z_entries = [1.5, 2.0, 2.5, 3.0]
    z_exits = [0.1, 0.5, 1.0, 1.5]
    
    best_sharpe = -np.inf
    best_params = {}
    best_result = None
    
    print(f"🔧 Optimizing {pair_name}...")
    
    for z_entry, z_exit in product(z_entries, z_exits):
        if z_exit >= z_entry:  # Exit must be less than entry
            continue
            
        backtester = PairsBacktester(z_entry=z_entry, z_exit=z_exit)
        result = backtester.backtest_pair(pair_name, pair_df, hedge_ratio)
        
        if result and result['sharpe_ratio'] > best_sharpe:
            best_sharpe = result['sharpe_ratio']
            best_params = {'z_entry': z_entry, 'z_exit': z_exit}
            best_result = result
    
    if best_result and best_sharpe > -np.inf:
        print(f"   ✅ Best: z_entry={best_params['z_entry']}, z_exit={best_params['z_exit']}")
        print(f"   📊 Sharpe: {best_sharpe:.2f}, Return: {best_result['total_return']:.2%}")
        return best_params, best_result
    else:
        print(f"   ❌ No profitable parameters found")
        return None, None

def main():
    print("=== PARAMETER OPTIMIZATION ===")
    
    # Load cointegration results
    results_df = pd.read_csv('results/cointegration_results.csv')
    cointegrated_pairs = results_df[results_df['is_cointegrated']]
    
    optimized_results = []
    
    for _, row in cointegrated_pairs.iterrows():
        pair_name = row['pair']
        hedge_ratio = row['hedge_ratio']
        
        try:
            pair_df = pd.read_csv(f'pairs/{pair_name}.csv', 
                                index_col='timestamp', parse_dates=True)
            
            best_params, result = optimize_parameters(pair_name, pair_df, hedge_ratio)
            if result:
                result['optimized_z_entry'] = best_params['z_entry']
                result['optimized_z_exit'] = best_params['z_exit']
                optimized_results.append(result)
            
        except Exception as e:
            print(f"Error optimizing {pair_name}: {e}")
    
    # Save optimized results
    if optimized_results:
        summary_data = []
        for res in optimized_results:
            summary_data.append({
                'pair': res['pair'],
                'optimized_z_entry': res['optimized_z_entry'],
                'optimized_z_exit': res['optimized_z_exit'],
                'total_return': res['total_return'],
                'sharpe_ratio': res['sharpe_ratio'],
                'max_drawdown': res['max_drawdown'],
                'win_rate': res['win_rate'],
                'num_trades': res['num_trades']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('results/optimized_backtest.csv', index=False)
        print(f"\n✅ Optimized results saved for {len(optimized_results)} pairs")
    
    return optimized_results

if __name__ == "__main__":
    results = main()