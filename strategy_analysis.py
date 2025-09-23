import pandas as pd
import matplotlib.pyplot as plt

def analyze_strategy_performance():
    """Compare original vs optimized parameters"""
    
    # Load results
    original_df = pd.read_csv('results/backtest_summary.csv')
    optimized_df = pd.read_csv('results/optimized_backtest.csv')
    
    print("=== STRATEGY COMPARISON ===")
    
    # Merge results
    comparison = original_df.merge(optimized_df, on='pair', suffixes=('_original', '_optimized'))
    
    for _, row in comparison.iterrows():
        print(f"\n📈 {row['pair']}:")
        print(f"   Original: Sharpe={row['sharpe_ratio_original']:.2f}, Return={row['total_return_original']:.2%}")
        print(f"   Optimized: Sharpe={row['sharpe_ratio_optimized']:.2f}, Return={row['total_return_optimized']:.2%}")
        print(f"   Parameters: z_entry={row['optimized_z_entry']}, z_exit={row['optimized_z_exit']}")
        
        improvement = row['sharpe_ratio_optimized'] - row['sharpe_ratio_original']
        print(f"   Improvement: {improvement:+.2f} Sharpe ratio")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sharpe ratio comparison
    pairs = comparison['pair']
    original_sharpe = comparison['sharpe_ratio_original']
    optimized_sharpe = comparison['sharpe_ratio_optimized']
    
    x = range(len(pairs))
    ax1.bar(x, original_sharpe, width=0.4, label='Original', alpha=0.7)
    ax1.bar([i + 0.4 for i in x], optimized_sharpe, width=0.4, label='Optimized', alpha=0.7)
    ax1.set_xticks([i + 0.2 for i in x])
    ax1.set_xticklabels(pairs, rotation=45)
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Returns comparison
    original_returns = comparison['total_return_original'] * 100
    optimized_returns = comparison['total_return_optimized'] * 100
    
    ax2.bar(x, original_returns, width=0.4, label='Original', alpha=0.7)
    ax2.bar([i + 0.4 for i in x], optimized_returns, width=0.4, label='Optimized', alpha=0.7)
    ax2.set_xticks([i + 0.2 for i in x])
    ax2.set_xticklabels(pairs, rotation=45)
    ax2.set_ylabel('Total Return (%)')
    ax2.set_title('Return Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_strategy_performance()