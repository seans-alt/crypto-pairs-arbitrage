import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtester import PairsBacktester

class PortfolioBacktester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.optimized_params = {
            'LINK-ETH': {'z_entry': 2.5, 'z_exit': 1.5},
            'DOT-BTC': {'z_entry': 3.0, 'z_exit': 1.5},
            'ADA-BTC': {'z_entry': 3.0, 'z_exit': 0.1}
        }
    
    def run_portfolio_backtest(self):
        """Backtest optimized portfolio"""
        portfolio_returns = None
        individual_returns = {}
        
        print("=== PORTFOLIO BACKTEST ===")
        
        for pair_name, params in self.optimized_params.items():
            try:
                # Load pair data
                pair_df = pd.read_csv(f'pairs/{pair_name}.csv', 
                                    index_col='timestamp', parse_dates=True)
                
                # Load hedge ratio
                results_df = pd.read_csv('results/cointegration_results.csv')
                hedge_ratio = results_df[results_df['pair'] == pair_name]['hedge_ratio'].iloc[0]
                
                # Backtest with optimized parameters
                backtester = PairsBacktester(
                    z_entry=params['z_entry'], 
                    z_exit=params['z_exit']
                )
                
                result = backtester.backtest_pair(pair_name, pair_df, hedge_ratio)
                
                if result:
                    # Equal weight portfolio
                    pair_returns = result['returns'] / len(self.optimized_params)
                    
                    if portfolio_returns is None:
                        portfolio_returns = pair_returns
                    else:
                        portfolio_returns = portfolio_returns.add(pair_returns, fill_value=0)
                    
                    individual_returns[pair_name] = result['returns']
                    print(f"✅ {pair_name}: Sharpe={result['sharpe_ratio']:.2f}")
                    
            except Exception as e:
                print(f"❌ Error with {pair_name}: {e}")
        
        if portfolio_returns is None:
            return None
        
        # Calculate portfolio metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(24 * 365)
        
        # Max drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'portfolio_returns': portfolio_returns,
            'individual_returns': individual_returns,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'cumulative_returns': cumulative_returns
        }
    
    def plot_portfolio_performance(self, results):
        """Plot portfolio vs individual performance"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Cumulative returns
        ax1.plot(results['cumulative_returns'].index, 
                results['cumulative_returns'].values, 
                label='Portfolio', linewidth=2.5, color='black')
        
        for pair_name, returns in results['individual_returns'].items():
            cumulative = (1 + returns).cumprod()
            ax1.plot(cumulative.index, cumulative.values, label=pair_name, alpha=0.7)
        
        ax1.set_title('Portfolio vs Individual Pair Performance')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        running_max = results['cumulative_returns'].expanding().max()
        portfolio_drawdown = (results['cumulative_returns'] - running_max) / running_max
        
        ax2.fill_between(portfolio_drawdown.index, portfolio_drawdown.values, 0, 
                        alpha=0.3, color='red', label='Portfolio Drawdown')
        ax2.set_title('Portfolio Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/portfolio_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("=== OPTIMIZED PORTFOLIO BACKTEST ===\n")
    
    backtester = PortfolioBacktester(initial_capital=10000)
    results = backtester.run_portfolio_backtest()
    
    if results:
        print(f"\n📊 PORTFOLIO RESULTS:")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
        
        # Individual performance
        print(f"\n🔍 INDIVIDUAL PERFORMANCE:")
        for pair_name, returns in results['individual_returns'].items():
            sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)
            total_ret = (1 + returns).cumprod().iloc[-1] - 1
            print(f"   {pair_name}: Return={total_ret:.2%}, Sharpe={sharpe:.2f}")
        
        # Plot results
        backtester.plot_portfolio_performance(results)
        
        # Save portfolio results
        portfolio_summary = pd.DataFrame([{
            'portfolio_return': results['total_return'],
            'portfolio_sharpe': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'num_pairs': len(results['individual_returns'])
        }])
        portfolio_summary.to_csv('results/portfolio_summary.csv', index=False)
        
        print(f"\n✅ Portfolio analysis complete!")
        
    return results

if __name__ == "__main__":
    results = main()