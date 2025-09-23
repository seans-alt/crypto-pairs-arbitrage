import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class PairsBacktester:
    def __init__(self, initial_capital=10000, z_entry=2.0, z_exit=0.5, z_stop=3.0):
        self.initial_capital = initial_capital
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.results = {}
        
    def calculate_zscore(self, spread, window=24):
        """Calculate rolling z-score with minimum periods"""
        mean = spread.rolling(window=window, min_periods=1).mean()
        std = spread.rolling(window=window, min_periods=1).std()
        # Avoid division by zero
        std = std.replace(0, np.nan).ffill().bfill()
        return (spread - mean) / std
    
    def generate_signals(self, pair_df, hedge_ratio):
        """Generate trading signals with stop loss"""
        asset1, asset2 = pair_df.iloc[:, 0], pair_df.iloc[:, 1]
        spread = asset1 - hedge_ratio * asset2
        zscore = self.calculate_zscore(spread)
        
        signals = pd.DataFrame(index=pair_df.index)
        signals['zscore'] = zscore
        signals['position'] = 0
        signals['entry_price'] = 0.0
        
        in_position = False
        position_direction = 0  # 1 for long, -1 for short
        
        for i in range(1, len(signals)):
            current_z = signals['zscore'].iloc[i]
            prev_z = signals['zscore'].iloc[i-1]
            
            if not in_position:
                # Entry conditions
                if current_z < -self.z_entry and prev_z >= -self.z_entry:
                    signals.iloc[i, signals.columns.get_loc('position')] = 1
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = current_z
                    in_position = True
                    position_direction = 1
                elif current_z > self.z_entry and prev_z <= self.z_entry:
                    signals.iloc[i, signals.columns.get_loc('position')] = -1
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = current_z
                    in_position = True
                    position_direction = -1
            else:
                # Check exit conditions
                exit_condition = (
                    (position_direction == 1 and current_z > -self.z_exit) or
                    (position_direction == -1 and current_z < self.z_exit) or
                    abs(current_z) > self.z_stop  # Stop loss
                )
                
                if exit_condition:
                    signals.iloc[i, signals.columns.get_loc('position')] = 0
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = 0
                    in_position = False
                    position_direction = 0
                else:
                    # Maintain position
                    signals.iloc[i, signals.columns.get_loc('position')] = position_direction
                    signals.iloc[i, signals.columns.get_loc('entry_price')] = signals['entry_price'].iloc[i-1]
        
        return signals
    
    def calculate_returns(self, pair_df, signals, hedge_ratio, transaction_cost=0.001):
        """Calculate strategy returns with transaction costs"""
        asset1, asset2 = pair_df.iloc[:, 0], pair_df.iloc[:, 1]
        
        # Price returns
        ret1 = asset1.pct_change()
        ret2 = asset2.pct_change()
        
        # Spread returns (without costs)
        spread_returns = signals['position'].shift(1) * (ret1 - hedge_ratio * ret2)
        
        # Transaction costs (apply when position changes)
        position_changes = signals['position'].diff().abs()
        transaction_costs = position_changes * transaction_cost
        
        # Net returns
        net_returns = spread_returns - transaction_costs
        
        return net_returns.dropna()
    
    def calculate_metrics(self, returns):
        """Calculate performance metrics"""
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        
        # Annualized Sharpe ratio
        sharpe = returns.mean() / returns.std() * np.sqrt(24 * 365)
        
        # Max drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': returns.std(),
            'num_trades': total_trades
        }
    
    def backtest_pair(self, pair_name, pair_df, hedge_ratio):
        """Complete backtest for one pair"""
        signals = self.generate_signals(pair_df, hedge_ratio)
        returns = self.calculate_returns(pair_df, signals, hedge_ratio)
        
        if len(returns) == 0:
            return None
        
        metrics = self.calculate_metrics(returns)
        metrics.update({
            'pair': pair_name,
            'returns': returns,
            'signals': signals,
            'cumulative_returns': (1 + returns).cumprod()
        })
        
        return metrics
    
    def run_backtest(self):
        """Backtest all cointegrated pairs"""
        # Load cointegration results
        results_df = pd.read_csv('results/cointegration_results.csv')
        cointegrated_pairs = results_df[results_df['is_cointegrated']]
        
        backtest_results = []
        
        print("=== Running Backtests ===")
        for _, row in cointegrated_pairs.iterrows():
            pair_name = row['pair']
            hedge_ratio = row['hedge_ratio']
            
            # Load pair data
            try:
                pair_df = pd.read_csv(f'pairs/{pair_name}.csv', 
                                    index_col='timestamp', parse_dates=True)
                
                result = self.backtest_pair(pair_name, pair_df, hedge_ratio)
                if result:
                    backtest_results.append(result)
                    
                    status = "✓" if result['sharpe_ratio'] > 1.0 else "✗"
                    print(f"{status} {pair_name}: Sharpe={result['sharpe_ratio']:.2f}, Return={result['total_return']:.2%}")
                    
            except Exception as e:
                print(f"✗ Error backtesting {pair_name}: {e}")
        
        # Sort by Sharpe ratio
        backtest_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        return backtest_results

def plot_results(backtest_results):
    """Plot comprehensive results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Cumulative returns
    for result in backtest_results:
        axes[0,0].plot(result['cumulative_returns'].index, 
                      result['cumulative_returns'].values,
                      label=f"{result['pair']} (Sharpe: {result['sharpe_ratio']:.2f})")
    
    axes[0,0].set_title('Cumulative Returns')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Sharpe ratios
    pairs = [r['pair'] for r in backtest_results]
    sharpes = [r['sharpe_ratio'] for r in backtest_results]
    axes[0,1].bar(pairs, sharpes)
    axes[0,1].set_title('Sharpe Ratios')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Drawdowns
    for result in backtest_results:
        running_max = result['cumulative_returns'].expanding().max()
        drawdown = (result['cumulative_returns'] - running_max) / running_max
        axes[1,0].plot(drawdown.index, drawdown.values, label=result['pair'])
    
    axes[1,0].set_title('Drawdowns')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Returns distribution
    returns = pd.concat([r['returns'] for r in backtest_results], axis=1)
    returns.columns = [r['pair'] for r in backtest_results]
    returns.plot.kde(ax=axes[1,1])
    axes[1,1].set_title('Returns Distribution')
    
    plt.tight_layout()
    plt.savefig('results/backtest_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=== Pairs Trading Backtest Engine ===\n")
    
    # Initialize backtester
    backtester = PairsBacktester(
        initial_capital=10000,
        z_entry=2.0,
        z_exit=0.5,
        z_stop=3.0
    )
    
    # Run backtests
    results = backtester.run_backtest()
    
    # Display results
    print(f"\n=== Backtest Summary ===")
    print(f"Tested {len(results)} cointegrated pairs")
    
    summary_data = []
    for result in results:
        summary_data.append({
            'pair': result['pair'],
            'total_return': result['total_return'],
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'volatility': result['volatility'],
            'num_trades': result['num_trades']
        })
        
        print(f"\n{result['pair']}:")
        print(f"  Return: {result['total_return']:.2%}")
        print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"  Max DD: {result['max_drawdown']:.2%}")
        print(f"  Win Rate: {result['win_rate']:.2%}")
        print(f"  Trades: {result['num_trades']}")
    
    # Save results
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/backtest_summary.csv', index=False)
    
    # Plot results
    if results:
        plot_results(results)
        print(f"\n✓ Results saved to results/backtest_summary.csv")
        print(f"✓ Charts saved to results/backtest_analysis.png")
    
    return results

if __name__ == "__main__":
    results = main()