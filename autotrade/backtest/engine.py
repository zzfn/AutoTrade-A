import pandas as pd
import numpy as np
import vectorbt as vbt
from loguru import logger
from typing import Optional, Dict, Any

from autotrade.strategies.signal_generator import SignalGenerator

class BacktestEngine:
    """
    Vectorized Backtesting Engine using vectorbt.
    Executes strategies defined by SignalGenerator.
    """

    def __init__(
        self, 
        signal_generator: SignalGenerator, 
        initial_capital: float = 100000.0,
        commission: float = 0.0005,  # 0.05% commission
        slippage: float = 0.0005     # 0.05% slippage
    ):
        self.signal_generator = signal_generator
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, data: pd.DataFrame) -> vbt.Portfolio:
        """
        Run backtest on provided data.
        
        Args:
            data: MultiIndex DataFrame with levels (timestamp, symbol) 
                  and columns (open, high, low, close, volume).
                  
        Returns:
            vbt.Portfolio object
        """
        logger.info(f"Starting backtest on {len(data)} rows of data...")
        
        # 1. Generate Features
        # Ensure column names are lower case
        data.columns = [c.lower() for c in data.columns]
        
        logger.info("Generating features...")
        features = self.signal_generator.feature_generator.generate(data)
        
        # 2. Predict Returns
        logger.info("Predicting returns using ML model...")
        if self.signal_generator.trainer is None:
            logger.warning("No ML model loaded. Using Close Price Momentum as fallback.")
            # Fallback: 5-day return momentum
            # We need to compute it on the pivot table to be safe or group by symbol
            # features index is (timestamp, symbol)
            # But let's check if 'close' is in features or we need to use data
            # FeatureGenerator adds raw columns if configured, but let's use 'data' for fallback metrics
            close_prices = data['close'].unstack()
            predictions_df = close_prices.pct_change(5)
            # Stack back to align with features structure if needed, but we proceed to pivoting anyway
        else:
            # Predict
            # Note: predict() expects DataFrame matching feature_names
            # We filter features to match model's expected input if possible, 
            # effectively just passing 'features' usually works if columns match.
            try:
                preds = self.signal_generator.trainer.predict(features)
                # Create a Series with MultiIndex
                predictions_series = pd.Series(preds, index=features.index)
                predictions_df = predictions_series.unstack()
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise e

        # 3. Form Portfolio (Select Top K)
        # predictions_df is Index=Timestamp, Columns=Symbol
        
        # Rank assets cross-sectionally
        logger.info(f"Selecting Top-{self.signal_generator.top_k} assets...")
        ranks = predictions_df.rank(axis=1, ascending=False)
        
        # Create signals (Long Top-K)
        # We assume long-only for now as per original strategy
        long_signals = ranks <= self.signal_generator.top_k
        
        # 4. Filter for A-share rules (if market is CN)
        if self.signal_generator.market == "cn":
            # Just a placeholder strict check, might be slow loop
            # But vectorbt can filter signals too.
            # For now, we assume data provided excludes suspended stocks or we handle it via price being NaN
            pass
            
        # 5. Calculate Weights
        # Equal weight for selected assets
        weights = long_signals.astype(float)
        row_sums = weights.sum(axis=1)
        # Avoid division by zero
        weights = weights.div(row_sums.replace(0, 1), axis=0)
        
        # Cash buffer (95% invested)
        weights = weights * 0.95
        
        # 6. Execute with vectorbt
        logger.info("Simulating trades with vectorbt...")
        
        # Prepare close prices aligned with weights
        close = data['close'].unstack()
        close = close.reindex(weights.index).ffill()
        
        # Align columns
        common_cols = weights.columns.intersection(close.columns)
        weights = weights[common_cols]
        close = close[common_cols]
        
        # Run Portfolio simulation
        # Use targetpercent size_type to rebalance to weights
        # size_granularity=100: 强制成交股数为 100 的整数倍 (A股一手规则)
        # min_size=100: 最少购买 100 股
        pf = vbt.Portfolio.from_orders(
            close=close,
            size=weights,
            size_type='targetpercent',
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage,
            freq='1D' if self.signal_generator.interval == 'day' else '1h',
            group_by=True,
            size_granularity=100, 
            min_size=100
        )
        
        # When grouped, total_return() returns a scalar (or single item Series)
        # We need to handle potential scalar or series output.
        total_ret = pf.total_return()
        if isinstance(total_ret, pd.Series):
             total_ret = total_ret.iloc[0] if not total_ret.empty else 0.0
             
        logger.info(f"Backtest complete. Total Return: {total_ret:.2%}")
        return pf

    def get_stats(self, pf: vbt.Portfolio) -> Dict[str, Any]:
        """Extract key statistics from portfolio."""
        # Handle scalar/series outputs for grouped portfolios
        def _get_scalar(val):
            if isinstance(val, pd.Series):
                return val.iloc[0] if not val.empty else 0.0
            return val

        return {
            "total_return": _get_scalar(pf.total_return()),
            "sharpe_ratio": _get_scalar(pf.sharpe_ratio()),
            "max_drawdown": _get_scalar(pf.max_drawdown()),
            "total_trades": pf.trades.count(), # Fix: use trades.count()
            "win_rate": _get_scalar(pf.trades.win_rate()), # Fix: use trades.win_rate()
            "stats": pf.stats().to_dict() # Full stats
        }
