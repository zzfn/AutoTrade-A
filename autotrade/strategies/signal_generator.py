from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Union

import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm

from autotrade.data.providers import AKShareDataProvider, DataProviderFactory
from autotrade.features import QlibFeatureGenerator
from autotrade.models import LightGBMTrainer, ModelManager


class SignalGenerator:
    """
    Signal Generator responsible for:
    1. Loading ML models.
    2. Fetching data (features).
    3. Generating predictions (signals).
    
    This component is independent of the backtesting engine.
    """

    def __init__(
        self,
        symbols: List[str],
        model_name: Optional[str] = None,
        models_dir: str = "artifacts/models",
        market: str = "us",
        lookback_period: int = 60,
        interval: str = "day",
        top_k: int = 3,
    ):
        self.symbols = symbols
        self.model_name = model_name
        self.models_dir = models_dir
        self.market = market.lower()
        self.lookback_period = lookback_period
        self.interval = interval
        self.top_k = min(top_k, len(symbols))

        # Initialize components
        self.feature_generator = QlibFeatureGenerator()
        self.model_manager = ModelManager(self.models_dir)
        self.trainer: Optional[LightGBMTrainer] = None
        
        # A-share provider (lazy loaded)
        self._cn_provider: Optional[AKShareDataProvider] = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the ML model."""
        try:
            if self.model_name:
                model_path = Path(self.models_dir) / self.model_name
            else:
                model_path = self.model_manager.get_current_model_path()
                if model_path:
                    self.model_name = model_path.name

            if not model_path or not model_path.exists():
                logger.warning(f"ML model not found, will use mock predictions if enabled.")
                return

            self.trainer = LightGBMTrainer(model_dir=self.models_dir)
            self.trainer.load(model_path)

            # Sync interval if available in metadata
            if hasattr(self.trainer, "metadata") and "interval" in self.trainer.metadata:
                model_interval = self.trainer.metadata["interval"]
                if model_interval == "1h":
                    self.interval = "hour"
                elif model_interval == "1d":
                    self.interval = "day"
            
            logger.info(f"Loaded model: {model_path.name} (Interval: {self.interval})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.trainer = None

    def _get_cn_provider(self) -> AKShareDataProvider:
        """Get or create A-share data provider."""
        if self._cn_provider is None:
            self._cn_provider = DataProviderFactory.get_provider("cn")
        return self._cn_provider

    def _is_valid_candidate(self, symbol: str, date: datetime) -> bool:
        """Check if stock is valid for trading (not ST, not suspended)."""
        if self.market != "cn":
            return True
        
        provider = self._get_cn_provider()
        try:
            if provider.is_st_stock(symbol):
                return False
            if provider.is_suspended(symbol, date):
                return False
            return True
        except Exception:
            # If check fails, assume valid to avoid blocking
            return True

    def generate_signals(
        self, 
        current_date: datetime, 
        data_lookup: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """
        Generate predictions for a specific date.
        
        Args:
            current_date: Date to generate signals for.
            data_lookup: Optional dictionary mapping symbol to DataFrame of historical data.
                         If not provided, data will be fetched via backtester/provider logic
                         (Not fully implemented here as we assume this method is called 
                         with data available or we use a data provider).
                         
                         For now, we'll try to use a fetch mechanism if `data_lookup` is missing,
                         similar to QlibMLStrategy.
                         
        Returns:
            Dict of {symbol: prediction_score}
        """
        predictions = {}
        
        for symbol in self.symbols:
            if not self._is_valid_candidate(symbol, current_date):
                continue

            # Need history
            df = None
            if data_lookup and symbol in data_lookup:
                df = data_lookup[symbol]
            else:
                # TODO: Implement data fetching extraction if needed here.
                # For `vectorbt`, we usually pass the full history.
                # But for `daily prediction`, we might need to fetch live data.
                pass
            
            # Since we are extracting from QlibMLStrategy, we need a replacement. 
            # Ideally SignalGenerator should be passed the data or have its own data provider instance.
            # For this step, I will assume the caller provides data or I will implement a fetcher in a later step.
            
            # To keep it simple for now, let's assume `data_lookup` is mandatory or we strictly use it for backtesting context where data is passed.
            # BUT: For daily usage, we need to fetch data.
            
            # Let's add a placeholder for fetching if data_lookup missing.
            # I can use `AKShareDataProvider` for CN or YFinance for US (via some provider).
            
            pass 
            
        return predictions

    def predict_single(self, df: pd.DataFrame) -> float:
        """
        Predict score for a single symbol using its dataframe.
        Assumes df has enough history.
        """
        if len(df) < 30:
            return -np.inf

        # Generate features
        try:
             # Ensure columns match what FeatureGenerator expects
             # FeatureGenerator expects: Open, High, Low, Close, Volume (capitalized or not depending on impl)
             # QlibMLStrategy renamed them to lowercase.
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            
            features = self.feature_generator._generate_single_symbol(df)
            if features is None or features.empty:
                return -np.inf
            
            latest_features = features.iloc[[-1]]
            
            if self.trainer:
                pred = self.trainer.predict(latest_features)[0]
            else:
                # Fallback: Momentum
                pred = df["close"].pct_change(5).iloc[-1]
                
            return float(pred)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return -np.inf

    def generate_latest_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate signals for the latest timestamp in the provided DataFrame.
        
        Args:
            df: MultiIndex DataFrame (timestamp, symbol) with OHLCV data.
            
        Returns:
            List of prediction dictionaries.
        """
        predictions = []
        
        # 1. Generate features for all data
        # FeatureGenerator handles MultiIndex automatically
        try:
            features = self.feature_generator.generate(df)
        except Exception as e:
            logger.error(f"Feature geneation failed: {e}")
            return []

        # 2. Get latest timestamp globally to identify "current" predictions
        if features.empty:
            return []
            
        latest_date = features.index.get_level_values(0).max()
        
        # 3. Filter valid symbols and prepare batch
        valid_rows = []
        valid_symbols = []
        
        # 优化：不再使用 tqdm 逐个打印进度，而是快速过滤
        # 但为了用户感知，简单打印一下
        logger.info("Filtering valid candidates...")
        
        provider = self._get_cn_provider()
        
        # Iterate over symbols present in the data
        symbols = features.index.get_level_values("symbol").unique()
        
        for symbol in symbols:
            try:
                # Validity checks
                # 1. ST Check (Fast via memory/disk cache)
                if provider.is_st_stock(symbol):
                    continue
                
                # 2. Suspension Check (Fast via memory df)
                if (latest_date, symbol) in df.index:
                    vol = df.loc[(latest_date, symbol), "volume"]
                    if vol == 0 or pd.isna(vol):
                        continue
                else:
                    continue

                # Get features
                if (latest_date, symbol) not in features.index:
                     continue
                
                row = features.loc[[(latest_date, symbol)]]
                valid_rows.append(row)
                valid_symbols.append(symbol)
                
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
                continue

        if not valid_rows:
            return []
            
        # 4. Batch Prediction
        logger.info(f"Batch predicting for {len(valid_rows)} symbols...")
        try:
            batch_features = pd.concat(valid_rows)
            
            if self.trainer:
                scores = self.trainer.predict(batch_features)
            else:
                scores = [0.0] * len(batch_features)
                
            # 5. Assemble results
            for i, symbol in enumerate(valid_symbols):
                score = float(scores[i])
                
                if score > 0.01:
                    signal = "BUY"
                elif score < -0.01:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                predictions.append({
                    "symbol": symbol,
                    "signal": signal,
                    "score": score,
                    "confidence": abs(score) * 100,
                    "date": latest_date.strftime("%Y-%m-%d %H:%M:%S"),
                })
                
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return []
        
        return predictions

