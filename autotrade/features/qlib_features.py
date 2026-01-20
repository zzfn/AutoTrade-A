"""
Qlib 特征生成器 - 生成 ML 模型所需的技术指标特征

任务 2.1 - 2.3: 实现特征生成、预处理和 Alpha158 基础因子
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from autotrade.features.smc import calculate_smc_features
from autotrade.features.price_action import calculate_price_action_features


class BaseFeatureGenerator(ABC):
    """特征生成器基类"""

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成特征"""
        pass


class QlibFeatureGenerator(BaseFeatureGenerator):
    """
    Qlib 风格特征生成器

    生成类似 Alpha158 的技术指标因子，包括：
    - 价格动量因子
    - 均线因子
    - 波动率因子
    - 成交量因子
    - 技术指标因子（RSI, MACD 等）
    - SMC 因子 (FVG, BOS, OB)
    - 价格行为因子 (Candlestick Patterns)
    """

    def __init__(
        self,
        window_sizes: list[int] | None = None,
        include_raw: bool = False,
        use_log: bool = True,
        use_log_returns: bool = True,
        use_cross_sectional_rank: bool = True,
        fill_method: str = "ffill",
    ):
        """
        初始化特征生成器 (Dimensionless Mode)

        Args:
            window_sizes: 滚动窗口大小列表
            include_raw: 是否包含原始 OHLCV 特征 (False 以消除量纲)
            use_log: 原始特征是否对数化 (仅当 include_raw=True 时有效)
            use_log_returns: 是否使用对数收益率 log(P_t/P_{t-1})
            use_cross_sectional_rank: 是否使用截面排名 (映射到 [0, 1])
            fill_method: 缺失值填充方法
        """
        self.window_sizes = window_sizes or [5, 10, 20, 30, 60]
        self.include_raw = include_raw
        self.use_log = use_log
        self.use_log_returns = use_log_returns
        self.use_cross_sectional_rank = use_cross_sectional_rank
        self.fill_method = fill_method


    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成全部特征

        Args:
            df: 输入数据，包含 open, high, low, close, volume 列
                可以是单股票 (DatetimeIndex) 或多股票 (MultiIndex)

        Returns:
            包含所有特征的 DataFrame
        """
        logger.info(f"开始生成特征，输入数据 shape: {df.shape}")

        if isinstance(df.index, pd.MultiIndex):
            # 多股票：按股票分组处理
            return self._generate_multi_symbol(df)
        else:
            # 单股票
            return self._generate_single_symbol(df)

    def _generate_single_symbol(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """为单个股票生成特征"""
        features = {}

        # 原始特征
        if self.include_raw:
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    val = df[col]
                    # 对数化处理
                    if self.use_log:
                        val = np.log1p(val)
                    features[f"${col}"] = val

        # 价格回报率
        close = df["close"]
        if self.use_log_returns:
            # Log Return: ln(P_t / P_{t-1})
            log_close = np.log(close)
            features["$return_1d"] = log_close.diff(1)
            features["$return_5d"] = log_close.diff(5)
            features["$return_10d"] = log_close.diff(10)
            features["$return_20d"] = log_close.diff(20)
        else:
            # Simple Return
            features["$return_1d"] = close.pct_change(1)
            features["$return_5d"] = close.pct_change(5)
            features["$return_10d"] = close.pct_change(10)
            features["$return_20d"] = close.pct_change(20)

        # 高低价比率
        features["$high_low_ratio"] = df["high"] / df["low"]
        features["$close_open_ratio"] = close / df["open"]

        # 均线特征
        for w in self.window_sizes:
            # 简单移动平均
            sma = close.rolling(window=w).mean()
            features[f"$sma_{w}"] = sma

            # 收盘价相对于均线的位置
            features[f"$close_sma_{w}_ratio"] = close / sma

            # 均线斜率 (动量)
            features[f"$sma_{w}_slope"] = sma.diff(5) / sma

        # 波动率特征
        for w in self.window_sizes:
            # 收益率标准差
            returns = close.pct_change()
            features[f"$volatility_{w}"] = returns.rolling(window=w).std()

            # ATR (真实波动范围)
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - close.shift(1))
            low_close = abs(df["low"] - close.shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f"$atr_{w}"] = tr.rolling(window=w).mean()

        # 成交量特征
        volume = df["volume"]
        for w in self.window_sizes:
            # 成交量均值
            vol_sma = volume.rolling(window=w).mean()
            features[f"$volume_sma_{w}"] = vol_sma

            # 相对成交量
            features[f"$rel_volume_{w}"] = volume / vol_sma

        # VWAP (成交量加权平均价)
        typical_price = (df["high"] + df["low"] + close) / 3
        features["$vwap"] = (typical_price * volume).cumsum() / volume.cumsum()

        # RSI (相对强弱指标)
        for w in [6, 12, 24]:
            features[f"$rsi_{w}"] = self._calculate_rsi(close, w)

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        features["$macd"] = macd
        features["$macd_signal"] = signal
        features["$macd_hist"] = macd - signal

        # Bollinger Bands
        for w in [20]:
            sma = close.rolling(window=w).mean()
            std = close.rolling(window=w).std()
            features[f"$bb_upper_{w}"] = sma + 2 * std
            features[f"$bb_lower_{w}"] = sma - 2 * std
            features[f"$bb_width_{w}"] = (2 * std) / sma
            features[f"$bb_position_{w}"] = (close - sma) / (2 * std)

        # SMC 因子
        smc_df = calculate_smc_features(df)
        for col in smc_df.columns:
            features[col] = smc_df[col]

        # 价格行为因子
        pa_df = calculate_price_action_features(df)
        for col in pa_df.columns:
            features[col] = pa_df[col]

        # 构建 DataFrame
        result = pd.DataFrame(features, index=df.index)

        # 处理缺失值
        result = self._handle_missing(result)

        # TODO: 可以添加其他的时序处理

        log_msg = f"生成了 {len(result.columns)} 个特征"
        if symbol:
            log_msg = f"[{symbol}] {log_msg}"
        logger.debug(log_msg)
        
        return result

    def _generate_multi_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """为多个股票生成特征"""
        results = []

        # 获取所有股票
        symbols = df.index.get_level_values("symbol").unique()
        logger.info(f"开始为 {len(symbols)} 只股票生成特征...")

        # 1. 先生成基础特征
        for symbol in tqdm(symbols, desc="Feature Gen", unit="stock"):
            try:
                # 提取单个股票数据
                symbol_df = df.xs(symbol, level="symbol")
                features = self._generate_single_symbol(symbol_df, symbol=symbol)
                features["symbol"] = symbol
                results.append(features.reset_index())
            except Exception as e:
                logger.error(f"生成 {symbol} 特征失败: {e}")

        if not results:
            return pd.DataFrame()

        # 合并结果
        result = pd.concat(results, ignore_index=True)
        result = result.set_index(["timestamp", "symbol"])
        result = result.sort_index()

        # 2. 截面排名 (Cross-Sectional Rank)
        if self.use_cross_sectional_rank:
            logger.info("执行截面排名 (Cross-Sectional Rank)...")
            # 对每一列特征，按时间分组进行排名映射到 [0, 1]
            # 注意：不应该对 symbol 列排名，也不应该处理非数值列
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            
            # 使用 groupby().rank(pct=True)
            ranked = result[numeric_cols].groupby("timestamp").rank(pct=True, method="first")
            
            # 替换原始值
            result[numeric_cols] = ranked
            
            # 填充可能的 NaN (如果某天只有一只股票，rank 可能是 NaN 或 1.0)
            result = result.fillna(0.5)

        return result

    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """计算 RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        if self.fill_method == "ffill":
            df = df.ffill()
        elif self.fill_method == "bfill":
            df = df.bfill()
        elif self.fill_method == "zero":
            df = df.fillna(0)
        elif self.fill_method == "mean":
            df = df.fillna(df.mean())

        # 再次确保所有 NaN 被填充
        df = df.fillna(0)

        # 替换无穷大/小值
        df = df.replace([np.inf, -np.inf], 0)
        
        # 再次裁减极值，防止溢出导致后续计算异常
        df = df.clip(lower=-1e8, upper=1e8)

        return df


    def get_feature_names(self) -> list[str]:
        """获取所有特征名称（用于模型训练）"""
        # 生成一个示例数据来获取特征名
        sample_data = pd.DataFrame(
            {
                "open": [1.0] * 100,
                "high": [1.1] * 100,
                "low": [0.9] * 100,
                "close": [1.0] * 100,
                "volume": [1000] * 100,
            },
            index=pd.date_range("2020-01-01", periods=100),
        )
        features = self._generate_single_symbol(sample_data)
        return list(features.columns)


class FeaturePreprocessor:
    """
    特征预处理器

    负责：
    1. 缺失值处理
    2. 异常值处理
    3. 特征标准化
    4. 特征选择
    """

    def __init__(
        self,
        clip_outliers: bool = True,
        outlier_std: float = 3.0,
        normalize_method: str = "rank",
    ):
        """
        初始化预处理器

        Args:
            clip_outliers: 是否截断异常值
            outlier_std: 异常值判定标准（几倍标准差）
            normalize_method: 标准化方法 ('minmax', 'rank')
        """
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        self.normalize_method = normalize_method

        # 存储拟合参数
        self._fit_params: Optional[dict] = None

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """拟合预处理参数"""
        self._fit_params = {}

        for col in df.columns:
            self._fit_params[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
            }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用预处理"""
        result = df.copy()

        for col in result.columns:
            if self._fit_params and col in self._fit_params:
                params = self._fit_params[col]
            else:
                params = {
                    "mean": result[col].mean(),
                    "std": result[col].std(),
                    "min": result[col].min(),
                    "max": result[col].max(),
                }

            # 截断异常值
            if self.clip_outliers and params["std"] > 0:
                lower = params["mean"] - self.outlier_std * params["std"]
                upper = params["mean"] + self.outlier_std * params["std"]
                result[col] = result[col].clip(lower, upper)

            # 标准化
            if self.normalize_method == "minmax":
                range_val = params["max"] - params["min"]
                if range_val > 0:
                    result[col] = (result[col] - params["min"]) / range_val
            elif self.normalize_method == "rank":
                result[col] = result[col].rank(pct=True)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """拟合并转换"""
        return self.fit(df).transform(df)
