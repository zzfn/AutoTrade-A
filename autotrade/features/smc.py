import numpy as np
import pandas as pd
from loguru import logger

def calculate_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Smart Money Concepts (SMC) 相关因子
    包含: FVG (Fair Value Gap), Swing Highs/Lows, BOS/CHoCH (简化版), Order Blocks (简化版)
    """
    res = pd.DataFrame(index=df.index)
    
    # 1. Fair Value Gap (FVG)
    # Bullish FVG: Low(i) > High(i-2)
    # Bearish FVG: High(i) < Low(i-2)
    res["$fvg_bullish"] = (df["low"] > df["high"].shift(2)).astype(float)
    res["$fvg_bearish"] = (df["high"] < df["low"].shift(2)).astype(float)
    
    # FVG Gap Size (归一化)
    res["$fvg_gap_size"] = np.where(
        res["$fvg_bullish"] > 0, 
        (df["low"] - df["high"].shift(2)) / df["close"],
        np.where(
            res["$fvg_bearish"] > 0,
            (df["low"].shift(2) - df["high"]) / df["close"],
            0
        )
    )

    # 2. Swing Highs / Lows (寻找局部极值)
    window = 5
    # 为了避免未来函数 (Lookahead bias), 我们认定在 N 天后的高点低于它时，才认定它是 Swing High
    # 这里使用简单的滞后确认
    res["$is_swing_high"] = (
        (df["high"].shift(window) == df["high"].rolling(window=window*2+1).max())
    ).astype(float)
    res["$is_swing_low"] = (
        (df["low"].shift(window) == df["low"].rolling(window=window*2+1).min())
    ).astype(float)

    # 3. Market Structure (简化版: 价格是否刷新了最近确认的 Swing 点)
    # 记录上一个确认的 swing high/low
    last_confirmed_high = df["high"].where(res["$is_swing_high"] > 0).ffill()
    last_confirmed_low = df["low"].where(res["$is_swing_low"] > 0).ffill()
    
    # Break of Structure (BOS) - 顺势突破最近确认的高/低点
    res["$bos_bullish"] = (df["close"] > last_confirmed_high.shift(1)).astype(float)
    res["$bos_bearish"] = (df["close"] < last_confirmed_low.shift(1)).astype(float)

    # 4. Order Blocks (简化版: BOS 发生前的反向 K 线区域)
    # Bullish OB: 产生 BOS Bullish 之前的最近一个阴线
    is_down_candle = df["close"] < df["open"]
    is_up_candle = df["close"] > df["open"]
    
    res["$potential_bullish_ob"] = ((res["$bos_bullish"] > 0) & is_down_candle.shift(1)).astype(float)
    res["$potential_bearish_ob"] = ((res["$bos_bearish"] > 0) & is_up_candle.shift(1)).astype(float)
    
    return res.fillna(0)
