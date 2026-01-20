import talib
import pandas as pd
import numpy as np

def calculate_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 TA-Lib 计算价格行为 (Candlestick Patterns) 因子
    """
    res = pd.DataFrame(index=df.index)
    
    # 常用形态识别
    # 返回值: 100 代表看涨, -100 代表看跌, 0 代表无形态
    
    # 1. 锤头/倒锤头 (Hammer / Inverted Hammer)
    res["$hammer"] = talib.CDLHAMMER(df["open"], df["high"], df["low"], df["close"])
    res["$inverted_hammer"] = talib.CDLINVERTEDHAMMER(df["open"], df["high"], df["low"], df["close"])
    
    # 2. 吞没形态 (Engulfing Pattern)
    res["$engulfing"] = talib.CDLENGULFING(df["open"], df["high"], df["low"], df["close"])
    
    # 3. 十字星 (Doji)
    res["$doji"] = talib.CDLDOJI(df["open"], df["high"], df["low"], df["close"])
    
    # 4. 晨星/暮星 (Morning Star / Evening Star)
    res["$morning_star"] = talib.CDLMORNINGSTAR(df["open"], df["high"], df["low"], df["close"])
    res["$evening_star"] = talib.CDLEVENINGSTAR(df["open"], df["high"], df["low"], df["close"])
    
    # 5. 射击之星 (Shooting Star)
    res["$shooting_star"] = talib.CDLSHOOTINGSTAR(df["open"], df["high"], df["low"], df["close"])
    
    # 6. 吊颈线 (Hanging Man)
    res["$hanging_man"] = talib.CDLHANGINGMAN(df["open"], df["high"], df["low"], df["close"])
    
    # 7. 乌云盖顶 / 刺透形态 (Dark Cloud Cover / Piercing Line)
    res["$dark_cloud_cover"] = talib.CDLDARKCLOUDCOVER(df["open"], df["high"], df["low"], df["close"])
    res["$piercing"] = talib.CDLPIERCING(df["open"], df["high"], df["low"], df["close"])

    # 归一化处理到 [-1, 1]
    for col in res.columns:
        res[col] = res[col] / 100.0
        
    return res.fillna(0)
