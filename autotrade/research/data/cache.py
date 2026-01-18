import os
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class ParquetCache:
    """
    Parquet 格式的本地持久化缓存
    """

    def __init__(self, cache_dir: str = "data/cache/akshare"):
        """
        初始化缓存

        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"缓存存储路径: {self.cache_dir.absolute()}")

    def _get_path(self, symbol: str) -> Path:
        """获取股票对应的缓存文件路径"""
        return self.cache_dir / f"{symbol}.parquet"

    def load(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        加载缓存数据

        Args:
            symbol: 股票代码

        Returns:
            DataFrame 或 None (如果不存在)
        """
        path = self._get_path(symbol)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            # 确保 timestamp 是索引或列
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.warning(f"加载缓存失败 {symbol}: {e}")
            return None

    def save(self, symbol: str, df: pd.DataFrame):
        """
        保存数据到缓存，如果已存在则合并

        Args:
            df: 要保存的数据 (包含 timestamp 列)
        """
        if df.empty:
            return

        path = self._get_path(symbol)
        
        # 确保 timestamp 为 datetime 类型以便后续合并
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        if path.exists():
            try:
                existing_df = pd.read_parquet(path)
                if "timestamp" in existing_df.columns:
                    existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
                
                # 合并并去重
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"], keep="last")
                df = df.sort_values("timestamp")
            except Exception as e:
                logger.warning(f"合并缓存失败 {symbol}: {e}，将直接覆盖")

        try:
            df.to_parquet(path, index=False)
            logger.debug(f"缓存已保存: {symbol}, 共 {len(df)} 行")
        except Exception as e:
            logger.error(f"保存缓存失败 {symbol}: {e}")

    def clear(self, symbol: Optional[str] = None):
        """清理缓存"""
        if symbol:
            path = self._get_path(symbol)
            if path.exists():
                path.unlink()
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
