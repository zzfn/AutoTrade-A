"""
数据提供者模块 - A 股数据获取

AutoTrade-A 专用：仅支持 A 股 (AKShare) 数据源
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import akshare as ak
import pandas as pd
from loguru import logger

from autotrade.research.data.cache import ParquetCache


class BaseDataProvider(ABC):
    """数据提供者基类"""

    @abstractmethod
    def fetch_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        获取历史数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 频率 ('1d')

        Returns:
            包含 OHLCV 数据的 DataFrame，MultiIndex: (datetime, symbol)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查数据源是否可用"""
        pass


class AKShareDataProvider(BaseDataProvider):
    """
    AKShare 数据提供者 - A 股市场

    支持功能：
    - 日线 OHLCV 数据获取
    - 前复权处理
    - 停牌/涨跌停状态识别
    - ST 股票识别
    """

    # A 股代码格式正则: 6位数字 + .SZ 或 .SH
    SYMBOL_PATTERN = re.compile(r"^(\d{6})\.(SZ|SH)$")

    # 涨跌停幅度限制
    LIMIT_PCTS = {
        "main": 0.10,  # 主板/中小板 ±10%
        "gem": 0.20,   # 创业板(300xxx) ±20%
        "star": 0.20,  # 科创板(688xxx) ±20%
    }

    def __init__(self, adjust: str = "qfq", cache_dir: str = "data/cache/akshare"):
        """
        初始化 AKShare 数据提供者

        Args:
            adjust: 复权类型 - "qfq" 前复权, "hfq" 后复权, "" 不复权
            cache_dir: 缓存目录路径
        """
        self.adjust = adjust
        self.cache_manager = ParquetCache(cache_dir)
        self._stock_info_cache: Optional[pd.DataFrame] = None

    def is_available(self) -> bool:
        """检查 AKShare 是否可用（始终可用，无需 API 密钥）"""
        return True

    def _validate_symbol(self, symbol: str) -> tuple[str, str]:
        """
        验证 A 股代码格式

        Args:
            symbol: 股票代码，如 "000001.SZ"

        Returns:
            (股票代码不带后缀, 市场标识)

        Raises:
            ValueError: 格式错误
        """
        match = self.SYMBOL_PATTERN.match(symbol.upper())
        if not match:
            if len(symbol) == 6 and symbol.isdigit():
                # 自动推断市场
                market = "SH" if symbol.startswith(("6", "9")) else "SZ"
                raise ValueError(
                    f"缺少市场后缀，应为 {symbol}.{market}（上海）或 {symbol}.SZ（深圳）"
                )
            raise ValueError(
                f"A 股代码格式错误: {symbol}。正确格式: 6位数字.SZ/SH，如 000001.SZ"
            )
        return match.group(1), match.group(2)

    def _get_board_type(self, symbol: str) -> str:
        """
        获取股票板块类型

        Args:
            symbol: 股票代码

        Returns:
            板块类型: "main", "gem", "star"
        """
        code, _ = self._validate_symbol(symbol)
        if code.startswith("300"):
            return "gem"  # 创业板
        elif code.startswith("688"):
            return "star"  # 科创板
        else:
            return "main"  # 主板/中小板

    def fetch_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        force_update: bool = False,
    ) -> pd.DataFrame:
        """
        获取 A 股历史数据（带持久化缓存）

        Args:
            symbols: A 股代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: "1d"
            force_update: 是否强制从网络获取

        Returns:
            DataFrame with MultiIndex (timestamp, symbol)
        """
        if interval != "1d":
            logger.warning(f"A 股数据仅支持日线，忽略 interval={interval}")

        all_data = []

        for symbol in symbols:
            try:
                # 1. 检查缓存
                cached_df = None
                if not force_update:
                    cached_df = self.cache_manager.load(symbol)

                # 2. 判断是否需要网络获取
                needs_fetch = True
                if cached_df is not None and not cached_df.empty:
                    # 确保 timestamp 为 datetime 类型并移除时区信息
                    cached_df["timestamp"] = pd.to_datetime(cached_df["timestamp"]).dt.tz_localize(None)
                    
                    # 检查缓存是否覆盖了请求的范围
                    cache_min = cached_df["timestamp"].min()
                    cache_max = cached_df["timestamp"].max()
                    
                    # 比较请求范围
                    req_start = pd.to_datetime(start_date).tz_localize(None)
                    req_end = pd.to_datetime(end_date).tz_localize(None)

                    if cache_min <= req_start and cache_max >= req_end:
                        now = datetime.now()
                        is_today = req_end.date() == now.date()
                        if not is_today or now.hour >= 16:
                            needs_fetch = False
                            logger.debug(f"{symbol} 缓存命中: {start_date.date()} - {end_date.date()}")
                        else:
                            logger.debug(f"{symbol} 包含今日且非收盘后，刷新今日数据")
                    else:
                        logger.debug(f"{symbol} 缓存范围不全: {cache_min.date()}~{cache_max.date()}")

                # 3. 执行网络获取
                if needs_fetch:
                    code, _ = self._validate_symbol(symbol)
                    start_str = start_date.strftime("%Y%m%d")
                    end_str = end_date.strftime("%Y%m%d")
                    
                    df = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_str,
                        end_date=end_str,
                        adjust=self.adjust,
                    )

                    if not df.empty:
                        df = df.rename(
                            columns={
                                "日期": "timestamp",
                                "开盘": "open",
                                "最高": "high",
                                "最低": "low",
                                "收盘": "close",
                                "成交量": "volume",
                            }
                        )
                        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
                        
                        # 保存到持久化缓存
                        self.cache_manager.save(symbol, df)
                        
                        # 重新加载完整数据
                        cached_df = self.cache_manager.load(symbol)
                        if cached_df is not None:
                            cached_df["timestamp"] = pd.to_datetime(cached_df["timestamp"]).dt.tz_localize(None)
                    else:
                        logger.warning(f"{symbol} AKShare 返回空数据")

                # 4. 截取范围内的数据
                if cached_df is not None and not cached_df.empty:
                    req_start = pd.to_datetime(start_date).tz_localize(None)
                    req_end = pd.to_datetime(end_date).tz_localize(None)
                    
                    mask = (cached_df["timestamp"] >= req_start) & (cached_df["timestamp"] <= req_end)
                    symbol_df = cached_df[mask].copy()
                    if not symbol_df.empty:
                        symbol_df["symbol"] = symbol.upper()
                        all_data.append(symbol_df)

            except Exception as e:
                logger.error(f"处理 {symbol} 数据失败: {e}")

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.set_index(["timestamp", "symbol"])
        result = result.sort_index()

        return result

    def get_stock_names(self, symbols: list[str]) -> dict[str, str]:
        """
        获取股票名称映射

        Args:
            symbols: 股票代码列表 (带后缀，如 000001.SZ)

        Returns:
            {symbol: name} 映射字典
        """
        try:
            if self._stock_info_cache is None:
                self._stock_info_cache = ak.stock_zh_a_spot_em()

            mapping = {}
            for symbol in symbols:
                try:
                    code, _ = self._validate_symbol(symbol)
                    stock_info = self._stock_info_cache[
                        self._stock_info_cache["代码"] == code
                    ]
                    if not stock_info.empty:
                        mapping[symbol] = stock_info.iloc[0]["名称"]
                    else:
                        mapping[symbol] = symbol
                except Exception:
                    mapping[symbol] = symbol
            return mapping
        except Exception as e:
            logger.warning(f"获取股票名称失败: {e}")
            return {s: s for s in symbols}

    def is_st_stock(self, symbol: str) -> bool:
        """
        检查是否 ST 股票

        Args:
            symbol: 股票代码

        Returns:
            True 如果是 ST 股票
        """
        try:
            code, _ = self._validate_symbol(symbol)

            # 获取股票名称
            if self._stock_info_cache is None:
                self._stock_info_cache = ak.stock_zh_a_spot_em()

            stock_info = self._stock_info_cache[
                self._stock_info_cache["代码"] == code
            ]

            if stock_info.empty:
                return False

            name = stock_info.iloc[0]["名称"]
            return "ST" in name or "*ST" in name

        except Exception as e:
            logger.warning(f"检查 ST 状态失败 {symbol}: {e}")
            return False

    def is_suspended(self, symbol: str, date: datetime) -> bool:
        """
        检查是否停牌

        Args:
            symbol: 股票代码
            date: 日期

        Returns:
            True 如果停牌（当天无成交数据或成交量为0）
        """
        try:
            # 使用 fetch_data 自动利用缓存
            df = self.fetch_data([symbol], date, date)

            if df.empty:
                return True  # 无数据视为停牌

            # index 是 (timestamp, symbol)，列有 volume
            volume = df.iloc[0].get("volume", 0)
            return volume == 0

        except Exception as e:
            logger.warning(f"检查停牌状态失败 {symbol}: {e}")
            return False

    def is_limit_up(self, symbol: str, date: datetime) -> bool:
        """
        检查是否涨停

        涨停条件：收盘价 ≈ 最高价 且 涨幅接近限制

        Args:
            symbol: 股票代码
            date: 日期

        Returns:
            True 如果涨停
        """
        return self._check_limit(symbol, date, is_up=True)

    def is_limit_down(self, symbol: str, date: datetime) -> bool:
        """
        检查是否跌停

        跌停条件：收盘价 ≈ 最低价 且 跌幅接近限制

        Args:
            symbol: 股票代码
            date: 日期

        Returns:
            True 如果跌停
        """
        return self._check_limit(symbol, date, is_up=False)

    def _check_limit(self, symbol: str, date: datetime, is_up: bool) -> bool:
        """
        检查涨跌停状态

        Args:
            symbol: 股票代码
            date: 日期
            is_up: True 检查涨停，False 检查跌停

        Returns:
            True 如果达到涨跌停
        """
        try:
            board = self._get_board_type(symbol)
            limit_pct = self.LIMIT_PCTS[board]

            # 获取近期数据以计算涨跌幅
            # 为了确保有前一交易日数据，我们向前多取几天
            df = self.fetch_data(
                [symbol], 
                date - timedelta(days=10), 
                date
            )

            if len(df) < 2:
                return False

            # 重置索引以便操作
            df = df.reset_index()
            df = df.sort_values("timestamp")
            
            # 找到 date 对应的位置
            # 使用 .dt.date 进行比较以避免时区/具体时间影响
            target_date = date.date()
            df_dates = df["timestamp"].dt.date
            today_mask = (df_dates == target_date)
            
            if not today_mask.any():
                return False
            
            idx = df[today_mask].index[0]
            # 获取在排序后 df 中的位置
            loc = df.index.get_loc(idx)
            
            if loc == 0:
                return False

            prev_close = df.iloc[loc - 1]["close"]
            today_data = df.iloc[loc]

            close = today_data["close"]
            high = today_data["high"]
            low = today_data["low"]

            # 计算涨跌幅
            change_pct = (close - prev_close) / prev_close

            # 涨停：涨幅接近限制 且 收盘=最高
            if is_up:
                return (
                    change_pct >= limit_pct * 0.98  # 接近涨停幅度
                    and abs(close - high) < 0.01  # 收盘价接近最高价
                )
            # 跌停：跌幅接近限制 且 收盘=最低
            else:
                return (
                    change_pct <= -limit_pct * 0.98  # 接近跌停幅度
                    and abs(close - low) < 0.01  # 收盘价接近最低价
                )

        except Exception as e:
            logger.warning(f"检查涨跌停状态失败 {symbol}: {e}")
            return False
    def get_index_constituents(self, index_code: str = "000300") -> list[str]:
        """
        获取指数成分股

        Args:
            index_code: 指数代码，默认 "000300" (沪深300)

        Returns:
            成分股代码列表 (带后缀)
        """
        try:
            # 移除可能的后缀
            if "." in index_code:
                index_code = index_code.split(".")[0]
            
            # CSI 300, 500, etc.
            # ak.index_stock_cons 接受 6 位代码
            df = ak.index_stock_cons(symbol=index_code)
            
            if df.empty:
                logger.warning(f"获取指数 {index_code} 成分股为空")
                return []
                
            symbols = []
            for code in df["品种代码"]:
                code = str(code).zfill(6)
                # 简单推断后缀
                if code.startswith(("6", "9")):
                    suffix = "SH"
                elif code.startswith(("0", "3")):
                    suffix = "SZ"
                elif code.startswith(("4", "8")):
                    suffix = "BJ" # 北交所
                else:
                    suffix = "SZ" # 默认 SZ
                
                symbols.append(f"{code}.{suffix}")
                
            return symbols
            
        except Exception as e:
            logger.error(f"获取指数 {index_code} 成分股失败: {e}")
            return []

class DataProviderFactory:
    """
    数据提供者工厂 - A 股专用

    AutoTrade-A 仅支持 A 股市场，默认返回 AKShareDataProvider
    """
    _providers = {}

    @staticmethod
    def get_provider(market: str = "cn") -> BaseDataProvider:
        """
        获取 A 股数据提供者

        Args:
            market: 市场代码（仅支持 "cn"）

        Returns:
            AKShareDataProvider 实例

        Raises:
            ValueError: 不支持的市场
        """
        market = market.lower()

        if market not in DataProviderFactory._providers:
            if market == "cn":
                logger.debug("初始化 AKShare 数据源")
                DataProviderFactory._providers[market] = AKShareDataProvider()
            else:
                raise ValueError(
                    f"AutoTrade-A 仅支持 A 股市场 (cn)，不支持: {market}"
                )

        return DataProviderFactory._providers[market]
