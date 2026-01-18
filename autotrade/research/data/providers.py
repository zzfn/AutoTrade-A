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

    def __init__(self, adjust: str = "qfq"):
        """
        初始化 AKShare 数据提供者

        Args:
            adjust: 复权类型 - "qfq" 前复权, "hfq" 后复权, "" 不复权
        """
        self.adjust = adjust
        self._cache: dict[str, pd.DataFrame] = {}
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
    ) -> pd.DataFrame:
        """
        获取 A 股历史数据

        Args:
            symbols: A 股代码列表，如 ["000001.SZ", "600000.SH"]
            start_date: 开始日期
            end_date: 结束日期
            interval: "1d"（仅支持日线）

        Returns:
            DataFrame with MultiIndex (timestamp, symbol)
            Columns: open, high, low, close, volume
        """
        if interval != "1d":
            logger.warning(f"A 股数据仅支持日线，忽略 interval={interval}")

        logger.info(f"从 AKShare 获取 A 股数据: {symbols}, {start_date} - {end_date}")

        all_data = []
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        for symbol in symbols:
            try:
                # 验证代码格式
                code, _ = self._validate_symbol(symbol)

                # 调用 AKShare 获取数据
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_str,
                    end_date=end_str,
                    adjust=self.adjust,
                )

                if df.empty:
                    logger.warning(f"{symbol} 无数据")
                    continue

                # 转换列名为标准格式
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

                # 只保留需要的列
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                df["symbol"] = symbol.upper()
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                all_data.append(df)
                logger.debug(f"获取 {symbol} 数据: {len(df)} 条记录")

            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")

        if not all_data:
            logger.warning("AKShare 返回空数据")
            return pd.DataFrame()

        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)
        result = result.set_index(["timestamp", "symbol"])
        result = result.sort_index()

        logger.info(f"从 AKShare 获取了 {len(result)} 条记录")
        return result

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
            code, _ = self._validate_symbol(symbol)
            date_str = date.strftime("%Y%m%d")

            # 获取当天数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=date_str,
                end_date=date_str,
                adjust=self.adjust,
            )

            if df.empty:
                return True  # 无数据视为停牌

            volume = df.iloc[0].get("成交量", 0)
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
            code, _ = self._validate_symbol(symbol)
            board = self._get_board_type(symbol)
            limit_pct = self.LIMIT_PCTS[board]

            date_str = date.strftime("%Y%m%d")
            prev_date = (date - timedelta(days=10)).strftime("%Y%m%d")

            # 获取近期数据以计算涨跌幅
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=prev_date,
                end_date=date_str,
                adjust=self.adjust,
            )

            if len(df) < 2:
                return False

            # 获取当日和前一日数据
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期")
            today = df[df["日期"] == pd.to_datetime(date)]
            if today.empty:
                return False

            idx = df[df["日期"] == pd.to_datetime(date)].index[0]
            if idx == 0:
                return False

            prev_close = df.loc[idx - 1, "收盘"]
            today_data = df.loc[idx]

            close = today_data["收盘"]
            high = today_data["最高"]
            low = today_data["最低"]

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


class DataProviderFactory:
    """
    数据提供者工厂 - A 股专用

    AutoTrade-A 仅支持 A 股市场，默认返回 AKShareDataProvider
    """

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

        if market == "cn":
            logger.info("使用 AKShare 作为 A 股数据源")
            return AKShareDataProvider()

        else:
            raise ValueError(
                f"AutoTrade-A 仅支持 A 股市场 (cn)，不支持: {market}"
            )
