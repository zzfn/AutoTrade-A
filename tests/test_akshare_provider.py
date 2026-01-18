"""
A股数据提供者测试

测试 AKShareDataProvider 的核心功能
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd

from autotrade.research.data.providers import (
    AKShareDataProvider,
    DataProviderFactory,
)


class TestAKShareDataProvider:
    """AKShareDataProvider 单元测试"""

    def setup_method(self):
        """每个测试前初始化"""
        self.provider = AKShareDataProvider()

    # ========================
    # 代码格式验证测试
    # ========================

    def test_validate_symbol_valid_sz(self):
        """测试有效的深圳代码"""
        code, market = self.provider._validate_symbol("000001.SZ")
        assert code == "000001"
        assert market == "SZ"

    def test_validate_symbol_valid_sh(self):
        """测试有效的上海代码"""
        code, market = self.provider._validate_symbol("600000.SH")
        assert code == "600000"
        assert market == "SH"

    def test_validate_symbol_lowercase(self):
        """测试小写代码"""
        code, market = self.provider._validate_symbol("000001.sz")
        assert code == "000001"
        assert market == "SZ"

    def test_validate_symbol_missing_suffix(self):
        """测试缺少后缀应报错"""
        with pytest.raises(ValueError) as exc_info:
            self.provider._validate_symbol("000001")
        assert "缺少市场后缀" in str(exc_info.value)

    def test_validate_symbol_invalid_format(self):
        """测试无效格式应报错"""
        with pytest.raises(ValueError) as exc_info:
            self.provider._validate_symbol("AAPL")
        assert "格式错误" in str(exc_info.value)

    def test_validate_symbol_wrong_length(self):
        """测试错误长度应报错"""
        with pytest.raises(ValueError) as exc_info:
            self.provider._validate_symbol("00001.SZ")
        assert "格式错误" in str(exc_info.value)

    # ========================
    # 板块类型测试
    # ========================

    def test_get_board_type_main(self):
        """测试主板识别"""
        assert self.provider._get_board_type("600000.SH") == "main"
        assert self.provider._get_board_type("000001.SZ") == "main"

    def test_get_board_type_gem(self):
        """测试创业板识别"""
        assert self.provider._get_board_type("300750.SZ") == "gem"

    def test_get_board_type_star(self):
        """测试科创板识别"""
        assert self.provider._get_board_type("688981.SH") == "star"

    # ========================
    # 可用性测试
    # ========================

    def test_is_available(self):
        """测试可用性（AKShare 无需 API 密钥）"""
        assert self.provider.is_available() is True


class TestDataProviderFactory:
    """DataProviderFactory 单元测试"""

    def test_get_provider_cn(self):
        """测试获取 A 股提供者"""
        provider = DataProviderFactory.get_provider("cn")
        assert isinstance(provider, AKShareDataProvider)

    def test_get_provider_cn_uppercase(self):
        """测试大写市场代码"""
        provider = DataProviderFactory.get_provider("CN")
        assert isinstance(provider, AKShareDataProvider)

    def test_get_provider_invalid_market(self):
        """测试无效市场应报错"""
        with pytest.raises(ValueError) as exc_info:
            DataProviderFactory.get_provider("jp")
        assert "不支持" in str(exc_info.value)

    def test_get_provider_us_not_supported(self):
        """测试 US 市场不再支持"""
        with pytest.raises(ValueError) as exc_info:
            DataProviderFactory.get_provider("us")
        assert "不支持" in str(exc_info.value)


class TestQlibMLStrategyRules:
    """A股交易规则测试"""

    def test_round_to_lots(self):
        """测试100股取整"""
        from autotrade.execution.strategies import QlibMLStrategy

        strategy = QlibMLStrategy.__new__(QlibMLStrategy)
        strategy.CN_LOT_SIZE = 100

        assert strategy._round_to_lots(150) == 100
        assert strategy._round_to_lots(99) == 0
        assert strategy._round_to_lots(300) == 300
        assert strategy._round_to_lots(250) == 200
        assert strategy._round_to_lots(0) == 0


# ========================
# 集成测试（需要网络）
# ========================

@pytest.mark.integration
class TestAKShareIntegration:
    """AKShare 集成测试（需要网络连接）"""

    def test_fetch_single_stock(self):
        """测试获取单只股票数据"""
        provider = AKShareDataProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        df = provider.fetch_data(
            symbols=["000001.SZ"],
            start_date=start_date,
            end_date=end_date,
        )

        assert not df.empty
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert isinstance(df.index, pd.MultiIndex)

    def test_fetch_multiple_stocks(self):
        """测试获取多只股票数据"""
        provider = AKShareDataProvider()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)

        df = provider.fetch_data(
            symbols=["000001.SZ", "600000.SH"],
            start_date=start_date,
            end_date=end_date,
        )

        assert not df.empty
        symbols_in_data = df.index.get_level_values("symbol").unique()
        assert "000001.SZ" in symbols_in_data
        assert "600000.SH" in symbols_in_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
