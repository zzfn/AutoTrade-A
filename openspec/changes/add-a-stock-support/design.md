# Design: 添加A股支持

## Architecture Overview

### 当前架构（美股）
```
┌─────────────────────────────────────────────────────┐
│                  Web Interface                      │
│              (Strategy Selection, Config)            │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              TradeManager                           │
│         (Strategy Execution Orchestrator)           │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│         QlibMLStrategy (LumiBot Strategy)           │
│      (Backtesting / Signal Generation)              │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│          DataProvider (Alpaca Only)                 │
│              ↓                                       │
│          QlibDataAdapter                            │
│              ↓                                       │
│          Qlib Format Data                           │
└─────────────────────────────────────────────────────┘
```

### 目标架构（美股 + A股）
```
┌─────────────────────────────────────────────────────┐
│          Web Interface (Market Selector)            │
│       US Stocks (Alpaca) | A Stocks (AKShare)       │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              TradeManager                           │
│         (market parameter: "us" | "cn")             │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│         QlibMLStrategy (Market-Aware)               │
│   - US: No trading restrictions                     │
│   - CN: Min 100 shares, Price limits                │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│       DataProviderFactory                           │
│     AlpacaDataProvider | AKShareDataProvider        │
│              ↓                                       │
│       QlibDataAdapter (Multi-Market)                │
│              ↓                                       │
│       Qlib Format Data (market tagged)              │
└─────────────────────────────────────────────────────┘
```

## Component Design

### 1. DataProvider Extension

#### AKShareDataProvider

**位置**: `autotrade/research/data/providers.py`

**核心职责**:
- 从AKShare获取A股日线数据
- 前复权处理
- 数据格式标准化

**接口设计**:
```python
class AKShareDataProvider(BaseDataProvider):
    """
    AKShare数据提供者 - A股市场

    支持功能：
    - 日线OHLCV数据获取
    - 前复权处理
    - 停牌/涨跌停状态识别
    """

    def __init__(self, adjust: str = "qfq"):  # qfq = 前复权
        self.adjust = adjust
        self._cache = {}

    def fetch_data(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",  # 仅支持日线
    ) -> pd.DataFrame:
        """
        获取A股历史数据

        Args:
            symbols: A股代码列表，如 ["000001.SZ", "600000.SH"]
            start_date: 开始日期
            end_date: 结束日期
            interval: "1d"（仅支持日线）

        Returns:
            DataFrame with MultiIndex (timestamp, symbol)
            Columns: open, high, low, close, volume
        """
        # 1. 验证symbol格式（.SZ 或 .SH 后缀）
        # 2. 调用ak.share.stock_zh_a_hist()
        # 3. 前复权处理
        # 4. 统一格式：(timestamp, symbol) MultiIndex
        pass

    def is_suspended(self, symbol: str, date: datetime) -> bool:
        """检查是否停牌"""
        # 停牌特征：当天无成交数据
        pass

    def is_limit_up(self, symbol: str, date: datetime) -> bool:
        """检查是否涨停"""
        # 涨停特征：close ≈ high 且 price increase ≈ 10%/20%
        pass

    def is_limit_down(self, symbol: str, date: datetime) -> bool:
        """检查是否跌停"""
        # 跌停特征：close ≈ low 且 price decrease ≈ 10%/20%
        pass

    def is_st_stock(self, symbol: str) -> bool:
        """检查是否ST股票"""
        # 通过symbol或名称判断
        pass
```

**数据格式示例**:
```python
# AKShare原始数据
     日期          开盘    最高    最低    收盘      成交量
2024-01-01    10.50   10.80   10.45   10.75   1000000

# 转换后（与Alpaca一致）
                        open    high     low   close  volume
timestamp           symbol
2024-01-01 000001.SZ  10.50   10.80   10.45   10.75  1000000
```

#### DataProviderFactory 扩展

**修改位置**: `autotrade/research/data/providers.py`

```python
class DataProviderFactory:
    """
    数据提供者工厂 - 支持多市场

    市场映射：
    - "us": AlpacaDataProvider
    - "cn": AKShareDataProvider
    """

    @staticmethod
    def get_provider(market: str = "us") -> BaseDataProvider:
        """
        根据市场获取对应数据提供者

        Args:
            market: 市场代码 ("us" 或 "cn")

        Returns:
            对应市场的数据提供者实例
        """
        if market == "cn":
            return AKShareDataProvider()
        elif market == "us":
            alpaca = AlpacaDataProvider()
            if alpaca.is_available():
                return alpaca
            raise RuntimeError("Alpaca API不可用")

        raise ValueError(f"不支持的市场: {market}")
```

### 2. QlibDataAdapter 扩展

**修改位置**: `autotrade/research/data/qlib_adapter.py`

**核心变更**:
```python
class QlibDataAdapter:
    """
    Qlib数据适配器 - 支持多市场

    目录结构：
    data/qlib/
    ├── us/           # 美股数据
    │   ├── stock_data/
    │   ├── calendars/
    │   └── instruments/
    └── cn/           # A股数据
        ├── stock_data/
        ├── calendars/
        └── instruments/
    """

    def __init__(self, market: str = "us"):
        self.market = market
        self.base_dir = Path(f"data/qlib/{market}")

    def fetch_and_store(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
    ):
        """
        获取并存储数据到Qlib格式

        根据 market 参数：
        - 选择对应的数据提供者
        - 存储到对应的目录
        - 生成对应市场的元数据
        """
        provider = DataProviderFactory.get_provider(self.market)
        df = provider.fetch_data(symbols, start_date, end_date)
        self._convert_and_store(df, symbols)

    def _update_instruments(self, symbols: list[str]):
        """
        更新股票列表文件

        A股格式：
        000001.SZ    平安银行        SZ     CSI100
        600000.SH    浦发银行        SH     CSI100
        """
        instruments_dir = self.base_dir / "instruments"
        # ... 实现细节
```

### 3. 策略层适配

**修改位置**: `autotrade/execution/strategies/qlib_ml_strategy.py`

**核心变更**:

```python
class QlibMLStrategy(Strategy):
    """
    Qlib ML 策略 - 支持多市场

    新增参数：
    - market: 市场代码 ("us" 或 "cn")
    """

    parameters = {
        # ... 现有参数
        "market": "us",  # 新增：市场选择
    }

    def _parse_parameters(self):
        # ... 现有逻辑
        self.market = self.parameters.get("market", "us")

    def _rebalance_portfolio(self, target_symbols: list):
        """
        再平衡投资组合 - 市场感知

        A股交易规则：
        - 最小交易单位：100股
        - 价格限制：涨跌停
        """
        if self.market == "cn":
            self._rebalance_portfolio_cn(target_symbols)
        else:
            self._rebalance_portfolio_us(target_symbols)

    def _rebalance_portfolio_cn(self, target_symbols: list):
        """
        A股再平衡逻辑

        特殊处理：
        1. 交易数量必须是100的整数倍
        2. 检查涨跌停状态
        3. 剔除ST股票
        """
        # 1. 过滤ST股票
        target_symbols = [
            s for s in target_symbols
            if not self._is_st_stock(s)
        ]

        # 2. 获取当前持仓
        current_positions = {...}

        # 3. 卖出逻辑
        for symbol in to_sell:
            price = self.get_last_price(symbol)
            if self._is_limit_up(symbol):  # 涨停无法买入
                continue
            # ... 卖出逻辑
            qty = self._round_to_lots(qty, 100)  # 100股整数倍

        # 4. 买入逻辑
        for symbol in target_symbols:
            price = self.get_last_price(symbol)
            if self._is_limit_down(symbol):  # 跌停无法卖出
                continue

            target_qty = int(target_per_stock / price)
            target_qty = self._round_to_lots(target_qty, 100)  # 向下取整到100

            # ... 买入逻辑

    def _round_to_lots(self, quantity: int, lot_size: int) -> int:
        """
        向下取整到交易单位

        A股：lot_size = 100
        """
        return (quantity // lot_size) * lot_size

    def _is_st_stock(self, symbol: str) -> bool:
        """检查是否ST股票"""
        provider = DataProviderFactory.get_provider("cn")
        return provider.is_st_stock(symbol)

    def _is_limit_up(self, symbol: str) -> bool:
        """检查是否涨停"""
        # ... 实现逻辑
        pass

    def _is_limit_down(self, symbol: str) -> bool:
        """检查是否跌停"""
        # ... 实现逻辑
        pass
```

### 4. 配置文件扩展

**位置**: `configs/universe.yaml`

```yaml
# 美股股票池
us_stocks:
  - SPY
  - AAPL
  - MSFT
  - GOOGL
  - AMZN

# A股股票池
cn_stocks:
  - 000001.SZ  # 平安银行
  - 600000.SH  # 浦发银行
  - 600519.SH  # 贵州茅台
  - 000002.SZ  # 万科A
  - 600036.SH  # 招商银行

# A股指数成分股（可选）
cn_indices:
  CSI100:  # 中证100
    - 000001.SZ
    - 600000.SH
    # ...
```

### 5. 前端界面设计

**组件位置**: `frontend/src/components/MarketSelector.tsx`

```typescript
interface MarketSelectorProps {
  value: "us" | "cn";
  onChange: (market: "us" | "cn") => void;
}

const MarketSelector: React.FC<MarketSelectorProps> = ({ value, onChange }) => {
  return (
    <Select value={value} onChange={onChange}>
      <option value="us">🇺🇸 US Stocks (Alpaca)</option>
      <option value="cn">🇨🇳 A股 (AKShare)</option>
    </Select>
  );
};

// 股票池输入提示组件
const SymbolInputHelper: React.FC<{ market: string }> = ({ market }) => {
  const examples = market === "us"
    ? ["AAPL", "MSFT", "GOOGL"]
    : ["000001.SZ", "600000.SH", "600519.SH"];

  return (
    <div>
      <label>Symbols (comma-separated)</label>
      <Input placeholder={examples.join(", ")} />
      <Hint>
        {market === "us"
          ? "Format: Ticker symbol (e.g., AAPL)"
          : "Format: Code.Market (e.g., 000001.SZ, 600000.SH)"}
      </Hint>
    </div>
  );
};
```

## Data Flow

### A股回测流程

```
1. 用户在前端选择"CN Market"
   ↓
2. 配置A股股票池：000001.SZ, 600000.SH, ...
   ↓
3. TradeManager.set_market("cn")
   ↓
4. DataProviderFactory.get_provider("cn")
   → 返回 AKShareDataProvider
   ↓
5. 获取数据：
   AKShareDataProvider.fetch_data()
   → 前复权处理
   → 格式标准化
   ↓
6. QlibDataAdapter存储数据
   → data/qlib/cn/stock_data/
   ↓
7. 特征生成（通用）
   → Alpha158因子
   ↓
8. 模型预测（通用）
   → LightGBM模型
   ↓
9. QlibMLStrategy执行
   → 应用A股交易规则
   → 100股整数倍
   → 涨跌停检查
   ↓
10. 回测结果
    → tearsheet
    → 交易记录
```

## Trade-offs and Decisions

### 决策1：使用AKShare而非Tushare

**理由**:
- ✅ 免费无限制
- ✅ 无需注册和token管理
- ✅ 数据覆盖全面

**权衡**:
- ⚠️ 稳定性依赖开源项目
- **缓解**: 版本锁定，定期更新

### 决策2：仅支持日线数据

**理由**:
- ✅ 日线频率自然规避T+1限制
- ✅ 回测性能好
- ✅ AKShare日线数据最稳定

**权衡**:
- ❌ 无法支持日内策略
- **可接受**: 用户需求明确不需要

### 决策3：复权数据

**选择**: 使用前复权

**理由**:
- ✅ 当前价格真实，便于策略决策
- ✅ 回测时技术指标计算准确
- ⚠️ 历史价格会失真

**权衡**:
- 历史价格不反映真实历史价格
- **缓解**: 文档明确说明，标注使用前复权

### 决策4：最小改动原则

**策略**:
- 现有特征生成器完全复用
- 现有ML模型框架完全复用
- 仅在数据层和策略执行层做最小适配

**理由**:
- ✅ 开发速度快
- ✅ 代码稳定性高
- ✅ 维护成本低

## Testing Strategy

### 单元测试

```python
# tests/test_akshare_provider.py
def test_fetch_data():
    provider = AKShareDataProvider()
    df = provider.fetch_data(
        symbols=["000001.SZ"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
    )
    assert not df.empty
    assert df.columns.tolist() == ["open", "high", "low", "close", "volume"]

def test_lot_rounding():
    strategy = QlibMLStrategy()
    assert strategy._round_to_lots(150, 100) == 100
    assert strategy._round_to_lots(99, 100) == 0
    assert strategy._round_to_lots(300, 100) == 300
```

### 集成测试

```python
# tests/test_a_stock_backtest.py
def test_a_stock_backtest():
    """测试A股回测完整流程"""
    # 1. 准备数据
    provider = AKShareDataProvider()
    df = provider.fetch_data(...)

    # 2. 运行回测
    strategy = QlibMLStrategy(
        parameters={
            "symbols": ["000001.SZ", "600000.SH"],
            "market": "cn",
            "top_k": 2,
        }
    )
    strategy.backtest(...)

    # 3. 验证结果
    assert len(strategy.trades) > 0
    assert all(qty % 100 == 0 for qty in trade.quantity for trade in strategy.trades)
```

### 回测验证

**对比验证**:
- 使用相同策略在A股和美股回测
- 验证交易规则正确应用
- 验证数据质量

## Migration Path

### 阶段1: 数据层（不破坏现有功能）
1. 添加AKShareDataProvider
2. 扩展DataProviderFactory（向后兼容）
3. 单元测试

### 阶段2: 回测支持
1. QlibMLStrategy添加market参数
2. 实现A股交易规则
3. 回测验证

### 阶段3: 前端界面
1. 添加MarketSelector组件
2. 更新配置界面
3. 用户体验测试

### 阶段4: 文档和清理
1. 更新README
2. 添加A股使用示例
3. 清理临时代码

## Performance Considerations

### AKShare性能
- API调用较慢（需爬取网页）
- **优化**: 本地缓存，增量更新

### 数据存储
- A股股票数量多（~5000只）
- **优化**: 仅存储股票池股票，非全市场

### 回测性能
- 日线数据量小，性能不是瓶颈
- 特征生成和模型推理与美股相同

## Security Considerations

- ✅ AKShare无需API密钥，无泄露风险
- ✅ 数据来源公开透明
- ⚠️ 需验证AKShare代码安全性（开源项目）

## Future Enhancements

### 可能的后续功能
1. 支持更多数据源（Tushare、JoinQuant）
2. 支持分钟级回测
3. 添加A股特有因子（北向资金等）
4. 股指期货/期权支持
5. 实盘交易支持（如有需要）

### 扩展点
- `DataProvider` 接口易于扩展新数据源
- `Strategy` 交易规则可插拔
- 前端市场选择器可扩展到更多市场
