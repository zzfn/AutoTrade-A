# a-stock-data-provider Spec Delta

## ADDED Requirements

### Requirement: A股数据源支持

系统 MUST 支持通过AKShare获取A股市场历史数据，作为现有Alpaca数据源的补充。

#### Scenario: 获取A股日线数据

- **GIVEN** 用户配置市场为"cn"（A股）
- **AND** 配置A股股票代码列表（如 ["000001.SZ", "600000.SH"]）
- **WHEN** 请求数据获取
- **THEN** 系统 MUST 使用 `AKShareDataProvider` 而非 `AlpacaDataProvider`
- **AND** 数据 MUST 包含 OHLCV 字段（开高低收成交量）
- **AND** 数据 MUST 使用前复权处理
- **AND** 数据格式 MUST 与美股数据一致（MultiIndex: timestamp, symbol）

#### Scenario: AKShare不可用时的降级处理

- **GIVEN** 用户配置市场为"cn"
- **WHEN** AKShare API调用失败或返回错误
- **THEN** 系统 MUST 抛出清晰的错误信息
- **AND** 系统 MUST NOT 自动切换到其他数据源（与美股不同）

---

### Requirement: 前复权数据处理

系统 MUST 对A股历史数据进行前复权处理，以避免分红送股导致的价格跳空影响回测结果。

#### Scenario: 前复权数据验证

- **GIVEN** 获取的A股历史数据包含分红或送股事件
- **WHEN** 检查数据连续性
- **THEN** 价格序列 MUST 不存在异常跳空
- **AND** 技术指标（如MA）计算 MUST 连续平滑

#### Scenario: 复权数据标注

- **GIVEN** 使用前复权数据进行回测
- **WHEN** 生成回测报告
- **THEN** 报告 MUST 明确标注使用"前复权数据"
- **AND** 报告 MUST 说明历史价格并非真实历史价格

---

### Requirement: A股代码格式验证

系统 MUST 验证A股代码格式，确保符合市场规范。

#### Scenario: 有效A股代码格式

- **GIVEN** 用户输入股票代码 "000001.SZ" 或 "600000.SH"
- **WHEN** 验证代码格式
- **THEN** 系统 MUST 接受该格式（6位数字 + . + 市场后缀）
- **AND** 市场后缀 MUST 是 "SZ"（深交所）或 "SH"（上交所）

#### Scenario: 无效A股代码格式

- **GIVEN** 用户输入股票代码 "000001"（无后缀）或 "AAPL"（美股格式）
- **WHEN** 验证代码格式
- **THEN** 系统 MUST 拒绝该格式
- **AND** 系统 MUST 提供格式提示（如 "000001.SZ"）

---

### Requirement: 停牌状态识别

系统 MUST 能够识别A股停牌状态，并在回测中正确处理。

#### Scenario: 检测停牌股票

- **GIVEN** 股票在特定日期处于停牌状态
- **WHEN** 检查该日期的数据
- **THEN** 系统 MUST 识别停牌状态（无成交数据或成交量=0）
- **AND** 系统 MUST 在策略执行时跳过该股票的交易

#### Scenario: 停牌期间的交易尝试

- **GIVEN** 策略在日期T尝试买入/卖出停牌股票
- **WHEN** 执行交易逻辑
- **THEN** 系统 MUST 跳过该股票
- **AND** 系统 MUST 记录日志说明跳过原因（停牌）

---

### Requirement: ST股票识别与过滤

系统 MUST 能够识别ST（特别处理）股票，并在回测中支持过滤。

#### Scenario: 识别ST股票

- **GIVEN** 股票代码为 "ST000001" 或 "*ST000001"
- **WHEN** 检查股票状态
- **THEN** 系统 MUST 识别该股票为ST股票
- **AND** 系统 MUST 提供过滤选项

#### Scenario: 过滤ST股票（默认行为）

- **GIVEN** 用户启用ST股票过滤（默认开启）
- **AND** 股票池包含ST股票
- **WHEN** 执行回测
- **THEN** 系统 MUST 自动剔除ST股票
- **AND** 系统 MUST 记录剔除的股票列表

---

### Requirement: 涨跌停状态识别

系统 MUST 能够识别A股涨跌停状态，用于风控和因子计算。

#### Scenario: 检测涨停

- **GIVEN** 股票当日价格上涨达到10%或20%限制
- **AND** 收盘价等于最高价
- **WHEN** 检查涨跌停状态
- **THEN** 系统 MUST 识别为涨停
- **AND** 系统 MUST 在策略执行时阻止买入该股票（无法成交）

#### Scenario: 检测跌停

- **GIVEN** 股票当日价格下跌达到10%或20%限制
- **AND** 收盘价等于最低价
- **WHEN** 检查涨跌停状态
- **THEN** 系统 MUST 识别为跌停
- **AND** 系统 MUST 在策略执行时阻止卖出该股票（无法成交）

#### Scenario: 创业板/科创板20%涨跌停

- **GIVEN** 股票属于创业板（300xxx.SZ）或科创板（688xxx.SH）
- **WHEN** 计算涨跌停阈值
- **THEN** 系统 MUST 使用20%而非10%

---

### Requirement: 数据存储隔离

系统 MUST 将美股数据和A股数据存储在不同目录，避免混淆。

#### Scenario: 数据目录结构

- **GIVEN** 系统同时支持美股和A股
- **WHEN** 存储Qlib格式数据
- **THEN** 美股数据 MUST 存储在 `data/qlib/us/`
- **AND** A股数据 MUST 存储在 `data/qlib/cn/`
- **AND** 两个目录 MUST 有独立的 calendars/ 和 instruments/

#### Scenario: 市场标识

- **GIVEN** Qlib数据文件
- **WHEN** 读取数据
- **THEN** 每个数据点 MUST 包含市场标识（us 或 cn）
- **AND** 系统 MUST 根据市场标识选择对应的数据提供者
