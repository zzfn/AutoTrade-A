## MODIFIED Requirements

### Requirement: ML 策略执行

系统 SHALL 提供 `QlibMLStrategy`，基于 ML 模型预测驱动交易决策：

- 继承 Qlib `BaseStrategy` 基类 (原为 LumiBot Strategy)
- 使用 Top-K 排名策略选股
- 支持配置持仓数量（top_k）和再平衡周期
- 使用 Qlib 内置回测引擎执行回测

#### Scenario: 基于预测信号交易

- **WHEN** 交易迭代开始
- **THEN** 策略接收最新预测分数
- **AND** 选择预测分数最高的 Top-K 股票
- **AND** 生成目标持仓权重的买卖订单

#### Scenario: 再平衡持仓

- **WHEN** 到达再平衡周期
- **THEN** 系统根据新一轮预测更新目标持仓
- **AND** 生成调仓订单 (Sell losers, Buy winners)
