# a-stock-backtesting Spec Delta

## ADDED Requirements

### Requirement: 最小交易单位限制

系统 MUST 在A股回测中强制执行最小交易单位限制（100股/手）。

#### Scenario: 买入数量向下取整到100股

- **GIVEN** A股市场回测
- **AND** 策略计算目标买入数量为150股
- **WHEN** 执行买入订单
- **THEN** 系统 MUST 将买入数量调整为100股（向下取整）
- **AND** 系统 MUST 记录调整日志

#### Scenario: 卖出数量必须是100股整数倍

- **GIVEN** A股市场回测
- **AND** 当前持仓250股（非标准持仓）
- **WHEN** 执行卖出订单
- **THEN** 系统 MUST 卖出200股（100的整数倍）
- **AND** 系统 MUST 保留50股剩余持仓
- **AND** 系统 MUST 记录部分卖出日志

#### Scenario: 目标数量不足100股

- **GIVEN** A股市场回测
- **AND** 策略计算目标买入数量为50股
- **WHEN** 执行买入订单
- **THEN** 系统 MUST 跳过该买入操作
- **AND** 系统 MUST 记录"数量不足100股，跳过买入"日志

---

### Requirement: 涨跌停交易限制

系统 MUST 在回测中阻止涨停时买入、跌停时卖出的不切实际订单。

#### Scenario: 涨停无法买入

- **GIVEN** A股市场回测
- **AND** 股票A在日期T处于涨停状态
- **WHEN** 策略尝试买入股票A
- **THEN** 系统 MUST 取消买入订单
- **AND** 系统 MUST 记录"涨停无法买入"日志
- **AND** 系统 MUST 继续处理其他股票

#### Scenario: 跌停无法卖出

- **GIVEN** A股市场回测
- **AND** 股票B在日期T处于跌停状态
- **WHEN** 策略尝试卖出股票B
- **THEN** 系统 MUST 取消卖出订单
- **AND** 系统 MUST 记录"跌停无法卖出"日志
- **AND** 系统 MUST 保留股票B的持仓

#### Scenario: 涨跌停后的次日正常交易

- **GIVEN** 股票C在日期T涨停
- **AND** 策略在T尝试买入但被阻止
- **WHEN** 日期T+1，股票C未涨停
- **THEN** 系统 MUST 允许买入股票C
- **AND** 交易 MUST 正常执行

---

### Requirement: 日线频率T+1处理

系统 MUST 确保日线回测正确处理A股T+1规则。

#### Scenario: 再平衡周期≥1天时自然规避T+1

- **GIVEN** A股市场回测
- **AND** 再平衡周期设置为1天或更长
- **AND** 策略在日期T卖出股票A、买入股票B
- **WHEN** 执行交易逻辑
- **THEN** 卖出A获得的资金 MUST 可以在同一天（T）用于买入B
- **AND** 系统 MUST 不需要额外的T+1限制逻辑
- **AND** 交易 MUST 正常执行

#### Scenario: 日线频率下的持仓股票可卖出

- **GIVEN** A股市场回测
- **AND** 持仓股票X是在日期T-1买入的
- **WHEN** 在日期T执行卖出股票X
- **THEN** 卖出订单 MUST 成功执行
- **AND** 系统 MUST 允许该交易

---

### Requirement: 市场参数配置

系统 MUST 支持通过参数指定市场，以应用对应的交易规则。

#### Scenario: 配置A股市场参数

- **GIVEN** 策略初始化
- **AND** 参数 `market` 设置为 "cn"
- **WHEN** 策略执行
- **THEN** 系统 MUST 应用A股交易规则（100股限制、涨跌停等）
- **AND** 系统 MUST 使用AKShare数据源
- **AND** 回测报告 MUST 标注"A股市场"

#### Scenario: 配置美股市场参数（默认）

- **GIVEN** 策略初始化
- **AND** 参数 `market` 未设置或设置为 "us"
- **WHEN** 策略执行
- **THEN** 系统 MUST 应用美股交易规则（无最小单位限制）
- **AND** 系统 MUST 使用Alpaca数据源
- **AND** 回测报告 MUST 标注"美股市场"

#### Scenario: 市场参数验证

- **GIVEN** 策略初始化
- **AND** 参数 `market` 设置为无效值（如 "jp"）
- **WHEN** 验证参数
- **THEN** 系统 MUST 抛出 `ValueError`
- **AND** 错误信息 MUST 说明支持的市场（"us", "cn"）

---

### Requirement: 回测报告市场标注

系统 MUST 在回测报告中清晰标注市场类型和使用的交易规则。

#### Scenario: A股回测报告

- **GIVEN** A股市场回测完成
- **WHEN** 生成回测报告
- **THEN** 报告 MUST 包含"市场：A股"字段
- **AND** 报告 MUST 列出应用的交易规则：
  - 最小交易单位：100股
  - 涨跌停限制：10%/20%
  - T+1规则：通过日线频率自然满足
- **AND** 报告 MUST 标注数据来源：AKShare
- **AND** 报告 MUST 标注复权方式：前复权

#### Scenario: 美股回测报告（保持不变）

- **GIVEN** 美股市场回测完成
- **WHEN** 生成回测报告
- **THEN** 报告 MUST 包含"市场：美股"字段
- **AND** 报告 MUST 说明无特殊交易限制
- **AND** 报告 MUST 标注数据来源：Alpaca

---

### Requirement: 股票筛选规则集成

系统 MUST 在策略执行前应用股票筛选规则。

#### Scenario: 策略执行前过滤ST股票

- **GIVEN** A股市场回测
- **AND** 股票池包含ST股票
- **AND** ST股票过滤已启用
- **WHEN** 策略执行交易逻辑
- **THEN** 系统 MUST 在获取预测分数前剔除ST股票
- **AND** 系统 MUST 仅对非ST股票生成预测和排序
- **AND** Top-K选股 MUST 不包含ST股票

#### Scenario: 策略执行前跳过停牌股票

- **GIVEN** A股市场回测
- **AND** 股票池包含停牌股票
- **WHEN** 策略在日期T执行
- **THEN** 系统 MUST 在交易决策时跳过停牌股票
- **AND** 系统 MUST 不对停牌股票生成买卖订单
- **AND** 系统 MUST 记录停牌股票列表

---

### Requirement: 回测性能不劣化

系统 MUST 确保A股回测性能与美股回测相当。

#### Scenario: 回测执行时间对比

- **GIVEN** A股和美股使用相同数量的股票（如10只）
- **AND** 相同的回测时间范围（如1年）
- **WHEN** 分别运行A股和美股回测
- **THEN** A股回测时间 MUST 在美股回测时间的150%以内
- **AND** 数据获取时间 MUST 在合理范围（< 60秒）

#### Scenario: 内存使用对比

- **GIVEN** A股和美股回测
- **WHEN** 加载数据
- **THEN** 内存使用 MUST 在系统可承受范围（< 4GB）
- **AND** A股回测内存使用 MUST 不显著高于美股
