# multi-market-ui Spec Delta

## ADDED Requirements

### Requirement: 市场选择器

系统 MUST 在前端界面提供市场选择器，允许用户在美股和A股之间切换。

#### Scenario: 显示市场选择器

- **GIVEN** 用户访问回测配置页面
- **WHEN** 页面渲染
- **THEN** 系统 MUST 显示市场选择下拉框
- **AND** 选项 MUST 包含：
  - 🇺🇸 US Stocks (Alpaca)
  - 🇨🇳 A股 (AKShare)
- **AND** 默认值 MUST 为"US Stocks"

#### Scenario: 切换到A股市场

- **GIVEN** 用户在回测配置页面
- **AND** 当前选择为"US Stocks"
- **WHEN** 用户选择"A股 (AKShare)"
- **THEN** 系统 MUST 更新市场参数为"cn"
- **AND** 系统 MUST 更新股票池输入提示（显示A股格式示例）
- **AND** 系统 MUST 清空或验证现有股票池配置

#### Scenario: 市场选择持久化

- **GIVEN** 用户选择"A股"市场
- **AND** 配置了A股股票池
- **WHEN** 用户刷新页面
- **THEN** 系统 MUST 记住用户的市场选择
- **AND** 系统 MUST 恢复A股股票池配置

---

### Requirement: 股票池输入市场感知

系统 MUST 根据选择的市场调整股票池输入的提示和验证规则。

#### Scenario: 美股市场股票池提示

- **GIVEN** 用户选择"US Stocks"市场
- **WHEN** 显示股票池输入框
- **THEN** 系统 MUST 显示美股格式示例：
  - "AAPL, MSFT, GOOGL, AMZN"
- **AND** 占位符 MUST 为 "AAPL, MSFT, ..."
- **AND** 帮助文本 MUST 说明"使用股票代码，如 AAPL"

#### Scenario: A股市场股票池提示

- **GIVEN** 用户选择"A股"市场
- **WHEN** 显示股票池输入框
- **THEN** 系统 MUST 显示A股格式示例：
  - "000001.SZ, 600000.SH, 600519.SH"
- **AND** 占位符 MUST 为 "000001.SZ, 600000.SH, ..."
- **AND** 帮助文本 MUST 说明"使用代码.市场格式，如 000001.SZ"

#### Scenario: 实时格式验证

- **GIVEN** 用户在"A股"市场
- **WHEN** 输入股票代码 "000001"（缺少后缀）
- **THEN** 系统 MUST 显示格式错误提示
- **AND** 错误信息 MUST 说明正确格式（如 "000001.SZ"）
- **AND** 系统 MUST 阻止提交错误的配置

---

### Requirement: 预设股票池配置

系统 MUST 为不同市场提供预设的股票池配置。

#### Scenario: 美股预设股票池

- **GIVEN** 用户选择"US Stocks"市场
- **AND** 点击"加载预设股票池"按钮
- **WHEN** 加载预设配置
- **THEN** 系统 MUST 从 `configs/universe.yaml` 加载 `us_stocks` 列表
- **AND** 系统 MUST 填充股票池输入框

#### Scenario: A股预设股票池

- **GIVEN** 用户选择"A股"市场
- **AND** 点击"加载预设股票池"按钮
- **WHEN** 加载预设配置
- **THEN** 系统 MUST 从 `configs/universe.yaml` 加载 `cn_stocks` 列表
- **AND** 系统 MUST 填充股票池输入框

#### Scenario: 预设配置不存在

- **GIVEN** 用户选择某个市场
- **AND** `configs/universe.yaml` 中缺少对应市场的预设配置
- **WHEN** 点击"加载预设股票池"
- **THEN** 系统 MUST 显示"预设配置不存在"提示
- **AND** 系统 MUST 提供"手动输入"选项

---

### Requirement: 回测结果市场标注

系统 MUST 在回测结果页面清晰标注市场类型。

#### Scenario: A股回测结果显示

- **GIVEN** 用户运行A股回测
- **WHEN** 回测完成
- **THEN** 页面标题 MUST 包含"🇨🇳 A股回测"
- **AND** 回测摘要 MUST 显示：
  - 市场：A股
  - 数据源：AKShare
  - 复权方式：前复权
  - 交易规则：100股限制、涨跌停限制

#### Scenario: 美股回测结果显示（保持不变）

- **GIVEN** 用户运行美股回测
- **WHEN** 回测完成
- **THEN** 页面标题 MUST 包含"🇺🇸 美股回测"
- **AND** 回测摘要 MUST 显示：
  - 市场：美股
  - 数据源：Alpaca
  - 交易规则：无特殊限制

---

### Requirement: 交易记录市场信息

系统 MUST 在交易记录中显示市场相关的特殊信息。

#### Scenario: A股交易记录显示交易单位

- **GIVEN** A股回测完成
- **WHEN** 查看交易记录
- **THEN** 每条交易记录 MUST 显示：
  - 股票代码（如 000001.SZ）
  - 交易数量（100的整数倍）
  - 交易价格
  - 交易日期
- **AND** 如果交易因涨跌停被阻止，MUST 显示原因

#### Scenario: A股交易记录显示特殊事件

- **GIVEN** A股回测期间股票发生涨跌停
- **WHEN** 查看交易记录
- **THEN** 系统 MUST 标注受影响的交易
- **AND** 标注 MUST 说明"因涨停无法买入"或"因跌停无法卖出"

---

### Requirement: 错误提示市场相关

系统 MUST 为A股市场提供特定的错误提示和帮助信息。

#### Scenario: A股数据获取失败

- **GIVEN** 用户运行A股回测
- **AND** AKShare API调用失败
- **WHEN** 数据获取错误
- **THEN** 系统 MUST 显示"A股数据获取失败"错误
- **AND** 错误信息 MUST 说明可能的原因：
  - 网络连接问题
  - 股票代码格式错误
  - AKShare服务暂时不可用
- **AND** 系统 MUST 提供"重试"按钮

#### Scenario: A股市场配置帮助

- **GIVEN** 用户首次使用A股功能
- **WHEN** 选择"A股"市场
- **THEN** 系统 MUST 显示引导提示
- **AND** 提示 MUST 说明：
  - 股票代码格式要求
  - 预设股票池位置
  - 数据初始化步骤（如需要）

---

### Requirement: UI性能和响应性

系统 MUST 确保市场切换和配置更新的响应性。

#### Scenario: 市场切换即时响应

- **GIVEN** 用户在回测配置页面
- **WHEN** 切换市场选择
- **THEN** UI MUST 在100ms内更新股票池提示
- **AND** 输入框占位符 MUST 立即更新
- **AND** 帮助文本 MUST 立即切换

#### Scenario: 股票池验证实时反馈

- **GIVEN** 用户在A股市场输入股票池
- **WHEN** 输入股票代码
- **THEN** 系统 MUST 在500ms内提供格式验证反馈
- **AND** 正确格式 MUST 显示绿色✓
- **AND** 错误格式 MUST 显示红色✗和错误说明
