# Proposal: 添加A股回测和信号支持

## Meta
- **ID**: add-a-stock-support
- **Status**: Draft
- **Created**: 2025-01-19
- **Owner**: C.Chen

## Problem Statement

当前系统仅支持美股市场，使用Alpaca API获取数据和回测。用户希望扩展系统以支持A股市场的回测和信号生成功能，但不需要实盘和paper交易支持。

**当前限制：**
- 数据源仅支持Alpaca（美股）
- 无A股数据获取能力
- 回测系统未考虑A股交易规则
- 前端界面仅支持美股市场

## Goals

### 主要目标
1. 支持从AKShare获取A股历史数据
2. 支持A股日线数据的复权处理（前复权）
3. 实现A股基础回测功能（日线频率）
4. 支持A股股票筛选（ST、停牌、涨跌停等）
5. 前端添加市场选择器（美股/A股切换）

### 非目标
- 实盘交易支持
- Paper交易支持
- 分钟级回测
- 复杂的A股交易成本模型（使用简化版）

## Success Criteria

1. **数据获取**
   - ✅ 能够从AKShare获取A股日线OHLCV数据
   - ✅ 支持前复权处理
   - ✅ 数据格式与现有系统兼容

2. **回测功能**
   - ✅ 支持日线级别回测
   - ✅ 实现最小交易单位（100股）
   - ✅ 实现涨跌停限制
   - ✅ T+1规则通过日线频率自然规避

3. **股票筛选**
   - ✅ 支持剔除ST股票
   - ✅ 支持剔除停牌股票
   - ✅ 支持识别涨跌停状态

4. **用户体验**
   - ✅ 前端可切换美股/A股市场
   - ✅ 配置文件支持A股股票池
   - ✅ 回测报告区分市场

## Proposed Solution

### 1. 数据层扩展

**新增AKShare数据提供者**
- 在 `autotrade/research/data/providers.py` 中添加 `AKShareDataProvider`
- 实现前复权数据获取
- 数据格式标准化为现有格式 `(timestamp, symbol)` MultiIndex

**数据适配器扩展**
- 修改 `QlibDataAdapter` 支持A股市场标识
- 添加A股特有的元数据管理

### 2. 回测系统适配

**交易规则实现**
- 最小交易单位：100股（1手）
- 涨跌停价格限制：
  - 主板/中小板：±10%
  - 创业板/科创板：±20%
- T+1规则：通过日线频率自然规避（rebalance_period >= 1天）

**策略参数扩展**
- 添加 `market` 参数（"us" 或 "cn"）
- 根据 `market` 自动应用对应交易规则

### 3. 股票筛选机制

**筛选规则**
- 剔除股票名称包含 "ST"、"*ST" 的股票
- 剔除停牌股票（当日无成交数据）
- 识别涨跌停状态（用于因子和风控）

### 4. 前端界面

**市场选择器**
- 在回测配置页面添加市场选择下拉框
- 股票池输入框根据市场切换格式提示
- 示例数据展示对应市场的股票代码格式

### 5. 配置文件

**股票池配置扩展**
- `configs/universe.yaml` 添加A股股票池
- 支持市场分组（us_stock / cn_stock）

## Affected Components

### 需要修改的文件
1. `autotrade/research/data/providers.py` - 新增 AKShareDataProvider
2. `autotrade/research/data/qlib_adapter.py` - 支持A股市场
3. `autotrade/execution/strategies/qlib_ml_strategy.py` - 交易规则适配
4. `configs/universe.yaml` - 添加A股股票池
5. `autotrade/web_server.py` - 前端市场选择器
6. `frontend/src/components/...` - UI组件更新

### 新增文件
1. `autotrade/research/data/akshare_provider.py` - AKShare数据提供者（可选独立文件）
2. `scripts/init_a_stock_data.py` - A股数据初始化脚本

## Dependencies

### 外部依赖
- **akshare**: Python库（需要添加到pyproject.toml）
- 无需API密钥或注册

### 内部依赖
- 现有数据适配器
- 现有特征生成器（基本通用）
- 现有ML模型框架（完全通用）

## Risks and Mitigations

### 风险1：AKShare数据稳定性
- **风险**: AKShare是开源项目，接口可能变化
- **缓解**: 版本锁定akshare，定期更新兼容性

### 风险2：A股数据质量
- **风险**: 历史数据可能缺失或异常
- **缓解**: 数据验证和清洗逻辑，异常值处理

### 风险3：复权数据准确性
- **风险**: 前复权可能导致历史价格失真
- **缓解**: 仅用于回测，明确标注使用复权数据

## Rollout Plan

### 阶段1：数据层（核心）
- 实现AKShareDataProvider
- 数据格式验证
- 复权处理

### 阶段2：回测支持
- 交易规则实现
- 策略适配
- 回测验证

### 阶段3：股票筛选
- ST/停牌识别
- 涨跌停检测
- 筛选逻辑集成

### 阶段4：前端界面
- 市场选择器
- 配置界面更新
- 用户体验优化

## Open Questions

1. **Q**: AKShare的API调用频率限制是多少？
   - **A**: 待调研，如有限制需要实现请求队列

2. **Q**: 是否需要支持分钟级数据？
   - **A**: 不需要（已确认），仅支持日线

3. **Q**: A股特有因子是否需要立即实现？
   - **A**: 需要支持（用户选择），但可作为后续增强

## Alternatives Considered

### 方案A：Tushare Pro
- ✅ 数据质量高，API稳定
- ❌ 需要注册和token管理
- ❌ 免费额度有限

### 方案B：AKShare（已选）
- ✅ 完全免费，无需注册
- ✅ 数据覆盖全面
- ⚠️ 稳定性依赖开源项目维护

### 方案C：JoinQuant/米筐
- ✅ 专业量化平台
- ❌ 需要付费
- ❌ 过于重量级

## Timeline Estimate

- 数据层实现: 2-3天
- 回测适配: 1-2天
- 股票筛选: 1天
- 前端界面: 1-2天
- 测试验证: 1-2天

**总计**: 6-10天

## References

- AKShare官方文档: https://akshare.akfamily.xyz/
- A股交易规则: 上交所/深交所官方规则
- 现有系统架构: openspec/specs/
