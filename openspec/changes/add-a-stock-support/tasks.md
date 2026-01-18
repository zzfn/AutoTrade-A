# Tasks: 添加A股支持

## Overview

按优先级排序的实施任务列表，每个任务都是小且可验证的。

---

## Phase 1: 数据层实现（核心功能）

### ✅ Task 1.1: 添加AKShare依赖
**Priority**: P0 (阻塞)
**Effort**: 5分钟
**Description**: 在项目依赖中添加akshare库

**Steps**:
1. 编辑 `pyproject.toml`
2. 在dependencies中添加 `akshare = "^1.12.0"`
3. 运行 `uv sync` 安装依赖

**Validation**:
- ✅ `uv run python -c "import akshare; print(akshare.__version__)"` 成功执行

---

### ✅ Task 1.2: 实现AKShareDataProvider基础框架
**Priority**: P0 (阻塞)
**Effort**: 2-3小时
**Description**: 创建AKShare数据提供者类，实现基础数据获取

**Steps**:
1. 在 `autotrade/research/data/providers.py` 中添加 `AKShareDataProvider` 类
2. 继承 `BaseDataProvider` 抽象基类
3. 实现 `fetch_data()` 方法：
   - 调用 `akshare.stock_zh_a_hist()` 获取日线数据
   - 参数：`adjust="qfq"`（前复权）
4. 实现数据格式转换：
   - 输出 MultiIndex (timestamp, symbol)
   - 列：open, high, low, close, volume

**Validation**:
- ✅ 单元测试：获取单只股票（000001.SZ）1年数据
- ✅ 验证输出格式与Alpaca一致
- ✅ 数据包含前复权处理

---

### ✅ Task 1.3: 实现A股代码格式验证
**Priority**: P0 (阻塞)
**Effort**: 1小时
**Description**: 验证A股代码格式（6位数字 + .SZ/.SH）

**Steps**:
1. 在 `AKShareDataProvider` 中添加 `_validate_symbol()` 方法
2. 验证规则：
   - 长度9位（6位数字 + 点 + 2位字母）
   - 前6位是数字
   - 后缀是 .SZ 或 .SH
3. 在 `fetch_data()` 中调用验证
4. 抛出清晰的错误信息

**Validation**:
- ✅ "000001.SZ" → 通过
- ✅ "600000.SH" → 通过
- ❌ "000001" → 报错："缺少市场后缀，应为 000001.SZ"
- ❌ "AAPL" → 报错："格式错误，应为 6位数字.SZ/SH"

---

### ✅ Task 1.4: 扩展DataProviderFactory支持多市场
**Priority**: P0 (阻塞)
**Effort**: 30分钟
**Description**: 修改工厂类支持根据市场参数返回对应数据提供者

**Steps**:
1. 修改 `DataProviderFactory.get_provider()` 方法签名，添加 `market` 参数
2. 逻辑：
   - `market="cn"` → 返回 `AKShareDataProvider`
   - `market="us"` 或默认 → 返回 `AlpacaDataProvider`
3. 更新错误提示，说明支持的市场

**Validation**:
- ✅ `get_provider("cn")` 返回 `AKShareDataProvider` 实例
- ✅ `get_provider("us")` 返回 `AlpacaDataProvider` 实例
- ❌ `get_provider("jp")` 抛出 `ValueError`

---

## Phase 2: 股票筛选和状态识别

### ✅ Task 2.1: 实现ST股票识别
**Priority**: P1 (重要)
**Effort**: 30分钟
**Description**: 识别和过滤ST股票

**Steps**:
1. 在 `AKShareDataProvider` 中添加 `is_st_stock()` 方法
2. 通过股票代码或名称判断（名称包含 "ST"、"*ST"）
3. 返回布尔值

**Validation**:
- ✅ `is_st_stock("ST000001.SZ")` → True
- ✅ `is_st_stock("000001.SZ")` → False

---

### ✅ Task 2.2: 实现停牌状态检测
**Priority**: P1 (重要)
**Effort**: 1小时
**Description**: 检测股票停牌状态

**Steps**:
1. 在 `AKShareDataProvider` 中添加 `is_suspended(symbol, date)` 方法
2. 判断逻辑：
   - 查询指定日期的数据
   - 如果数据不存在或成交量=0 → 停牌
3. 返回布尔值

**Validation**:
- ✅ 测试已知停牌日期返回True
- ✅ 测试正常交易日返回False

---

### ✅ Task 2.3: 实现涨跌停检测
**Priority**: P1 (重要)
**Effort**: 1.5小时
**Description**: 检测涨停和跌停状态

**Steps**:
1. 添加 `is_limit_up(symbol, date)` 方法
   - 判断逻辑：收盘价=最高价 且 涨幅≈10%/20%
2. 添加 `is_limit_down(symbol, date)` 方法
   - 判断逻辑：收盘价=最低价 且 跌幅≈10%/20%
3. 判断创业板/科创板：
   - 300xxx.SZ → 创业板（20%）
   - 688xxx.SH → 科创板（20%）
   - 其他 → 10%

**Validation**:
- ✅ 测试已知涨停日期返回True
- ✅ 测试已知跌停日期返回True
- ✅ 测试正常日期返回False

---

## Phase 3: 回测系统适配

### ✅ Task 3.1: 添加市场参数到策略
**Priority**: P0 (阻塞)
**Effort**: 30分钟
**Description**: 在QlibMLStrategy中添加market参数

**Steps**:
1. 在 `QlibMLStrategy.parameters` 中添加 `"market": "us"`
2. 在 `_parse_parameters()` 中解析 `self.market`
3. 传递给数据提供者

**Validation**:
- ✅ 策略初始化时 `market="cn"` 正确解析
- ✅ `market="us"` 为默认值

---

### ✅ Task 3.2: 实现最小交易单位限制
**Priority**: P0 (阻塞)
**Effort**: 1小时
**Description**: A股交易数量必须是100股的整数倍

**Steps**:
1. 在 `QlibMLStrategy` 中添加 `_round_to_lots(quantity, lot_size)` 方法
2. 向下取整到100的倍数
3. 在 `_rebalance_portfolio()` 中：
   - 仅在 `market="cn"` 时应用
   - 买入数量：`_round_to_lots(target_qty, 100)`
   - 卖出数量：`_round_to_lots(current_qty, 100)`
4. 记录调整日志

**Validation**:
- ✅ 买入150股 → 调整为100股
- ✅ 买入50股 → 跳过（不足100）
- ✅ 卖出250股 → 卖出200股，保留50股

---

### ✅ Task 3.3: 实现涨跌停交易限制
**Priority**: P1 (重要)
**Effort**: 1.5小时
**Description**: 涨停时阻止买入，跌停时阻止卖出

**Steps**:
1. 在 `_rebalance_portfolio()` 中添加涨跌停检查
2. 买入前检查：
   - 调用 `is_limit_up(symbol)`
   - 如果涨停，跳过买入并记录日志
3. 卖出前检查：
   - 调用 `is_limit_down(symbol)`
   - 如果跌停，跳过卖出并记录日志

**Validation**:
- ✅ 涨停股票无法买入，日志记录"涨停无法买入"
- ✅ 跌停股票无法卖出，日志记录"跌停无法卖出"
- ✅ 次日正常股票可正常交易

---

### ✅ Task 3.4: 集成股票筛选
**Priority**: P1 (重要)
**Effort**: 1小时
**Description**: 在策略执行前过滤ST和停牌股票

**Steps**:
1. 在 `_get_predictions()` 中添加过滤逻辑
2. 跳过ST股票：`if provider.is_st_stock(symbol): continue`
3. 跳过停牌股票：`if provider.is_suspended(symbol, date): continue`
4. 记录过滤的股票列表

**Validation**:
- ✅ ST股票不在预测结果中
- ✅ 停牌股票不在预测结果中
- ✅ 日志记录过滤的股票

---

## Phase 4: 数据存储和适配器

### ✅ Task 4.1: 修改QlibDataAdapter支持多市场
**Priority**: P0 (阻塞)
**Effort**: 2小时
**Description**: 适配器支持市场参数和数据目录隔离

**Steps**:
1. 在 `QlibDataAdapter.__init__()` 中添加 `market` 参数
2. 数据目录：
   - `us` → `data/qlib/us/`
   - `cn` → `data/qlib/cn/`
3. 在 `fetch_and_store()` 中根据 `market` 选择对应的数据提供者
4. 更新 `_update_instruments()` 支持A股代码格式

**Validation**:
- ✅ `market="cn"` 时数据存储到 `data/qlib/cn/`
- ✅ `market="us"` 时数据存储到 `data/qlib/us/`
- ✅ 两个市场数据不混淆

---

### ✅ Task 4.2: 创建A股数据初始化脚本
**Priority**: P1 (重要)
**Effort**: 1小时
**Description**: 提供脚本初始化A股历史数据

**Steps**:
1. 创建 `scripts/init_a_stock_data.py`
2. 功能：
   - 读取 `configs/universe.yaml` 中的 `cn_stocks`
   - 调用 `AKShareDataProvider` 获取数据
   - 调用 `QlibDataAdapter` 存储数据
   - 显示进度条
3. 添加命令行参数：
   - `--start-date`: 开始日期
   - `--end-date`: 结束日期

**Validation**:
- ✅ 成功获取并存储A股历史数据
- ✅ 数据可用于回测

---

## Phase 5: 配置文件

### ✅ Task 5.1: 扩展universe.yaml添加A股股票池
**Priority**: P0 (阻塞)
**Effort**: 30分钟
**Description**: 在配置文件中添加A股股票池

**Steps**:
1. 编辑 `configs/universe.yaml`
2. 添加 `cn_stocks` 部分：
   - 选择10-20只知名A股
   - 包含主板、创业板、科创板
   - 格式：000001.SZ
3. 添加注释说明市场标识

**Validation**:
- ✅ 配置文件包含 `cn_stocks` 列表
- ✅ 代码格式正确（.SZ/.SH后缀）

---

## Phase 6: 前端界面

### ✅ Task 6.1: 实现市场选择器组件
**Priority**: P1 (重要)
**Effort**: 2小时
**Description**: 添加市场选择下拉框

**Steps**:
1. 创建 `frontend/src/components/MarketSelector.tsx`
2. 功能：
   - 下拉框：US Stocks / A股
   - 选择回调 `onChange(market)`
3. 集成到回测配置页面
4. 添加样式和图标（🇺🇸 / 🇨🇳）

**Validation**:
- ✅ 选择器显示正确
- ✅ 切换市场触发回调
- ✅ 选择持久化到localStorage

---

### ✅ Task 6.2: 实现股票池输入市场感知
**Priority**: P1 (重要)
**Effort**: 2小时
**Description**: 根据市场切换股票池提示

**Steps**:
1. 创建 `frontend/src/components/SymbolInput.tsx`
2. 接受 `market` 参数
3. 根据 `market` 动态显示：
   - 占位符
   - 示例代码
   - 帮助文本
4. 添加实时格式验证

**Validation**:
- ✅ 美股市场显示 "AAPL, MSFT" 示例
- ✅ A股市场显示 "000001.SZ, 600000.SH" 示例
- ✅ 输入错误格式时显示红色错误提示

---

### ✅ Task 6.3: 添加预设股票池加载按钮
**Priority**: P2 (可选)
**Effort**: 1小时
**Description**: 一键加载预设股票池

**Steps**:
1. 添加"加载预设"按钮
2. 点击时从API获取对应市场的预设股票池
3. 后端接口：`GET /api/universe?market=cn`
4. 填充到股票池输入框

**Validation**:
- ✅ 点击按钮加载对应市场的预设股票池
- ✅ API返回正确的股票列表

---

### ✅ Task 6.4: 回测结果页面市场标注
**Priority**: P2 (可选)
**Effort**: 1小时
**Description**: 在回测报告中显示市场信息

**Steps**:
1. 修改回测结果页面
2. 添加"市场"字段显示
3. 显示数据源、交易规则等信息

**Validation**:
- ✅ A股回测显示"市场：A股"
- ✅ 美股回测显示"市场：美股"

---

## Phase 7: 测试和验证

### ✅ Task 7.1: 编写单元测试
**Priority**: P0 (阻塞)
**Effort**: 2小时
**Description**: 数据提供者和工具函数的单元测试

**Steps**:
1. 创建 `tests/test_akshare_provider.py`
2. 测试用例：
   - 数据获取格式验证
   - 代码格式验证
   - ST股票识别
   - 停牌检测
   - 涨跌停检测
3. 创建 `tests/test_strategy_cn.py`
4. 测试用例：
   - 100股取整
   - 涨跌停交易限制

**Validation**:
- ✅ 所有测试通过
- ✅ 测试覆盖率 > 80%

---

### ✅ Task 7.2: 端到端回测验证
**Priority**: P0 (阻塞)
**Effort**: 2小时
**Description**: 完整的A股回测流程测试

**Steps**:
1. 准备A股数据（5-10只股票）
2. 运行完整回测（1年时间范围）
3. 验证：
   - 数据获取成功
   - 交易数量是100的倍数
   - 涨跌停正确处理
   - ST股票被过滤
   - 回测报告生成

**Validation**:
- ✅ 回测成功完成无报错
- ✅ 交易记录符合A股规则
- ✅ 回测报告包含市场信息

---

### ✅ Task 7.3: 性能测试
**Priority**: P2 (可选)
**Effort**: 1小时
**Description**: 对比A股和美股回测性能

**Steps**:
1. 使用相同数量股票（10只）
2. 相同时间范围（1年）
3. 分别运行A股和美股回测
4. 记录执行时间和内存使用

**Validation**:
- ✅ A股回测时间在美股150%以内
- ✅ 内存使用在合理范围

---

## Phase 8: 文档和清理

### ✅ Task 8.1: 更新README
**Priority**: P2 (可选)
**Effort**: 1小时
**Description**: 添加A股功能说明

**Steps**:
1. 在README中添加"A股支持"章节
2. 说明：
   - 支持的市场
   - 数据源（AKShare）
   - 股票代码格式
   - 配置方法
3. 添加使用示例

**Validation**:
- ✅ README包含A股功能说明
- ✅ 用户可根据文档完成配置

---

### ✅ Task 8.2: 添加CHANGELOG
**Priority**: P2 (可选)
**Effort**: 30分钟
**Description**: 记录本次变更

**Steps**:
1. 在CHANGELOG.md中添加新版本
2. 列出主要变更：
   - 新增A股数据源（AKShare）
   - 支持A股回测
   - 前端市场选择器

**Validation**:
- ✅ CHANGELOG清晰记录变更内容

---

## 任务统计

- **总任务数**: 24
- **P0（阻塞）**: 10
- **P1（重要）**: 10
- **P2（可选）**: 4
- **估计总工时**: 30-40小时

## 并行化建议

可以并行执行的任务组：
1. **Phase 1-2**: 数据层和筛选功能可并行
2. **Phase 3-4**: 回测适配和数据存储可并行
3. **Phase 6**: 前端开发和Phase 5（配置）可并行
4. **Phase 7**: 测试必须在功能开发完成后进行

## 关键路径

```
Task 1.1 → Task 1.2 → Task 1.4 → Task 3.1 → Task 3.2 → Task 7.2
                    ↓
                Task 4.1 → Task 4.2
```

这条路径必须按顺序完成，其他任务可以并行。
