# Project Context

## Purpose

AutoTrade-A 是一个基于 **VectorBT** 的 A 股量化预测平台，专注于：

- 使用机器学习模型进行 A 股市场预测
- 提供回测功能验证策略有效性
- 展示每日预测信号辅助投资决策

**注意**: 本系统仅提供预测信号，不进行实际交易。

## Tech Stack

### 核心框架

- **Python 3.11+** - 主要开发语言
- **uv** - 包管理与虚拟环境管理
- **VectorBT** - 向量化回测与策略框架（核心）

### 数据源

- **AKShare** - A 股历史数据（免费、无需 API）

### 机器学习

- **LightGBM** - 梯度提升模型
- **Qlib** - 量化特征工程

### 数据与分析

- **TA-Lib** - 技术分析指标
- **Pandas** - 数据处理
- **NumPy** - 数值计算

### Web 框架

- **FastAPI** - 后端 API
- **Jinja2** - 模板渲染
- **React** - 前端 UI

### 日志

- **Loguru** - 日志记录

### 开发工具

- **Ruff** - 代码检查和格式化

## Project Conventions

### Code Style

- 使用 **Ruff** 进行代码检查和格式化
- 行长度限制：88 字符
- 引号风格：双引号
- Import 排序：使用 isort（通过 Ruff）

### 命名规范

- 策略类名：`PascalCase`，以 `Strategy` 结尾（如 `QlibMLStrategy`）
- 模块名：`snake_case`
- 函数/变量：`snake_case`
- 常量：`UPPER_SNAKE_CASE`

### A 股代码规范

- 格式: `6位数字.交易所`
- 深圳: `.SZ` (如 `000001.SZ`)
- 上海: `.SH` (如 `600000.SH`)

### Architecture Patterns

- **策略模式**：每个交易策略是独立的类
- **配置驱动**：参数通过配置文件和参数字典管理
- **生命周期钩子**：
  - `initialize()` - 初始化
  - `next()` - 信号生成逻辑 (VectorBT style)

### Testing Strategy

- 策略必须通过历史回测验证
- 使用 AKShare 数据进行回测

### Git Workflow

- 主分支：`main`
- 功能分支：`feature/<description>`
- Commit 格式：`<type>: <description>`
  - `feat`, `fix`, `docs`, `refactor`, `test`

## Domain Context

### A 股交易规则

- **交易单位**: 最小 100 股（1 手）
- **涨跌停限制**:
  - 主板/中小板: ±10%
  - 创业板(300xxx): ±20%
  - 科创板(688xxx): ±20%
- **ST 股票**: \*ST 表示退市风险警示，建议过滤
- **交易时间**: 9:30-11:30, 13:00-15:00

### VectorBT 核心概念

- **Portfolio**: 投资组合对象，包含回测结果
- **Signals**: 信号矩阵 (Entries/Exits)
- **IndicatorFactory**: 指标构建工厂

### LumiBot 常用方法

```python
# 生成信号
entries = close > ma
exits = close < ma

# 运行回测
portfolio = vbt.Portfolio.from_signals(close, entries, exits)

# 获取统计
stats = portfolio.stats()
portfolio.plot().show()
```

## Important Constraints

### 技术约束

- Python 代码必须使用 `uv run python` 执行
- 所有策略必须先通过回测
- A 股数据仅支持日线频率

### 业务约束

- **本系统仅提供预测信号，不进行实际交易**
- 预测仅供参考，投资需谨慎

## External Dependencies

### AKShare

- **文档**: https://akshare.akfamily.xyz/
- **GitHub**: https://github.com/akfamily/akshare
- 免费开源，无需 API Key

### VectorBT

- **文档**: https://vectorbt.dev/
- **GitHub**: https://github.com/polakowo/vectorbt
- **Cookbook**: https://vectorbt.dev/api/portfolio/base/
