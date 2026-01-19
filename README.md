# AutoTrade-A

基于机器学习的 A 股量化交易回测系统。

## 功能特性

- 🇨🇳 **A 股专用** - 专注于 A 股市场，使用 AKShare 获取免费行情数据
- 📊 **VectorBT 回测** - 基于 VectorBT 的高性能回测引擎
- 🤖 **ML 策略** - 基于 LightGBM 的机器学习选股策略
- 🌐 **Web 界面** - 实时监控仪表盘和模型管理
- 🔄 **模型训练** - 支持特征工程和模型在线训练
- 💾 **数据缓存** - Parquet 格式的本地持久化缓存

## 快速开始

### 1. 安装依赖

```bash
# 安装 Python 依赖
uv sync
```

### 2. 配置股票池

编辑 `configs/universe.yaml` 设置要交易的股票池：

```yaml
symbols:
  - CSI300  # 沪深300成分股
  # 或指定具体股票
  # - 000001.SZ
  # - 600000.SH
```

### 3. 启动服务

```bash
# 启动 Web 服务器
uv run uvicorn autotrade.web_server:app --reload
```

访问 http://localhost:8000 查看仪表盘。

## ML 策略使用指南

### 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Web UI)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Backtest    │  │ Model       │  │ Train Model         │  │
│  │ Config      │  │ Management  │  │ Trigger             │  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     TradeManager                             │
│                   (Signal Generator)                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│   Data Provider  ──►  Features  ──►  ML Model  ──►  VectorBT│
└─────────────────────────────────────────────────────────────┘
```

### 1. 初始化数据

首次使用需要获取 A 股历史数据：

```bash
# 使用默认股票池（从 configs/universe.yaml 读取）
uv run python scripts/init_a_stock_data.py

# 或指定股票和时间范围
uv run python scripts/init_a_stock_data.py \
  --symbols 000001.SZ,600000.SH \
  --start-date 2023-01-01 \
  --end-date 2024-12-31
```

### 2. 训练模型

```bash
# 基础训练（使用默认股票池）
uv run python scripts/train_model.py

# 指定股票
uv run python scripts/train_model.py --symbols 000001.SZ,600000.SH

# 训练后设置为当前模型
uv run python scripts/train_model.py --set-current
```

### 3. 通过 Web 界面使用

1. 访问 http://localhost:8000
2. 进入「回测」页面配置参数
3. 进入「模型」页面查看和选择模型
4. 开始回测验证策略表现

### 4. API 端点

| 端点                      | 方法 | 说明             |
| ------------------------- | ---- | ---------------- |
| `/api/backtest/run`       | POST | 运行回测         |
| `/api/models`             | GET  | 列出所有可用模型 |
| `/api/models/train`       | POST | 训练新模型       |
| `/api/predict`            | POST | 生成预测信号     |

## 策略说明

### ML 策略

基于机器学习模型预测的策略：

- 使用 LightGBM 预测未来收益率
- Top-K 排名选股（选择预测分数最高的 K 只股票）
- 定期再平衡
- 支持前端配置和模型热切换

#### 特征工程

采用类似 Alpha158 的技术指标因子：

- 价格回报率（1/5/10/20 天）
- 移动平均线及斜率
- 波动率（ATR、标准差）
- 成交量因子
- RSI、MACD、布林带等技术指标

## 项目结构

```
autotrade/
├── backtesting/          # VectorBT 回测引擎
├── core/                 # 核心模块（信号生成器等）
├── research/             # 研究模块
│   ├── data/            # 数据获取（AKShare）
│   ├── features/        # 特征工程
│   └── models/          # 模型训练和管理
├── shared/               # 共享工具
│   └── config/          # 配置管理
├── ui/                   # Web 界面
│   └── templates/
├── trade_manager.py      # 交易管理器
└── web_server.py         # Web 服务器

scripts/
├── init_a_stock_data.py  # A 股数据初始化
└── train_model.py        # 模型训练

configs/
└── universe.yaml         # 股票池配置

models/                   # 训练好的模型
data/cache/               # 数据缓存（Parquet 格式）
```

## A 股交易规则

回测中自动应用以下 A 股特殊规则：

| 规则         | 说明                          |
| ------------ | ----------------------------- |
| 最小交易单位 | 100 股（1 手）                |
| 涨跌停限制   | 主板 ±10%，创业板/科创板 ±20% |
| ST 股票过滤  | 自动剔除 ST/\*ST 股票         |
| T+1 规则     | 通过日线频率自然规避          |

### 股票代码格式

A 股代码格式为 `6位数字.市场后缀`：

- **SZ** (深圳): `000001.SZ`, `300750.SZ`
- **SH** (上海): `600000.SH`, `688981.SH`

### 数据源

A 股数据使用 [AKShare](https://akshare.akfamily.xyz/) 获取，免费且无需注册。

## 开发

```bash
# 运行测试
uv run pytest

# 代码格式化
uv run ruff format .
uv run ruff check . --fix
```

## 许可证

MIT
