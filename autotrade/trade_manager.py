"""
TradeManager - A 股交易管理器

AutoTrade-A 专用：仅支持 A 股预测和回测
"""

import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autotrade.models import ModelManager
from autotrade.common.config.loader import default_config

class TradeManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.active_strategy = None
        self.strategy_thread: threading.Thread | None = None
        self.is_running = False

        # State storage
        self.state: dict[str, Any] = {
            "status": "stopped",
            "logs": [],
            "orders": [],
            "portfolio": {"cash": 0.0, "value": 0.0, "positions": []},
            "market_status": "unknown",
            "last_update": None,
        }
        
        # 预测锁，防止并发重复计算
        self.prediction_lock = threading.Lock()

        # ML 策略配置
        self.ml_config: dict[str, Any] = {
            "model_name": None,  # None 表示使用最优模型（由 ModelManager 自动选择）
            "top_k": 3,
            "rebalance_period": 1,
        }
        self.model_manager = ModelManager()

        # 模型训练状态
        self.training_status = {
            "in_progress": False,
            "progress": 0,
            "message": "",
        }

        # 数据同步状态
        self.data_sync_status = {
            "in_progress": False,
            "progress": 0,
            "message": "",
        }

        self._initialized = True

    def _resolve_symbols(self, symbols: list[str], refresh: bool = False) -> list[str]:
        """
        解析股票列表，将指数代码转换为成分股列表

        Args:
            symbols: 股票代码列表，可能包含指数代码（如 "sse50", "CSI300"）
            refresh: 是否强制刷新指数成分股

        Returns:
            实际的股票代码列表
        """
        from autotrade.data import DataProviderFactory

        provider = DataProviderFactory.get_provider("cn")
        resolved = []

        # 指数代码映射
        index_mapping = {
            "sse50": "000016",  # 上证50
            "csi300": "000300",  # 沪深300
            "csi500": "000905",  # 中证500
            "sz50": "399330",    # 深证50
        }

        for symbol in symbols:
            symbol_lower = symbol.lower()
            if symbol_lower in index_mapping:
                # 获取指数成分股
                index_code = index_mapping[symbol_lower]
                self.log(f"正在获取 {symbol} ({index_code}) 的成分股...")
                try:
                    constituents = provider.get_index_constituents(index_code)
                    if constituents:
                        self.log(f"{symbol} 包含 {len(constituents)} 只股票")
                        resolved.extend(constituents)
                    else:
                        self.log(f"警告：无法获取 {symbol} 的成分股")
                except Exception as e:
                    self.log(f"获取 {symbol} 成分股失败: {e}")
            else:
                # 直接添加股票代码
                resolved.append(symbol.upper())

        # 去重
        resolved = list(dict.fromkeys(resolved).keys())
        return resolved

    # Below is the modified get_latest_predictions

    def get_latest_predictions(self, symbols: list[str] | None = None, refresh: bool = False) -> dict:
        """
        获取最新的预测信号
        
        通过加锁和双重检查锁定 (Double-Checked Locking) 防止并发重复计算。
        """
        use_default_symbols = (symbols is None)
        
        # 1. 第一次检查缓存 (无锁)
        if not refresh and use_default_symbols:
            cached = self._load_prediction_cache()
            current_model = self.model_manager.get_current_model()
            
            if cached and cached.get("model") == current_model:
                self.log(f"使用缓存的预测结果 (日期: {cached.get('date')}, 模型: {cached.get('model')})")
                return cached
            elif cached:
                if cached.get("model") != current_model:
                    self.log(f"模型已变更 (缓存: {cached.get('model')}, 当前: {current_model})，准备重新计算...")

        # 2. 获取锁
        self.log("正在请求预测资源锁...")
        with self.prediction_lock:
            # 3. 第二次检查缓存
            if not refresh and use_default_symbols:
                cached = self._load_prediction_cache()
                current_model = self.model_manager.get_current_model()
                
                if cached and cached.get("model") == current_model:
                    self.log(f"使用新生成的缓存预测结果")
                    return cached

            try:
                from autotrade.data import QlibDataAdapter
                from autotrade.strategies.signal_generator import SignalGenerator

                self.log("开始执行预测计算...")

                # 1. 加载配置的股票列表
                if symbols is None:
                    symbols = self._get_default_universe()

                # 解析可能的指数代码
                symbols = self._resolve_symbols(symbols, refresh=refresh)

                # 2. 加载数据
                adapter = QlibDataAdapter(interval="1d", market="cn")
                end_date = datetime.now()
                # 确保有足够的数据生成特征 (Lookback)
                # 默认 SignalGenerator lookback 是 60
                start_date = end_date - timedelta(days=120)  

                df = adapter.load_data(symbols, start_date, end_date)

                should_fetch = df.empty
                
                # ... Data freshness check logic (simplified) ...
                if not df.empty:
                    try:
                        latest_date = df.index.get_level_values(0).max().date()
                        today = datetime.now().date()
                        if latest_date < today:
                             # Check if we checked recently
                             last_check = self.state.get("_last_data_check")
                             if not last_check or (datetime.now() - datetime.fromisoformat(last_check)).total_seconds() > 3600:
                                 should_fetch = True
                                 self.log(f"数据滞后 (最新: {latest_date})，尝试同步...")
                    except Exception as e:
                        self.log(f"检查数据时效性失败: {e}")

                if should_fetch:
                    try:
                        if df.empty:
                            self.log("本地无数据，正在从 AKShare 获取...")
                        
                        adapter.fetch_and_store(symbols, start_date, end_date, update_mode="append")
                        df = adapter.load_data(symbols, start_date, end_date)
                        self.state["_last_data_check"] = datetime.now().isoformat()
                    except Exception as e:
                        self.log(f"数据同步失败: {e}")

                if df.empty:
                    return {
                        "status": "error",
                        "message": "无法获取数据",
                        "predictions": [],
                    }

                # 3. Initialize SignalGenerator
                model_name = self.ml_config.get("model_name")
                # If model_name is None, SignalGenerator uses current default or fallback
                
                sig_gen = SignalGenerator(
                    symbols=symbols,
                    model_name=model_name,
                    models_dir="models",
                    market="cn",
                    top_k=self.ml_config.get("top_k", 3)
                )
                
                # Check model status
                if sig_gen.trainer is None:
                     self.log("警告: 未加载 ML 模型，使用默认 fallback 策略")
                
                # 4. Generate Predictions
                predictions = sig_gen.generate_latest_signals(df)
                
                # 5. Enrich with stock names
                stock_names = adapter.provider.get_stock_names(symbols)
                
                for p in predictions:
                    p["name"] = stock_names.get(p["symbol"], p["symbol"])
                    # Try to add current price
                    try:
                        p["price"] = float(df.xs(p["symbol"], level="symbol")["close"].iloc[-1])
                    except:
                        p["price"] = 0.0

                # Sort
                predictions.sort(key=lambda x: (x["signal"] in ["BUY", "SELL", "HOLD", "SUSPENDED"], x["score"]), reverse=True)
                
                self.log(f"预测完成: {len(predictions)} 只股票")
                
                model_used = sig_gen.model_name or "fallback"
                latest_data_date = df.index.get_level_values(0).max().strftime("%Y-%m-%d %H:%M:%S")

                result = {
                    "status": "success",
                    "model": model_used,
                    "date": latest_data_date,
                    "predictions": predictions,
                }
                
                if use_default_symbols:
                    self._save_prediction_cache(result)
                    
                return result

            except Exception as e:
                import traceback
                self.log(f"预测失败: {e}")
                traceback.print_exc()
                return {
                    "status": "error",
                    "message": str(e),
                    "predictions": [],
                }
    
    def _get_default_universe(self):
        """从配置文件读取默认股票池"""
        try:
            if default_config:
                return default_config.symbols
        except Exception as e:
            self.log(f"读取配置失败: {e}")
        return ["CSI300"]

    def run_backtest(self, params: dict):
        """Run a backtest using vectorbt in a separate thread."""

        def _backtest_task():
            try:
                self.log("Starting backtest (VectorBT)...")

                from autotrade.strategies.signal_generator import SignalGenerator
                from autotrade.backtest.engine import BacktestEngine
                from autotrade.data import QlibDataAdapter

                # 1. Parse dates and config
                start_date = datetime.strptime(params.get("start_date", "2024-01-01"), "%Y-%m-%d")
                end_date = datetime.strptime(params.get("end_date", "2025-12-31"), "%Y-%m-%d")
                
                # Always use configured symbols for now
                symbols = self._get_universe_symbols()
                symbols = self._resolve_symbols(symbols)
                self.log(f"回测股票池规模: {len(symbols)}")

                # 2. Load Data
                adapter = QlibDataAdapter(interval="1d", market="cn")
                # Add buffer for features
                data_start = start_date - timedelta(days=120) 
                
                self.log("正在加载数据...")
                df = adapter.load_data(symbols, data_start, end_date)

                if df.empty:
                    self.log("本地无数据，正在从 AKShare 获取...")
                    try:
                        adapter.fetch_and_store(symbols, data_start, end_date, update_mode="append")
                        df = adapter.load_data(symbols, data_start, end_date)
                    except Exception as e:
                        self.log(f"数据获取失败: {e}")
                        return

                if df.empty:
                    self.log("回测失败: 无数据")
                    return

                # 3. Setup Strategy and Engine
                model_name = params.get("model_name", self.ml_config.get("model_name"))
                top_k = int(params.get("top_k", self.ml_config.get("top_k", 3)))
                
                sig_gen = SignalGenerator(
                    symbols=symbols,
                    model_name=model_name,
                    market="cn",
                    top_k=top_k
                )
                
                engine = BacktestEngine(
                    signal_generator=sig_gen,
                    initial_capital=float(params.get("initial_capital", 100000.0)),
                    commission=0.0005,
                    slippage=0.0005
                )

                # 4. Run Backtest
                pf = engine.run(df)

                # 5. Generate Report and Save
                stats = engine.get_stats(pf)
                self.log(f"Backtest Complete. Total Return: {stats['total_return']:.2%}, Sharpe: {stats['sharpe_ratio']:.2f}")

                # Save report
                base_dir = os.path.dirname(os.path.abspath(__file__))
                logs_dir = os.path.join(os.path.dirname(base_dir), "logs")
                os.makedirs(logs_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = os.path.join(logs_dir, f"backtest_{timestamp}.html")

                # Generate VectorBT report with plots
                try:
                    import plotly.graph_objects as go

                    # Create subplots
                    fig = go.Figure()

                    # 1. Equity Curve
                    idx = pf.index()
                    equity = pf.value()
                    fig.add_trace(go.Scatter(
                        x=idx,
                        y=equity,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#2E86AB', width=2)
                    ))

                    fig.update_layout(
                        title=f"Backtest Report - {timestamp}",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value",
                        template="plotly_white",
                        height=400
                    )

                    # Save to HTML
                    fig.write_html(report_path)

                    # Append stats to the HTML file
                    with open(report_path, "r+") as f:
                        content = f.read()
                        # Insert stats before closing body
                        stats_html = f"""
                        <div style="margin: 20px; padding: 20px; background: #f5f5f5; border-radius: 5px;">
                            <h2>Performance Statistics</h2>
                            <table style="border-collapse: collapse; width: 100%;">
                                <tr style="background: #ddd;">
                                    <th style="padding: 10px; text-align: left;">Metric</th>
                                    <th style="padding: 10px; text-align: left;">Value</th>
                                </tr>
                                <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">Total Return</td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['total_return']:.2%}</td></tr>
                                <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">Sharpe Ratio</td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['sharpe_ratio']:.2f}</td></tr>
                                <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">Max Drawdown</td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['max_drawdown']:.2%}</td></tr>
                                <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">Total Trades</td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['total_trades']}</td></tr>
                                <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">Win Rate</td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{stats['win_rate']:.2%}</td></tr>
                                <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">Model</td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{sig_gen.model_name}</td></tr>
                                <tr><td style="padding: 8px; border-bottom: 1px solid #ddd;">Top K</td><td style="padding: 8px; border-bottom: 1px solid #ddd;">{sig_gen.top_k}</td></tr>
                            </table>
                        </div>
                        """
                        content = content.replace("</body>", stats_html + "</body>")
                        f.seek(0)
                        f.write(content)
                        f.truncate()

                except Exception as e:
                    # Fallback to simple HTML if plotting fails
                    with open(report_path, "w") as f:
                        f.write(f"<html><head><style>body{{font-family:Arial;}} table{{border-collapse:collapse;width:50%;}} th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd;}} th{{background:#ddd;}}</style></head><body>")
                        f.write(f"<h1>Backtest Report</h1>")
                        f.write(f"<p><strong>Date:</strong> {timestamp}</p>")
                        f.write(f"<p><strong>Model:</strong> {sig_gen.model_name}</p>")
                        f.write(f"<h2>Performance Statistics</h2>")
                        f.write(f"<table><tr><th>Metric</th><th>Value</th></tr>")
                        f.write(f"<tr><td>Total Return</td><td>{stats['total_return']:.2%}</td></tr>")
                        f.write(f"<tr><td>Sharpe Ratio</td><td>{stats['sharpe_ratio']:.2f}</td></tr>")
                        f.write(f"<tr><td>Max Drawdown</td><td>{stats['max_drawdown']:.2%}</td></tr>")
                        f.write(f"<tr><td>Total Trades</td><td>{stats['total_trades']}</td></tr>")
                        f.write(f"<tr><td>Win Rate</td><td>{stats['win_rate']:.2%}</td></tr>")
                        f.write(f"</table></body></html>")
                
                self.state["last_backtest"] = {
                    "tearsheet": f"/logs/backtest_{timestamp}.html", # Simplified path mapping
                    "timestamp": datetime.now().isoformat(),
                    "stats": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) for k,v in stats.items() if k != 'stats'}
                }

            except Exception as e:
                import traceback
                self.log(f"Backtest error: {e}")
                traceback.print_exc()

        thread = threading.Thread(target=_backtest_task, daemon=True)
        thread.start()
        return {"status": "backtest_started"}

    def initialize_and_start(self):
        """
        Initialize the system and start the default strategy/monitor.
        For A-share prediction, this ensures models are loaded and schedules daily checks.
        """
        self.log("Initializing AutoTrade-A...")
        
        # In A-share mode, we might not have a continuous loop strategy like crypto.
        # But we can start a scheduler or just ensure everything is ready.
        # For compatibility with web_server, we'll mark it as running.
        
        if self.is_running:
            self.log("Strategy is already running.")
            return {"status": "running"}

        # Define a simpler runner for A-shares that periodically updates or just sleeps/waits for commands
        def _daily_runner():
            import time
            while self.is_running:
                # Perform daily checks (e.g., is data updated?)
                # For now, just a heartbeat
                self.log("Heartbeat: System is active.")
                
                # Check for daily predictions if within market hours?
                # ...
                
                for _ in range(3600): # Sleep 1 hour, check stop flag
                    if not self.is_running: break
                    time.sleep(1)
        
        return self.start_strategy(runner=_daily_runner)


    def start_strategy(self, runner=None):
        """Start the strategy in a separate thread and begin monitoring."""
        if self.is_running:
            return False

        def run_target():
            try:
                self.log("Starting strategy...")
                if runner:
                    runner()
                elif self.active_strategy:
                    if hasattr(self.active_strategy, "run_all"):
                        self.active_strategy.run_all()
                    elif hasattr(self.active_strategy, "run"):
                        self.active_strategy.run()
                    else:
                        raise AttributeError(
                            "Strategy has no run() or run_all() method and no runner provided."
                        )
                else:
                    raise ValueError("No strategy set and no runner provided.")
            except Exception as e:
                self.log(f"Strategy error: {str(e)}")
            finally:
                self.is_running = False
                self.update_status("stopped")
                self.log("Strategy stopped.")

        self.is_running = True
        self.update_status("running")

        self.strategy_thread = threading.Thread(target=run_target, daemon=True)
        self.strategy_thread.start()
        return True

    def stop_strategy(self):
        """Stop the running strategy."""
        self.log("Stopping strategy...")
        self.update_status("stopping")

        # Force thread to end if possible and mark as stopped
        self.is_running = False
        self.update_status("stopped")
        self.log("Strategy stopped manually.")
        return {"status": "success", "message": "策略已停止"}

    def update_status(self, status: str):
        self.state["status"] = status
        self.state["last_update"] = datetime.now().isoformat()

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.state["logs"].append(f"[{timestamp}] {message}")
        print(f"[TM LOG] {message}")
        if len(self.state["logs"]) > 100:
            self.state["logs"].pop(0)

    def update_portfolio(self, cash, value, positions, market_status="unknown"):
        self.state["portfolio"] = {"cash": cash, "value": value, "positions": positions}
        self.state["market_status"] = market_status
        self.state["last_update"] = datetime.now().isoformat()

    def add_order(self, order_info):
        self.state["orders"].insert(0, order_info)
        if len(self.state["orders"]) > 50:
            self.state["orders"].pop()

    def get_state(self):
        return self.state

    # ==================== ML 策略相关 API ====================

    def set_ml_config(self, config: dict) -> dict:
        """
        设置 ML 策略配置

        Args:
            config: 包含 model_name, top_k, rebalance_period 等参数

        Returns:
            操作结果
        """
        if self.is_running:
            return {"status": "error", "message": "策略运行中，请先停止"}

        # 更新配置
        if "model_name" in config:
            self.ml_config["model_name"] = config["model_name"]
        if "top_k" in config:
            self.ml_config["top_k"] = int(config["top_k"])
        if "rebalance_period" in config:
            self.ml_config["rebalance_period"] = int(config["rebalance_period"])

        self.log(f"ML 配置更新: {self.ml_config}")
        return {"status": "success", "config": self.ml_config}

    def get_strategy_config(self) -> dict:
        """获取当前策略配置"""
        return {
            "strategy_type": "qlib_ml",  # 固定使用 ML 策略
            "ml_config": self.ml_config.copy(),
            "is_running": self.is_running,
            "status": self.state["status"],
            "universe_symbols": self._get_universe_symbols(),
        }

    def _get_universe_symbols(self) -> list[str]:
        """从配置文件读取默认股票池"""
        try:
            if default_config:
                return default_config.symbols
        except Exception as e:
            self.log(f"读取 universe 配置失败: {e}")
        return ["CSI300"]

    def list_models(self) -> list:
        """列出所有可用的 ML 模型"""
        return self.model_manager.list_models()

    def get_current_model(self) -> dict:
        """获取当前选择的模型信息"""
        model_name = self.model_manager.get_current_model()
        if model_name:
            info = self.model_manager.get_model_info(model_name)
            return {"status": "success", "model": info}
        return {"status": "success", "model": None, "message": "未选择模型"}

    def select_model(self, model_name: str) -> dict:
        """
        选择要使用的模型

        Args:
            model_name: 模型名称

        Returns:
            操作结果
        """
        success = self.model_manager.set_current_model(model_name)
        if success:
            # 同时更新 ML 配置
            self.ml_config["model_name"] = model_name
            self.log(f"模型选择: {model_name}")
            return {"status": "success", "model_name": model_name}
        return {"status": "error", "message": f"模型不存在: {model_name}"}

    def delete_model(self, model_name: str) -> dict:
        """
        删除模型

        Args:
            model_name: 模型名称

        Returns:
            操作结果
        """
        success = self.model_manager.delete_model(model_name)
        if success:
            self.log(f"删除模型: {model_name}")
            return {"status": "success", "model_name": model_name}
        return {"status": "error", "message": f"删除模型失败: {model_name}"}

    def start_model_training(self, config: dict = None) -> dict:
        """
        启动模型训练

        Args:
            config: 可选的训练配置

        Returns:
            操作结果
        """
        if self.training_status["in_progress"]:
            return {"status": "error", "message": "模型训练已在进行中"}

        def _training_task():
            try:
                from autotrade.data import QlibDataAdapter
                from autotrade.features import QlibFeatureGenerator
                from autotrade.models import LightGBMTrainer

                self.training_status["in_progress"] = True
                self.training_status["progress"] = 0
                self.training_status["message"] = "开始模型训练..."
                self.log("开始模型训练")

                # 默认配置 - A 股股票
                train_config = config or {}
                symbols = train_config.get("symbols")
                if not symbols:
                    symbols = self._get_universe_symbols()
                
                # Resolve symbols
                symbols = self._resolve_symbols(symbols)

                train_days = train_config.get("train_days", 252)
                target_horizon = train_config.get("target_horizon", 5)
                interval = "1d"  # A 股仅支持日线

                # 固定时间段配置
                train_start_str = "2010-01-01"
                valid_start_str = "2022-01-01"
                valid_end_str = "2023-12-31"

                # 1. 加载数据 (20%)
                self.training_status["progress"] = 10
                self.training_status["message"] = f"加载数据 ({interval})..."

                adapter = QlibDataAdapter(interval=interval, market="cn")
                
                # 数据加载结束时间为验证集结束时间
                end_date_obj = datetime.strptime(valid_end_str, "%Y-%m-%d")
                # 数据加载开始时间为训练集开始时间 - 60天 (warmup)
                start_date_obj = datetime.strptime(train_start_str, "%Y-%m-%d") - timedelta(days=60)

                # 尝试获取新数据
                try:
                    adapter.fetch_and_store(
                        symbols, start_date_obj, end_date_obj, update_mode="append"
                    )
                except Exception as e:
                    self.log(f"获取新数据失败（将使用现有数据）: {e}")

                df = adapter.load_data(symbols, start_date_obj, end_date_obj)
                self.training_status["progress"] = 20

                if df.empty:
                    raise ValueError("没有可用的数据")

                # 2. 生成特征 (40%)
                self.training_status["message"] = "生成特征..."
                feature_gen = QlibFeatureGenerator()
                features = feature_gen.generate(df)
                self.training_status["progress"] = 40

                # 3. 生成目标变量
                self.training_status["message"] = "准备训练数据..."
                import pandas as pd

                if isinstance(df.index, pd.MultiIndex):
                    close_prices = df["close"].unstack("symbol")
                    future_returns = close_prices.pct_change(target_horizon).shift(
                        -target_horizon
                    )
                    target = future_returns.stack().reindex(features.index)
                else:
                    target = (
                        df["close"].pct_change(target_horizon).shift(-target_horizon)
                    )
                    target = target.reindex(features.index)

                # 移除 NaN
                import numpy as np

                valid_mask = ~(features.isna().any(axis=1) | target.isna())
                features = features[valid_mask]
                target = target[valid_mask]
                self.training_status["progress"] = 50

                # 4. 训练模型 (80%)
                self.training_status["message"] = "训练模型..."

                # 分割训练/验证集 (按日期切分)
                split_date = datetime.strptime(valid_start_str, "%Y-%m-%d")
                
                # features index 是 (date, symbol) MultiIndex
                dates = features.index.get_level_values(0)
                
                train_mask = dates < split_date
                valid_mask = (dates >= split_date) & (dates <= end_date_obj)
                
                X_train = features.loc[train_mask]
                y_train = target.loc[train_mask]
                
                X_valid = features.loc[valid_mask]
                y_valid = target.loc[valid_mask]

                trainer = LightGBMTrainer(
                    model_name="lightgbm_rolling",
                    num_boost_round=300,
                )
                trainer.train(X_train, y_train, X_valid, y_valid)
                self.training_status["progress"] = 80

                # 5. 评估并保存 (100%)
                self.training_status["message"] = "保存模型..."
                metrics = trainer.evaluate(X_valid, y_valid)
                trainer.metadata.update(
                    {
                        "symbols": symbols,
                        "train_start_date": train_start_str,
                        "valid_start_date": valid_start_str,
                        "valid_end_date": valid_end_str,
                        "interval": interval,
                        "market": "cn",
                        "ic": metrics["ic"],
                        "icir": metrics["icir"],
                        "trained_via_ui": True,
                        "updated_at": datetime.now().isoformat(),
                        # 注意：不再保存 normalization_params
                        # 因为使用截面排名 (cross_sectional_rank) 时，
                        # 特征已经归一化到 [0, 1]，不需要额外的 Z-score 标准化
                    }
                )

                model_path = trainer.save()
                self.training_status["progress"] = 100
                self.training_status["message"] = (
                    f"完成！模型: {model_path.name}, IC: {metrics['ic']:.4f}"
                )

                self.log(f"模型训练完成: {model_path.name}, IC={metrics['ic']:.4f}")

            except Exception as e:
                import traceback

                self.training_status["message"] = f"错误: {e}"
                self.log(f"模型训练失败: {e}")
                traceback.print_exc()
            finally:
                self.training_status["in_progress"] = False

        # 启动后台任务
        thread = threading.Thread(target=_training_task, daemon=True)
        thread.start()

        return {"status": "started", "message": "模型训练已启动"}

    def get_training_status(self) -> dict:
        """获取模型训练状态"""
        return self.training_status

    def start_data_sync(self, config: dict = None) -> dict:
        """
        启动数据同步

        Args:
            config: 包含 symbols, days, interval, update_mode 等参数
        """
        if self.data_sync_status["in_progress"]:
            return {"status": "error", "message": "数据同步已在进行中"}

        def _data_sync_task():
            try:
                from autotrade.research.data import QlibDataAdapter

                self.data_sync_status["in_progress"] = True
                self.data_sync_status["progress"] = 0
                self.data_sync_status["message"] = "准备同步数据..."
                self.log("开始数据同步")

                sync_config = config or {}
                symbols = sync_config.get("symbols")
                if not symbols:
                    symbols = self._get_universe_symbols()
                
                # Resolve symbols
                symbols = self._resolve_symbols(symbols)

                days = sync_config.get("days", 365)
                interval = "1d"  # A 股仅支持日线
                update_mode = sync_config.get("update_mode", "append")

                adapter = QlibDataAdapter(interval=interval, market="cn")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                self.data_sync_status["message"] = (
                    f"正在从 AKShare 获取 {len(symbols)} 只股票的数据..."
                )
                self.data_sync_status["progress"] = 10

                adapter.fetch_and_store(
                    symbols, start_date, end_date, update_mode=update_mode
                )

                self.data_sync_status["progress"] = 100
                self.data_sync_status["message"] = (
                    f"成功同步 {len(symbols)} 只股票的数据 ({interval})"
                )
                self.log(f"数据同步完成: {len(symbols)} symbols")

            except Exception as e:
                self.data_sync_status["message"] = f"同步失败: {e}"
                self.log(f"数据同步失败: {e}")
            finally:
                self.data_sync_status["in_progress"] = False

        thread = threading.Thread(target=_data_sync_task, daemon=True)
        thread.start()

        return {"status": "started", "message": "数据同步已启动"}

    def get_data_sync_status(self) -> dict:
        """获取数据同步状态"""
        return self.data_sync_status
