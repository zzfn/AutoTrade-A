"""
TradeManager - A 股交易管理器

AutoTrade-A 专用：仅支持 A 股预测和回测
"""

import os
import threading
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lumibot.backtesting import YahooDataBacktesting

from autotrade.execution.strategies import QlibMLStrategy
from autotrade.research.models import ModelManager


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

    def set_strategy(self, strategy_instance):
        """Set the strategy instance to be managed."""
        self.active_strategy = strategy_instance

    def initialize_and_start(self):
        """
        初始化策略（A 股模式：仅用于预测信号，不进行实际交易）

        AutoTrade-A 不支持实时交易，仅提供预测信号和回测功能。
        """
        if self.is_running:
            return {"status": "already_running"}

        self.log("AutoTrade-A 已启动 - A 股预测信号模式")
        self.log("说明: 本系统仅提供预测信号和回测功能，不进行实际交易")
        self.update_status("ready")

        return {"status": "ready", "message": "系统就绪，可使用预测和回测功能"}

    def _resolve_symbols(self, symbols: list[str]) -> list[str]:
        """
        解析股票代码，支持指数代码扩展（如 CSI300）
        """
        if not symbols:
            return []

        resolved = []
        try:
            from autotrade.research.data.providers import DataProviderFactory
            # 获取 provider
            provider = DataProviderFactory.get_provider("cn")
            
            for s in symbols:
                s_upper = s.upper().strip()
                if s_upper in ["CSI300", "000300", "300", "HS300"]:
                    if hasattr(provider, "get_index_constituents"):
                        self.log(f"正在获取沪深300成分股...")
                        cons = provider.get_index_constituents("000300")
                        self.log(f"已获取 {len(cons)} 只成分股")
                        resolved.extend(cons)
                    else:
                        self.log("Provider 不支持 get_index_constituents")
                elif s_upper in ["CSI500", "000905", "500", "ZZ500"]:
                    if hasattr(provider, "get_index_constituents"):
                        self.log(f"正在获取中证500成分股...")
                        cons = provider.get_index_constituents("000905")
                        self.log(f"已获取 {len(cons)} 只成分股")
                        resolved.extend(cons)
                else:
                    resolved.append(s)
        except Exception as e:
            self.log(f"解析股票代码失败: {e}")
            # Fallback to original
            return symbols

        # 去重
        return sorted(list(set(resolved)))

    def get_latest_predictions(self, symbols: list[str] | None = None) -> dict:
        """
        获取最新的预测信号

        Args:
            symbols: 股票代码列表，如果为 None 则使用默认配置

        Returns:
            包含预测信号的字典
        """
        try:
            from autotrade.research.data import QlibDataAdapter
            from autotrade.research.features import QlibFeatureGenerator

            self.log("正在获取预测信号...")

            # 1. 加载配置的股票列表
            if symbols is None:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(base_dir, "../configs/universe.yaml")
                symbols = ["CSI300"]  # 默认 A 股 CSI300

                try:
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            config = yaml.safe_load(f)
                            if config and "symbols" in config:
                                symbols = config["symbols"]
                except Exception as e:
                    self.log(f"读取配置失败，使用默认股票列表: {e}")

            # 解析可能的指数代码
            symbols = self._resolve_symbols(symbols)

            # 2. 加载数据
            adapter = QlibDataAdapter(interval="1d", market="cn")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # 获取60天数据用于特征计算

            df = adapter.load_data(symbols, start_date, end_date)

            df = adapter.load_data(symbols, start_date, end_date)

            should_fetch = df.empty
            
            # 检查数据是否过旧
            if not df.empty:
                try:
                    latest_date = df.index.get_level_values(0).max().date()
                    today = datetime.now().date()
                    # 如果最新数据不是今天，且距离上次检查超过 1 小时（避免频繁请求），尝试同步
                    if latest_date < today:
                        last_check = self.state.get("_last_data_check")
                        # 检查间隔：3600秒 (1小时)
                        if not last_check or (datetime.now() - datetime.fromisoformat(last_check)).total_seconds() > 3600:
                            should_fetch = True
                            self.log(f"数据滞后 (最新: {latest_date})，尝试同步...")
                except Exception as e:
                    self.log(f"检查数据时效性失败: {e}")

            if should_fetch:
                try:
                    # 获取数据 (append 模式支持增量更新)
                    if df.empty:
                        self.log("本地无数据，正在从 AKShare 获取...")
                    
                    adapter.fetch_and_store(symbols, start_date, end_date, update_mode="append")
                    df = adapter.load_data(symbols, start_date, end_date)
                    
                    # 更新检查时间
                    self.state["_last_data_check"] = datetime.now().isoformat()
                except Exception as e:
                    self.log(f"数据同步失败: {e}")

            if df.empty:
                return {
                    "status": "error",
                    "message": "无法获取数据",
                    "predictions": [],
                }

            # 3. 生成特征
            feature_gen = QlibFeatureGenerator(normalize=True)
            features = feature_gen.generate(df)

            # 4. 加载模型
            model_name = self.ml_config.get("model_name")
            if model_name is None:
                model_name = self.model_manager.get_current_model()

            if not model_name:
                return {
                    "status": "error",
                    "message": "未找到可用模型，请先训练模型",
                    "predictions": [],
                }

            model_info = self.model_manager.get_model_info(model_name)
            if not model_info or "path" not in model_info:
                return {
                    "status": "error",
                    "message": f"模型 {model_name} 不可用",
                    "predictions": [],
                }

            # 5. 加载模型并预测
            import pickle
            import numpy as np

            model_path = Path(model_info["path"]) / "model.pkl"
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # 获取股票名称映射
            stock_names = adapter.provider.get_stock_names(symbols)

            # 获取每只股票最新的特征
            predictions = []
            latest_date = features.index.get_level_values(0).max()

            for symbol in symbols:
                try:
                    # 获取该股票最新日期的特征
                    if (latest_date, symbol) in features.index:
                        symbol_features = features.loc[(latest_date, symbol)]
                        X = symbol_features.values.reshape(1, -1)
                        pred_score = model.predict(X)[0]

                        # 转换为信号
                        if pred_score > 0.01:
                            signal = "BUY"
                        elif pred_score < -0.01:
                            signal = "SELL"
                        else:
                            signal = "HOLD"

                        predictions.append({
                            "symbol": symbol,
                            "name": stock_names.get(symbol, symbol),
                            "signal": signal,
                            "score": float(pred_score),
                            "confidence": abs(float(pred_score)) * 100,
                            "date": latest_date.strftime("%Y-%m-%d"),
                        })
                except Exception as e:
                    self.log(f"预测 {symbol} 失败: {e}")

            # 按得分排序
            predictions.sort(key=lambda x: x["score"], reverse=True)

            self.log(f"预测完成: {len(predictions)} 只股票")

            return {
                "status": "success",
                "model": model_name,
                "date": latest_date.strftime("%Y-%m-%d"),
                "predictions": predictions,
            }

        except Exception as e:
            import traceback
            self.log(f"预测失败: {e}")
            traceback.print_exc()
            return {
                "status": "error",
                "message": str(e),
                "predictions": [],
            }

    def run_backtest(self, params: dict):
        """Run a backtest in a separate thread."""

        def _backtest_task():
            try:
                self.log("Starting backtest...")

                # 1. Parse dates
                backtesting_start = datetime.strptime(
                    params.get("start_date", "2023-01-01"), "%Y-%m-%d"
                )
                backtesting_end = datetime.strptime(
                    params.get("end_date", "2023-01-31"), "%Y-%m-%d"
                )

                # 2. Parse symbols (clean up quotes and spaces)
                symbol_input = params.get("symbol", "CSI300")
                symbols = [
                    s.strip().replace('"', "").replace("'", "")
                    for s in symbol_input.split(",")
                    if s.strip()
                ]
                if not symbols:
                    symbols = ["CSI300"]

                # Resolve symbols (handle CSI300 etc.)
                symbols = self._resolve_symbols(symbols)

                # 3. Parse interval - A 股仅支持日线
                interval = "1d"

                # 4. A 股市场固定
                market = "cn"

                self.log(
                    f"Backtesting [A股] {symbols} from {backtesting_start} to {backtesting_end} (Interval: {interval})"
                )

                # 5. Execute backtest with ML strategy
                strategy_class = QlibMLStrategy

                try:
                    lumibot_interval = "day"

                    # 如果未指定模型，使用当前最优模型
                    model_name = params.get("model_name", self.ml_config.get("model_name"))
                    if model_name is None:
                        model_name = self.model_manager.get_current_model()

                    backtest_params = {
                        "symbols": symbols,
                        "model_name": model_name,
                        "top_k": params.get("top_k", self.ml_config.get("top_k", 3)),
                        "rebalance_period": params.get("rebalance_period", 1),
                        "sleeptime": "0S",
                        "timestep": "1D",
                        "market": market,
                    }

                    # A 股使用第一个股票作为基准
                    benchmark = symbols[0] if symbols else "000001.SZ"

                    # Start time to identify new files
                    start_time = datetime.now()

                    strategy_class.backtest(
                        YahooDataBacktesting,
                        backtesting_start,
                        backtesting_end,
                        benchmark_asset=benchmark,
                        parameters=backtest_params,
                        time_unit=lumibot_interval,
                    )

                    # Find newly generated reports in logs/
                    import glob

                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    logs_dir = os.path.join(os.path.dirname(base_dir), "logs")

                    # Look for html files generated recently
                    html_files = glob.glob(os.path.join(logs_dir, "*.html"))
                    new_reports = []
                    for f in html_files:
                        if (
                            os.path.getmtime(f) >= start_time.timestamp() - 5
                        ):  # 5s buffer
                            new_reports.append(os.path.basename(f))

                    tearsheet = next((f for f in new_reports if "tearsheet" in f), None)
                    trades_report = next(
                        (f for f in new_reports if "trades" in f), None
                    )

                    if tearsheet or trades_report:
                        self.state["last_backtest"] = {
                            "tearsheet": f"/reports/{tearsheet}" if tearsheet else None,
                            "trades": f"/reports/{trades_report}"
                            if trades_report
                            else None,
                            "timestamp": datetime.now().isoformat(),
                        }
                        self.log(f"Backtest reports generated: {new_reports}")

                    self.log("Backtest finished successfully.")
                except Exception as e:
                    import traceback

                    self.log(f"Backtest execution failed: {e}")
                    print(traceback.format_exc())

            except Exception as e:
                self.log(f"Backtest error: {e}")

        # Start backtest in background thread
        thread = threading.Thread(target=_backtest_task, daemon=True)
        thread.start()
        return {"status": "backtest_started"}

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
        }

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
                from autotrade.research.data import QlibDataAdapter
                from autotrade.research.features import QlibFeatureGenerator
                from autotrade.research.models import LightGBMTrainer

                self.training_status["in_progress"] = True
                self.training_status["progress"] = 0
                self.training_status["message"] = "开始模型训练..."
                self.log("开始模型训练")

                # 默认配置 - A 股股票
                train_config = config or {}
                symbols = train_config.get("symbols", ["CSI300"])
                
                # Resolve symbols
                symbols = self._resolve_symbols(symbols)

                train_days = train_config.get("train_days", 252)
                target_horizon = train_config.get("target_horizon", 5)
                interval = "1d"  # A 股仅支持日线

                # 1. 加载数据 (20%)
                self.training_status["progress"] = 10
                self.training_status["message"] = f"加载数据 ({interval})..."

                adapter = QlibDataAdapter(interval=interval, market="cn")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=train_days + 60)

                # 尝试获取新数据
                try:
                    adapter.fetch_and_store(
                        symbols, start_date, end_date, update_mode="append"
                    )
                except Exception as e:
                    self.log(f"获取新数据失败（将使用现有数据）: {e}")

                df = adapter.load_data(symbols, start_date, end_date)
                self.training_status["progress"] = 20

                if df.empty:
                    raise ValueError("没有可用的数据")

                # 2. 生成特征 (40%)
                self.training_status["message"] = "生成特征..."
                feature_gen = QlibFeatureGenerator(normalize=True)
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

                # 分割训练/验证集 (80/20)
                split_idx = int(len(features) * 0.8)
                X_train, X_valid = features.iloc[:split_idx], features.iloc[split_idx:]
                y_train, y_valid = target.iloc[:split_idx], target.iloc[split_idx:]

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
                        "train_days": train_days,
                        "interval": interval,
                        "market": "cn",
                        "ic": metrics["ic"],
                        "icir": metrics["icir"],
                        "trained_via_ui": True,
                        "updated_at": datetime.now().isoformat(),
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
                symbols = sync_config.get("symbols", ["CSI300"])
                
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
