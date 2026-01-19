"""
AutoTrade-A Web Server - A 股预测信号系统

提供 Web UI 和 API 接口
"""

import asyncio
import logging
import signal
import threading
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from autotrade.trade_manager import TradeManager


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Monkey patch signal.signal to prevent ValueError in non-main threads
_original_signal = signal.signal


def _thread_safe_signal(signum, handler):
    if threading.current_thread() is not threading.main_thread():
        logging.warning(
            f"Ignored signal registration for {signum} from non-main thread."
        )
        return
    return _original_signal(signum, handler)


signal.signal = _thread_safe_signal


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理系统生命周期的上下文管理器"""
    logger.info("正在启动 AutoTrade-A 系统...")

    # 初始化 TradeManager（A 股模式，无需连接交易所）
    async def startup_task():
        try:
            await asyncio.sleep(1)
            logger.info("初始化 A 股预测系统...")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tm.initialize_and_start)
            logger.info(f"系统启动结果: {result}")
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    init_task = asyncio.create_task(startup_task())

    yield  # 应用运行期间

    # 应用关闭时的清理逻辑
    logger.info("正在关闭系统...")
    init_task.cancel()
    try:
        await asyncio.wait_for(asyncio.to_thread(tm.stop_strategy), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("策略清理超时，强制关闭...")
    except Exception as e:
        logger.error(f"清理过程中发生错误: {e}")


app = FastAPI(lifespan=lifespan)


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")
TEMPLATES_DIR = os.path.join(UI_DIR, "templates")
STATIC_DIR = os.path.join(UI_DIR, "static")

# Mounts
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount(
    "/reports",
    StaticFiles(directory=os.path.join(os.path.dirname(BASE_DIR), "logs")),
    name="reports",
)
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Manager
tm = TradeManager()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/backtest", response_class=HTMLResponse)
async def read_backtest(request: Request):
    return templates.TemplateResponse(request, "backtest.html")


@app.post("/api/run_backtest")
async def run_backtest(request: Request):
    params = await request.json()
    return tm.run_backtest(params)


# ==================== 预测 API ====================


@app.get("/api/predict")
async def get_predictions(refresh: bool = False):
    """获取最新的预测信号"""
    return tm.get_latest_predictions(refresh=refresh)


@app.post("/api/predict")
async def get_predictions_with_symbols(request: Request):
    """获取指定股票的预测信号"""
    try:
        data = await request.json()
        symbols = data.get("symbols")
        refresh = data.get("refresh", False)
        return tm.get_latest_predictions(symbols, refresh=refresh)
    except Exception:
        return tm.get_latest_predictions()


# ==================== ML 策略相关 API ====================


@app.get("/api/strategy/config")
async def get_strategy_config():
    """获取当前策略配置"""
    return tm.get_strategy_config()


@app.post("/api/strategy/start")
async def start_strategy():
    """手动启动策略"""
    return tm.initialize_and_start()


@app.post("/api/strategy/stop")
async def stop_strategy():
    """手动停止策略"""
    return tm.stop_strategy()


@app.post("/api/strategy/ml_config")
async def set_ml_config(request: Request):
    """设置 ML 策略配置"""
    config = await request.json()
    return tm.set_ml_config(config)


@app.get("/api/models")
async def list_models():
    """列出所有可用的 ML 模型"""
    return {"status": "success", "models": tm.list_models()}


@app.get("/api/models/current")
async def get_current_model():
    """获取当前选择的模型"""
    return tm.get_current_model()


@app.post("/api/models/select")
async def select_model(request: Request):
    """选择要使用的模型"""
    data = await request.json()
    model_name = data.get("model_name")
    if not model_name:
        return {"status": "error", "message": "缺少 model_name 参数"}
    return tm.select_model(model_name)


@app.post("/api/models/delete")
async def delete_model(request: Request):
    """删除模型"""
    data = await request.json()
    model_name = data.get("model_name")
    if not model_name:
        return {"status": "error", "message": "缺少 model_name 参数"}
    return tm.delete_model(model_name)


@app.post("/api/models/train")
async def start_model_training(request: Request):
    """启动模型训练"""
    try:
        config = await request.json()
    except Exception:
        config = None
    return tm.start_model_training(config)


@app.get("/api/models/train/status")
async def get_training_status():
    """获取模型训练状态"""
    return tm.get_training_status()


@app.post("/api/data/sync")
async def start_data_sync(request: Request):
    """启动数据同步"""
    try:
        config = await request.json()
    except Exception:
        config = None
    return tm.start_data_sync(config)


@app.get("/api/data/sync/status")
async def get_data_sync_status():
    """获取数据同步状态"""
    return tm.get_data_sync_status()


# ==================== 模型管理页面 ====================


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """模型管理页面"""
    return templates.TemplateResponse(request, "models.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            try:
                state = {
                    "status": tm.state.get("status", "unknown"),
                    "logs": list(tm.state.get("logs", [])),
                    "orders": [dict(o) for o in tm.state.get("orders", [])],
                    "portfolio": {
                        "cash": tm.state.get("portfolio", {}).get("cash", 0.0),
                        "value": tm.state.get("portfolio", {}).get("value", 0.0),
                        "positions": [dict(p) for p in tm.state.get("portfolio", {}).get("positions", [])],
                    },
                    "market_status": tm.state.get("market_status", "unknown"),
                    "last_update": tm.state.get("last_update"),
                    "strategy_config": tm.get_strategy_config(),
                    "training_status": tm.get_training_status().copy() if isinstance(tm.get_training_status(), dict) else tm.get_training_status(),
                    "data_sync_status": tm.get_data_sync_status().copy() if isinstance(tm.get_data_sync_status(), dict) else tm.get_data_sync_status(),
                }

                await websocket.send_json(state)
            except (WebSocketDisconnect, RuntimeError):
                logger.info("WebSocket connection closed")
                break
            except Exception as e:
                logger.error(f"Error preparing or sending WS data: {e}")

            await asyncio.sleep(1)  # 1Hz update
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WS Connection Error: {e}")
