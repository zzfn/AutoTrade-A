import os
import uvicorn


def main():
    """
    启动 AutoTrade Web 服务
    所有的交互（训练、回测、交易）现在都通过 Web UI 进行。
    """
    print("Starting AutoTrade Web Server (FastAPI + React)...")
    
    # 获取配置（支持环境变量）
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    # 默认开启热重载以方便开发
    reload = os.getenv("RELOAD", "true").lower() == "true"

    uvicorn.run(
        "autotrade.web_server:app", 
        host=host, 
        port=port, 
        reload=reload
    )


if __name__ == "__main__":
    main()
