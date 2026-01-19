import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="AutoTrade CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command (Web Server)
    subparsers.add_parser("run", help="Start web server")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, help="Path to config file")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("--symbols", nargs="+", help="List of symbols")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--start", type=str, default="2024-01-01", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", type=str, default="2025-01-01", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.command == "run" or args.command is None:
        print("Starting AutoTrade Web Server (FastAPI + React)...")
        # Using reload=True for dev experience as requested
        uvicorn.run("autotrade.web_server:app", host="0.0.0.0", port=8000, reload=True)
    
    elif args.command == "train":
        print("Starting Model Training...")
        from autotrade.trade_manager import TradeManager
        import asyncio
        tm = TradeManager()
        # Run in a simple loop
        result = tm.start_model_training(config={} if not args.config else {"config_path": args.config})
        print(f"Training initiated: {result}")
        # Note: In a real CLI we might want to wait for completion or show progress. 
        # But TradeManager runs in background thread. 
        # For simple CLI usage, we might just exit or wait.
        # Let's just wait a bit to show it started.
        import time
        time.sleep(2)
        print("Training running in background (check logs or web UI for status)...")

    elif args.command == "predict":
        print("Generating Predictions...")
        from autotrade.trade_manager import TradeManager
        tm = TradeManager()
        result = tm.get_latest_predictions(symbols=args.symbols)
        print(result)

    elif args.command == "backtest":
        print("Running Backtest...")
        from autotrade.trade_manager import TradeManager
        tm = TradeManager()
        params = {
            "start_date": args.start,
            "end_date": args.end
        }
        result = tm.run_backtest(params)
        print(f"Backtest initiated: {result}")
        import time
        time.sleep(2)
        print("Backtest running in background (check logs or web UI for status)...")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
