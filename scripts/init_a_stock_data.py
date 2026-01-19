#!/usr/bin/env python3
"""
A股数据初始化脚本

从AKShare获取A股历史数据并存储为Qlib格式

用法:
    uv run python scripts/init_a_stock_data.py
    uv run python scripts/init_a_stock_data.py --start-date 2023-01-01 --end-date 2024-12-31
    uv run python scripts/init_a_stock_data.py --symbols 000001.SZ,600000.SH
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from autotrade.research.data.providers import AKShareDataProvider
from autotrade.research.data.qlib_adapter import QlibDataAdapter


def load_universe_config(config_path: Path) -> list[str]:
    """从配置文件加载A股股票池
    
    优先读取 symbols，如果没有则 fallback 到 cn_stocks
    """
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}")
        return []

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 优先读取 symbols，fallback 到 cn_stocks
    return config.get("symbols", config.get("cn_stocks", []))


def main():
    parser = argparse.ArgumentParser(description="初始化A股历史数据")
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d"),
        help="开始日期 (YYYY-MM-DD)，默认2年前",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="结束日期 (YYYY-MM-DD)，默认今天",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="股票代码，逗号分隔 (如 000001.SZ,600000.SH)，默认从 configs/universe.yaml 读取",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/qlib",
        help="数据存储目录",
    )
    parser.add_argument(
        "--adjust",
        type=str,
        default="qfq",
        choices=["qfq", "hfq", ""],
        help="复权类型: qfq-前复权, hfq-后复权, 空-不复权",
    )

    args = parser.parse_args()

    # 解析日期
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # 获取股票列表
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        config_path = project_root / "configs" / "universe.yaml"
        symbols = load_universe_config(config_path)
        if not symbols:
            logger.error("未指定股票，请使用 --symbols 或在 configs/universe.yaml 中配置 cn_stocks")
            sys.exit(1)

    logger.info(f"📊 初始化A股数据")
    logger.info(f"   股票数量: {len(symbols)}")
    logger.info(f"   日期范围: {args.start_date} ~ {args.end_date}")
    logger.info(f"   复权类型: {args.adjust or '不复权'}")
    logger.info(f"   数据目录: {args.data_dir}")

    # 创建数据提供者和适配器
    provider = AKShareDataProvider(adjust=args.adjust)
    adapter = QlibDataAdapter(
        data_dir=args.data_dir,
        provider=provider,
        interval="1d",
        market="cn",
    )

    # 逐个获取数据（显示进度条）
    success_count = 0
    fail_count = 0
    failed_symbols = []

    for symbol in tqdm(symbols, desc="获取数据"):
        try:
            result = adapter.fetch_and_store(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                update_mode="replace",
            )
            if result.get("status") == "success":
                success_count += 1
                logger.debug(f"✅ {symbol}: {result.get('total_records', 0)} 条记录")
            else:
                fail_count += 1
                failed_symbols.append(symbol)
                logger.warning(f"❌ {symbol}: {result.get('message', '未知错误')}")
        except Exception as e:
            fail_count += 1
            failed_symbols.append(symbol)
            logger.error(f"❌ {symbol}: {e}")

    # 输出统计
    print("\n" + "=" * 50)
    print(f"📈 数据初始化完成")
    print(f"   成功: {success_count} 只股票")
    print(f"   失败: {fail_count} 只股票")
    if failed_symbols:
        print(f"   失败列表: {', '.join(failed_symbols)}")
    print(f"   数据目录: {Path(args.data_dir) / 'cn' / '1d'}")
    print("=" * 50)

    # 显示可用数据
    available = adapter.get_available_symbols()
    if available:
        print(f"\n可用股票 ({len(available)} 只):")
        for s in sorted(available)[:10]:
            date_range = adapter.get_date_range(s)
            if date_range:
                print(f"  - {s}: {date_range[0].strftime('%Y-%m-%d')} ~ {date_range[1].strftime('%Y-%m-%d')}")
        if len(available) > 10:
            print(f"  ... 还有 {len(available) - 10} 只股票")


if __name__ == "__main__":
    main()
