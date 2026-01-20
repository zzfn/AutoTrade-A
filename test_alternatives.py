
import akshare as ak
import pandas as pd
import time

def test_sina():
    print("Testing Sina Spot (stock_zh_a_spot)...")
    try:
        # 新浪接口，数据量可能较少或字段不同
        df = ak.stock_zh_a_spot() 
        print(f"Sina success: {len(df)} rows")
        print("Columns:", df.columns.tolist())
    except Exception as e:
        print(f"Sina failed: {e}")

def test_tx():
    print("\nTesting Tencent/Other (stock_zh_a_spot_tx)... (if available)")
    # Akshare might not have a direct full market TX API exposed simply as spot
    pass

if __name__ == "__main__":
    test_sina()
