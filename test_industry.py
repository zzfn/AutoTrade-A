
import akshare as ak
import pandas as pd
import time

def check_industry():
    print("Fetching industry list...")
    try:
        # 获取行业板块列表
        df_ind = ak.stock_board_industry_name_em()
        print(f"Got {len(df_ind)} industries.")
        print(df_ind.head())
        
        # Try fetching constituents of the first industry
        first_ind = df_ind.iloc[0]['板块名称']
        print(f"\nFetching constituents for: {first_ind}")
        df_cons = ak.stock_board_industry_cons_em(symbol=first_ind)
        print(f"Got {len(df_cons)} stocks.")
        print(df_cons.head())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_industry()
