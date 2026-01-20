
import akshare as ak
import pandas as pd
import json

def check_spot_data():
    print("Fetching spot data...")
    try:
        df = ak.stock_zh_a_spot_em()
        print("Columns:", df.columns.tolist())
        print("First row:", df.iloc[0].to_dict())
    except Exception as e:
        print("Error fetching spot data:", e)

def check_industry_data():
    print("\nFetching industry data...")
    try:
        df = ak.stock_board_industry_name_em()
        print("Columns:", df.columns.tolist())
        print("First 5 rows:", df.head(5).to_dict('records'))
    except Exception as e:
        print("Error fetching industry data:", e)

if __name__ == "__main__":
    check_spot_data()
    check_industry_data()
