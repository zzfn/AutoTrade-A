
import pandas as pd
import os

cache_path = "data/cache/index_constituents.parquet"

if os.path.exists(cache_path):
    try:
        df = pd.read_parquet(cache_path)
        print(f"Cache file exists at {cache_path}")
        print(f"Columns: {df.columns}")
        print("Counts per index:")
        print(df["index_code"].value_counts())
    except Exception as e:
        print(f"Error reading cache: {e}")
else:
    print(f"Cache file does not exist at {cache_path}")
