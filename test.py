import akshare as ak

stock_zh_a_hist_df = ak.stock_zh_a_hist(
    symbol="002050",
    period="daily",
    start_date="20260101",
    end_date="20260119",
    adjust="qfq"
)
print(stock_zh_a_hist_df)