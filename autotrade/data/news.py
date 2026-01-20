
import logging
import akshare as ak
import pandas as pd
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

class NewsManager:
    def __init__(self):
        self._cache = []
        self._last_update = None
        self._lock = threading.Lock()
        self._update_interval = 60 # seconds

    def get_latest_news(self, limit=50):
        """
        Get latest financial news.
        Uses a simple cache to avoid spamming the API.
        """
        current_time = time.time()
        
        should_update = False
        with self._lock:
            if self._last_update is None or (current_time - self._last_update > self._update_interval):
                should_update = True

        if should_update:
            self._update_cache()

        with self._lock:
            return self._cache[:limit]

    def _update_cache(self):
        try:
            logger.info("Fetching latest news from Cailian Press (stock_info_global_cls)...")
            # 财联社电报
            df = ak.stock_info_global_cls()
            
            if df.empty:
                return
            
            news_list = []
            for _, row in df.iterrows():
                # Columns: ['标题', '内容', '发布日期', '发布时间']
                # Construct timestamp string
                d = str(row.get("发布日期", ""))
                t = str(row.get("发布时间", ""))
                full_time = f"{d} {t}".strip()
                
                # Generate a pseudo ID from time if not present
                news_id = str(abs(hash(full_time + row.get("标题", ""))))[:8]

                news_item = {
                    "title": row.get("标题", "No Title"),
                    "content": row.get("内容", ""),
                    "time": full_time,
                    "id": news_id
                }
                news_list.append(news_item)
            
            # Ensure sorting by time descending (latest first)
            news_list.sort(key=lambda x: x["time"], reverse=True)

            with self._lock:
                self._cache = news_list
                self._last_update = time.time()
                
            logger.info(f"Updated news cache with {len(news_list)} items.")
            
        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
