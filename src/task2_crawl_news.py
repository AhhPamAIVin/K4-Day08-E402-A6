"""Task 2: crawl public Shopee help articles to structured JSON."""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from .task1_collect_legal_docs import fetch_page
except ImportError:  # Allow: python src/task2_crawl_news.py
    from task1_collect_legal_docs import fetch_page

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/188931?seo=1",
    "https://help.shopee.vn/portal/4/article/79233?seo=1",
    "https://help.shopee.vn/portal/4/article/189477?seo=1",
    "https://help.shopee.vn/portal/4/article/189473?seo=1",
    "https://help.shopee.vn/portal/4/article/79198?seo=1",
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _to_markdown(title: str, text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    output = [f"# {title}", ""]
    for line in lines:
        if line == title or "Shopee Trung tâm trợ giúp" in line:
            continue
        if re.match(r"^\d+(?:\.\d+)*[.)]?\s+", line):
            output.append(f"## {line}")
        else:
            output.append(line)
    return "\n\n".join(output)


async def crawl_article(url: str) -> dict:
    title, text = await asyncio.to_thread(fetch_page, url)
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "customer_role": "buyer",
        "content_markdown": _to_markdown(title, text),
    }


async def crawl_all():
    setup_directory()
    results = await asyncio.gather(*(crawl_article(url) for url in ARTICLE_URLS), return_exceptions=True)
    failures = []
    for index, (url, result) in enumerate(zip(ARTICLE_URLS, results), 1):
        if isinstance(result, Exception):
            failures.append(f"{url}: {result}")
            print(f"  [ERROR] Failed: {url}: {result}")
            continue
        filepath = DATA_DIR / f"article_{index:02d}.json"
        filepath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath}")
    if failures:
        raise RuntimeError("Không crawl đủ bài:\n" + "\n".join(failures))


if __name__ == "__main__":
    asyncio.run(crawl_all())
