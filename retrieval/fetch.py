import json
import time
import requests
from pathlib import Path

BASE_URL = "https://support.vagaro.com/api/v2/help_center/en-us/articles.json"
FIELDS = ["id", "title", "body", "section_id", "html_url", "label_names", "updated_at"]
LIMIT = None  # set to None to fetch all 809 articles


def fetch_articles(limit: int | None = None) -> list[dict]:
    articles = []
    url = BASE_URL + "?per_page=30"

    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for article in data["articles"]:
            articles.append({k: article[k] for k in FIELDS})
            if limit and len(articles) >= limit:
                return articles

        url = data["next_page"]
        time.sleep(0.3)

    return articles


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    articles = fetch_articles(limit=LIMIT)
    Path("data/articles.json").write_text(json.dumps(articles, indent=2))
    print(f"Fetched {len(articles)} articles -> data/articles.json")
