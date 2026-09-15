"""End-to-end scraping, cleaning, SQLite loading, and query demonstration."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "books.db"
RATE_GBP_TO_INR = 105.50
BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
RATING_MAP = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def scrape_pages(page_count: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "ZeptoCapstone/1.0 educational scraper"})
    for page_number in range(1, page_count + 1):
        response = session.get(BASE_URL.format(page_number), timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for card in soup.select("article.product_pod"):
            title_node = card.select_one("h3 a")
            price_node = card.select_one(".price_color")
            stock_node = card.select_one(".availability")
            rating_node = card.select_one("p.star-rating")
            category_node = card.select_one(".product_price")
            # The listing page does not expose the category directly. The detail
            # page is the authoritative source and is fetched once per book.
            detail_url = card.select_one("h3 a")["href"]
            detail_url = requests.compat.urljoin(response.url, detail_url)
            detail = session.get(detail_url, timeout=30)
            detail.raise_for_status()
            detail_soup = BeautifulSoup(detail.text, "html.parser")
            breadcrumb = detail_soup.select("ul.breadcrumb li a")
            category = breadcrumb[-1].get_text(strip=True) if breadcrumb else "Unknown"
            rows.append(
                {
                    "title_raw": title_node.get("title", "") if title_node else "",
                    "price_raw": price_node.get_text(" ", strip=True) if price_node else "",
                    "star_rating_raw": " ".join(rating_node.get("class", [])[1:]) if rating_node else "",
                    "availability_raw": stock_node.get_text(" ", strip=True) if stock_node else "",
                    "category": category,
                }
            )
    return pd.DataFrame(rows)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["title"] = df["title_raw"].fillna("").astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df = df[(df["title"] != "") & (df["category"] != "")].copy()

    df["price_gbp"] = (
        df["price_raw"].astype(str).str.replace("£", "", regex=False).str.replace(",", "", regex=False)
    )
    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    def parse_rating(value: str) -> float:
        words = str(value).lower().split()
        return float(next((RATING_MAP[word] for word in words if word in RATING_MAP), float("nan")))

    df["rating"] = df["star_rating_raw"].map(parse_rating)
    df["rating"] = df["rating"].fillna(df["rating"].median()).clip(1, 5).astype(int)
    df["in_stock"] = df["availability_raw"].str.contains("in stock", case=False, na=False)
    df["price_inr"] = (df["price_gbp"] * RATE_GBP_TO_INR).round(2)
    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def load_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            CREATE TABLE categories(
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE books(
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK(in_stock IN (0,1)),
                category_id INTEGER NOT NULL REFERENCES categories(category_id)
            );
            """
        )
        categories = sorted(df["category"].unique().tolist())
        conn.executemany("INSERT INTO categories(category_name) VALUES (?)", [(x,) for x in categories])
        lookup = pd.read_sql_query("SELECT category_id, category_name FROM categories", conn)
        category_ids = dict(zip(lookup.category_name, lookup.category_id))
        records = [
            (row.title, float(row.price_gbp), float(row.price_inr), int(row.rating), int(row.in_stock), category_ids[row.category])
            for row in df.itertuples(index=False)
        ]
        conn.executemany(
            "INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id) VALUES (?,?,?,?,?,?)",
            records,
        )
        conn.commit()


def run_queries(db_path: Path = DB_PATH) -> None:
    query_text = (ROOT / "queries.sql").read_text(encoding="utf-8")
    queries = [q.strip() for q in query_text.split(";") if q.strip() and not q.lstrip().startswith("--")]
    # Keep comments attached to statements in the source file, but execute the SQL body.
    statements = []
    for chunk in query_text.split(";"):
        body = "\n".join(line for line in chunk.splitlines() if not line.strip().startswith("--")).strip()
        if body:
            statements.append(body)
    with sqlite3.connect(db_path) as conn:
        for index, statement in enumerate(statements, 1):
            result = pd.read_sql_query(statement, conn)
            print(f"\nSQL query {index}:\n{statement}\n{result.to_string(index=False)}")
        join_sql = statements[-1]
        sql_join = pd.read_sql_query(join_sql, conn)
        books = pd.read_sql_query("SELECT * FROM books", conn)
        cats = pd.read_sql_query("SELECT * FROM categories", conn)
        pandas_join = books.merge(cats, on="category_id")[
            ["category_name", "title", "rating", "price_inr"]
        ].sort_values(["rating", "price_inr"], ascending=[False, False]).head(10)
        print("\nJOIN equality via pd.merge:", sql_join.reset_index(drop=True).equals(pandas_join.reset_index(drop=True)))


def main() -> None:
    raw = scrape_pages()
    clean = clean_data(raw)
    if len(clean) < 60:
        raise RuntimeError(f"Expected at least 60 books; scraped {len(clean)}")
    print(f"Scraped {len(clean)} books across {clean['category'].nunique()} categories")
    load_sqlite(clean)
    run_queries()


if __name__ == "__main__":
    main()


