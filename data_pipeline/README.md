# Module 1 — Data Pipeline

`pipeline.py` scrapes the first five paginated pages of `books.toscrape.com`, which yields at least 60 books across the catalogue. It captures title, GBP price, star rating, availability, and category; parses the fields defensively; applies the fixed project conversion `1 GBP = 105.50 INR`; and writes a normalized SQLite database.

The schema is:

```text
categories(category_id PRIMARY KEY, category_name UNIQUE)
books(book_id PRIMARY KEY, title, price_gbp, price_inr, rating,
      in_stock, category_id REFERENCES categories(category_id))
```

Cleaning decisions are explicit in the code. Invalid numeric prices are median-imputed, unknown star ratings are median-imputed and clipped to 1–5, and rows with an empty title/category are dropped. Availability is converted to a boolean using the word `In stock`; SQLite stores it as `0`/`1`.

Run:

```bash
python data_pipeline/pipeline.py
```

The script writes `books.db`, executes the five required SQL query patterns in `queries.sql`, prints their outputs, reads two results with `pd.read_sql`, and reproduces the join with `pd.merge`.


