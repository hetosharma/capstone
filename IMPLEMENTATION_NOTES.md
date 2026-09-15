# Implementation notes

The capstone is organized as three independently runnable Python modules with one consolidated `requirements.txt`.

- The data pipeline uses the public `books.toscrape.com` practice site, a fixed `1 GBP = 105.50 INR` conversion, defensive parsing, normalized SQLite tables, and SQL/pandas query equivalence.
- The analytics module includes the committed `analytics/titanic.csv` fallback, train-only preprocessing, classifier comparisons, imbalance experiments, Random Forest tuning with OOB scoring, and a regression side task.
- The support assistant is offline-first: `MOCK_LLM` defaults to `1`, embeddings and Chroma retrieval remain real, and the optional live path is isolated behind `MOCK_LLM=0` and `GROQ_API_KEY`.

Static validation completed before publication: all Python source files compile with the bundled Python runtime, the Titanic fallback contains 891 data rows, and the repository contains all eight policy documents plus the Dockerfile and prompt template.


