-- SELECT / WHERE
SELECT title, price_gbp, rating FROM books WHERE rating >= 4;

-- ORDER BY / LIMIT
SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;

-- DISTINCT
SELECT DISTINCT rating FROM books ORDER BY rating;

-- IN / BETWEEN
SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30;

-- JOIN
SELECT c.category_name, b.title, b.rating, b.price_inr
FROM books AS b
JOIN categories AS c ON c.category_id = b.category_id
ORDER BY b.rating DESC, b.price_inr DESC
LIMIT 10;


