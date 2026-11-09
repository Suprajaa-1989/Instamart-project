-- Q1 WHERE then HAVING: completed orders > 500, customers whose total of those orders > 8000
SELECT customer_id, COUNT(*) AS n_orders, ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- Q2 three-table join: revenue by city and product category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3 scalar subquery: products priced above the catalogue average
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4 LEFT JOIN: customers with no completed order
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5 CTE: customers whose total spend is above the average of customer totals
WITH cust AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders WHERE status = 'completed' GROUP BY customer_id
)
SELECT customer_id,
       ROUND(total_spend, 2) AS total_spend,
       (SELECT ROUND(AVG(total_spend), 2) FROM cust) AS avg_spend
FROM cust
WHERE total_spend > (SELECT AVG(total_spend) FROM cust)
ORDER BY total_spend DESC;

-- Q6 window functions: RANK, DENSE_RANK, ROW_NUMBER of customers inside each city
WITH cust AS (
    SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders o JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM cust
ORDER BY city, row_num;

-- Q7 running total: daily completed revenue with a cumulative column
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
ORDER BY day;
