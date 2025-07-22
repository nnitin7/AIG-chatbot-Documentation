SELECT
    query_id,
    user_id,
    response_latency,
    AVG(response_latency) OVER (PARTITION BY user_id ORDER BY timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS rolling_avg_latency,
    COUNT(*) OVER (PARTITION BY user_id) AS total_queries
FROM customer_responses
WHERE timestamp > '2024-01-01'
    AND response_accuracy >= 0.9
GROUP BY query_id, user_id, response_latency, timestamp
HAVING total_queries > 50
ORDER BY rolling_avg_latency DESC;