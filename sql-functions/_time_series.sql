-- Query 7: Month-over-month (MoM) trip volume variation using the LAG window function
-- Evaluates sequential monthly volume trends and computes percentage change across periods.
WITH MonthlyTrips AS (
  SELECT
    EXTRACT(MONTH FROM trip_start_timestamp) AS month_num,
    COUNT(unique_key) AS trip_count
  FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
  WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2023
  GROUP BY month_num
)
SELECT
  month_num,
  trip_count,
  LAG(trip_count) OVER(ORDER BY month_num) AS prev_month_count,
  trip_count - LAG(trip_count) OVER(ORDER BY month_num) AS mom_volume_change,
  ROUND(((trip_count - LAG(trip_count) OVER(ORDER BY month_num)) / LAG(trip_count) OVER(ORDER BY month_num)) * 100, 2) AS mom_percent_change
FROM MonthlyTrips
ORDER BY month_num;
