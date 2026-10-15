-- Query 1: Trip volume and aggregate revenue by hour of day
-- Extracts the hour component from the trip start timestamp and aggregates total revenue.
SELECT
  EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,
  COUNT(unique_key) AS trip_count,
  ROUND(SUM(trip_total), 2) AS total_revenue
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2023
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- Query 2: Trip volume distribution by day of week
-- EXTRACT(DAYOFWEEK) returns integers 1 (Sunday) through 7 (Saturday).
SELECT
  EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS day_of_week_num,
  COUNT(unique_key) AS trip_count
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2023
GROUP BY day_of_week_num
ORDER BY day_of_week_num;

-- Query 4: Top 10 taxi companies by trip volume and average fare
-- Aggregates volume and average fare per operator, filtering null company records.
SELECT
  company,
  COUNT(unique_key) AS trip_count,
  ROUND(AVG(fare), 2) AS avg_fare
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2023
  AND company IS NOT NULL
GROUP BY company
ORDER BY trip_count DESC
LIMIT 10;
