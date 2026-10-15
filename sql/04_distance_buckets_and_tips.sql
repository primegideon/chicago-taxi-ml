-- Query 3: Average fare by trip distance bucket
-- Categorizes trips into discrete mileage bands using CASE WHEN logic.
-- Ordering by MIN(trip_miles) ensures the result rows follow logical distance progression rather than alphabetical order.
SELECT
  CASE
    WHEN trip_miles >= 0 AND trip_miles <= 2 THEN '0-2 miles'
    WHEN trip_miles > 2 AND trip_miles <= 5 THEN '2-5 miles'
    WHEN trip_miles > 5 AND trip_miles <= 10 THEN '5-10 miles'
    WHEN trip_miles > 10 THEN '10+ miles'
  END AS distance_bucket,
  COUNT(unique_key) AS trip_count,
  ROUND(AVG(fare), 2) AS avg_fare
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2023
  AND trip_miles IS NOT NULL
GROUP BY distance_bucket
ORDER BY MIN(trip_miles);

-- Query 5: Gratuity rate analysis by transaction method
-- Evaluates tipping frequencies and average gratuity values across Cash and Credit Card payments.
SELECT
  payment_type,
  COUNT(unique_key) AS total_trips,
  COUNTIF(tips > 0) AS trips_with_tip,
  ROUND((COUNTIF(tips > 0) / COUNT(unique_key)) * 100, 2) AS percent_of_trips_tipped,
  ROUND(AVG(tips), 2) AS avg_tip_amount
FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2023
  AND payment_type IN ('Cash', 'Credit Card')
GROUP BY payment_type;
