# Chicago Taxi Transit Dataset — Data Dictionary & Schema Specification

This document outlines the data schema, feature specifications, and target variable definitions for the Chicago Taxi transit dataset (`bigquery-public-data.chicago_taxi_trips.taxi_trips`) utilized across exploratory data analysis and supervised predictive modeling.

---

## Feature Specifications

| Field Name | Description | Data Type | Role / Scope | Example Values |
| :--- | :--- | :--- | :--- | :--- |
| **`trip_miles`** | Total transit distance traversed during the trip in statute miles. | Float (Numeric) | Continuous Feature | `2.50`, `18.20` |
| **`trip_seconds`** | Total elapsed elapsed transit duration in seconds. | Integer (Numeric) | Continuous Feature | `720`, `2280` |
| **`trip_hour`** | Discrete hour of departure extracted from `trip_start_timestamp` (`0`–`23`). | Integer (Numeric) | Engineered Feature | `8`, `14`, `17` |
| **`trip_day`** | Day of the week for trip commencement (`0` = Monday through `6` = Sunday). | Integer (Numeric) | Engineered Feature | `0`, `4`, `6` |
| **`payment_type`** | Transaction settlement method recorded by meter telemetry. | String (Categorical) | Categorical Feature | `'Credit Card'`, `'Cash'` |
| **`fare`** | Base meter fare amount assessed for the trip in USD. | Float (Numeric) | **Target Variable** (Regression) | `14.50`, `45.00` |
| **`tips`** | Recorded gratuity amount remitted by passenger in USD. | Float (Numeric) | Analytical Metric | `0.00`, `5.00` |
| **`tip_given`** | Binary indicator denoting remittance of non-zero gratuity (`1` = Gratuity remitted, `0` = No gratuity recorded). | Integer (Binary) | **Target Variable** (Classification) | `0`, `1` |

---

## Data Ingestion & Transformation Notes

* **Temporal Extraction:** `trip_hour` and `trip_day` are derived directly from the ISO timestamp `trip_start_timestamp`.
* **One-Hot Encoding:** For linear and ensemble classification pipelines, `payment_type` is transformed using dummy variable encoding with the reference category dropped to avoid multicollinearity.
* **Outlier Filtration:** Records exhibiting non-positive values (`fare <= 0`, `trip_seconds <= 0`, `trip_miles <= 0`) are filtered prior to model ingestion.