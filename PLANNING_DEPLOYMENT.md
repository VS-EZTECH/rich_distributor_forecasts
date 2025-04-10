# PLANNING_DEPLOYMENT.md: SKU Sales Forecasting System Deployment

## 1. Objective

To deploy and operationalize the SKU sales forecasting system on Google Cloud Platform (GCP). The system will:

1.  **Automate Weekly Forecasting:** Generate 4-week ahead sales forecasts for ~525 SKU-Channel-Unit combinations every Wednesday night using pre-trained models.
2.  **Automate Performance Monitoring:** Continuously track forecast accuracy (vs. actuals) for each combination on a weekly/monthly basis.
3.  **Facilitate Periodic Retraining:** Enable efficient (initially manual) retraining and hyperparameter tuning of models based on performance monitoring results.

## 2. Architecture Overview

The deployment leverages a serverless, event-driven architecture on GCP, primarily using Cloud Run Jobs for batch processing triggered by Cloud Scheduler.

**Core Components:**

*   **Data Sources:**
    *   `BigQuery`: Stores historical sales/promo data, actuals for monitoring, generated forecasts, and performance metrics.
    *   `Source TBD`: A defined, reliable source for *future* promotional plans (critical dependency).
    *   `Open-Meteo API / GCS Cache`: Source for historical and potentially future weather data.
*   **Model Storage:**
    *   `Google Cloud Storage (GCS)`: Stores the ~525 serialized Prophet model files (`.pkl` or `.joblib`), organized by `Unit/Channel/SKU_ID.pkl`.
*   **Compute:**
    *   `Cloud Run Jobs`: Executes containerized Python scripts for batch prediction, performance monitoring, and retraining tasks. Scales to zero when inactive.
*   **Scheduling:**
    *   `Cloud Scheduler`: Triggers the Cloud Run Jobs on a defined schedule (e.g., weekly for prediction, weekly/monthly for monitoring).
*   **Monitoring & Visualization:**
    *   `Looker Studio`: Connects to BigQuery to visualize performance metrics via dashboards.
    *   `Cloud Logging`: Captures logs from Cloud Run Jobs for debugging.
    *   `(Optional) Cloud Monitoring`: Provides alerting based on performance metric thresholds stored in BigQuery.
*   **Container Registry:**
    *   `Artifact Registry`: Stores the Docker container images for the Python scripts.

**Workflows:**

1.  **Initial Bulk Training (Pre-deployment):**
    *   Run `run_all_skus.py` (potentially modified for cloud environment) locally or on a Compute Engine instance.
    *   **Input:** Historical data from BigQuery, SKU combinations.
    *   **Output:** ~525 trained model files uploaded to GCS.
2.  **Weekly Forecasting Pipeline (Automated):**
    *   `Cloud Scheduler` (Wed night trigger) -> `Cloud Run Job (Prediction)`
    *   **Job Logic:**
        *   Fetches list of combinations to forecast.
        *   For each combination:
            *   Loads model from GCS.
            *   Fetches latest actual sales from BigQuery (for `y_lag1`).
            *   Fetches/constructs future regressor data (promo, weather) for 4 weeks.
            *   Runs iterative `model.predict()`.
        *   Writes all forecasts (combination, forecast date, target date, value) to a dedicated BigQuery table.
3.  **Performance Monitoring Pipeline (Automated):**
    *   `Cloud Scheduler` (e.g., Weekly trigger, offset from forecast/actuals availability) -> `Cloud Run Job (Monitoring)`
    *   **Job Logic:**
        *   Fetches recent forecasts from the forecast BigQuery table.
        *   Fetches corresponding actual sales from the source BigQuery table.
        *   Joins forecasts and actuals by combination and target date.
        *   Calculates performance metrics (WAPE, sMAPE, MAE, RMSE).
        *   Writes metrics (combination, calculation date, period, metric name, value) to a dedicated metrics BigQuery table.
4.  **Retraining Pipeline (Manual Trigger / Future Automation):**
    *   `Manual Trigger` (e.g., `gcloud run jobs execute`) or `Cloud Scheduler` -> `Cloud Run Job (Retraining)`
    *   **Job Logic:**
        *   Accepts a list of combinations to retrain (identified via monitoring).
        *   Fetches full, updated historical data from BigQuery for specified combinations.
        *   Performs hyperparameter tuning (optional) and model retraining.
        *   Overwrites the corresponding model file in GCS with the new version.
        *   Logs the outcome.

## 3. Key GCP Services & Rationale

*   **BigQuery:** Chosen for its scalability, cost-effectiveness for large datasets, and direct integration with Looker Studio and other GCP services. Ideal for storing structured time-series data, forecasts, and metrics.
*   **Cloud Storage (GCS):** Standard, durable, and cost-effective object storage, suitable for storing potentially large model files.
*   **Cloud Run Jobs:** Preferred serverless compute option for batch tasks. Automatically scales, pay-per-use, handles containerized workloads, simplifying dependency management. More suitable than Cloud Functions for potentially longer-running batch processes like retraining or forecasting many models.
*   **Cloud Scheduler:** Reliable and fully managed cron job service for triggering automated workflows.
*   **Artifact Registry:** Secure, managed repository for storing and managing Docker container images used by Cloud Run.
*   **Looker Studio:** Free and powerful tool for creating interactive dashboards directly from BigQuery data, enabling easy visualization of model performance.
*   **Cloud Logging/Monitoring:** Standard GCP services for observability and potential alerting.

## 4. Data Flow Summary

1.  **Training:** BQ (Historical Data) -> Compute (Training Script) -> GCS (Models)
2.  **Prediction:** Scheduler -> Cloud Run Job -> [GCS (Models), BQ (Latest Actuals), Source TBD (Future Regressors)] -> BQ (Forecasts Table)
3.  **Monitoring:** Scheduler -> Cloud Run Job -> BQ (Forecasts Table, Actuals Table) -> BQ (Metrics Table) -> Looker Studio
4.  **Retraining:** Manual/Scheduler -> Cloud Run Job -> BQ (Historical Data) -> GCS (Overwrite Models)

## 5. Critical Considerations & Dependencies

*   **Future Regressor Data Source:** **This is the most critical dependency.** A robust, automated process must be established to provide accurate future promotion plans (and weather forecasts, if used) for the 4-week horizon *before* the weekly forecast job runs. The format and retrieval method need definition.
*   **Lagged Target (`y_lag1`) Data:** The weekly prediction job depends on having access to the *most recent actual sales data* from BigQuery to initialize the iterative prediction loop. Data freshness is key.
*   **Containerization:** All Python scripts (`predict_weekly.py`, `monitor_performance.py`, `retrain_models.py`) need to be containerized using Docker, including all dependencies from `requirements.txt`.
*   **IAM Permissions:** Service Accounts for Cloud Run Jobs need appropriate roles/permissions to access BigQuery (read/write), GCS (read/write), and potentially other services. Principle of least privilege should be followed.
*   **Error Handling & Logging:** Cloud Run Jobs must implement comprehensive `try...except` blocks and detailed logging (using `google-cloud-logging` library) to facilitate debugging and monitoring of failures.
*   **Configuration Management:** How will configurations (BQ table names, GCS bucket names, combination lists, API keys) be managed? Environment variables in Cloud Run Jobs are a standard approach. Consider using Secret Manager for sensitive values.
*   **Cost Management:** Monitor GCP service usage (Cloud Run execution time, GCS storage, BigQuery queries/storage). Set up billing alerts if necessary.
*   **Scalability:** While Cloud Run scales automatically, consider potential bottlenecks: BigQuery query performance, external API rate limits (weather), GCS access patterns. For very large scales, prediction might need parallelization within the Cloud Run Job or across multiple jobs. 