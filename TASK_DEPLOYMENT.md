# TASK_DEPLOYMENT.md: SKU Sales Forecasting System Deployment Checklist

This checklist tracks the tasks required to deploy the forecasting system to GCP based on `PLANNING_DEPLOYMENT.md`.

## Phase 1: GCP Setup & Foundational Configuration

*   `[X]` **1.1 GCP Project:** Confirm target GCP project ID and ensure billing is enabled.
*   `[X]` **1.2 APIs:** Enable necessary GCP APIs:
    *   `[X]` Cloud Run API
    *   `[X]` Cloud Scheduler API
    *   `[X]` Cloud Build API (if using Cloud Build for containers)
    *   `[X]` Artifact Registry API
    *   `[X]` BigQuery API
    *   `[X]` Cloud Storage API
    *   `[X]` Secret Manager API (if used)
    *   `[X]` Cloud Logging API
    *   `[X]` Cloud Monitoring API (optional)
*   `[X]` **1.3 GCS Buckets:**
    *   `[X]` Create GCS bucket for storing trained models (`gs://eztech-442521-rich-distributor-forecast-models`). Define folder structure (`Unit/Channel/SKU_ID.pkl`).
    *   `[X]` Create GCS bucket for potential temporary data or staging (`gs://eztech-442521-rich-distributor-forecast-staging`).
*   `[X]` **1.4 BigQuery Setup:**
    *   `[X]` Create a new BigQuery Dataset (`rich`).
    *   `[X]` Define and create table schema for storing forecasts (`distributor_forecasts_weekly`).
    *   `[X]` Define and create table schema for storing performance metrics (`distributor_model_performance_metrics`).
    *   `[X]` Confirm access to source historical/actuals table (`rich.distributor_sales_frozen_sample_1`).
*   `[X]` **1.5 Service Accounts & IAM:**
    *   `[X]` Create dedicated Service Account(s) for Cloud Run Jobs (`rich-dist-forecast-runner@eztech-442521.iam.gserviceaccount.com`).
    *   `[X]` Grant necessary IAM roles to the Service Account(s):
        *   `[X]` BigQuery Data Editor (for `rich` dataset)
        *   `[X]` BigQuery User (for project `eztech-442521`)
        *   `[X]` Storage Object Admin (for model bucket `gs://eztech-442521-rich-distributor-forecast-models`)
        *   `[X]` Logs Writer (for project `eztech-442521`)
        *   `[X]` Secret Manager Secret Accessor (for project `eztech-442521`)
*   `[X]` **1.6 Artifact Registry:**
    *   `[X]` Create an Artifact Registry repository (Docker type) to store container images (`rich-distributor-images` in `asia-south1`).

## Phase 2: Model Preparation & Storage

*   `[X]` **2.1 Finalize Training Script:**
    *   `[X]` Ensure `run_all_skus.py` (or equivalent) correctly serializes models (using `joblib` or `pickle`).
    *   `[X]` Add logic to upload trained models directly to the GCS bucket (`gs://eztech-442521-rich-distributor-forecast-models/Unit/Channel/SKU_ID.pkl`). Authenticate using Application Default Credentials (ADC) or a service account key.
*   `[ ]` **2.2 Execute Bulk Training:**
    *   `[ ]` Run the finalized training script for all ~525 combinations.
*   `[ ]` **2.3 Verify Models in GCS:**
    *   `[ ]` Confirm all ~525 model files exist in the GCS bucket with the correct naming convention and structure.

## Phase 3: Weekly Forecasting Pipeline Implementation

*   `[ ]` **3.1 Define Future Data Processes:**
    *   `[ ]` **CRITICAL:** Define, document, and implement the *automated* process for sourcing/generating **future promotion data** for the next 4 weeks for all combinations. Where will this data live (BQ table, GCS file, API)?
    *   `[ ]` Define/confirm process for fetching **future weather data** (if used).
    *   `[ ]` Confirm BQ query/process to get the **latest actual sales** for initializing `y_lag1`.
*   `[ ]` **3.2 Create Prediction Script (`predict_weekly.py`):**
    *   `[ ]` Create a new script or adapt existing ones.
    *   `[ ]` Add argument parsing/environment variable reading for configuration (BQ tables, GCS bucket, target combinations - potentially read from a config file or BQ table).
    *   `[ ]` Implement logic to load models from GCS based on combination ID.
    *   `[ ]` Implement logic to fetch/prepare future regressors (promos, weather) and latest actuals (for lag).
    *   `[ ]` Implement the core iterative prediction logic using Prophet's `predict`.
    *   `[ ]` Implement logic to write forecast results (dataframe) to the target BigQuery `weekly_forecasts` table using `pandas-gbq` or `google-cloud-bigquery` client library.
    *   `[ ]` Integrate robust error handling (`try...except`) per combination.
    *   `[ ]` Integrate Cloud Logging (`google-cloud-logging` library) for detailed logs.
*   `[ ]` **3.3 Containerize Prediction Script:**
    *   `[ ]` Create `Dockerfile.predict` specifying Python base image, copying script/`src` code, installing `requirements.txt`, and setting entrypoint/cmd.
    *   `[ ]` Update `requirements.txt` with necessary libraries (`google-cloud-storage`, `google-cloud-bigquery`, `google-cloud-logging`, `pandas-gbq`, `prophet`, `joblib`, etc.).
    *   `[ ]` Build the Docker image locally (`docker build ...`).
    *   `[ ]` Push the image to Artifact Registry (`docker push ...`).
*   `[ ]` **3.4 Create Cloud Run Job (Prediction):**
    *   `[ ]` Create a new Cloud Run Job (`gcloud run jobs create predict-weekly --image ...`).
    *   `[ ]` Configure job settings:
        *   `--image`: Point to the image in Artifact Registry.
        *   `--service-account`: Specify the created service account.
        *   `--region`: Specify GCP region.
        *   Set necessary Environment Variables (BQ table names, GCS bucket, etc.).
        *   Configure CPU/memory resources, task timeout (ensure it's sufficient).
    *   `[ ]` Manually execute the job (`gcloud run jobs execute predict-weekly`) and verify successful completion and data written to BigQuery. Debug using Cloud Logging.
*   `[ ]` **3.5 Create Cloud Scheduler (Prediction):**
    *   `[ ]` Create a Cloud Scheduler job (`gcloud scheduler jobs create http predict-weekly-trigger --schedule "0 22 * * 3" ...`).
    *   `[ ]` Set target to "Cloud Run".
    *   `[ ]` Configure the job to trigger the `predict-weekly` Cloud Run Job.
    *   `[ ]` Specify timezone (e.g., `TZ=Europe/Moscow`).
    *   `[ ]` Ensure the scheduler's service account has permission to invoke the Cloud Run Job.

## Phase 4: Performance Monitoring Pipeline Implementation

*   `[ ]` **4.1 Create Monitoring Script (`monitor_performance.py`):**
    *   `[ ]` Create a new script.
    *   `[ ]` Add argument parsing/env vars for configuration (BQ tables).
    *   `[ ]` Implement logic to:
        *   Query recent forecasts from `weekly_forecasts` table for a defined lookback period (e.g., last 4-5 weeks).
        *   Query corresponding actuals from the source sales table.
        *   Join the dataframes.
        *   Calculate performance metrics (WAPE, sMAPE, MAE, RMSE) per combination for the period. Use functions from `sklearn.metrics` or custom implementations.
        *   Write the calculated metrics to the `model_performance_metrics` BigQuery table.
    *   `[ ]` Add error handling and Cloud Logging.
*   `[ ]` **4.2 Containerize Monitoring Script:**
    *   `[ ]` Create `Dockerfile.monitor`.
    *   `[ ]` Build and push the image to Artifact Registry.
*   `[ ]` **4.3 Create Cloud Run Job (Monitoring):**
    *   `[ ]` Create a new Cloud Run Job (`monitor-performance`).
    *   `[ ]` Configure image, service account, environment variables, resources, timeout.
    *   `[ ]` Manually execute and verify successful run and data in metrics table.
*   `[ ]` **4.4 Create Cloud Scheduler (Monitoring):**
    *   `[ ]` Create a Cloud Scheduler job (`monitor-performance-trigger`).
    *   `[ ]` Set schedule (e.g., weekly `0 0 * * 5`, running after actuals are likely available).
    *   `[ ]` Configure target to trigger the `monitor-performance` Cloud Run Job.
*   `[ ]` **4.5 Build Looker Studio Dashboard:**
    *   `[ ]` Connect Looker Studio to the `model_performance_metrics` BigQuery table.
    *   `[ ]` Create an initial dashboard showing:
        *   Overall average WAPE/sMAPE trends.
        *   Table of worst-performing models (by recent WAPE/sMAPE).
        *   Ability to filter by Unit/Channel/SKU.
        *   Time series plot of metrics for a selected model.
*   `[ ]` **4.6 (Optional) Configure Cloud Monitoring Alerts:**
    *   `[ ]` Define alert conditions (e.g., WAPE > 0.7 for 2 consecutive weeks).
    *   `[ ]` Create alerts in Cloud Monitoring based on queries against the `model_performance_metrics` table.

## Phase 5: Retraining Setup (Manual Trigger Focus)

*   `[ ]` **5.1 Create/Adapt Retraining Script (`retrain_models.py`):**
    *   `[ ]` Adapt existing training/tuning code (`run_all_skus.py`, `src/...`).
    *   `[ ]` Add argument parsing/env vars to accept a list/pattern of specific `Unit/Channel/SKU_ID` combinations to retrain.
    *   `[ ]` Ensure it fetches the *full updated* historical data from BigQuery for the specified combinations.
    *   `[ ]` Ensure it performs tuning (optional) and retraining.
    *   `[ ]` Ensure it *overwrites* the correct model file in the GCS bucket.
    *   `[ ]` Add comprehensive logging of the process and outcome (e.g., new hyperparameters, metrics on internal validation split).
*   `[ ]` **5.2 (Recommended) Containerize Retraining Script:**
    *   `[ ]` Create `Dockerfile.retrain`.
    *   `[ ]` Build and push the image to Artifact Registry.
*   `[ ]` **5.3 Define Manual Trigger Process:**
    *   `[ ]` Document the command/process for manually triggering a retraining run for specific models (e.g., using `gcloud run jobs execute retrain-models --update-env-vars TARGET_COMBINATIONS=...` if containerized, or steps for running locally with credentials).

## Phase 6: Testing, Documentation & Go-Live

*   `[ ]` **6.1 Testing:**
    *   `[ ]` Perform unit tests on helper functions within the new scripts.
    *   `[ ]` Perform integration tests: run the weekly forecast job for a small sample of models and verify output.
    *   `[ ]` Perform integration tests: run the monitoring job and verify metrics calculation.
    *   `[ ]` Perform end-to-end test: let the scheduled jobs run for a cycle, check forecasts, wait for actuals, check monitoring results.
*   `[ ]` **6.2 Documentation:**
    *   `[ ]` Update `README.md` with:
        *   Overview of the deployed architecture on GCP.
        *   Instructions for setting up credentials/environment locally (if needed for debugging/manual runs).
        *   Instructions on how to manually trigger prediction, monitoring, or retraining jobs.
        *   Location of key resources (GCS buckets, BQ tables, Looker Studio dashboard).
        *   Troubleshooting common issues / how to check logs.
*   `[ ]` **6.3 Final Review:**
    *   `[ ]` Review all configurations (Cloud Run env vars, Scheduler times, IAM permissions).
    *   `[ ]` Review estimated costs.
*   `[ ]` **6.4 Go-Live:**
    *   `[ ]` Ensure all necessary model files are in GCS.
    *   `[ ]` Enable the Cloud Scheduler jobs.
    *   `[ ]` Monitor the first few automated runs closely via Cloud Logging and the Looker Studio dashboard. 