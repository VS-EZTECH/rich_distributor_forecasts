# TASKS.md: SKU Sales Forecasting (v1 - Single SKU Focus)

This checklist tracks the tasks outlined in the project plan for the initial **single SKU** development. Mark items as complete (`[x]`) as they are finished.

## Phase 1: Data Acquisition and Preparation

* `[x]` **1. Environment Setup:**
    * `[x]` Set up a dedicated Python virtual environment.
    * `[x]` Install required Python libraries (pandas, prophet, google-cloud-bigquery, scikit-learn, requests/weather-api-client, etc.).
    * `[x]` Configure Google Cloud authentication for BigQuery access.
    * `[x]` (If using weather API) Secure necessary API keys/credentials. (Using Open-Meteo - No key required)
* `[x]` **2. Define Scope & Timeframes:**
    * `[x]` **Select the single target SKU ID for initial development.** (Selected SKU_ID: 'П-00006477')
    * `[x]` Define the exact historical data range needed for this SKU (min 6 months training + 28 days validation). (Date range: 2023-12-29 to 2025-03-29)
    * `[x]` Set specific start/end dates for training and validation periods. (Training: 2023-12-29 to 2025-03-01, Validation: 2025-03-02 to 2025-03-29)
* `[x]` **3. Data Retrieval:**
    * `[x]` Write and test SQL query to fetch **Daily** Sales and Promotion data for the **single target SKU**, including date scaffold.
    * `[x]` Handle NULLs appropriately in SQL (e.g., `COALESCE`).
    * `[x]` Load query results into a pandas DataFrame.
    * `[x]` Perform initial data type checks/conversions (esp. dates).
    * `[x]` Fetch historical **daily** weather data for the relevant period and location using an API client.
    * `[x]` Merge **daily** weather data with the daily sales data.
* `[x]` **4. Feature Engineering & Aggregation:**
    * `[x]` **4a.** Remove rows with negative sales (`y < 0`) from **daily** data. Document this step.
    * `[x]` **4b.** **Weekly Aggregation:** Aggregate daily data to weekly level.
        * `[x]` Sum daily `y` for weekly `y`.
        * `[x]` Aggregate regressors (e.g., mean `Promo_discount_perc`, max `is_promo`, mean/sum weather).
    * `[-]` **4c. Analyze Variance:** Initial analysis on daily data showed high skew/variance. Weekly aggregation significantly reduced this. **Log transformation deemed unnecessary.**
    * `[x]` **4d.** Validate aggregated regressor columns: `Promo_discount_perc`, `is_promo`.
    * `[x]` **4e.** Validate aggregated weather-based regressor columns (e.g., `temperature_2m_max`, `precipitation_sum`).
    * `[x]` **4f.** Prepare holiday data structure for Prophet (Russia Holidays, custom events) **aligned with weekly frequency**.
* `[x]` **5. Data Filtering & Splitting:**
    * `[x]` Verify the single target SKU has sufficient **weekly** training data (e.g., > 24 weeks).
    * `[x]` Split the SKU's **weekly aggregated** data into `train_df` and `validation_df` based on the defined validation period (last 4 weeks).

## Phase 2: Model Training, Tuning, and Prediction (Weekly Data)

* `[x]` **6. Initial Model Training (Single SKU, Weekly):** (Completed 2025-03-31)
    * `[x]` Instantiate `Prophet` model with baseline settings (seasonality modes, holidays from 4f).
    * `[x]` Add defined regressors (from 4d, 4e) to the model instance.
    * `[x]` Fit the model to the **weekly** `train_df` (`m.fit`), handling potential errors.
* `[x]` **7. Hyperparameter Tuning (Single SKU, Weekly):** (Completed 2025-03-31, Refactored 2025-04-01)
    * `[x]` Define a parameter grid for tuning (e.g., `changepoint_prior_scale`, `seasonality_prior_scale`, `holidays_prior_scale`).
    * `[x]` Use `diagnostics.cross_validation` on the **weekly** `train_df` with appropriate `initial`, `period`, `horizon` settings (e.g., horizon = '4 W').
    * `[x]` Use `diagnostics.performance_metrics` to evaluate results across the parameter grid based on a chosen metric (sMAPE).
    * `[x]` Identify the best combination of hyperparameters. Document the results.
    * `[x]` **Refactoring (2025-04-01):** Moved tuning logic into `src/hyperparameter_tuning.py` module.
* `[x]` **8. Final Model Training (Single SKU, Weekly):** (Completed 2025-03-31)
    * `[x]` Instantiate a *new* `Prophet` model instance using the **best hyperparameters** found in Step 7.
    * `[x]` Add the same holidays and regressors as in Step 6.
    * `[x]` Re-fit this tuned model to the entire **weekly** `train_df`.
    * `[x]` Store the final, tuned model object.
* `[x]` **9. Generate Future Predictions (4 weeks):** (Completed 2025-04-01)
    * `[x]` Get the final tuned model (from Step 8).
    * `[x]` Create the `future_df` for the 4-week forecast horizon **using weekly frequency** (`freq='W-SUN'`).
    * `[x]` Populate `future_df` with values for all regressors for the future **weeks**.
    * `[x]` **Added (2025-04-01):** Implemented iterative prediction loop to handle `y_lag1` regressor for future dates.
    * `[x]` Generate **weekly** predictions using `m.predict` on the tuned model.
    * `[x]` Filter the forecast output to keep only the 4 future **weeks**.
* `[x]` **10. Post-Process Predictions:** (Completed 2025-04-01)
    * `[-]` ~~Apply inverse transformation (`np.expm1`)~~ **(N/A - Log transform not used).**
    * `[x]` Clip final `yhat` values at zero.
    * `[-]` ~~Aggregate daily predictions into weekly sums~~ **(N/A - Predictions are weekly).**
    * `[x]` Store the final **weekly** forecasts for the single SKU.

## Phase 3: Evaluation and Analysis (Single SKU, Weekly)

* `[x]` **11. Prepare Validation Actuals:** (Completed 2025-04-01)
    * `[x]` Get the actual weekly sales data (`y`) from the **weekly** `validation_df`.
    * `[-]` ~~Aggregate actual daily sales into weekly sums~~ **(N/A - Data is weekly).**
* `[x]` **12. Calculate Performance Metrics:** (Updated 2025-04-01)
    * `[x]` Merge **weekly** forecasts (from Step 10) with **weekly** actuals (from Step 11).
    * `[x]` Calculate metrics (MAE, RMSE, WAPE, sMAPE).
    * `[x]` **Metrics (Validation - Auto Tuning):** MAE=217.61, RMSE=242.34, sMAPE=1.1705, WAPE=0.7296 (Params: `{'cps': 1.0, 'hps': 10.0, 'sps': 0.1}`)
    * `[x]` **Metrics (Validation - Manual Override):** MAE=158.20, RMSE=189.45, sMAPE=0.7136, WAPE=0.5304 (Params: `{'cps': 0.5, 'hps': 0.1, 'sps': 10.0}`)
    * `[x]` **Note:** Manual parameter check yielded significantly better results than automated tuning. Reverted to auto-tuning for now but noted the discrepancy.
* `[x]` **13. Analyze Results & Visualize:** (Updated 2025-04-01)
    * `[x]` Generate plots of **weekly** actuals vs. forecasts (validation period).
    * `[x]` Generate Prophet component plots using the final tuned model and forecast object.
    * `[x]` Analyze error patterns.
    * `[x]` **Fix (2025-04-01):** Corrected logic for generating combined history+future forecast object needed for `plot_components` and `plot` when using `y_lag1`.
* `[x]` **14. Review & Plan Next Steps:**
    * `[x]` Document key findings, model limitations, and performance results for the single SKU.
    * `[x]` **Plan for v1.1:** Scaling the validated pipeline to the remaining ~9 SKUs.
    * `[x]` Outline any further improvements identified (e.g., feature refinement, revisiting weather data integration).

## Phase 4: Model Improvement (Post-Refactoring) (Started 2025-03-31)

*Triggered by poor validation results (sMAPE ~1.57 -> 1.17) for SKU 'П-00006477' after initial refactoring/y_lag1 addition.*

* `[ ]` **15. Diagnose & Refine Model:**
    * `[ ]` **15a. Diagnose Failure Mode:**
        * `[x]` Analyze component plots. *(Completed 2025-03-31)*
            *   Finding: Zero predictions likely caused by combination of very strong negative *yearly seasonality* and negative *regressor effect*.
        * `[ ]` Check raw `yhat` values before clipping for the validation period.
        * `[ ]` Examine regressor contributions in the forecast period.
    * `[ ]` **15b. Revisit Hyperparameter Tuning:**
        * `[x]` Analyze `output/cv_tuning_results.csv`: Identify parameters yielding best **sMAPE/WAPE** during CV. *(Completed 2025-03-31)*
            *   Finding: Best MAE params (`{'cps': 1.0, 'hps': 10.0, 'sps': 0.1}`) differ slightly from prior best sMAPE params (`{'cps': 1.0, 'hps': 5.0/10.0, 'sps': 0.1}`). sMAPE = 0.9349.
        * `[x]` **Manual Check:** Re-trained final model with specific parameters (`{'cps': 0.5, 'hps': 0.1, 'sps': 10.0}`). *(Completed 2025-03-31)*
            * **Finding:** Achieved much better validation metrics (MAE=158, sMAPE=0.71). Automated tuning did not find these.
        * `[x]` **Decision:** Reverted script to use automated tuning for consistency, but documented the better manual parameters. *(Completed 2025-03-31)*
        * `[ ]` **(Future)** Consider adjusting parameter grid or CV strategy if auto-tuning continues to underperform manual checks.
    * `[ ]` **15c. Feature Engineering & Refinement (Iterative):**
        * `[x]` Add lagged target (`y_lag1`) as a regressor. *(Completed 2025-03-31)*
        * `[ ]` Refine holiday features (e.g., specific holiday names instead of generic `RU_Holiday`, potentially add `lower/upper_window`).
        * `[ ]` **(Consider Later)** Review regressor aggregation methods (promo/weather).
        * `[ ]` **(Consider Later)** Explore multiplicative seasonality if variance suggests it.
    * `[ ]` **15d. Iterate & Evaluate:** Re-run evaluation (Task 11-13 steps) after each significant modeling change.

### Discovered During Work (To Be Prioritized)
- DONE (2025-03-31): Update `fetch_sales_data` to use `distributor_sales_frozen_sample_1` table with filters unit='Пермь', channel='TF' and derive `is_promo` from `Promo_discount_perc`.
- DONE (2025-03-31): Re-run the entire forecasting pipeline (retrain, re-tune, predict) due to the data source change.
- DONE (2025-03-31): Correct weekly aggregation to Monday-Sunday (labeled Monday) using `resample('W-SUN', label='left', closed='left')`.
- DONE (2025-04-02): Updated model to use actual promo values from validation data instead of fixed assumptions, resulting in significantly improved metrics across all SKUs: MAE=237.90, RMSE=285.19, sMAPE=0.623, WAPE=0.550. This meets our primary goal of reducing WAPE below 70%.

## Phase 5: Alternative Approach - Log Transformation (Completed 2025-04-01)

*Objective: Implement and evaluate the forecasting pipeline using `np.log1p` transformation on the target variable `y`.*

* `[x]` **16. Prepare Data with Log Transform:**
    * `[x]` Load the prepared weekly aggregated data (`weekly_df` from Phase 1).
    * `[x]` Apply `np.log1p` transformation to the `y` column. Keep the original `y` column for later evaluation.
    * `[x]` Split data into `train_log_df` and `validation_log_df` (target `y` is log-transformed).
* `[x]` **17. Hyperparameter Tuning (Log Transformed Data):**
    * `[x]` Define parameter grid (similar to Step 7, potentially adjusted).
    * `[x]` Run `diagnostics.cross_validation` on `train_log_df`.
    * `[x]` Evaluate CV results using `diagnostics.performance_metrics` (metrics calculated on log scale).
    * `[x]` Identify best hyperparameters based on a suitable metric (log RMSE=2.1226: `{'cps': 0.5, 'hps': 1.0, 'sps': 0.1}`).
* `[x]` **18. Final Model Training (Log Transformed Data):**
    * `[x]` Instantiate `Prophet` with best hyperparameters from Step 17.
    * `[x]` Add holidays and regressors (same as Step 6/8).
    * `[x]` Fit the model to the full `train_log_df`.
    * `[x]` Store the final log-transformed model (`output_log_transformed/`).
* `[x]` **19. Generate Future Predictions (Log Scale & Inverse Transform):**
    * `[x]` Create `future_log_df` (same regressors as Step 9).
    * `[x]` Generate predictions (`yhat`, `yhat_lower`, `yhat_upper`) using the log-transformed model.
    * `[x]` Apply inverse transformation (`np.expm1`) to `yhat`, `yhat_lower`, `yhat_upper`.
    * `[x]` Clip inverse-transformed `yhat` at zero.
    * `[x]` Filter forecast to keep only the 4 future weeks.
* `[x]` **20. Evaluate Log Transformation Approach:**
    * `[x]` Prepare validation actuals (original scale `y` from `validation_df`).
    * `[x]` Merge inverse-transformed weekly forecasts (from Step 19) with original-scale actuals.
    * `[x]` Calculate performance metrics (MAE, RMSE, WAPE, sMAPE) on the original scale.
    * `[x]` Compare these metrics to the results from the non-transformed approach (Step 12 / Phase 4 info).
        * **Metrics (Original Scale):** MAE=290.84, RMSE=299.11, sMAPE=1.8983, WAPE=0.9752
* `[x]` **21. Analyze & Compare:**
    * `[x]` Generate plots (actuals vs. forecast, components) for the log-transformed model results (using inverse-transformed values).
    * `[x]` Compare plots and metrics with the non-transformed approach.
    * `[x]` **Finding:** Log transformation resulted in significantly worse performance. Abandoning this approach. *(Completed 2025-04-01)*

## Phase 6: Pipeline Scaling (All SKU-Channel-Unit Combinations)

*Objective: Adapt the single-SKU pipeline (from `src/forecasting.py` and related modules) to run forecasts for all unique combinations of `SKU_ID`, `Channel`, and `Unit` provided in a **user-specified Google Sheet**, storing the performance metrics for each.*

*   `[x]` **22. Load Combinations & Fetch Data:**
    *   `[x]` **22a. Load Combinations:** Implement logic to load a user-provided **Google Sheet** (Sheet ID and Tab specified via arguments) containing `(SKU_ID, Channel, Unit)` combinations.
    *   `[x]` **22b. Fetch & Filter Data (Per Combination):** Modify `src/data_loader.py` or create logic in the main script (`run_all_skus.py`) to:
        *   Iterate through each row (`sku_id`, `channel`, `unit`) from the loaded combinations.
        *   For *each* combination, query `distributor_sales_frozen_sample_1` filtering by the *specific* `sku_id`, `channel`, `unit` AND `date >= '2023-01-01'`.
        *   Store fetched daily data (e.g., in a dictionary keyed by the combination tuple).
    *   `[x]` **22c. Determine Date Range & Check Data:**
        *   For the fetched data of *each* combination:
            *   Find the minimum (`min_date`) and maximum (`max_date`) sales dates.
            *   Check if the duration (`max_date` - `min_date`) meets the minimum requirement (e.g., > 28 weeks total for train+validation).
            *   Log and mark combinations with insufficient data to be skipped.

*   `[x]` **23. Ensure Pipeline Reusability:**
    *   `[x]` **23a. Review Modules:** Verify functions within `src/data_loader.py`, `src/weather_handler.py`, `src/hyperparameter_tuning.py`, and `src/forecasting_v1.py` are suitable for processing data for a single combination passed as input.
    *   `[x]` **23b. Parameterize Core Logic:** Refactor relevant functions/methods (primarily in `src/forecasting_v1.py`) to accept the pre-fetched/filtered data, `min_date`, `max_date`, `sku_id`, `channel`, `unit` as parameters.
    *   `[x]` **23c. Parameterize Paths:** Modify relevant functions to save outputs (models, plots, temporary files) to combination-specific subdirectories within `output/` using the structure `output/forecasts/[Unit]/[Channel]/`, with model files named after the SKU_ID.
    *   `[x]` **23d. Configuration:** Ensure any configuration files (`config.yaml` if used) support or are adapted for batch processing.

*   `[x]` **24. Implement Iteration and Execution Logic:**
    *   `[x]` **24a. Create Main Script:** Create a new script `run_all_skus.py` (or similar) in the root directory.
    *   `[x]` **24b. Iteration Loop:** In `run_all_skus.py`, implement a loop iterating through the combinations that passed the data sufficiency check (from 22c).
    *   `[x]` **24c. Pipeline Execution:** Inside the loop, call the main processing function (refactored from `src/forecasting_v1.py`, see 23b), passing the specific combination's data and metadata.
    *   `[x]` **24d. Error Handling:** Implement `try...except` blocks for each combination's processing. Log errors (e.g., data prep error, model fit failure) with the combination ID and continue to the next.
    *   `[x]` **24e. (Optional) Parallel Processing:** Consider using `multiprocessing` if performance becomes an issue.

*   `[x]` **25. Aggregate and Store Results:**
    *   `[x]` **25a. Initialize Collector:** Before the loop in `run_all_skus.py`, initialize a list or DataFrame to store results.
    *   `[x]` **25b. Collect Results:** Within the loop, after processing each combination, append a record including: `SKU_ID`, `Channel`, `Unit`, `min_date`, `max_date`, `Status` (e.g., 'Success', 'Skipped - Insufficient Data', 'Error: Model Fit Failed'), Evaluation Metrics (MAE, RMSE, WAPE, sMAPE if successful), Best Hyperparameters (if successful).
    *   `[x]` **25c. Save Aggregated Results:** After the loop completes, save the collected results to a CSV file (e.g., `output/all_skus_validation_metrics_*.csv`).

*   `[x]` **26. Testing and Final Run:**
    *   `[x]` **26a. Sample Test:** Test the entire `run_all_skus.py` pipeline using the user-provided **Google Sheet** but initially filtered using `--max_combinations` (e.g., 1-5 combinations).
    *   `[x]` **26b. Debug:** Address any issues identified during the sample test.
    *   `[x]` **26c. Full Execution:** Run `run_all_skus.py` for all combinations specified in the input **Google Sheet**.
    *   `[x]` **26d. Review Output:** Verify the contents and format of the final aggregated results file (`output/all_skus_validation_metrics_*.csv`).