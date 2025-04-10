# PLANNING.md: SKU Sales Forecasting

## 1. Project Objective

To develop and validate a forecasting system predicting **weekly gross sales volume** for **multiple SKU-Channel-Unit combinations**. The forecast horizon is **4 weeks (28 days)**. The system uses a batch processing pipeline to handle different combinations efficiently.

## 2. Input Data

Historical daily data sourced from BigQuery for each SKU-Channel-Unit combination, including:

* **Sales Data:** Daily sales units.
* **Promotion Data:** Daily discount percentage.
* **(Optional/Future) Weather Data:** Historical daily weather metrics (e.g., temperature, precipitation) for the relevant location.

## 3. Methodology & Architecture

The solution employs individual time series models (one per combination) using FB Prophet.

**3.1. Modeling Choice:**

* **Model:** FB Prophet (chosen for its ability to handle seasonality, holidays, and external regressors).
* **Granularity:** Model trained on **weekly aggregated** data. Daily data is fetched and aggregated (sum sales, average/max regressors). Predictions are generated **weekly**.
* **Scope:** Multiple models, one for each valid SKU-Channel-Unit combination provided in the input file.

**3.2. Architecture Overview:**

1.  **Data Pipeline (BQ -> Python/Pandas):**
    * Fetch relevant **daily** Sales and Promotion data for the **single target SKU** using `google-cloud-bigquery`.
    * Construct a complete date scaffold for the analysis period for this SKU (implicitly handled by BQ).
    * Handle missing values (e.g., `COALESCE(Sales, 0)`, `COALESCE(Discount, 0)`).
    * **(Optional/Future)** Integrate Weather Data: Fetch historical **daily** weather data via API and merge with daily sales/promo data based on date.

2.  **Feature Engineering & Aggregation (Pandas):**
    * **Data Cleaning (Daily):** Remove rows with negative sales values from the daily data.
    * **Weekly Aggregation:** Aggregate the cleaned daily data to a weekly frequency (e.g., starting Monday).
        * Target (`y`): Sum of daily sales.
        * Regressors (`Promo_discount_perc`, weather): Average or other appropriate aggregation (e.g., max for `is_promo`, sum for precipitation).
    * **Target Transformation:** **No transformation applied.** Initial analysis showed weekly aggregation sufficiently stabilized variance. *Update (YYYY-MM-DD): A separate test run using log transformation (`np.log1p`) confirmed it did not improve performance for the target SKU.*
    * **Regressor Handling:** Ensure regressor columns are correctly aggregated. *Update (YYYY-MM-DD): Added lagged target (`y_lag1`) as a key regressor.*
    * **Holiday/Event Data:** Prepare standard country holidays (Russia specified) and potentially custom events for Prophet, aligned with the weekly frequency.

3.  **Modeling (Prophet):**
    * **Data Splitting:** Split the **weekly aggregated** data temporally into training and validation sets.
    * **Initial Training:** Train an initial Prophet model on the training data with default or estimated hyperparameters.
    * **Hyperparameter Tuning:** Perform systematic tuning using Prophet's built-in cross-validation. *Update (YYYY-MM-DD): This logic has been refactored into the `src/hyperparameter_tuning.py` module.*
    * **Final Training:** Re-train the Prophet model on the full training dataset using the best hyperparameters identified during tuning.
    * Store the final, tuned model object.

4.  **Prediction & Post-processing:**
    * **Future Dataframe:** Create a dataframe for the 4-week forecast horizon.
    * **Populate Future Regressors:** Fill the future dataframe with known/assumed values for all regressors, including iterative calculation for `y_lag1`.
    * **Prediction:** Generate weekly forecasts using the **tuned** model.
    * **Inverse Transform (If Applicable):** Apply `expm1` to the predicted values if log transformation was used (Not applicable based on tests).
    * **Clean-up:** Clip final `yhat` predictions at zero.
    * **Improved Regressor Values (2025-04-02):** Modified the pipeline to use actual promotional values from validation data instead of fixed assumptions, resulting in significantly improved forecasting accuracy (average WAPE: 55.0%, average sMAPE: 62.3%).

5.  **Evaluation:**
    * Compare **weekly** forecasts against actual **weekly** sales data from the held-out validation set.
    * Calculate key performance indicators (KPIs): WAPE (primary), MAE, RMSE, sMAPE.
    * Analyze model performance. *Update (YYYY-MM-DD): Manual parameter setting (`{'cps': 0.5, 'hps': 0.1, 'sps': 10.0}`) yielded significantly better validation results (MAE ~158, sMAPE ~0.71) than the current automated tuning best (MAE ~218, sMAPE ~1.17). This suggests potential improvements needed in the tuning strategy or parameter grid.*

## 4. Key Assumptions & Decisions

* **Multi-SKU Pipeline:** The system is designed to process a list of SKU-Channel-Unit combinations provided via a **Google Sheet** (specified by Sheet ID and Tab Name).
* **Gross Sales Focus:** Negative sales rows are removed; the forecast represents gross sales potential, not net sales.
* **Minimum Data:** ~6 months of daily history (translating to ~24+ weeks) is deemed sufficient for initial model training per combination.
* **Future Regressors:** Assumes future promotion plans are available or reasonable assumptions can be made. Future weather data requires a separate forecast or assumption. Uses actual promo values from validation period for improved short-term forecast accuracy.
* **Deployment:** Model deployment and operationalization are deferred.

## 5. Alternative Approach Exploration: Log Transformation (Completed YYYY-MM-DD)

*Objective: Explore using `np.log1p` transformation on the weekly target variable (`y`).*

**Methodology Differences:**

*   **Target Transformation:** Apply `np.log1p` to the weekly `y` column before training.
*   **Prediction Post-processing:** Apply the inverse transformation `np.expm1` to the forecast outputs (`yhat`, `yhat_lower`, `yhat_upper`) before clipping at zero.
*   **Evaluation:** Performance metrics (MAE, RMSE, WAPE, sMAPE) will be calculated on the *inverse-transformed* predictions against the original-scale actuals.
*   **Hyperparameter Tuning:** Cross-validation and tuning (Step 7 in the main plan) will be performed on the log-transformed target.

*This exploration was completed and found that log transformation significantly worsened performance for the target SKU. It will not be pursued further for this SKU at this time.*


