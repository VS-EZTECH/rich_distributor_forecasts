# Distributor Forecasting 2.0

A forecasting system to predict weekly gross sales volume for multiple SKU-Channel-Unit combinations using FB Prophet. It features a batch processing pipeline that reads combination inputs from **Google Sheets** and supports parallel execution.

## Important Note (2025-04-05)

**Python Environment**: Always use the specific Python path for this project:
```bash
/Users/vandit/miniconda3/envs/fbprophet-env/bin/python
```

**Google Sheets Credentials**: Ensure your service account JSON key file for Google Sheets API access is placed at `.secrets/eztech-442521-sheets.json` or provide the path using the `--gsheet_creds_path` argument.

Example command:
```bash
/Users/vandit/miniconda3/envs/fbprophet-env/bin/python run_all_skus.py --combinations_gsheet_id YOUR_SHEET_ID_HERE
```

## Recent Updates (2025-04-05)

- **Input Source Changed**: The pipeline now reads SKU-Channel-Unit combinations directly from a specified Google Sheet instead of a CSV file. Requires `google-api-python-client` and `google-auth-oauthlib` libraries.
- **Model Accuracy Improvement**: Implemented using actual promotional values from validation data instead of fixed assumptions. This resulted in significantly better forecast accuracy:
  - Average MAE: 237.90
  - Average RMSE: 285.19
  - Average sMAPE: 0.623 (62.3%)
  - Average WAPE: 0.550 (55.0%)
  - These results meet our primary goal of reducing WAPE below 70%

- **Multi-SKU Pipeline:** Added functionality to process multiple SKU-Channel-Unit combinations in batch mode (`run_all_skus.py`).
- **Improved Data Handling:** Modified data loader to support different units and channels.
- **Parallel Processing:** Added support for parallel processing of combinations.
- **Refactoring:** Modularized the forecasting code for better reusability.
- **Previous Updates:**
  - **Refactoring:** Hyperparameter tuning logic was moved from `forecasting_v1.py` into a dedicated module `src/hyperparameter_tuning.py`.
  - **Feature Engineering:** Added a lagged target feature (`y_lag1`) as a regressor and implemented iterative prediction for future values.
  - **Tuning Findings:** Initial automated hyperparameter tuning yielded suboptimal results. Manual testing with parameters `{'changepoint_prior_scale': 0.5, 'holidays_prior_scale': 0.1, 'seasonality_prior_scale': 10.0}` showed significantly better performance (MAE ~158 vs ~218, sMAPE ~0.71 vs ~1.17) on the validation set for the target SKU. The script currently uses automated tuning, but this discrepancy is noted.
  - **Log Transformation Test:** A parallel test using log transformation (`np.log1p`) on the target variable was conducted but resulted in significantly worse performance and was abandoned for this SKU.
  - **Plotting Fix:** Resolved issues with generating Prophet plots (`plot_components`, `plot`) when using the `y_lag1` feature.

## Project Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Distributor\ Forecasting\ 2.0
```

### 2. Create a Virtual Environment

```bash
# For macOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Google Cloud Authentication

1. Create a service account in Google Cloud Console with BigQuery permissions
2. Download the service account key JSON file
3. Set the environment variable:

```bash
# For macOS/Linux
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# For Windows (Command Prompt)
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account-key.json

# For Windows (PowerShell)
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account-key.json"
```

### 5. Verify Setup

Run the setup verification script:

```bash
python setup.py
```

## Project Structure

- `PLANNING.md` - Project architecture, goals, and methodology
- `TASK.md` - Task checklist and project roadmap
- `setup.py` - Environment setup verification script
- `requirements.txt` - Required Python packages
- `run_all_skus.py` - Main script for running forecasts for multiple SKU-Channel-Unit combinations
- `/src` - Source code for the forecasting system
  - `forecasting.py` - Modular forecasting functions for batch processing
  - `hyperparameter_tuning.py` - Module for Prophet CV tuning
  - `data_loader.py` - Module for fetching sales data
  - `weather_handler.py` - Module for fetching weather data
  - `sku_combination_loader.py` - Module for loading SKU combinations from Google Sheets and filtering them.
- `test_tuning.py` - Standalone script for testing hyperparameter tuning
- `/data` - **DEPRECATED for input combinations.** Was used for `sku_combinations.csv`.
- `/output` - Generated files (model pickles, forecasts, plots, CV results)
  - `/forecasts` - SKU-specific forecasts organized by unit and channel
- `/sql` - SQL queries for data extraction
  - `rich_distributor_queries.sql` - SQL queries for distributor data

## Running Multiple SKU Forecasts

Use the `run_all_skus.py` script to process multiple SKU-Channel-Unit combinations read from a Google Sheet:

```bash
/Users/vandit/miniconda3/envs/fbprophet-env/bin/python run_all_skus.py --combinations_gsheet_id YOUR_SHEET_ID_HERE
```

Replace `YOUR_SHEET_ID_HERE` with the actual ID of your Google Sheet (e.g., `1BTCZsliXDbEsj9XOdCfETxFhCs2LdrrDko0UQu9w-PQ`).

### Command-Line Arguments

- `--combinations_gsheet_id`: (Required) Google Sheet ID containing the SKU_ID, Channel, Unit columns.
- `--combinations_gsheet_name`: Name of the tab in the Google Sheet (default: "Target").
- `--gsheet_creds_path`: Path to the Google Sheets service account credentials JSON file (default: `.secrets/eztech-442521-sheets.json`).
- `--output_dir`: Base output directory (default: "output")
- `--project_id`: Google Cloud Project ID (default: "eztech-442521")
- `--dataset_id`: BigQuery dataset ID (default: "rich")
- `--start_date`: Optional start date (YYYY-MM-DD)
- `--end_date`: Optional end date (YYYY-MM-DD)
- `--validation_days`: Days for validation period (default: 28)
- `--weather_latitude`: Latitude for weather data (default: 58.0 - Perm, Russia)
- `--weather_longitude`: Longitude for weather data (default: 56.3 - Perm, Russia)
- `--max_combinations`: Optional limit on combinations to process
- `--specific_sku`: Optional filter for a specific SKU_ID
- `--specific_unit`: Optional filter for a specific Unit
- `--specific_channel`: Optional filter for a specific Channel
- `--parallel`: Run processing in parallel
- `--workers`: Number of worker processes when parallel=True (default: 4)

### Sample Usage for Testing

Test with the first 3 combinations from the Google Sheet:

```bash
/Users/vandit/miniconda3/envs/fbprophet-env/bin/python run_all_skus.py --combinations_gsheet_id YOUR_SHEET_ID_HERE --max_combinations 3
```

Run for a specific SKU (П-00006477) found in the Google Sheet:

```bash
/Users/vandit/miniconda3/envs/fbprophet-env/bin/python run_all_skus.py --combinations_gsheet_id YOUR_SHEET_ID_HERE --specific_sku "П-00006477"
```

## External Data Sources

- **Weather Data:** Historical daily weather data (temperature, precipitation, snowfall) for relevant locations (e.g., Perm, Russia) is sourced from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). This API is free for non-commercial use and does not require an API key.

## Methodology

This project uses Facebook Prophet for time series forecasting with the following workflow:

1. Data acquisition from BigQuery (daily)
2. Feature engineering and **weekly aggregation**
3. Model training and hyperparameter tuning (**weekly data**)
4. Prediction generation (**weekly**)
5. Evaluation and analysis (**weekly**)

For more details, refer to `PLANNING.md`. 