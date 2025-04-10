import pandas as pd
import numpy as np
from google.cloud import bigquery
import os
from datetime import datetime, timedelta
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import logging
import traceback

# Import storage client
from google.cloud import storage

# Import functions from other modules - fix the imports to reference src directory
from src.data_loader import fetch_sales_data
from src.weather_handler import fetch_weather_data, fetch_forecast_weather_data
from src.hyperparameter_tuning import tune_hyperparameters

# Define log file path and ensure directory exists
log_dir = 'output'
log_file = os.path.join(log_dir, 'forecasting_process.log')
os.makedirs(log_dir, exist_ok=True) # <-- Create directory if it doesn't exist

# Create logger for this module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file) # <-- Use variable for path
    ]
)
logger = logging.getLogger('forecasting')

# --- Helper Functions ---
def _smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)"""
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the numerator and denominator, handling potential zeros
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    ratio = np.zeros_like(denominator)
    mask = denominator != 0
    ratio[mask] = numerator[mask] / denominator[mask]
    
    return np.mean(ratio)

def create_output_directory(base_dir, unit, channel):
    """Create output directory structure for a specific SKU-Channel-Unit combination"""
    output_dir = os.path.join(base_dir, 'forecasts', unit, channel)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_combination(client, project_id, dataset_id, sku_id, channel, unit, 
                        weather_lat, weather_lon, start_date=None, end_date=None,
                        validation_days=28, base_output_dir='output'):
    """
    Process a single SKU-Channel-Unit combination through the entire forecasting pipeline.
    
    Args:
        client (bigquery.Client): Initialized BigQuery client
        project_id (str): GCP project ID
        dataset_id (str): BigQuery dataset ID
        sku_id (str): Target SKU ID
        channel (str): Target channel
        unit (str): Target unit
        weather_lat (float): Latitude for weather data
        weather_lon (float): Longitude for weather data
        start_date (str, optional): Start date in YYYY-MM-DD format. If None, all available data will be used.
        end_date (str, optional): End date in YYYY-MM-DD format. If None, all available data will be used.
        validation_days (int): Number of days for validation period at end of data
        base_output_dir (str): Base directory for outputs
        
    Returns:
        dict: Results dictionary with metrics and status
    """
    result = {
        'SKU_ID': sku_id,
        'Channel': channel,
        'Unit': unit,
        'Status': 'Unknown',
        'min_date': None,
        'max_date': None,
        'validation_start': None,
        'MAE': None,
        'RMSE': None,
        'sMAPE': None,
        'WAPE': None,
        'best_params': None,
        'error_message': None
    }
    
    try:
        # Create output directory for this combination
        output_dir = create_output_directory(base_output_dir, unit, channel)
        logger.info(f"Processing SKU_ID={sku_id}, Channel={channel}, Unit={unit}")
        
        # 1. Fetch data - if dates not provided, use all available data
        logger.info(f"Fetching data for {sku_id}-{channel}-{unit}...")
        
        # First, fetch without date filters to determine available date range if not provided
        if start_date is None or end_date is None:
            min_end_date = "2025-12-31"  # Far future date to get all data
            min_start_date = "2023-01-01"  # Set a reasonable minimum date
            
            temp_df = fetch_sales_data(
                client=client,
                project_id=project_id,
                dataset_id=dataset_id,
                sku_id=sku_id,
                unit=unit,
                channel=channel,
                start_date=min_start_date,
                end_date=min_end_date
            )
            
            if temp_df.empty:
                result['Status'] = 'Error: No Data'
                result['error_message'] = f"No data found for SKU {sku_id}, Channel {channel}, Unit {unit}"
                logger.error(result['error_message'])
                return result
                
            # Determine actual date range
            actual_min_date = temp_df['ds'].min().strftime("%Y-%m-%d")
            actual_max_date = temp_df['ds'].max().strftime("%Y-%m-%d")
            
            result['min_date'] = actual_min_date
            result['max_date'] = actual_max_date
            
            # Use discovered dates if not explicitly provided
            if start_date is None:
                start_date = actual_min_date
            if end_date is None:
                end_date = actual_max_date
            
        # Now fetch with specified date range
        df_daily = fetch_sales_data(
            client=client,
            project_id=project_id,
            dataset_id=dataset_id,
            sku_id=sku_id,
            unit=unit,
            channel=channel,
            start_date=start_date,
            end_date=end_date
        )
        
        if df_daily.empty:
            result['Status'] = 'Error: No Data'
            result['error_message'] = f"No data found in specified date range for SKU {sku_id}, Channel {channel}, Unit {unit}"
            logger.error(result['error_message'])
            return result
            
        # Check if we have enough data (at least validation_days + 28 more days)
        min_required_days = validation_days + 28
        actual_days = (df_daily['ds'].max() - df_daily['ds'].min()).days + 1
        
        if actual_days < min_required_days:
            result['Status'] = 'Skipped: Insufficient Data'
            result['error_message'] = f"Insufficient data: {actual_days} days available, {min_required_days} required"
            logger.warning(result['error_message'])
            return result
            
        # 2. Fetch weather data
        df_weather = fetch_weather_data(
            latitude=weather_lat, 
            longitude=weather_lon, 
            start_date=start_date, 
            end_date=end_date
        )
        
        if df_weather is None or df_weather.empty:
            logger.warning(f"No weather data retrieved for {sku_id}-{channel}-{unit}. Proceeding without weather features.")
            df_merged = df_daily
        else:
            # Merge sales and weather data
            df_merged = pd.merge(df_daily, df_weather, on='ds', how='left')
            
            # Handle potential missing weather values
            weather_cols = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'snowfall_sum']
            for col in weather_cols:
                if col in df_merged.columns:
                    df_merged[col].fillna(0, inplace=True)
        
        # 3. Remove negative sales
        initial_rows = len(df_merged)
        df_merged = df_merged[df_merged['y'] >= 0]
        rows_removed = initial_rows - len(df_merged)
        logger.info(f"Removed {rows_removed} rows with negative sales.")
        
        # 4. Weekly aggregation
        # Ensure 'ds' is the index for resampling
        df_merged_daily_indexed = df_merged.set_index('ds')
        
        # Define aggregations
        aggregation_rules = {
            'y': 'sum', 
            'Promo_discount_perc': 'mean', 
            'is_promo': 'max'
        }
        
        # Add weather aggregations if columns exist
        weather_regressors = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'snowfall_sum']
        for reg in weather_regressors:
            if reg in df_merged_daily_indexed.columns:
                aggregation_rules[reg] = 'mean' if 'temp' in reg else 'sum'
        
        # Perform weekly resampling
        df_weekly = df_merged_daily_indexed.resample('W-SUN').agg(aggregation_rules)
        
        # Reset index to make 'ds' a column again
        df_weekly.reset_index(inplace=True)
        
        # Fill potential NaNs created by resampling
        df_weekly['y'].fillna(0, inplace=True)
        regressor_cols = [col for col in aggregation_rules.keys() if col != 'y']
        for col in regressor_cols:
            if col in df_weekly.columns:
                df_weekly[col].fillna(0, inplace=True)
        
        # Add y_lag1 feature
        df_weekly['y_lag1'] = df_weekly['y'].shift(1)
        df_weekly['y_lag1'].fillna(0, inplace=True)
        
        # Save the aggregated data
        df_weekly.to_csv(f"{output_dir}/weekly_aggregated_data_{sku_id}.csv", index=False)
        
        # 5. Prepare holiday data
        years = df_weekly['ds'].dt.year.unique()
        import holidays
        holidays_ru = holidays.RU(years=years)
        df_holidays = pd.DataFrame(list(holidays_ru.items()), columns=['ds', 'holiday'])
        df_holidays['ds'] = pd.to_datetime(df_holidays['ds'])
        
        # Filter holidays to match the range of df_weekly
        df_holidays = df_holidays[(df_holidays['ds'] >= df_weekly['ds'].min()) & 
                                  (df_holidays['ds'] <= df_weekly['ds'].max())]
        
        # Create holiday features for Prophet
        prophet_holidays = pd.DataFrame()
        prophet_holidays['ds'] = df_holidays['ds']
        prophet_holidays['holiday'] = 'RU_Holiday'
        prophet_holidays['lower_window'] = 0
        prophet_holidays['upper_window'] = 0
        
        # 6. Data splitting
        historical_end = df_weekly['ds'].max().date()
        validation_start = historical_end - timedelta(days=validation_days-1)
        validation_start_date = validation_start.strftime("%Y-%m-%d")
        
        result['validation_start'] = validation_start_date
        
        validation_start_dt = pd.to_datetime(validation_start_date)
        if validation_start_dt.weekday() == 6:  # Sunday
            first_validation_week_ds = validation_start_dt
        else:
            first_validation_week_ds = validation_start_dt + timedelta(days=(6 - validation_start_dt.weekday()))
            
        train_df = df_weekly[df_weekly['ds'] < validation_start_date].copy()
        validation_df = df_weekly[df_weekly['ds'] >= validation_start_date].copy()

        result['validation_start'] = validation_start_date # Store validation start date

        if train_df.empty or len(train_df) < 10: # Need some minimum data for training
             result['Status'] = 'Skipped: Insufficient Training Data'
             result['error_message'] = f"Insufficient training data after splitting: {len(train_df)} weeks"
             logger.warning(result['error_message'])
             return result

        logger.info(f"Train data: {len(train_df)} weeks ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
        logger.info(f"Validation data: {len(validation_df)} weeks ({validation_df['ds'].min().date()} to {validation_df['ds'].max().date()})")

        # Determine if yearly seasonality should be enabled
        training_duration_days = (train_df['ds'].max() - train_df['ds'].min()).days
        enable_yearly_seasonality = training_duration_days >= 365
        logger.info(f"Training data duration: {training_duration_days} days. Yearly seasonality enabled: {enable_yearly_seasonality}")

        # 7. Hyperparameter tuning
        logger.info("Starting hyperparameter tuning...")
        
        # Define CV settings - adjusted based on available data
        initial_weeks = max(len(train_df) - 8, int(len(train_df) * 0.6))  # Use at least 60% of data for initial training
        initial_period = f'{initial_weeks} W'
        period_interval = '4 W'
        cv_horizon = '4 W'
        
        logger.info(f"Starting hyperparameter tuning with CV settings: initial={initial_period}, period={period_interval}, horizon={cv_horizon}")
        
        best_params, df_cv_results = tune_hyperparameters(
            train_df=train_df,
            prophet_holidays=prophet_holidays,
            param_grid=None,  # Use default expanded parameter grid from tuning module
            initial_period=initial_period,
            period_interval=period_interval,
            cv_horizon=cv_horizon,
            weather_regressors=weather_regressors,
            use_lag_feature=True,
            rolling_window=3,  # Increase rolling window for smoother metrics
            optimize_param_grid=True,  # Use data-driven parameter grid optimization
            max_workers=4  # Use 4 workers for parallel hyperparameter tuning
        )
        
        # Save CV results
        if df_cv_results is not None and not df_cv_results.empty:
            df_cv_results.to_csv(f"{output_dir}/cv_tuning_results_{sku_id}.csv", index=False)
        
        result['best_params'] = str(best_params)
        
        # 8. Final model training
        logger.info(f"Training final model with best params: {best_params}")
        final_model = Prophet(
            holidays=prophet_holidays,
            changepoint_prior_scale=best_params['cps'],
            holidays_prior_scale=best_params['hps'],
            seasonality_prior_scale=best_params['sps'],
            weekly_seasonality=True,  # Always assume weekly
            daily_seasonality=False, # Data is weekly
            yearly_seasonality=enable_yearly_seasonality # Dynamically set based on data duration
        )

        # Add regressors used during training/tuning
        final_model.add_regressor('Promo_discount_perc')
        final_model.add_regressor('is_promo')
        for reg in weather_regressors:
            if reg in train_df.columns:
                final_model.add_regressor(reg)
        
        # Add lagged target regressor
        if 'y_lag1' in train_df.columns:
            final_model.add_regressor('y_lag1')
                
        # Fit the final model
        final_model.fit(train_df)
        
        # Save the final model locally
        # Use a consistent naming convention that matches the target GCS structure
        local_model_filename = f"{output_dir}/{unit}_{channel}_{sku_id}.pkl"
        with open(local_model_filename, 'wb') as f:
            pickle.dump(final_model, f)
        logger.info(f"Saved final model locally to {local_model_filename}")
        
        # --- Upload model to GCS --- 
        try:
            gcs_client = storage.Client()
            bucket_name = "eztech-442521-rich-distributor-forecast-models" # Use the correct bucket name
            # Ensure correct GCS path structure: Unit/Channel/SKU_ID.pkl
            destination_blob_name = f"{unit}/{channel}/{sku_id}.pkl"
            
            bucket = gcs_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)
            
            blob.upload_from_filename(local_model_filename)
            
            logger.info(f"Successfully uploaded model to gs://{bucket_name}/{destination_blob_name}")
            
        except Exception as gcs_error:
            logger.error(f"Failed to upload model to GCS: {gcs_error}")
            # Decide if this should be a critical error or just a warning
            # For now, log error but let the process continue if local save worked.
            # If GCS upload is mandatory, uncomment the line below to raise the error
            # raise gcs_error 
        # --------------------------

        # 9. Generate forecast for plotting
        logger.info("Generating full forecast for plotting")
        forecast_for_plotting = generate_forecast_for_plotting(
            model=final_model,
            train_df=train_df,
            validation_df=validation_df,
            weather_lat=weather_lat,
            weather_lon=weather_lon
        )
        
        # 10. Generate future predictions
        logger.info("Generating iterative predictions for 4 weeks")
        forecast_output = generate_future_predictions(
            model=final_model,
            train_df=train_df,
            validation_df=validation_df,
            weather_lat=weather_lat,
            weather_lon=weather_lon,
            forecast_periods=4
        )
        
        if forecast_output is not None and not forecast_output.empty:
            forecast_output.to_csv(f"{output_dir}/weekly_forecast_{unit}_{channel}_{sku_id}.csv", index=False)
        
        # 11. Evaluate forecast
        if not validation_df.empty and not forecast_output.empty:
            logger.info("Evaluating forecast against validation data")
            
            # Prepare validation actuals
            validation_actuals = validation_df[['ds', 'y']]
            
            # Merge forecasts with actuals
            comparison_df = pd.merge(validation_actuals, forecast_output, on='ds', how='inner')
            
            if not comparison_df.empty:
                # Calculate metrics
                mae = mean_absolute_error(comparison_df['y'], comparison_df['yhat'])
                rmse = np.sqrt(mean_squared_error(comparison_df['y'], comparison_df['yhat']))
                smape = _smape(comparison_df['y'], comparison_df['yhat'])
                
                # Calculate WAPE
                wape = np.sum(np.abs(comparison_df['y'] - comparison_df['yhat'])) / np.sum(np.abs(comparison_df['y'])) if np.sum(np.abs(comparison_df['y'])) != 0 else float('inf')
                
                # Update result with metrics
                result['MAE'] = mae
                result['RMSE'] = rmse
                result['sMAPE'] = smape
                result['WAPE'] = wape
                
                logger.info(f"Validation Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, sMAPE={smape:.4f}, WAPE={wape:.4f}")
                
                # Save comparison data
                comparison_df.to_csv(f"{output_dir}/forecast_vs_actuals_{unit}_{channel}_{sku_id}.csv", index=False)
                
                # Generate visualization
                if len(comparison_df) > 0:
                    generate_validation_plot(
                        comparison_df=comparison_df,
                        sku_id=sku_id,
                        output_dir=output_dir,
                        unit=unit,
                        channel=channel
                    )
            
            # Generate component plots if forecast_for_plotting exists
            if forecast_for_plotting is not None and not forecast_for_plotting.empty:
                generate_component_plots(
                    model=final_model,
                    forecast=forecast_for_plotting,
                    sku_id=sku_id,
                    output_dir=output_dir,
                    validation_df=validation_df,
                    unit=unit,
                    channel=channel
                )
        
        result['Status'] = 'Success'
        logger.info(f"Successfully completed processing for {sku_id}-{channel}-{unit}")
        
    except Exception as e:
        result['Status'] = 'Error: Processing Failed'
        result['error_message'] = str(e)
        logger.error(f"Error processing {sku_id}-{channel}-{unit}: {e}")
        logger.debug(traceback.format_exc())
    
    return result

def generate_forecast_for_plotting(model, train_df, validation_df=None, weather_lat=None, weather_lon=None):
    """
    Generate full forecast object including history for plotting purposes
    
    Args:
        model: Trained Prophet model
        train_df: Training data DataFrame
        validation_df: Validation data DataFrame with actual values for future periods (if available)
        weather_lat: Latitude for weather forecast data
        weather_lon: Longitude for weather forecast data
        
    Returns:
        DataFrame with forecast including history
    """
    try:
        # Create future dataframe including history
        future_df_for_plotting = model.make_future_dataframe(periods=4, freq='W-SUN', include_history=True)
        
        # Populate regressors for the entire dataframe (history + future)
        regressor_cols_to_merge = list(model.extra_regressors.keys())
        if 'ds' not in regressor_cols_to_merge:
            regressor_cols_to_merge = ['ds'] + regressor_cols_to_merge
        
        # Merge historical regressors
        df_for_plotting_merged = pd.merge(future_df_for_plotting, train_df[regressor_cols_to_merge], on='ds', how='left')
        
        # Identify future dates
        train_end_date = train_df['ds'].max()
        future_start_date = train_end_date + pd.Timedelta(weeks=1)
        future_mask = df_for_plotting_merged['ds'] >= future_start_date
        future_dates_for_plotting = df_for_plotting_merged.loc[future_mask, 'ds']
        
        # Prepare a template for future regressors
        plotting_future_regressors_df = pd.DataFrame({'ds': future_dates_for_plotting})
        
        # Check if validation data is available and use actual values instead of assumptions
        if validation_df is not None and not validation_df.empty:
            logger.info("Using actual promo values from validation data for forecasting")
            # Get promo columns from validation data
            promo_cols = [col for col in validation_df.columns if col in ['Promo_discount_perc', 'is_promo']]
            
            if promo_cols:
                # Merge validation data for promo values
                validation_promo_df = validation_df[['ds'] + promo_cols]
                plotting_future_regressors_df = pd.merge(
                    plotting_future_regressors_df, 
                    validation_promo_df,
                    on='ds', 
                    how='left'
                )
                
                logger.info(f"Found {len(promo_cols)} promo columns in validation data")
                # For any missing values (dates beyond validation period), use historical means
                for col in promo_cols:
                    if plotting_future_regressors_df[col].isnull().any():
                        historical_mean = train_df[col].mean()
                        plotting_future_regressors_df[col] = plotting_future_regressors_df[col].fillna(historical_mean)
                        logger.info(f"Filled missing {col} values with historical mean: {historical_mean:.2f}")
        else:
            # Use assumptions only when no validation data is available
            logger.info("No validation data available, using assumptions for promo values")
            plotting_future_regressors_df['Promo_discount_perc'] = 5.0  # Assumption
            plotting_future_regressors_df['is_promo'] = 1.0  # Assumption
        
        # Populate future weather
        df_weather_fcst_plotting = fetch_forecast_weather_data(weather_lat, weather_lon, forecast_days=35)
        if df_weather_fcst_plotting is not None and not df_weather_fcst_plotting.empty:
            df_weather_fcst_plotting_indexed = df_weather_fcst_plotting.set_index('ds')
            
            # Define weather aggregation rules
            weather_fcst_agg_rules = {
                'temperature_2m_max': 'mean', 
                'temperature_2m_min': 'mean', 
                'precipitation_sum': 'sum', 
                'snowfall_sum': 'sum'
            }
            
            valid_weather_fcst_agg_rules = {k: v for k, v in weather_fcst_agg_rules.items() 
                                            if k in df_weather_fcst_plotting_indexed.columns}
            
            df_weather_fcst_plotting_weekly = df_weather_fcst_plotting_indexed.resample('W-SUN').agg(valid_weather_fcst_agg_rules).reset_index()
            df_weather_fcst_plotting_weekly['ds'] = pd.to_datetime(df_weather_fcst_plotting_weekly['ds'])
            
            plotting_future_regressors_df = pd.merge(plotting_future_regressors_df, df_weather_fcst_plotting_weekly, on='ds', how='left')
        
        # Fill missing future weather with historical means
        weather_regressors = [col for col in model.extra_regressors.keys() 
                            if 'temp' in col or 'precip' in col or 'snow' in col]
        
        for reg in weather_regressors:
            if reg not in plotting_future_regressors_df.columns:
                plotting_future_regressors_df[reg] = np.nan
            if plotting_future_regressors_df[reg].isnull().any():
                if reg in train_df.columns:
                    historical_mean = train_df[reg].mean()
                    plotting_future_regressors_df[reg] = plotting_future_regressors_df[reg].fillna(historical_mean)
                else:
                    plotting_future_regressors_df[reg] = plotting_future_regressors_df[reg].fillna(0)
        
        # Merge these future regressors into the main plotting dataframe
        df_for_plotting_merged = pd.merge(df_for_plotting_merged, plotting_future_regressors_df, 
                                        on='ds', how='left', suffixes=('', '_future'))
        
        # Combine original and future columns
        for col in plotting_future_regressors_df.columns:
            if col != 'ds':
                future_col_name = col + '_future'
                if future_col_name in df_for_plotting_merged.columns:
                    df_for_plotting_merged[col] = df_for_plotting_merged[future_col_name].combine_first(df_for_plotting_merged[col])
                    df_for_plotting_merged.drop(columns=[future_col_name], inplace=True)
        
        # Ensure historical y_lag1 is correct
        df_for_plotting_merged['y_lag1'] = df_for_plotting_merged['y_lag1'].ffill()
        
        # Iteratively predict future y_lag1
        last_known_y = train_df.loc[train_df['ds'] == train_end_date, 'y'].iloc[0]
        
        # Ensure all required columns exist
        required_plotting_cols = list(model.extra_regressors.keys())
        for col in required_plotting_cols:
            if col not in df_for_plotting_merged.columns:
                df_for_plotting_merged[col] = 0
            # Fill NaNs
            df_for_plotting_merged[col] = df_for_plotting_merged[col].ffill().bfill().fillna(0)
        
        # Set initial lag for the first future date
        first_future_index = df_for_plotting_merged[df_for_plotting_merged['ds'] == future_start_date].index
        if not first_future_index.empty:
            df_for_plotting_merged.loc[first_future_index, 'y_lag1'] = last_known_y
        
        # Loop through future dates to predict and update next lag
        future_indices_to_iterate = df_for_plotting_merged[future_mask].index
        for i in range(len(future_indices_to_iterate)):
            current_index = future_indices_to_iterate[i]
            current_row = df_for_plotting_merged.loc[[current_index], ['ds'] + required_plotting_cols]
            
            # Fill NaNs
            if current_row.isnull().values.any():
                current_row = current_row.fillna(0)
            
            # Predict for current step
            predicted_values = model.predict(current_row)
            yhat_current = predicted_values['yhat'].iloc[0]
            
            # Update lag for next step
            if i + 1 < len(future_indices_to_iterate):
                next_index = future_indices_to_iterate[i+1]
                df_for_plotting_merged.loc[next_index, 'y_lag1'] = max(0, yhat_current)
        
        # Final check for NaNs
        if df_for_plotting_merged[required_plotting_cols].isnull().values.any():
            for col in required_plotting_cols:
                df_for_plotting_merged[col] = df_for_plotting_merged[col].ffill().bfill().fillna(0)
        
        # Generate final forecast
        forecast_for_plotting = model.predict(df_for_plotting_merged[['ds'] + required_plotting_cols])
        return forecast_for_plotting
        
    except Exception as e:
        logger.error(f"Error generating forecast for plotting: {e}")
        return None

def generate_future_predictions(model, train_df, validation_df=None, weather_lat=None, weather_lon=None, forecast_periods=4):
    """
    Generate future predictions iteratively using y_lag1 regressor
    
    Args:
        model: Trained Prophet model
        train_df: Training data DataFrame
        validation_df: Validation data DataFrame with actual values for future periods (if available)
        weather_lat: Latitude for weather forecast data
        weather_lon: Longitude for weather forecast data
        forecast_periods: Number of periods to forecast
        
    Returns:
        DataFrame with forecast
    """
    try:
        # Get the last date and last known 'y' from training data
        last_train_date = train_df['ds'].max()
        
        if 'y_lag1' not in train_df.columns:
            logger.error("y_lag1 column not found in train_df. Cannot perform iterative prediction.")
            return None
            
        last_known_y_lag1_for_pred = train_df.loc[train_df['ds'] == last_train_date, 'y'].iloc[0]
        
        # Prepare future regressor values
        future_dates = pd.date_range(start=last_train_date + timedelta(weeks=1), 
                                    periods=forecast_periods, freq='W-SUN')
        future_regressors_df = pd.DataFrame({'ds': future_dates})
        
        # Check if validation data is available and use actual values instead of assumptions
        if validation_df is not None and not validation_df.empty:
            logger.info("Using actual promo values from validation data for future predictions")
            # Get promo columns from validation data
            promo_cols = [col for col in validation_df.columns if col in ['Promo_discount_perc', 'is_promo']]
            
            if promo_cols:
                # Merge validation data for promo values
                validation_promo_df = validation_df[['ds'] + promo_cols]
                future_regressors_df = pd.merge(
                    future_regressors_df, 
                    validation_promo_df,
                    on='ds', 
                    how='left'
                )
                
                # For any missing values (dates beyond validation period), use historical means
                for col in promo_cols:
                    if future_regressors_df[col].isnull().any():
                        historical_mean = train_df[col].mean()
                        future_regressors_df[col] = future_regressors_df[col].fillna(historical_mean)
                        logger.info(f"Filled missing {col} values with historical mean: {historical_mean:.2f}")
        else:
            # Use assumptions only when no validation data is available
            logger.info("No validation data available, using assumptions for promo values")
            future_regressors_df['Promo_discount_perc'] = 5.0  # Assumption
            future_regressors_df['is_promo'] = 1.0  # Assumption
        
        # Populate future weather
        df_weather_fcst = fetch_forecast_weather_data(weather_lat, weather_lon, forecast_days=forecast_periods*7)
        if df_weather_fcst is not None and not df_weather_fcst.empty:
            df_weather_fcst_indexed = df_weather_fcst.set_index('ds')
            
            weather_fcst_agg_rules = {
                'temperature_2m_max': 'mean', 
                'temperature_2m_min': 'mean', 
                'precipitation_sum': 'sum', 
                'snowfall_sum': 'sum'
            }
            
            valid_weather_fcst_agg_rules = {k: v for k, v in weather_fcst_agg_rules.items() 
                                           if k in df_weather_fcst_indexed.columns}
            
            df_weather_fcst_weekly = df_weather_fcst_indexed.resample('W-SUN').agg(valid_weather_fcst_agg_rules).reset_index()
            df_weather_fcst_weekly['ds'] = pd.to_datetime(df_weather_fcst_weekly['ds'])
            
            future_regressors_df = pd.merge(future_regressors_df, df_weather_fcst_weekly, on='ds', how='left')
        
        # Fill missing future weather with historical means
        weather_regressors = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'snowfall_sum']
        for reg in weather_regressors:
            if reg not in future_regressors_df.columns:
                future_regressors_df[reg] = np.nan
            
            if future_regressors_df[reg].isnull().any():
                if reg in train_df.columns:
                    historical_mean = train_df[reg].mean()
                    future_regressors_df[reg] = future_regressors_df[reg].fillna(historical_mean)
                else:
                    future_regressors_df[reg] = future_regressors_df[reg].fillna(0)
        
        # Iterative prediction loop
        all_forecasts = []
        current_lag_value = last_known_y_lag1_for_pred
        
        for i in range(forecast_periods):
            # Create dataframe for single week
            current_date = future_dates[i]
            future_df_single = pd.DataFrame({'ds': [current_date]})
            
            # Merge regressors for this week
            regressors_for_week = future_regressors_df[future_regressors_df['ds'] == current_date]
            if regressors_for_week.empty:
                logger.error(f"Could not find pre-calculated regressors for week {current_date}")
                break
                
            future_df_single = pd.merge(future_df_single, regressors_for_week, on='ds', how='left')
            
            # Add lag value
            future_df_single['y_lag1'] = current_lag_value
            
            # Get required columns
            required_regressors = list(model.extra_regressors.keys())
            required_cols = ['ds'] + required_regressors
            
            # Check for missing columns
            missing_cols = False
            for col in required_cols:
                if col not in future_df_single.columns:
                    logger.error(f"Column '{col}' required by model is missing")
                    missing_cols = True
                    break
                    
            if missing_cols:
                break
            
            # Predict this week
            forecast_single = model.predict(future_df_single[required_cols])
            all_forecasts.append(forecast_single[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
            
            # Update lag for next iteration
            current_lag_value = max(0, forecast_single['yhat'].iloc[0])
            logger.info(f"Predicted week {i+1}/{forecast_periods}: {current_date}. Next lag: {current_lag_value:.2f}")
        
        # Combine forecasts
        if all_forecasts:
            forecast_output = pd.concat(all_forecasts, ignore_index=True)
            
            # Clip values at zero
            forecast_output['yhat'] = forecast_output['yhat'].clip(lower=0)
            forecast_output['yhat_lower'] = forecast_output['yhat_lower'].clip(lower=0)
            forecast_output['yhat_upper'] = forecast_output['yhat_upper'].clip(lower=0)
            
            return forecast_output
        else:
            logger.warning("No successful iterative predictions were made")
            return pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
            
    except Exception as e:
        logger.error(f"Error generating future predictions: {e}")
        return None

def generate_validation_plot(comparison_df, sku_id, output_dir, unit=None, channel=None):
    """Generate plot comparing actuals vs forecast for validation period"""
    try:
        fig = go.Figure()
        
        # Actuals
        fig.add_trace(go.Scatter(
            x=comparison_df['ds'], 
            y=comparison_df['y'], 
            mode='lines+markers', 
            name='Actual Sales'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=comparison_df['ds'], 
            y=comparison_df['yhat'], 
            mode='lines+markers', 
            name='Forecast (yhat)'
        ))
        
        # Uncertainty interval
        fig.add_trace(go.Scatter(
            x=comparison_df['ds'], 
            y=comparison_df['yhat_upper'], 
            mode='lines', 
            line=dict(width=0), 
            showlegend=False, 
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=comparison_df['ds'], 
            y=comparison_df['yhat_lower'], 
            mode='lines', 
            line=dict(width=0), 
            fill='tonexty', 
            fillcolor='rgba(68, 68, 68, 0.2)', 
            showlegend=False,
            name='Lower Bound'
        ))
        
        fig.update_layout(
            title=f'Weekly Actual Sales vs. Forecast (Validation) - SKU: {sku_id}, Channel: {channel}, Unit: {unit}',
            xaxis_title='Week (Sunday)',
            yaxis_title='Sales Quantity',
            legend_title="Legend"
        )
        
        file_prefix = f"{unit}_{channel}_{sku_id}" if unit and channel else sku_id
        fig.write_html(f"{output_dir}/validation_forecast_vs_actuals_{file_prefix}.html")
        logger.info(f"Validation plot saved to {output_dir}/validation_forecast_vs_actuals_{file_prefix}.html")
        
    except Exception as e:
        logger.error(f"Error generating validation plot: {e}")

def generate_component_plots(model, forecast, sku_id, output_dir, validation_df, unit=None, channel=None):
    """Generate Prophet component plots"""
    try:
        # Component plot
        fig_comp = model.plot_components(forecast)
        plt.suptitle(f'Prophet Model Components - SKU: {sku_id}, Channel: {channel}, Unit: {unit}', y=1.02)
        
        file_prefix = f"{unit}_{channel}_{sku_id}" if unit and channel else sku_id
        fig_comp.savefig(f"{output_dir}/prophet_model_components_{file_prefix}.png")
        plt.close(fig_comp)
        
        # Full history + forecast plot
        fig_full = model.plot(forecast)
        
        # Add vertical line for train/validation split
        if not validation_df.empty:
            split_date = validation_df['ds'].min()
            plt.axvline(split_date, color='r', linestyle='--', lw=2, 
                       label=f'Train/Validation Split ({split_date.date()})')
            plt.legend()
            
        plt.title(f'Full History and Forecast - SKU: {sku_id}, Channel: {channel}, Unit: {unit}')
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity (y)')
        fig_full.savefig(f"{output_dir}/full_history_forecast_{file_prefix}.png")
        plt.close(fig_full)
        
        logger.info(f"Component plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating component plots: {e}") 