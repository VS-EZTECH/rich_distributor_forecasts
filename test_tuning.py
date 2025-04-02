#!/usr/bin/env python
"""
Test script for improved hyperparameter tuning on a specific SKU.
Usage: /Users/vandit/miniconda3/envs/fbprophet-env/bin/python test_tuning.py
"""

import pandas as pd
import numpy as np
import logging
import os
import pickle
from google.cloud import bigquery
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import functions from the project modules
from src.data_loader import fetch_sales_data
from src.weather_handler import fetch_weather_data
from src.hyperparameter_tuning import tune_hyperparameters, get_default_param_grid, get_optimized_param_grid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/test_tuning.log')
    ]
)

logger = logging.getLogger('test_tuning')

def prepare_test_data(sku_id="П-00006477", channel="TF", unit="Пермь"):
    """
    Prepare test data for a specific SKU-Channel-Unit combination.
    
    Args:
        sku_id (str): Target SKU ID
        channel (str): Target channel
        unit (str): Target unit
        
    Returns:
        tuple: (train_df, validation_df, prophet_holidays)
    """
    logger.info(f"Preparing test data for SKU_ID={sku_id}, Channel={channel}, Unit={unit}")
    
    # Initialize BigQuery client
    client = bigquery.Client(project="eztech-442521")
    
    # Fetch data
    start_date = "2023-01-01"
    end_date = "2025-03-31"
    
    df_daily = fetch_sales_data(
        client=client,
        project_id="eztech-442521",
        dataset_id="rich",
        sku_id=sku_id,
        unit=unit,
        channel=channel,
        start_date=start_date,
        end_date=end_date
    )
    
    if df_daily.empty:
        logger.error(f"No data found for SKU {sku_id}, Channel {channel}, Unit {unit}")
        return None, None, None
    
    # Fetch weather data
    weather_lat = 58.0  # Perm, Russia
    weather_lon = 56.3  # Perm, Russia
    
    df_weather = fetch_weather_data(
        latitude=weather_lat,
        longitude=weather_lon,
        start_date=start_date,
        end_date=end_date
    )
    
    if df_weather is None or df_weather.empty:
        logger.warning(f"No weather data retrieved. Proceeding without weather features.")
        df_merged = df_daily
    else:
        # Merge sales and weather data
        df_merged = pd.merge(df_daily, df_weather, on='ds', how='left')
        
        # Handle potential missing weather values
        weather_cols = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'snowfall_sum']
        for col in weather_cols:
            if col in df_merged.columns:
                df_merged[col].fillna(0, inplace=True)
    
    # Remove negative sales
    initial_rows = len(df_merged)
    df_merged = df_merged[df_merged['y'] >= 0]
    rows_removed = initial_rows - len(df_merged)
    logger.info(f"Removed {rows_removed} rows with negative sales.")
    
    # Weekly aggregation
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
    
    # Perform weekly resampling (Monday-Sunday, labeled with Monday)
    df_weekly = df_merged_daily_indexed.resample('W-SUN').agg(aggregation_rules)
    
    # Reset index to make 'ds' a column again
    df_weekly.reset_index(inplace=True)
    
    # Fill potential NaNs
    df_weekly['y'].fillna(0, inplace=True)
    regressor_cols = [col for col in aggregation_rules.keys() if col != 'y']
    for col in regressor_cols:
        if col in df_weekly.columns:
            df_weekly[col].fillna(0, inplace=True)
    
    # Add y_lag1 feature
    df_weekly['y_lag1'] = df_weekly['y'].shift(1)
    df_weekly['y_lag1'].fillna(0, inplace=True)
    
    # Save the aggregated data for reference
    os.makedirs("output/test_tuning", exist_ok=True)
    df_weekly.to_csv(f"output/test_tuning/weekly_aggregated_data_{sku_id}.csv", index=False)
    
    # Prepare holiday data
    years = df_weekly['ds'].dt.year.unique()
    import holidays
    holidays_ru = holidays.RU(years=years)
    df_holidays = pd.DataFrame(list(holidays_ru.items()), columns=['ds', 'holiday'])
    df_holidays['ds'] = pd.to_datetime(df_holidays['ds'])
    
    # Filter holidays to match the range of df_weekly
    df_holidays = df_holidays[
        (df_holidays['ds'] >= df_weekly['ds'].min()) & 
        (df_holidays['ds'] <= df_weekly['ds'].max())
    ]
    
    # Create holiday features for Prophet
    prophet_holidays = pd.DataFrame()
    prophet_holidays['ds'] = df_holidays['ds']
    prophet_holidays['holiday'] = 'RU_Holiday'
    prophet_holidays['lower_window'] = 0
    prophet_holidays['upper_window'] = 0
    
    # Data splitting - use last 4 weeks as validation
    validation_days = 28
    historical_end = df_weekly['ds'].max().date()
    validation_start = historical_end - timedelta(days=validation_days-1)
    validation_start_date = validation_start.strftime("%Y-%m-%d")
    
    validation_start_dt = pd.to_datetime(validation_start_date)
    if validation_start_dt.weekday() == 6:  # Sunday
        first_validation_week_ds = validation_start_dt
    else:
        first_validation_week_ds = validation_start_dt + timedelta(days=(6 - validation_start_dt.weekday()))
        
    train_df = df_weekly[df_weekly['ds'] < first_validation_week_ds]
    validation_df = df_weekly[df_weekly['ds'] >= first_validation_week_ds]
    
    logger.info(f"Training set: {len(train_df)} weeks, Validation set: {len(validation_df)} weeks")
    
    return train_df, validation_df, prophet_holidays

def run_tuning_test():
    """Run the hyperparameter tuning test on the specified SKU."""
    # Prepare test data
    train_df, validation_df, prophet_holidays = prepare_test_data(
        sku_id="П-00006477",
        channel="TF",
        unit="Пермь"
    )
    
    if train_df is None:
        logger.error("Failed to prepare test data. Exiting.")
        return
    
    # Define weather regressors
    weather_regressors = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'snowfall_sum']
    available_weather_regressors = [reg for reg in weather_regressors if reg in train_df.columns]
    
    # 1. Get the default expanded parameter grid
    default_grid = get_default_param_grid()
    logger.info(f"Default expanded parameter grid has {len(default_grid)} parameters")
    
    # 2. Get an optimized parameter grid based on the data characteristics
    optimized_grid = get_optimized_param_grid(train_df)
    logger.info(f"Optimized parameter grid prioritizes certain parameters based on data characteristics")
    
    # Define CV settings
    initial_weeks = max(len(train_df) - 8, int(len(train_df) * 0.6))
    initial_period = f'{initial_weeks} W'
    period_interval = '4 W' 
    cv_horizon = '4 W'
    
    # 3. Run the tuning with optimized grid
    logger.info("Running hyperparameter tuning with optimized grid...")
    best_params, df_cv_results = tune_hyperparameters(
        train_df=train_df,
        prophet_holidays=prophet_holidays,
        param_grid=optimized_grid,
        initial_period=initial_period,
        period_interval=period_interval,
        cv_horizon=cv_horizon,
        weather_regressors=available_weather_regressors,
        use_lag_feature=True,
        rolling_window=3,
        optimize_param_grid=False  # We're already using an optimized grid
    )
    
    # Save results
    output_dir = "output/test_tuning"
    df_cv_results.to_csv(f"{output_dir}/cv_tuning_results.csv", index=False)
    
    # Also test with known good parameters
    logger.info("Testing with known good parameters...")
    manual_params = {
        'changepoint_prior_scale': 0.5,
        'holidays_prior_scale': 0.1,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative'
    }
    
    logger.info(f"Manual best parameters: {manual_params}")
    
    # Compare results
    if best_params:
        logger.info(f"Auto-tuned best parameters: {best_params}")
        
        # Compare parameter values
        for key in ['changepoint_prior_scale', 'holidays_prior_scale', 'seasonality_prior_scale', 'seasonality_mode']:
            if key in best_params and key in manual_params:
                auto_val = best_params[key]
                manual_val = manual_params[key]
                logger.info(f"Parameter {key}: Auto={auto_val}, Manual={manual_val}")
    
    logger.info("Hyperparameter tuning test completed successfully.")
    return best_params, df_cv_results

if __name__ == "__main__":
    logger.info("Starting hyperparameter tuning test script")
    os.makedirs("output/test_tuning", exist_ok=True)
    
    try:
        best_params, df_cv_results = run_tuning_test()
        
        # If we got results, create a simple visualization of parameter impacts
        if df_cv_results is not None and not df_cv_results.empty:
            # Clean up the results for analysis
            numeric_df = df_cv_results.copy()
            
            # Drop rows with infinity metrics
            numeric_df = numeric_df[numeric_df['mae'] < float('inf')]
            
            # Try to extract individual parameters if they exist
            param_cols = ['changepoint_prior_scale', 'holidays_prior_scale', 
                           'seasonality_prior_scale', 'seasonality_mode', 
                           'changepoint_range']
            
            param_cols_present = [col for col in param_cols if col in numeric_df.columns]
            
            if param_cols_present and len(numeric_df) > 0:
                # Create parameter impact visualizations
                fig, axes = plt.subplots(len(param_cols_present), 1, figsize=(10, 3*len(param_cols_present)))
                
                for i, param in enumerate(param_cols_present):
                    ax = axes[i] if len(param_cols_present) > 1 else axes
                    
                    # For categorical parameters like seasonality_mode
                    if numeric_df[param].dtype == 'object':
                        grouped = numeric_df.groupby(param)['smape'].mean().reset_index()
                        ax.bar(grouped[param], grouped['smape'])
                        ax.set_title(f'Impact of {param} on sMAPE')
                        ax.set_ylabel('sMAPE')
                    else:
                        # For numeric parameters
                        ax.scatter(numeric_df[param], numeric_df['smape'])
                        ax.set_title(f'Impact of {param} on sMAPE')
                        ax.set_xlabel(param)
                        ax.set_ylabel('sMAPE')
                        
                plt.tight_layout()
                plt.savefig('output/test_tuning/parameter_impact.png')
                logger.info("Created parameter impact visualization")
        
        logger.info("Test script completed successfully")
        
    except Exception as e:
        logger.error(f"Error in test script: {e}")
        import traceback
        logger.error(traceback.format_exc()) 