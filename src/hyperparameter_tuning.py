import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import traceback # Import traceback for more detailed error logging
import logging
import concurrent.futures

# Set up logger
logger = logging.getLogger('hyperparameter_tuning')

# Helper function for sMAPE - placed here for use within tuning
def _smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Handle potential division by zero if both true and pred are zero
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)
    return np.mean(ratio)

def get_default_param_grid():
    """
    Returns an expanded default parameter grid for Prophet hyperparameter tuning.
    This provides more granular options than the original grid.
    """
    return {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5, 1.0],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0, 20.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.8, 0.9, 0.95]
    }

def get_optimized_param_grid(time_series_df=None):
    """
    Returns a parameter grid optimized based on time series characteristics.
    If a dataframe is provided, analyzes it to suggest appropriate parameters.
    Otherwise, returns a reasonable default grid.
    
    Args:
        time_series_df (pd.DataFrame, optional): DataFrame with 'ds' and 'y' columns
        
    Returns:
        dict: Parameter grid optimized for the data characteristics
    """
    # Start with default expanded grid
    param_grid = get_default_param_grid()
    
    # If no dataframe provided, return default grid
    if time_series_df is None or 'y' not in time_series_df.columns:
        return param_grid
    
    # Analyze time series characteristics
    y = time_series_df['y']
    
    # Check coefficient of variation (CV) - high CV suggests multiplicative seasonality
    cv = y.std() / y.mean() if y.mean() > 0 else 0
    
    # Check for zeros - many zeros might need special handling
    zero_pct = (y == 0).mean()
    
    # Check length of time series - shorter series need more regularization
    n_points = len(y)
    
    logger.info(f"Time series characteristics: CV={cv:.2f}, zero_pct={zero_pct:.2f}, n_points={n_points}")
    
    # Adjust grid based on characteristics
    if cv > 0.5:
        # High variability - prioritize multiplicative seasonality
        param_grid['seasonality_mode'] = ['multiplicative', 'additive']
    else:
        # Lower variability - prioritize additive seasonality
        param_grid['seasonality_mode'] = ['additive', 'multiplicative']
    
    if zero_pct > 0.3:
        # Many zeros - need more regularization
        param_grid['seasonality_prior_scale'] = [0.01, 0.1, 1.0, 5.0]
    
    if n_points < 30:
        # Short time series - need more regularization
        param_grid['changepoint_prior_scale'] = [0.001, 0.01, 0.05, 0.1]
        
    return param_grid

def evaluate_params(params, train_df, prophet_holidays, initial_period, period_interval, cv_horizon, 
                    weather_regressors, use_lag_feature, rolling_window, enable_yearly_seasonality_for_tuning):
    """
    Evaluate a single parameter combination and return the results.
    
    Args:
        params (dict): Parameter combination to evaluate
        train_df (pd.DataFrame): Training data
        prophet_holidays (pd.DataFrame): Holiday data for Prophet
        initial_period (str): Initial training period
        period_interval (str): Period interval for CV
        cv_horizon (str): Forecast horizon for CV
        weather_regressors (list): List of weather regressor columns
        use_lag_feature (bool): Whether to use lagged target as regressor
        rolling_window (int): Rolling window size for metrics calculation
        enable_yearly_seasonality_for_tuning (bool): Whether to enable yearly seasonality
        
    Returns:
        dict: Dictionary with evaluation results
    """
    try:
        # Create Prophet model with current parameters
        prophet_kwargs = {k: v for k, v in params.items()}
        m_tune = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=enable_yearly_seasonality_for_tuning,
            daily_seasonality=False,
            holidays=prophet_holidays,
            **prophet_kwargs
        )
        
        # Add regressors dynamically based on presence in train_df
        if 'Promo_discount_perc' in train_df.columns:
            m_tune.add_regressor('Promo_discount_perc')
        if 'is_promo' in train_df.columns:
            m_tune.add_regressor('is_promo')
            
        # Add weather regressors
        for reg in weather_regressors:
            if reg in train_df.columns:
                m_tune.add_regressor(reg)
                
        # Add lag feature if requested and available
        if use_lag_feature and 'y_lag1' in train_df.columns:
            m_tune.add_regressor('y_lag1')
        
        # Fit the model and run cross-validation
        m_tune.fit(train_df)
        df_cv = cross_validation(
            m_tune,
            initial=initial_period,
            period=period_interval,
            horizon=cv_horizon,
            parallel="processes"  # Ensure parallel processing
        )
        
        # Calculate performance metrics
        df_p = performance_metrics(df_cv, rolling_window=rolling_window)
        mae = df_p['mae'].values[0]
        rmse = df_p['rmse'].values[0]
        smape_val = _smape(df_cv['y'], df_cv['yhat'])
        
        # Return results
        result_entry = {
            'params': str(params),
            'mae': mae,
            'rmse': rmse,
            'smape': smape_val
        }
        
        # Add individual parameter values for easier analysis
        for k, v in params.items():
            result_entry[k] = v
            
        return result_entry, None  # Return result and no error
    
    except Exception as e:
        error_msg = f"Error during CV for params {params}: {e}"
        result_entry = {
            'params': str(params),
            'mae': float('inf'),
            'rmse': float('inf'),
            'smape': float('inf'),
            'error': str(e)
        }
        
        # Add individual parameter values
        for k, v in params.items():
            result_entry[k] = v
            
        return result_entry, error_msg  # Return result and error message

def tune_hyperparameters(train_df, prophet_holidays, param_grid=None, initial_period=None, 
                         period_interval='4 W', cv_horizon='4 W', weather_regressors=None, 
                         use_lag_feature=True, rolling_window=1, optimize_param_grid=True, max_workers=4):
    """
    Performs hyperparameter tuning using Prophet's cross-validation.

    Args:
        train_df (pd.DataFrame): Training data with 'ds', 'y', and regressor columns.
        prophet_holidays (pd.DataFrame): DataFrame of holidays for Prophet.
        param_grid (dict, optional): Dictionary defining the parameter grid to search.
                                     If None, uses get_default_param_grid().
        initial_period (str, optional): Initial training period for CV (e.g., '60 W').
                                       If None, automatically calculated based on data length.
        period_interval (str): Period interval for CV cuts (e.g., '4 W').
        cv_horizon (str): Forecast horizon for CV (e.g., '4 W').
        weather_regressors (list): List of weather regressor column names.
        use_lag_feature (bool): Whether to include 'y_lag1' as a regressor.
        rolling_window (int): Rolling window size for calculating performance metrics.
        optimize_param_grid (bool): Whether to optimize parameter grid based on data characteristics.
        max_workers (int): Number of worker processes for parallel grid search.

    Returns:
        tuple: (best_params, df_cv_results)
               - best_params (dict): Dictionary of the best hyperparameters found.
               - df_cv_results (pd.DataFrame): DataFrame containing results for all parameter combinations.
    """
    logger.info("\nStarting Hyperparameter Tuning (Prophet Cross-Validation)... ")
    
    # Determine if yearly seasonality should be enabled based on full training data duration
    training_duration_days = (train_df['ds'].max() - train_df['ds'].min()).days
    enable_yearly_seasonality_for_tuning = training_duration_days >= 365
    logger.info(f"Overall training data duration: {training_duration_days} days. Yearly seasonality for tuning: {enable_yearly_seasonality_for_tuning}")
    
    # Set default parameter grid if not provided
    if param_grid is None:
        if optimize_param_grid:
            param_grid = get_optimized_param_grid(train_df)
            logger.info("Using optimized parameter grid based on data characteristics")
        else:
            param_grid = get_default_param_grid()
            logger.info("Using default expanded parameter grid")
    
    # If weather_regressors not provided, initialize as empty list
    if weather_regressors is None:
        weather_regressors = []
    
    # Auto-calculate initial period if not provided
    if initial_period is None:
        # Use 70% of data for initial training, but at least 12 weeks
        n_weeks = len(train_df)
        initial_weeks = max(int(n_weeks * 0.7), min(12, n_weeks - 4))
        initial_period = f'{initial_weeks} W'
        logger.info(f"Auto-calculated initial_period: {initial_period} based on {n_weeks} weeks of data")
    
    logger.info(f"CV Settings: initial='{initial_period}', period='{period_interval}', horizon='{cv_horizon}', rolling_window={rolling_window}")
    
    # Create parameter grid
    grid = ParameterGrid(param_grid)
    all_params = list(grid)
    num_combinations = len(all_params)
    logger.info(f"Evaluating {num_combinations} parameter combinations in parallel using {max_workers} workers")
    
    # Initialize results list
    results_list = []
    
    # Determine how many workers to use (don't use more than needed)
    effective_workers = min(max_workers, num_combinations)
    
    # Run grid search in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
        # Create a list of futures
        futures = [
            executor.submit(
                evaluate_params, 
                params, 
                train_df, 
                prophet_holidays, 
                initial_period, 
                period_interval, 
                cv_horizon, 
                weather_regressors, 
                use_lag_feature, 
                rolling_window,
                enable_yearly_seasonality_for_tuning
            ) 
            for params in all_params
        ]
        
        # Process results as they complete
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result, error = future.result()
                results_list.append(result)
                
                # Log progress
                if (i+1) % 5 == 0 or i+1 == num_combinations:
                    logger.info(f"Progress: {i+1}/{num_combinations} parameter combinations evaluated ({(i+1)/num_combinations*100:.1f}%)")
                
                # Log results
                if error:
                    logger.error(error)
                else:
                    params_str = result['params']
                    mae = result['mae']
                    rmse = result['rmse']
                    smape = result['smape']
                    logger.info(f"  Params: {params_str} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, sMAPE: {smape:.4f}")
                    
            except Exception as e:
                logger.error(f"Error processing future: {e}")
    
    # Convert results to DataFrame
    df_cv_results = pd.DataFrame(results_list)
    
    # Find best parameters based on sMAPE
    if not df_cv_results.empty and 'smape' in df_cv_results.columns:
        # Filter out rows with infinite or NaN metrics
        valid_results = df_cv_results[pd.notnull(df_cv_results['smape']) & ~np.isinf(df_cv_results['smape'])]
        
        if not valid_results.empty:
            # Get best parameters based on sMAPE
            best_idx = valid_results['smape'].idxmin()
            best_row = valid_results.loc[best_idx]
            
            # Extract original parameters
            best_params = {
                'changepoint_prior_scale': best_row['changepoint_prior_scale'],
                'seasonality_prior_scale': best_row['seasonality_prior_scale'],
                'holidays_prior_scale': best_row['holidays_prior_scale'],
                'seasonality_mode': best_row['seasonality_mode'],
                'changepoint_range': best_row['changepoint_range']
            }
            
            best_smape = best_row['smape']
            best_mae = best_row['mae']
            
            logger.info(f"\nBest parameters found (based on sMAPE={best_smape:.4f}): {best_params}")
            logger.info(f"  (MAE associated with best sMAPE: {best_mae:.2f})")
            
            # Find best MAE for reference
            best_mae_idx = valid_results['mae'].idxmin()
            best_mae_value = valid_results.loc[best_mae_idx, 'mae']
            logger.info(f"  (Best MAE found during tuning was: {best_mae_value:.2f})")
        else:
            logger.warning("No valid results found (all had NaN or infinite metrics)")
            best_params = {}
    else:
        logger.warning("No valid results found in the hyperparameter tuning")
        best_params = {}
    
    # Save results
    try:
        output_file = "output/cv_tuning_results.csv"
        df_cv_results.to_csv(output_file, index=False)
        logger.info(f"CV tuning results saved to {output_file}")
    except Exception as save_e:
        logger.error(f"Error saving CV results: {save_e}")

    return best_params, df_cv_results 