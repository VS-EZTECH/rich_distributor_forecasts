import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import traceback # Import traceback for more detailed error logging
import logging

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

def tune_hyperparameters(train_df, prophet_holidays, param_grid=None, initial_period=None, 
                         period_interval='4 W', cv_horizon='4 W', weather_regressors=None, 
                         use_lag_feature=True, rolling_window=1, optimize_param_grid=True):
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

    Returns:
        tuple: (best_params, df_cv_results)
               - best_params (dict): Dictionary of the best hyperparameters found.
               - df_cv_results (pd.DataFrame): DataFrame containing results for all parameter combinations.
    """
    logger.info("\nStarting Hyperparameter Tuning (Prophet Cross-Validation)... ")
    
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
    logger.info(f"Evaluating {len(all_params)} parameter combinations...")
    
    # Track best parameters
    best_params = None
    best_mae = float('inf')
    best_smape = float('inf')
    best_mae_at_best_smape = float('inf')
    results_list = []

    # Go through all parameter combinations
    for i, params in enumerate(all_params):
        logger.info(f"  Testing params {i+1}/{len(all_params)}: {params}")
        try:
            # Create Prophet model with current parameters
            prophet_kwargs = {k: v for k, v in params.items()}
            m_tune = Prophet(
                weekly_seasonality=True,
                yearly_seasonality=True,
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
            elif use_lag_feature:
                logger.warning(f"y_lag1 specified but not found in train_df for params {params}.")

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

            # Store results
            result_entry = {
                'params': str(params),
                'mae': mae,
                'rmse': rmse,
                'smape': smape_val
            }
            
            # Add individual parameter values for easier analysis
            for k, v in params.items():
                result_entry[k] = v
                
            results_list.append(result_entry)
            
            logger.info(f"    Params: {params} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, sMAPE: {smape_val:.4f}")

            # Track best parameters based on sMAPE
            if smape_val < best_smape:
                best_smape = smape_val
                best_mae_at_best_smape = mae
                best_params = params
                logger.info(f"    *** New best sMAPE found: {best_smape:.4f} (MAE: {best_mae_at_best_smape:.2f}) ***")

            # Also track best MAE for reference
            if mae < best_mae:
                best_mae = mae

        except Exception as e:
            logger.error(f"    Error during CV for params {params}: {e}")
            # Log traceback for detailed debugging
            logger.debug(traceback.format_exc())
            # Add failed parameters to results with infinity metrics
            result_entry = {
                'params': str(params),
                'mae': float('inf'),
                'rmse': float('inf'),
                'smape': float('inf'),
                'error': str(e)
            }
            for k, v in params.items():
                result_entry[k] = v
            results_list.append(result_entry)

    # Report results
    logger.info(f"\nHyperparameter tuning finished.")
    if best_params:
        logger.info(f"Best parameters found (based on sMAPE={best_smape:.4f}): {best_params}")
        logger.info(f"  (MAE associated with best sMAPE: {best_mae_at_best_smape:.2f})")
        logger.info(f"  (Best MAE found during tuning was: {best_mae:.2f})")
    else:
        logger.warning("No best parameters found, CV might have failed for all combinations.")
        best_params = {}  # Return empty dict to indicate failure or use defaults

    # Convert results to DataFrame and save
    df_cv_results = pd.DataFrame(results_list)
    try:
        output_file = "output/cv_tuning_results.csv"
        df_cv_results.to_csv(output_file, index=False)
        logger.info(f"CV tuning results saved to {output_file}")
    except Exception as save_e:
        logger.error(f"Error saving CV results: {save_e}")

    return best_params, df_cv_results 