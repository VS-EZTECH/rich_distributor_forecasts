# Model Improvement Plan for Distributor Forecasting

## Current Performance Analysis

Based on the most recent run (2025-04-02), the forecasting models are showing poor performance with concerning metrics:

- **Average Metrics Across 9 SKUs:**
  - MAE: 642.17
  - RMSE: 727.48
  - sMAPE: 0.85 (85%)
  - WAPE: 1.43 (143%)

The high sMAPE (85%) indicates that the average error is nearly as large as the average of the actual and predicted values. Even more concerning, the WAPE exceeding 100% suggests that the model is performing worse than a simple average prediction would.

When examining individual SKUs, we observe extreme cases:
- П-00006401 (ON TRADE, Пермь): WAPE of 6.34 (634%), with a forecast of 2389 against an actual value of 189
- Some SKUs perform reasonably (e.g., П-00006477 TF,Пермь with WAPE of 0.50)

## Identified Issues

1. **Poor Hyperparameter Selection**
   - The automated tuning algorithm is selecting suboptimal parameters for most SKUs
   - Manual parameter settings for П-00006477 (changepoint_prior_scale: 0.5, holidays_prior_scale: 0.1, seasonality_prior_scale: 10.0) performed better

2. **Inappropriate Seasonality Settings**
   - Mixed use of additive and multiplicative seasonality without clear reasoning
   - Possible overfitting of seasonal patterns with limited data

3. **Outliers and Data Issues**
   - Large discrepancies in some forecasts suggest outliers or data anomalies
   - Inadequate handling of sparse or zero values

4. **Regressor Problems**
   - Future regressors being set to assumed values (e.g., Promo_discount_perc=5.0, is_promo=1.0)
   - Weather regressors may be having unexpected effects on forecast

5. **Cross-Validation Strategy**
   - Current CV strategy may not be robust enough
   - Parameter grid may be too limited

## Practical Improvement Steps

### 1. Data Quality Enhancement

1. **Implement Outlier Detection and Handling**
   ```python
   def detect_and_handle_outliers(df, col='y', threshold=3):
       """Detect and handle outliers in a time series."""
       # Calculate median and MAD (Median Absolute Deviation)
       median = df[col].median()
       mad = np.median(np.abs(df[col] - median))
       
       # Flag outliers
       outlier_mask = (df[col] > median + threshold * mad) | (df[col] < median - threshold * mad)
       
       # Log outliers
       logger.info(f"Detected {outlier_mask.sum()} outliers in {col}")
       
       # Replace outliers with median or interpolated values
       df_handled = df.copy()
       if outlier_mask.sum() > 0:
           # Create a copy of original values for reference
           df_handled[f'{col}_original'] = df_handled[col]
           # Replace with rolling median or interpolation
           df_handled.loc[outlier_mask, col] = df_handled[col].rolling(
               window=5, center=True, min_periods=1).median().fillna(method='ffill').fillna(method='bfill')
       
       return df_handled, outlier_mask
   ```

2. **Add Data Quality Checks and Reporting**
   - Implement checks for data completeness, consistency, and distribution
   - Generate data quality reports for each SKU-Channel-Unit combination
   - Flag combinations with potential data issues

### 2. Hyperparameter Tuning Improvements

1. **Expand Parameter Grid**
   ```python
   param_grid = {
       'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
       'seasonality_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0, 20.0],
       'holidays_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0],
       'seasonality_mode': ['additive', 'multiplicative'],
       'changepoint_range': [0.8, 0.9, 0.95]  # New parameter
   }
   ```

2. **Implement Smarter CV Strategy**
   - Use time-based K-fold cross-validation
   - Adjust CV period based on data quantity
   - Consider expanding `rolling_window` in performance_metrics

3. **Create SKU-Specific Parameter Recommendations**
   - Develop logic to recommend parameters based on data characteristics
   - Use rules like "if high variance, use multiplicative seasonality"

### 3. Feature Engineering

1. **Improve Regressor Selection**
   ```python
   def evaluate_regressor_importance(train_df, target_col='y', regressors=None):
       """Evaluate importance of regressors using correlation and mutual information."""
       from sklearn.feature_selection import mutual_info_regression
       
       if regressors is None:
           # Use all numeric columns except target and datetime
           regressors = [col for col in train_df.columns 
                         if col != target_col and col != 'ds' 
                         and np.issubdtype(train_df[col].dtype, np.number)]
       
       # Calculate correlations
       correlations = {reg: train_df[reg].corr(train_df[target_col]) 
                      for reg in regressors if not train_df[reg].isna().all()}
       
       # Calculate mutual information for non-linear relationships
       X = train_df[regressors].fillna(0)
       y = train_df[target_col]
       mi_scores = mutual_info_regression(X, y)
       mi_values = {regressors[i]: score for i, score in enumerate(mi_scores)}
       
       return {'correlation': correlations, 'mutual_info': mi_values}
   ```

2. **Dynamic Lag Feature Selection**
   - Test multiple lag features (y_lag1, y_lag2, etc.)
   - Include rolling averages as features (e.g., 3-week moving average)
   - Evaluate each feature's predictive power

3. **Improve Future Regressor Prediction**
   - Instead of fixed assumptions, use historical patterns for promotions
   - Consider seasonality in promotions/discounts

### 4. Model Architecture Changes

1. **Implement Auto-Seasonality Detection**
   ```python
   def detect_seasonality(time_series, column='y'):
       """Detect appropriate seasonality mode based on data characteristics."""
       from statsmodels.tsa.seasonal import seasonal_decompose
       
       # Check if time series is long enough for yearly seasonality
       if len(time_series) >= 52:  # At least a year of weekly data
           # Decompose the time series
           decomposition = seasonal_decompose(time_series[column], model='additive', period=52)
           
           # Calculate relative magnitude of seasonality component
           seasonal_magnitude = decomposition.seasonal.std() / time_series[column].std()
           
           # Calculate coefficient of variation to help decide on seasonality mode
           cv = time_series[column].std() / time_series[column].mean()
           
           # Log findings
           logger.info(f"Seasonal magnitude: {seasonal_magnitude:.4f}, CV: {cv:.4f}")
           
           # Decision logic
           if cv > 0.5 and seasonal_magnitude > 0.3:
               return 'multiplicative'
           else:
               return 'additive'
       else:
           # Default to additive for short time series
           return 'additive'
   ```

2. **Implement Ensemble Forecasting**
   - Train multiple Prophet models with different parameters
   - Average predictions for improved stability
   - Consider weighted averages based on CV performance

### 5. Model Validation and Evaluation

1. **Implement Rolling Origin Evaluation**
   - Test model on multiple validation periods
   - Report stability of performance across periods

2. **Add Visual Diagnostics**
   - Plot prediction intervals vs. actuals
   - Visualize component contributions
   - Create residual plots for error analysis

3. **Develop Model Scoring System**
   ```python
   def score_model_fit(forecast_df, actual_df, metrics=['mae', 'rmse', 'smape', 'wape']):
       """Score model fit quality and flag potential issues."""
       scores = {}
       issues = []
       
       # Calculate basic metrics
       actuals = actual_df['y'].values
       predictions = forecast_df['yhat'].values
       
       if 'mae' in metrics:
           scores['mae'] = mean_absolute_error(actuals, predictions)
       if 'rmse' in metrics:
           scores['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))
       if 'smape' in metrics:
           scores['smape'] = _smape(actuals, predictions)
       if 'wape' in metrics:
           scores['wape'] = np.sum(np.abs(actuals - predictions)) / np.sum(np.abs(actuals))
       
       # Check for systematic bias
       mean_error = np.mean(predictions - actuals)
       if abs(mean_error) > 0.2 * np.mean(actuals):
           issues.append(f"Systematic bias detected: {mean_error:.2f}")
       
       # Check for outliers in errors
       errors = predictions - actuals
       error_std = np.std(errors)
       if np.any(np.abs(errors) > 3 * error_std):
           extreme_errors = np.sum(np.abs(errors) > 3 * error_std)
           issues.append(f"Extreme prediction errors found: {extreme_errors} points")
       
       # Check for performance deterioration
       if len(actuals) >= 4:
           first_half_wape = np.sum(np.abs(actuals[:len(actuals)//2] - predictions[:len(actuals)//2])) / np.sum(np.abs(actuals[:len(actuals)//2]))
           second_half_wape = np.sum(np.abs(actuals[len(actuals)//2:] - predictions[len(actuals)//2:])) / np.sum(np.abs(actuals[len(actuals)//2:]))
           if second_half_wape > 1.5 * first_half_wape:
               issues.append(f"Performance deterioration detected: WAPE increased from {first_half_wape:.2f} to {second_half_wape:.2f}")
       
       return {"scores": scores, "issues": issues}
   ```

## Implementation Plan

### Phase 1: Quick Wins (1-2 days)

1. **Fix Hyperparameter Selection**
   - Implement the expanded parameter grid
   - Test the successful manual parameters on all SKUs
   - Rerun forecast and evaluate improvements

2. **Improve Regressor Handling**
   - Analyze correlation between each regressor and target
   - Remove weakly correlated regressors
   - Improve future regressor values based on historical patterns

### Phase 2: Advanced Improvements (3-5 days)

1. **Implement Outlier Detection and Handling**
   - Add outlier detection to data preparation pipeline
   - Test impact on model performance

2. **Add Seasonality Auto-detection**
   - Implement seasonality detection
   - Compare results with fixed seasonality setting

3. **Enhance Validation Strategy**
   - Implement rolling origin evaluation
   - Create improved model scoring system

### Phase 3: Production Refinements (1 week)

1. **Implement Ensemble Forecasting**
   - Develop ensemble approach
   - Compare with single-model approach

2. **Create Automated Reporting**
   - Generate detailed performance reports
   - Identify problematic SKUs and recommend specific improvements

3. **Optimize Pipeline**
   - Refine parallel processing
   - Improve error handling and logging

## Success Criteria

- **Primary Goal**: Reduce average WAPE below 0.7 (70%)
- **Secondary Goal**: Achieve sMAPE below 0.5 (50%)
- **High-performing SKUs**: At least 75% of SKUs should have WAPE < 0.85
- **Extreme Failures**: No SKU should have WAPE > 1,0

These improvements focus on practical, implementable changes to the existing codebase while addressing the core issues identified in the current forecasting performance. 