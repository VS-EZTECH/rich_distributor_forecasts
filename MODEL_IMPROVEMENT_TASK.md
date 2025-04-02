# MODEL IMPROVEMENT TASK

## Goal
Improve the accuracy and reliability of the SKU sales forecasting models by implementing a series of targeted enhancements.

## Success Criteria
- **Primary Goal**: Reduce average WAPE below 0.7 (70%)
- **Secondary Goal**: Achieve sMAPE below 0.5 (50%) 
- **High-performing SKUs**: At least 75% of SKUs should have WAPE < 0.85
- **Extreme Failures**: No SKU should have WAPE > 1.0

## Phase 1: Quick Wins (1-2 days)

### 1. Fix Hyperparameter Selection

- [x] **1.1 Expand Parameter Grid**
  - [x] Modify `src/hyperparameter_tuning.py` to include finer-grained parameter values
  - [x] Add new `changepoint_range` parameter to the grid
  - [x] Add `seasonality_mode` to the cross-validation process
  - [x] Test the expanded grid with one sample SKU before full implementation
  - [x] Use specific Python path for all executions: `/Users/vandit/miniconda3/envs/fbprophet-env/bin/python`
  - [x] Run tests with command: `/Users/vandit/miniconda3/envs/fbprophet-env/bin/python run_all_skus.py --combinations_file data/sku_combinations.csv --specific_sku "П-00006477" --specific_channel "TF" --specific_unit "Пермь"`
  - [x] **HIGH PRIORITY:** Fix the pipeline to use actual promo values from validation data instead of fixed assumptions for forecasting

- [ ] **1.2 Test Manual Parameter Configuration**
  - [ ] Create a function to override automated parameters with manual best-known settings
  - [ ] Implement a configuration mechanism to specify per-SKU parameter overrides
  - [ ] Test the successful П-00006477 parameters (`cps=0.5, hps=0.1, sps=10.0`) on all SKUs
  - [ ] Compare results with the previous run and document improvements

- [ ] **1.3 Adjust CV Strategy**
  - [ ] Modify the cross-validation period calculation to be more robust for different data lengths
  - [ ] Increase `rolling_window` in `performance_metrics` call to smooth evaluation results
  - [ ] Add logic to dynamically select initial training period based on data characteristics
  - [ ] Add validation checks to ensure CV settings are appropriate for available data

- [ ] **1.4 Create Testing Harness**
  - [ ] Develop a quick testing module to evaluate parameter performance without full pipeline run
  - [ ] Add functionality to compare different parameter sets side-by-side
  - [ ] Implement visualization for hyperparameter sensitivity analysis
  - [ ] Create a report template for hyperparameter tuning results

### 2. Improve Regressor Handling

- [ ] **2.1 Analyze Feature Importance**
  - [ ] Implement `evaluate_regressor_importance()` function to assess correlation and mutual information
  - [ ] Apply this function to each SKU-Channel-Unit combination
  - [ ] Generate a report of regressor importance by SKU
  - [ ] Identify regressors with consistently low importance across SKUs

- [ ] **2.2 Implement Regressor Selection Logic**
  - [ ] Add logic to dynamically include/exclude regressors based on importance metrics
  - [ ] Set a minimum threshold for regressor inclusion (e.g., correlation > 0.2 or MI > 0.05)
  - [ ] Modify the model training pipeline to use only selected regressors
  - [ ] Add logging to track which regressors are used for each SKU

- [ ] **2.3 Improve Future Regressor Values**
  - [x] **HIGH PRIORITY:** Fix the pipeline to use actual promo values from validation data instead of fixed assumptions
  - [ ] Update `generate_future_predictions()` function to accept and use known promo values for forecast period
  - [ ] Modify data loading to ensure promotion data for validation period is preserved and utilized
  - [ ] Implement a fallback mechanism to use SKU-specific historical averages only when actual promo data is unavailable
  - [ ] Analyze historical patterns of promotion discounts by SKU-Channel-Unit for true future predictions
  - [ ] Implement seasonality detection for promotional activity for cases where no actual data exists

- [ ] **2.4 Weather Regressor Enhancement**
  - [ ] Analyze correlation between weather variables and sales by SKU
  - [ ] Test alternative aggregations for weather data (e.g., max vs. mean temperature)
  - [ ] Consider lagged weather effects (e.g., impact of previous week's weather)
  - [ ] Implement more robust handling of missing weather forecasts

### 3. Basic Data Quality Improvements

- [ ] **3.1 Identify and Handle Anomalies**
  - [ ] Implement a simple statistical outlier detection for weekly aggregated data
  - [ ] Add visualization of detected outliers for manual review
  - [ ] Create a mechanism to flag and report potential data quality issues
  - [ ] Test impact of outlier removal/adjustment on model performance

- [ ] **3.2 Add Weekly Pattern Detection**
  - [ ] Analyze weekly sales patterns by SKU to identify strong/weak days
  - [ ] Consider day-of-week effects when aggregating to weekly level
  - [ ] Test alternative weekly aggregation methods (weighted vs. simple sum)
  - [ ] Compare forecast accuracy with different aggregation approaches

### 4. Testing and Evaluation

- [ ] **4.1 Prepare Test Cases**
  - [ ] Select a representative subset of SKUs for quick testing (3-4 SKUs with varying performance)
  - [ ] Create a baseline of current performance metrics for these SKUs
  - [ ] Develop a standardized testing protocol for evaluating improvements
  - [ ] Set up automated comparison of before/after metrics

- [ ] **4.2 Implement Quick Validation**
  - [ ] Create a streamlined validation pipeline for rapid testing
  - [ ] Add functionality to run tests on historical data with known outcomes
  - [ ] Implement diagnostic visualizations for error analysis
  - [ ] Add detailed logging of model performance changes

- [ ] **4.3 Document Results**
  - [ ] Create a structured format for recording improvement outcomes
  - [ ] Set up version tracking for model parameters and results
  - [ ] Prepare a template for reporting findings to stakeholders
  - [ ] Develop a decision framework for implementing changes at scale

## Next Steps

After completing Phase 1, we will evaluate the results and proceed to Phase 2 (Advanced Improvements) which will focus on:

1. Implementing full outlier detection and handling
2. Adding seasonality auto-detection
3. Enhancing validation strategy with rolling origin evaluation
4. Implementing additional diagnostic tools

Phases 2 and 3 will be detailed after assessing the outcomes of Phase 1. 