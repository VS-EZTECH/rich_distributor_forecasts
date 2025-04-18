import os
import argparse
import pandas as pd
from datetime import datetime
from google.cloud import bigquery, storage
from google.oauth2 import service_account
import logging
import concurrent.futures
import traceback

# Import the combination loader functions
from src.sku_combination_loader import filter_combinations, load_sku_combinations_from_gsheet

# Import the main processing function
from src.forecasting import process_combination

# Configure logging
run_log_dir = 'output'
run_log_file = os.path.join(run_log_dir, 'run_all_skus.log')
os.makedirs(run_log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(run_log_file)
    ]
)
logger = logging.getLogger('run_all_skus')

# Define path for the other log file (relative to script execution)
forecasting_log_file = os.path.join(run_log_dir, 'forecasting_process.log')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the forecasting pipeline for multiple SKU-Channel-Unit combinations.")
    
    # Input source arguments (Google Sheet only)
    parser.add_argument("--combinations_gsheet_id", type=str, required=True,
                           help="Google Sheet ID for the combinations.")
    parser.add_argument("--combinations_gsheet_name", type=str, default="Target",
                      help="Name of the tab in the Google Sheet (default: Target).")
    parser.add_argument("--gsheet_creds_path", type=str, default=".secrets/eztech-442521-sheets.json",
                      help="Path to the Google Sheets service account credentials JSON file (default: .secrets/eztech-442521-sheets.json).")

    # Added BQ Creds Path argument
    parser.add_argument("--bq_creds_path", type=str, default=".secrets/eztech-442521-bigquery.json",
                      help="Path to the BigQuery service account credentials JSON file (default: .secrets/eztech-442521-bigquery.json).")

    parser.add_argument("--output_dir", type=str, default="output",
                      help="Base directory for outputs.")
    
    parser.add_argument("--project_id", type=str, default="eztech-442521",
                      help="Google Cloud Project ID.")
    
    parser.add_argument("--dataset_id", type=str, default="rich",
                      help="BigQuery dataset ID.")
    
    parser.add_argument("--start_date", type=str, default=None,
                      help="Optional start date for data retrieval (YYYY-MM-DD). If not provided, all available data will be used.")
    
    parser.add_argument("--end_date", type=str, default=None,
                      help="Optional end date for data retrieval (YYYY-MM-DD). If not provided, all available data will be used.")
    
    parser.add_argument("--validation_days", type=int, default=28,
                      help="Number of days for validation period. Default is 28 (4 weeks).")
    
    parser.add_argument("--weather_latitude", type=float, default=58.0,
                      help="Latitude for weather data (default: 58.0 - Perm, Russia).")
    
    parser.add_argument("--weather_longitude", type=float, default=56.3,
                      help="Longitude for weather data (default: 56.3 - Perm, Russia).")
    
    parser.add_argument("--max_combinations", type=int, default=None,
                      help="Optional limit on the number of combinations to process. Useful for testing.")
    
    parser.add_argument("--specific_sku", type=str, default=None,
                      help="Optional filter for a specific SKU_ID.")
    
    parser.add_argument("--specific_unit", type=str, default=None,
                      help="Optional filter for a specific Unit.")
    
    parser.add_argument("--specific_channel", type=str, default=None,
                      help="Optional filter for a specific Channel.")
    
    parser.add_argument("--parallel", action="store_true",
                      help="Run processing in parallel using multiple processors.")
    
    parser.add_argument("--workers", type=int, default=4,
                      help="Number of worker processes to use when parallel=True. Default is 4.")
    
    return parser.parse_args()

def initialize_bigquery_client(project_id, credentials_path):
    """Initialize a BigQuery client using service account credentials."""
    try:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = bigquery.Client(project=project_id, credentials=credentials)
        logger.info(f"BigQuery client initialized successfully using {credentials_path}.")
        return client
    except FileNotFoundError:
        logger.error(f"Error initializing BigQuery client: Credentials file not found at {credentials_path}")
        return None
    except Exception as e:
        logger.error(f"Error initializing BigQuery client: {e}")
        return None

def create_output_directories(base_dir):
    """Create necessary output directories."""
    # Create the base output directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directories for logs and results
    os.makedirs(os.path.join(base_dir, 'forecasts'), exist_ok=True)
    
    logger.info(f"Created output directories under {base_dir}")

def run_single_combination(args, client, combination_row):
    """Run the forecasting pipeline for a single combination."""
    sku_id = combination_row['SKU_ID']
    channel = combination_row['Channel']
    unit = combination_row['Unit']
    
    logger.info(f"Processing combination: SKU_ID={sku_id}, Channel={channel}, Unit={unit}")
    
    try:
        # Initialize a new BigQuery client if not provided (for parallel processing)
        local_client = client
        if local_client is None:
            try:
                # Use credentials path here as well
                credentials = service_account.Credentials.from_service_account_file(args.bq_creds_path)
                local_client = bigquery.Client(project=args.project_id, credentials=credentials)
                logger.info(f"Initialized new BigQuery client for {sku_id}-{channel}-{unit} using {args.bq_creds_path}")
            except FileNotFoundError:
                logger.error(f"Error initializing BigQuery client in worker process: Credentials file not found at {args.bq_creds_path}")
                raise # Re-raise to stop this worker
            except Exception as client_err:
                logger.error(f"Error initializing BigQuery client in worker process: {client_err}")
                raise # Re-raise to stop this worker
                
        result = process_combination(
            client=local_client,
            project_id=args.project_id,
            dataset_id=args.dataset_id,
            sku_id=sku_id,
            channel=channel,
            unit=unit,
            weather_lat=args.weather_latitude,
            weather_lon=args.weather_longitude,
            start_date=args.start_date,
            end_date=args.end_date,
            validation_days=args.validation_days,
            base_output_dir=args.output_dir
        )
        
        logger.info(f"Completed processing: SKU_ID={sku_id}, Channel={channel}, Unit={unit}, Status={result['Status']}")
        return result
        
    except Exception as e:
        error_msg = f"Error processing {sku_id}-{channel}-{unit}: {e}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        
        # Return a result with error status
        return {
            'SKU_ID': sku_id,
            'Channel': channel,
            'Unit': unit,
            'Status': 'Error: Uncaught Exception',
            'min_date': None,
            'max_date': None,
            'validation_start': None,
            'MAE': None,
            'RMSE': None,
            'sMAPE': None,
            'WAPE': None,
            'best_params': None,
            'error_message': str(e)
        }

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directories
    create_output_directories(args.output_dir)
    
    # Initialize BigQuery client - passing credentials path
    client = initialize_bigquery_client(args.project_id, args.bq_creds_path)
    if client is None:
        logger.error("Failed to initialize BigQuery client. Exiting.")
        return
    
    # Log the start of the process
    start_time = datetime.now()
    logger.info(f"Started run_all_skus.py at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and filter combinations from Google Sheet
    try:
        logger.info(f"Loading combinations from Google Sheet ID: {args.combinations_gsheet_id}, Tab: {args.combinations_gsheet_name}")
        combinations_df = load_sku_combinations_from_gsheet(
            creds_path=args.gsheet_creds_path,
            sheet_id=args.combinations_gsheet_id,
            sheet_name=args.combinations_gsheet_name
        )

        filtered_df = filter_combinations(
            combinations_df,
            max_combinations=args.max_combinations,
            specific_sku=args.specific_sku,
            specific_unit=args.specific_unit,
            specific_channel=args.specific_channel
        )
        
        logger.info(f"Processing {len(filtered_df)} filtered combinations")
        
        # Initialize results collector
        all_results = []
        
        if args.parallel and len(filtered_df) > 1:
            logger.info(f"Running in parallel mode with {args.workers} workers")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                # Create a list of futures - don't pass the client here
                futures = [executor.submit(run_single_combination, args, None, row) 
                          for _, row in filtered_df.iterrows()]
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                        logger.info(f"Completed combination: {result['SKU_ID']}-{result['Channel']}-{result['Unit']} (Status: {result['Status']})")
                    except Exception as e:
                        logger.error(f"Error in parallel execution: {e}")
        else:
            logger.info("Running in sequential mode")
            
            # Process each combination sequentially - here we can use the main client
            for _, row in filtered_df.iterrows():
                result = run_single_combination(args, client, row)
                all_results.append(result)
        
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(all_results)
        
        # Round numeric metrics to 4 decimal places
        numeric_cols = ['MAE', 'RMSE', 'sMAPE', 'WAPE']
        for col in numeric_cols:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"{args.output_dir}/all_skus_validation_metrics_{timestamp}.csv"
        results_df.to_csv(results_filename, index=False)
        logger.info(f"Saved results to {results_filename}")
        
        # Print summary statistics
        status_counts = results_df['Status'].value_counts()
        logger.info(f"Processing summary:\n{status_counts}")
        
        # Calculate average metrics for successful runs
        success_df = results_df[results_df['Status'] == 'Success']
        if not success_df.empty:
            avg_metrics = success_df[numeric_cols].mean()
            logger.info(f"Average metrics for successful runs:\n{avg_metrics}")

        # --- Upload Logs and Results to GCS Staging Bucket ---
        logger.info("Attempting to upload results and logs to GCS staging bucket...")
        staging_bucket_name = "eztech-442521-rich-distributor-forecast-staging"

        def upload_to_gcs(local_path, destination_blob_name, bucket_name):
            """Helper function to upload a file to GCS."""
            if not os.path.exists(local_path):
                logger.warning(f"Local file not found, skipping GCS upload: {local_path}")
                return
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(local_path)
                logger.info(f"Successfully uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")
            except Exception as upload_error:
                logger.error(f"Failed to upload {local_path} to GCS: {upload_error}")

        # 1. Rename log files with timestamp
        ts_run_log_file = os.path.join(run_log_dir, f'run_all_skus_{timestamp}.log')
        ts_forecasting_log_file = os.path.join(run_log_dir, f'forecasting_process_{timestamp}.log')

        try:
            if os.path.exists(run_log_file):
                os.rename(run_log_file, ts_run_log_file)
                logger.info(f"Renamed {run_log_file} to {ts_run_log_file}")
            else:
                logger.warning(f"Original log file not found: {run_log_file}")
                ts_run_log_file = None # Set to None if rename failed
        except Exception as rename_err:
            logger.error(f"Failed to rename {run_log_file}: {rename_err}")
            ts_run_log_file = None

        try:
            if os.path.exists(forecasting_log_file):
                os.rename(forecasting_log_file, ts_forecasting_log_file)
                logger.info(f"Renamed {forecasting_log_file} to {ts_forecasting_log_file}")
            else:
                 logger.warning(f"Original log file not found: {forecasting_log_file}")
                 ts_forecasting_log_file = None # Set to None if rename failed
        except Exception as rename_err:
            logger.error(f"Failed to rename {forecasting_log_file}: {rename_err}")
            ts_forecasting_log_file = None

        # 2. Upload results CSV
        results_base_name = os.path.basename(results_filename)
        upload_to_gcs(results_filename, f"results/{results_base_name}", staging_bucket_name)

        # 3. Upload renamed run_all_skus log
        if ts_run_log_file:
             run_log_base_name = os.path.basename(ts_run_log_file)
             upload_to_gcs(ts_run_log_file, f"logs/{run_log_base_name}", staging_bucket_name)

        # 4. Upload renamed forecasting log
        if ts_forecasting_log_file:
            forecast_log_base_name = os.path.basename(ts_forecasting_log_file)
            upload_to_gcs(ts_forecasting_log_file, f"logs/{forecast_log_base_name}", staging_bucket_name)
        # -----------------------------------------------------

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.debug(traceback.format_exc())
    
    # Log end time and duration
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")

if __name__ == "__main__":
    main() 