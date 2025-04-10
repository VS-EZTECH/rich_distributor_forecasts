import pandas as pd
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging

logger = logging.getLogger(__name__)

def load_sku_combinations_from_gsheet(creds_path, sheet_id, sheet_name):
    """
    Load SKU-Channel-Unit combinations from a Google Sheet.

    Args:
        creds_path (str): Path to the service account JSON key file.
        sheet_id (str): The Google Sheet ID.
        sheet_name (str): The name of the tab within the sheet.

    Returns:
        pd.DataFrame: DataFrame with columns SKU_ID, Channel, Unit

    Raises:
        FileNotFoundError: If the credentials file does not exist.
        Exception: If there's an error connecting to or reading from Google Sheets.
        ValueError: If the sheet doesn't contain the required columns.
    """
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Service account credentials file not found: {creds_path}")

    try:
        # Define the scope
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

        # Authenticate using the service account
        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=SCOPES)

        # Build the service client
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=sheet_id,
                                    range=sheet_name).execute()
        values = result.get('values', [])

        if not values:
            logger.warning(f"No data found in Google Sheet '{sheet_name}'.")
            return pd.DataFrame(columns=['SKU_ID', 'Channel', 'Unit']) # Return empty DataFrame

        # Assume the first row is the header
        header = values[0]
        data = values[1:]

        df = pd.DataFrame(data, columns=header)

        # Check for required columns (case-insensitive check)
        required_columns_map = {'sku_id': 'SKU_ID', 'channel': 'Channel', 'unit': 'Unit'}
        actual_columns_lower = {col.lower().strip(): col for col in df.columns}
        rename_map = {}
        missing_columns = []

        for req_col_lower, req_col_target in required_columns_map.items():
            if req_col_lower in actual_columns_lower:
                rename_map[actual_columns_lower[req_col_lower]] = req_col_target
            else:
                missing_columns.append(req_col_target)

        if missing_columns:
            raise ValueError(f"Missing required columns in Google Sheet '{sheet_name}': {missing_columns}")

        # Rename columns to the desired format
        df = df.rename(columns=rename_map)

        # Keep only the required columns
        df = df[['SKU_ID', 'Channel', 'Unit']]


        # Ensure SKU_ID is string type
        df['SKU_ID'] = df['SKU_ID'].astype(str)
        # Trim whitespace from string columns
        for col in ['SKU_ID', 'Channel', 'Unit']:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        logger.info(f"Loaded {len(df)} SKU-Channel-Unit combinations from Google Sheet ID {sheet_id}, Tab '{sheet_name}'")
        return df

    except FileNotFoundError:
        logger.error(f"Credentials file not found: {creds_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading combinations from Google Sheet ID {sheet_id}, Tab '{sheet_name}': {e}")
        raise ValueError(f"Error loading combinations from Google Sheet: {e}")

def filter_combinations(df, max_combinations=None, specific_sku=None, specific_unit=None, specific_channel=None):
    """
    Filter SKU-Channel-Unit combinations based on provided criteria.
    
    Args:
        df (pd.DataFrame): DataFrame with SKU_ID, Channel, Unit columns
        max_combinations (int, optional): Maximum number of combinations to return
        specific_sku (str, optional): Filter by specific SKU_ID
        specific_unit (str, optional): Filter by specific Unit
        specific_channel (str, optional): Filter by specific Channel
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Apply filters if provided
    if specific_sku:
        filtered_df = filtered_df[filtered_df['SKU_ID'] == specific_sku]
    
    if specific_unit:
        filtered_df = filtered_df[filtered_df['Unit'] == specific_unit]
    
    if specific_channel:
        filtered_df = filtered_df[filtered_df['Channel'] == specific_channel]
    
    # Limit number of combinations if requested
    if max_combinations and max_combinations > 0 and max_combinations < len(filtered_df):
        filtered_df = filtered_df.head(max_combinations)
    
    print(f"Filtered to {len(filtered_df)} SKU-Channel-Unit combinations")
    return filtered_df 