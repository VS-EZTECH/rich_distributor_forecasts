import pandas as pd
import os

def load_sku_combinations(file_path):
    """
    Load SKU-Channel-Unit combinations from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the combinations
        
    Returns:
        pd.DataFrame: DataFrame with columns SKU_ID, Channel, Unit
        
    Raises:
        FileNotFoundError: If the CSV file does not exist
        ValueError: If the CSV file doesn't contain required columns
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Combinations file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        # Check for required columns
        required_columns = ['SKU_ID', 'Channel', 'Unit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in combinations file: {missing_columns}")
        
        # Ensure SKU_ID is string type
        df['SKU_ID'] = df['SKU_ID'].astype(str)
        # Trim whitespace from string columns
        for col in required_columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        print(f"Loaded {len(df)} SKU-Channel-Unit combinations from {file_path}")
        return df
    
    except Exception as e:
        if isinstance(e, FileNotFoundError) or isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error loading combinations file: {e}")

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