import pandas as pd
from google.cloud import bigquery
from datetime import datetime

def fetch_sales_data(client: bigquery.Client, project_id: str, dataset_id: str, sku_id: str, 
                     unit: str, channel: str, start_date: str, end_date: str):
    """
    Fetch sales data for a specific SKU ID, unit, channel, and date range.
    Returns a DataFrame with columns needed for Prophet.
    
    Args:
        client (bigquery.Client): Initialized BigQuery client
        project_id (str): GCP project ID
        dataset_id (str): BigQuery dataset ID
        sku_id (str): Target SKU ID to fetch data for
        unit (str): Target unit (e.g., 'Пермь')
        channel (str): Target channel (e.g., 'TF')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: DataFrame with ds, y, Promo_discount_perc, is_promo, stock columns
    """
    # Ensure date parameters are in YYYY-MM-DD format
    try:
        if start_date:
            start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        if end_date:
            end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error formatting date parameters: {e}")
        return pd.DataFrame(columns=['ds', 'y', 'Promo_discount_perc', 'is_promo', 'stock'])
    
    # Updated table name
    sales_table_id = "distributor_sales_frozen_sample_1" 

    sql_query = f"""
    SELECT
        FORMAT_DATE('%Y-%m-%d', date) AS ds, -- Format date as YYYY-MM-DD
        quantity AS y,                      -- Target variable (sales quantity)
        COALESCE(Promo_discount_perc, 0) AS Promo_discount_perc, -- Handle potential NULLs, default to 0
        CASE 
            WHEN Promo_discount_perc IS NOT NULL AND Promo_discount_perc > 0
            THEN 1 
            ELSE 0 
        END AS is_promo,                    -- Binary indicator based on discount > 0
        COALESCE(stock, 0) AS stock         -- Stock level, default to 0 if NULL
    FROM
        `{project_id}.{dataset_id}.{sales_table_id}`
    WHERE
        SKU_ID = @sku_id
        AND unit = @unit
        AND channel = @channel
        AND date BETWEEN @start_date AND @end_date 
    ORDER BY
        date ASC                          
    """
    
    query_params = [
        bigquery.ScalarQueryParameter("sku_id", "STRING", sku_id),
        bigquery.ScalarQueryParameter("unit", "STRING", unit),
        bigquery.ScalarQueryParameter("channel", "STRING", channel),
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
    ]
    
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)
    
    try:
        if client is None:
             raise Exception("BigQuery client not initialized.")
        
        df = client.query(sql_query, job_config=job_config).to_dataframe()
        print(f"Successfully fetched {len(df)} sales rows for SKU {sku_id}, Unit {unit}, Channel {channel}.")
        
        # Ensure ds is datetime object with proper format for merging
        df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')
        
        # Ensure 'is_promo' is integer type
        df['is_promo'] = df['is_promo'].astype(int)
        
        # Ensure 'Promo_discount_perc' is float, handling potential NAs introduced by COALESCE if source had non-numeric
        df['Promo_discount_perc'] = pd.to_numeric(df['Promo_discount_perc'], errors='coerce').fillna(0).astype(float)
        
        # Ensure 'stock' is float
        df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0).astype(float)

        return df[['ds', 'y', 'Promo_discount_perc', 'is_promo', 'stock']] # Return all needed cols
    except Exception as e:
        print(f"Error fetching sales data: {e}")
        return pd.DataFrame(columns=['ds', 'y', 'Promo_discount_perc', 'is_promo', 'stock']) # Return empty df with correct columns 