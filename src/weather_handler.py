import pandas as pd
import requests
from datetime import datetime

def fetch_weather_data(latitude: float, longitude: float, start_date: str, end_date: str):
    """
    Fetch daily weather data from Open-Meteo API.
    Returns a DataFrame with ds and weather variables.
    """
    # Ensure date parameters are in YYYY-MM-DD format
    try:
        if start_date:
            start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        if end_date:
            end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error formatting date parameters: {e}")
        return None
        
    print(f"Fetching weather data for {latitude}, {longitude} from {start_date} to {end_date}...")
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
        "timezone": "Europe/Moscow"  # Use relevant timezone for Perm
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        df_weather = pd.DataFrame(data['daily'])
        df_weather.rename(columns={'time': 'ds'}, inplace=True)
        
        # Ensure date format is consistent
        df_weather['ds'] = pd.to_datetime(df_weather['ds'], format='%Y-%m-%d')
        
        print(f"Successfully fetched {len(df_weather)} weather rows.")
        return df_weather
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None # Returning None to match original logic, consider raising Exception
    except Exception as e:
        print(f"Error processing weather data: {e}")
        return None # Returning None to match original logic

def fetch_forecast_weather_data(latitude: float, longitude: float, forecast_days: int = 16):
    """
    Fetch daily weather forecast data from Open-Meteo API.
    Returns a DataFrame with ds and weather variables.
    """
    print(f"Fetching {forecast_days}-day weather forecast for {latitude}, {longitude}...")
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Ensure forecast_days is within the API limit (typically 16)
    forecast_days = min(forecast_days, 16)
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
        "timezone": "Europe/Moscow",  # Use relevant timezone
        "forecast_days": forecast_days
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        if 'daily' not in data:
            print(f"Warning: 'daily' key not found in forecast response. Data: {data}")
            return pd.DataFrame() # Return empty dataframe
            
        df_weather_fcst = pd.DataFrame(data['daily'])
        df_weather_fcst.rename(columns={'time': 'ds'}, inplace=True)
        
        # Ensure date format is consistent
        df_weather_fcst['ds'] = pd.to_datetime(df_weather_fcst['ds'], format='%Y-%m-%d')
        
        print(f"Successfully fetched {len(df_weather_fcst)} weather forecast rows.")
        # Ensure all requested columns are present, fill with 0 if missing
        expected_cols = ['ds', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'snowfall_sum']
        for col in expected_cols:
            if col != 'ds' and col not in df_weather_fcst.columns:
                 print(f"Warning: Forecast missing column '{col}'. Filling with 0.")
                 df_weather_fcst[col] = 0
        return df_weather_fcst[expected_cols] # Return only expected columns
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather forecast data: {e}")
        return pd.DataFrame() # Return empty dataframe on error
    except Exception as e:
        print(f"Error processing weather forecast data: {e}")
        return pd.DataFrame() # Return empty dataframe on error 