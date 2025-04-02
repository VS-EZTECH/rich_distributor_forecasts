#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Environment setup script for SKU Sales Forecasting project.
This script:
1. Tests the installation of required packages
2. Sets up Google Cloud authentication for BigQuery access
"""

import os
import sys
import subprocess
import importlib

def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def verify_packages():
    """Verify all required packages are installed"""
    required_packages = [
        'pandas',
        'prophet',
        'google.cloud.bigquery',
        'sklearn',
        'matplotlib',
        'numpy',
        'plotly',
        'holidays'
    ]
    
    all_installed = True
    for package in required_packages:
        if not check_package(package.split('.')[0]):
            all_installed = False
    
    return all_installed

def test_bigquery_connection():
    """Test BigQuery connection"""
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        print("✓ BigQuery client created successfully")
        
        # Try a simple query
        query = "SELECT 1"
        results = client.query(query).result()
        for row in results:
            print(f"Test query result: {row}")
        
        print("✓ BigQuery connection test successful")
        return True
    except Exception as e:
        print(f"✗ BigQuery connection test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("SKU Sales Forecasting - Environment Setup")
    print("=" * 60)
    
    # Check if packages are installed
    print("\nVerifying required packages:")
    packages_ok = verify_packages()
    
    if not packages_ok:
        print("\nSome packages are missing. Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✓ Packages installed successfully")
        except subprocess.CalledProcessError:
            print("✗ Failed to install packages")
            return
    
    # Test BigQuery connection
    print("\nTesting BigQuery connection:")
    bq_ok = test_bigquery_connection()
    
    if not bq_ok:
        print("\nBigQuery authentication failed. Please ensure:")
        print("1. You have set up a Google Cloud service account with appropriate permissions")
        print("2. The GOOGLE_APPLICATION_CREDENTIALS environment variable is set to the path of your service account key file")
        print("\nFor example:")
        print('export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"')
    
    # Summary
    print("\n" + "=" * 60)
    if packages_ok and bq_ok:
        print("✓ Environment setup complete! You're ready to start forecasting.")
    else:
        print("✗ Environment setup incomplete. Please address the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main() 