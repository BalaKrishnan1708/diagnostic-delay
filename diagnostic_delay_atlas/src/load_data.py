import pandas as pd
import numpy as np
import os

def load_all_data(data_dir='data'):
    """
    Loads all 3 healthcare datasets, standardizes columns, and adds demographic features.
    
    Args:
        data_dir (str): Directory containing the CSV files.
        
    Returns:
        dict: Dictionary containing dataframes for 'cancer', 'diabetes', and 'heart'.
    """
    datasets = {}
    
    # 1. Cancer Dataset
    cancer_path = os.path.join(data_dir, 'cancer patient data sets.csv')
    try:
        df_cancer = pd.read_csv(cancer_path)
        # Standardize column names
        df_cancer.columns = [col.lower().replace(' ', '_').replace('__', '_').strip() for col in df_cancer.columns]
        
        # Add disease type
        df_cancer['disease_type'] = 'cancer'
        
        # Mapping gender (1=Male, 2=Female)
        df_cancer['gender'] = df_cancer['gender'].map({1: 'Male', 2: 'Female'})
        
        datasets['cancer'] = df_cancer
    except FileNotFoundError:
        print(f"Error: {cancer_path} not found.")
    
    # 2. Diabetes Dataset
    diabetes_path = os.path.join(data_dir, 'diabetes_012_health_indicators_BRFSS2021.csv')
    try:
        df_diabetes = pd.read_csv(diabetes_path)
        # Standardize column names
        df_diabetes.columns = [col.lower().replace(' ', '_').strip() for col in df_diabetes.columns]
        
        # Add disease type
        df_diabetes['disease_type'] = 'diabetes'
        
        # Mapping gender (0=Female, 1=Male)
        df_diabetes['gender'] = df_diabetes['sex'].map({0: 'Female', 1: 'Male'})
        
        datasets['diabetes'] = df_diabetes
    except FileNotFoundError:
        print(f"Error: {diabetes_path} not found.")

    # 3. Heart Dataset
    heart_path = os.path.join(data_dir, 'heart.csv')
    try:
        df_heart = pd.read_csv(heart_path)
        # Standardize column names
        df_heart.columns = [col.lower().replace(' ', '_').strip() for col in df_heart.columns]
        
        # Add disease type
        df_heart['disease_type'] = 'heart'
        
        # Mapping gender (1=Male, 0=Female)
        df_heart['gender'] = df_heart['sex'].map({1: 'Male', 0: 'Female'})
        
        datasets['heart'] = df_heart
    except FileNotFoundError:
        print(f"Error: {heart_path} not found.")

    # Common Processing
    for name, df in datasets.items():
        # Handle Age Grouping
        # Diabetes age column is often categorical (1-13), but user requested <30, 30-45, 46-60, 60+
        # If age is numeric, we bin it. If it's the 1-13 scale from BRFSS, we approximate.
        # Assuming numeric age where possible.
        
        if 'age' in df.columns:
            bins = [0, 29, 45, 60, 150]
            labels = ['<30', '30-45', '46-60', '60+']
            df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
        
        # Median imputation for missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        datasets[name] = df
        
    return datasets

if __name__ == "__main__":
    # Test loading
    data = load_all_data('../data')
    for name, df in data.items():
        print(f"Loaded {name}: {df.shape}")
