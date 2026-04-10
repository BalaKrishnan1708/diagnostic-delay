import os
import sys
import pandas as pd

# Add src to path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))

from load_data import load_all_data
from feature_engineering import compute_ddi
from analysis import ddi_by_disease, ddi_by_gender, ddi_by_age_group, ddi_by_income, correlation_heatmap, equity_gap_table
from model import train_models

def main():
    print("--- Diagnostic Delay Atlas Pipeline ---")
    
    # 1. Load Data
    print("\nPhase 1: Loading datasets...")
    data_dir = os.path.join(script_dir, 'data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir} folder. Please ensure the CSV files are placed there.")
    
    datasets = load_all_data(data_dir)
    if not datasets:
        print("No datasets found in 'data/' folder. Please download the CSVs described in the prompt.")
        return

    # 2. Feature Engineering
    print("\nPhase 2: Computing Diagnostic Delay Index (DDI)...")
    datasets = compute_ddi(datasets)
    
    # Save processed data for dashboard
    output_base_dir = os.path.join(script_dir, 'outputs')
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    for name, df in datasets.items():
        df.to_csv(os.path.join(output_base_dir, f'{name}_processed.csv'), index=False)
    
    # 3. Statistical Analysis
    print("\nPhase 3: Running statistical analysis and generating figures...")
    fig_dir = os.path.join(script_dir, 'outputs', 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    ddi_by_disease(datasets, output_dir=fig_dir)
    ddi_by_gender(datasets, output_dir=fig_dir)
    ddi_by_age_group(datasets, output_dir=fig_dir)
    ddi_by_income(datasets, output_dir=fig_dir)
    correlation_heatmap(datasets, output_dir=fig_dir)
    equity_gap_table(datasets, output_path=os.path.join(script_dir, 'outputs', 'equity_gaps.csv'))
    
    # 4. Model Training
    print("\nPhase 4: Training machine learning models...")
    model_dir = os.path.join(script_dir, 'outputs', 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    results = train_models(datasets, output_dir=model_dir, fig_dir=fig_dir)
    
    print("\n" + "="*40)
    print("PIPELINE COMPLETE!")
    print("="*40)
    print("\nNext steps:")
    print("1. Ensure all figures were generated in outputs/figures/")
    print("2. Ensure all models were saved in outputs/models/")
    print("3. Run the interactive dashboard with:")
    print("   streamlit run src/dashboard.py")
    print("="*40)

if __name__ == "__main__":
    main()
