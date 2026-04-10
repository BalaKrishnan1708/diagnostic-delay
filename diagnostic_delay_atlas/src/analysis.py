import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu

def ddi_by_disease(datasets, output_dir='outputs/figures'):
    """Boxplot comparing DDI distribution across all 3 diseases."""
    combined = pd.concat([df[['ddi', 'disease_type']] for df in datasets.values()])
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='disease_type', y='ddi', data=combined, hue='disease_type', palette='viridis', legend=False)
    plt.title('DDI Distribution Across Diseases')
    plt.savefig(os.path.join(output_dir, 'ddi_by_disease.png'))
    plt.close()

def ddi_by_gender(datasets, output_dir='outputs/figures'):
    """Bar chart of mean DDI by gender with error bars and p-value annotation."""
    for name, df in datasets.items():
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x='gender', y='ddi', data=df, hue='gender', palette='muted', capsize=.1, legend=False)
        
        # Mann-Whitney U test
        males = df[df['gender'] == 'Male']['ddi']
        females = df[df['gender'] == 'Female']['ddi']
        if len(males) > 0 and len(females) > 0:
            stat, p = mannwhitneyu(males, females)
            plt.text(0.5, df['ddi'].max(), f'p-value: {p:.4f}', ha='center')
            
        plt.title(f'Mean DDI by Gender - {name.capitalize()}')
        plt.savefig(os.path.join(output_dir, f'ddi_by_gender_{name}.png'))
        plt.close()

def ddi_by_age_group(datasets, output_dir='outputs/figures'):
    """Grouped bar chart: age groups on x-axis, mean DDI as bars, colored by disease."""
    combined = pd.concat([df[['ddi', 'disease_type', 'age_group']] for df in datasets.values()])
    plt.figure(figsize=(12, 6))
    sns.barplot(x='age_group', y='ddi', hue='disease_type', data=combined)
    plt.title('DDI by Age Group and Disease')
    plt.savefig(os.path.join(output_dir, 'ddi_by_age.png'))
    plt.close()

def ddi_by_income(datasets, output_dir='outputs/figures'):
    """Line chart of mean DDI vs income level for diabetes dataset."""
    if 'diabetes' in datasets:
        df = datasets['diabetes']
        if 'income' in df.columns:
            income_stats = df.groupby('income')['ddi'].mean().reset_index()
            corr = df[['income', 'ddi']].corr().iloc[0, 1]
            plt.figure(figsize=(10, 6))
            sns.lineplot(x='income', y='ddi', data=income_stats, marker='o')
            plt.title(f'Mean DDI vs Income Level (Diabetes) - Corr: {corr:.2f}')
            plt.savefig(os.path.join(output_dir, 'ddi_by_income.png'))
            plt.close()

def correlation_heatmap(datasets, output_dir='outputs/figures'):
    """Heatmap of correlation between DDI and all numeric features."""
    for name, df in datasets.items():
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()[['ddi']].sort_values(by='ddi', ascending=False)
        plt.figure(figsize=(6, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Feature Correlation with DDI - {name.capitalize()}')
        plt.savefig(os.path.join(output_dir, f'correlation_{name}.png'))
        plt.close()

def equity_gap_table(datasets, output_path='outputs/equity_gaps.csv'):
    """Summary DataFrame showing max DDI gap between demographic groups per disease."""
    gaps = []
    for name, df in datasets.items():
        # Gender gap
        gender_means = df.groupby('gender')['ddi'].mean()
        gender_gap = gender_means.max() - gender_means.min() if not gender_means.empty else 0
        
        # Age group gap
        age_means = df.groupby('age_group')['ddi'].mean()
        age_gap = age_means.max() - age_means.min() if not age_means.empty else 0
        
        gaps.append({
            'disease': name,
            'max_gender_gap': gender_gap,
            'max_age_gap': age_gap,
            'highest_delay_gender': gender_means.idxmax() if not gender_means.empty else 'N/A',
            'highest_delay_age_group': age_means.idxmax() if not age_means.empty else 'N/A'
        })
    
    gap_df = pd.DataFrame(gaps)
    gap_df.to_csv(output_path, index=False)
    print("Equity Gap Table:")
    print(gap_df)
    return gap_df
