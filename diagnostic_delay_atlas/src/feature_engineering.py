import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def compute_ddi(datasets):
    """
    Calculates the Diagnostic Delay Index (DDI) score (0-100) and risk category for each patient.
    
    Args:
        datasets (dict): Dictionary of dataframes.
        
    Returns:
        dict: Enhanced dictionary of dataframes.
    """
    scaler = MinMaxScaler(feature_range=(0, 100))
    
    # 1. Cancer DDI
    if 'cancer' in datasets:
        df = datasets['cancer']
        
        # Mapping column names if needed based on standardization
        symptom_cols = ['chest_pain', 'coughing_of_blood', 'fatigue', 'weight_loss', 'shortness_of_breath', 'wheezing']
        risk_cols = ['air_pollution', 'smoking', 'passive_smoker', 'genetic_risk', 'alcohol_use']
        
        # Check if columns exist (graceful handling)
        existing_symptoms = [c for c in symptom_cols if c in df.columns]
        existing_risks = [c for c in risk_cols if c in df.columns]
        
        symptom_severity = df[existing_symptoms].mean(axis=1) if existing_symptoms else 0
        risk_factor_load = df[existing_risks].mean(axis=1) if existing_risks else 0
        
        # DDI = ((symptom_severity / 9) * 50) + ((risk_factor_load / 9) * 30) + (age / 100 * 20)
        age_val = df['age'] if 'age' in df.columns else pd.Series(0, index=df.index)
        df['ddi_raw'] = ((symptom_severity / 9) * 50) + ((risk_factor_load / 9) * 30) + (age_val / 100 * 20)
        
        df['ddi'] = scaler.fit_transform(df[['ddi_raw']])
        datasets['cancer'] = df

    # 2. Diabetes DDI
    if 'diabetes' in datasets:
        df = datasets['diabetes']
        
        # healthcare_barrier = (no_doc_bc_cost * 30) + ((5 - gen_hlth) * 10) + (any_healthcare == 0) * 20
        # symptom_burden = (phys_hlth / 30 * 25) + (ment_hlth / 30 * 15)
        # risk_factor_load = (high_bp + high_chol + smoker + stroke + diff_walk) * 4
        
        # gen_hlth is 1-5 where 1 is excellent, 5 is poor. 
        # (5 - gen_hlth) * 10: Mapping so higher is worse? Wait, user says (5 - gen_hlth) * 10.
        # If gen_hlth is 1 (Excellent), score is 40. If 5 (Poor), score is 0. 
        # Usually DDI should be higher for more delay/worse health indicators?
        # Let's stick to the user's specific formula.
        
        no_doc = df['no_docbc_cost'] if 'no_docbc_cost' in df.columns else pd.Series(0, index=df.index)
        gen_hlth = df['gen_hlth'] if 'gen_hlth' in df.columns else pd.Series(3, index=df.index)
        any_hc = df['any_healthcare'] if 'any_healthcare' in df.columns else pd.Series(1, index=df.index)
        
        hc_barrier = (no_doc * 30) + ((5 - gen_hlth) * 10) + (any_hc == 0).astype(int) * 20
        
        phys_hlth = df['phys_hlth'] if 'phys_hlth' in df.columns else pd.Series(0, index=df.index)
        ment_hlth = df['ment_hlth'] if 'ment_hlth' in df.columns else pd.Series(0, index=df.index)
        symptom_burden = (phys_hlth / 30 * 25) + (ment_hlth / 30 * 15)
        
        risk_cols = ['high_bp', 'high_chol', 'smoker', 'stroke', 'diff_walk']
        existing_risks = [c for c in risk_cols if c in df.columns]
        risk_factor_load = df[existing_risks].sum(axis=1) * 4 if existing_risks else 0
        
        df['ddi_raw'] = hc_barrier + symptom_burden + risk_factor_load
        df['ddi'] = scaler.fit_transform(df[['ddi_raw']])
        datasets['diabetes'] = df

    # 3. Heart DDI
    if 'heart' in datasets:
        df = datasets['heart']
        
        # symptom_severity = (cp / 3 * 30) + (trestbps / 200 * 20) + (chol / 600 * 20)
        # clinical_risk = (fbs * 10) + (exang * 15) + (oldpeak / 6 * 15) + (ca / 3 * 10)
        
        cp = df['cp'] if 'cp' in df.columns else pd.Series(0, index=df.index)
        trestbps = df['trestbps'] if 'trestbps' in df.columns else pd.Series(120, index=df.index)
        chol = df['chol'] if 'chol' in df.columns else pd.Series(200, index=df.index)
        
        symptom_severity = (cp / 3 * 30) + (trestbps / 200 * 20) + (chol / 600 * 20)
        
        fbs = df['fbs'] if 'fbs' in df.columns else pd.Series(0, index=df.index)
        exang = df['exang'] if 'exang' in df.columns else pd.Series(0, index=df.index)
        oldpeak = df['oldpeak'] if 'oldpeak' in df.columns else pd.Series(0, index=df.index)
        ca = df['ca'] if 'ca' in df.columns else pd.Series(0, index=df.index)
        
        clinical_risk = (fbs * 10) + (exang * 15) + (oldpeak / 6 * 15) + (ca / 3 * 10)
        
        df['ddi_raw'] = symptom_severity + clinical_risk
        df['ddi'] = scaler.fit_transform(df[['ddi_raw']])
        datasets['heart'] = df

    # Categorize DDI
    for name, df in datasets.items():
        def categorize(val):
            if val < 33: return 'Low'
            elif val < 66: return 'Medium'
            else: return 'High'
            
        df['ddi_risk_category'] = df['ddi'].apply(categorize)
        datasets[name] = df
        
    return datasets
