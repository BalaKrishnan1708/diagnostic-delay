import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score, classification_report

def train_models(datasets, output_dir='outputs/models', fig_dir='outputs/figures'):
    """Trains RF and XGBoost for each dataset and evaluates results."""
    results = []
    
    for name, df in datasets.items():
        print(f"Training models for {name}...")
        
        # 1. Preprocessing
        drop_cols = ['patient_id', 'id', 'index', 'ddi', 'ddi_raw', 'ddi_risk_category', 'disease_type', 'age_group', 'gender']
        X = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        y = df['ddi_risk_category']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. Models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        }
        
        best_model = None
        best_f1 = -1
        best_name = ""
        
        for m_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            auc = roc_auc_score(y_test, probs, multi_class='ovr')
            
            print(f"  {m_name} - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = m_name
                best_acc = acc
                best_auc = auc
        
        # 3. Best Model Save and SHAP
        joblib.dump(best_model, os.path.join(output_dir, f'{name}_model.pkl'))
        joblib.dump(scaler, os.path.join(output_dir, f'{name}_scaler.pkl'))
        joblib.dump(le, os.path.join(output_dir, f'{name}_encoder.pkl'))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix - {name.capitalize()} ({best_name})')
        plt.savefig(os.path.join(fig_dir, f'confusion_{name}.png'))
        plt.close()
        
        # SHAP (using a subset for speed on large datasets)
        explainer = shap.TreeExplainer(best_model)
        shap_subset_size = min(1000, X_test_scaled.shape[0])
        X_test_subset = X_test_scaled[:shap_subset_size]
        shap_values = explainer.shap_values(X_test_subset)
        
        plt.figure()
        shap.summary_plot(shap_values, X_test.iloc[:shap_subset_size], show=False)
        plt.title(f'SHAP Summary - {name.capitalize()}')
        plt.savefig(os.path.join(fig_dir, f'shap_{name}.png'))
        plt.close()
        
        results.append({
            'dataset': name,
            'best_model': best_name,
            'accuracy': best_acc,
            'f1': best_f1,
            'auc': best_auc
        })
        
    results_df = pd.DataFrame(results)
    print("\nModel Training Comparison:")
    print(results_df)
    return results_df
