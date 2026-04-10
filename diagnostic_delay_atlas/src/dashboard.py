import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Diagnostic Delay Atlas", 
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Design System (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --color-1: #335765; /* Dark Slate */
        --color-2: #74A8A4; /* Muted Teal */
        --color-3: #B6D9E0; /* Light Blue */
        --color-4: #DBE2DC; /* Soft Gray */
        --color-5: #7F543D; /* Terra Cotta */
        --bg: #DBE2DC;
        --text: #335765;
    }

    .stApp {
        background-color: var(--bg);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        color: var(--color-1) !important;
    }

    /* Glassmorphic containers with user colors */
    .stMetric, div[data-testid="stExpander"], .stTable, div[data-testid="stSidebarUserContent"], .stDataFrame {
        background: rgba(182, 217, 224, 0.4) !important; /* Semi-transparent color-3 */
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(116, 168, 164, 0.3) !important; /* color-2 */
        border-radius: 15px !important;
        color: var(--text) !important;
    }

    /* Custom metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        color: var(--color-1) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--color-5) !important;
        font-weight: 600 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--color-1) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: var(--color-4) !important;
    }

    /* Button Styling */
    .stButton > button {
        background-color: var(--color-1) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        transition: 0.3s;
    }
    
    .stButton > button:hover {
        background-color: var(--color-5) !important;
    }
</style>
""", unsafe_allow_html=True)

# Define custom plotly palette
USER_PALETTE = ['#335765', '#74A8A4', '#7F543D', '#B6D9E0', '#DBE2DC']

# Get the directory of this script
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src/ is one level down

@st.cache_data
def load_data():
    try:
        data = {
            'cancer': pd.read_csv(os.path.join(script_dir, 'outputs', 'cancer_processed.csv')),
            'diabetes': pd.read_csv(os.path.join(script_dir, 'outputs', 'diabetes_processed.csv')),
            'heart': pd.read_csv(os.path.join(script_dir, 'outputs', 'heart_processed.csv'))
        }
        gaps = pd.read_csv(os.path.join(script_dir, 'outputs', 'equity_gaps.csv'))
        return data, gaps
    except Exception as e:
        st.error(f"Error loading processed datasets: {e}")
        return None, None

def sidebar_nav():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Disease Explorer", "Equity Gap Atlas", "ML Predictions", "Raw Data"])
    return page

def overview_page(data, gaps):
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📊 Diagnostic Delay Atlas: Global Health Insights")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
        This platform provides a real-time, data-driven analysis of delays in clinical diagnosis. 
        All metrics shown are computed directly from the provided patient datasets.
    """)
    
    if data:
        all_df = pd.concat(data.values())
        total_pats = len(all_df)
        avg_ddi = all_df['ddi'].mean()
        high_risk_pct = (all_df['ddi_risk_category'] == 'High').mean() * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Clinical Records", f"{total_pats:,}")
        with col2:
            st.metric("Global Mean Delay Index", f"{avg_ddi:.2f}")
        with col3:
            st.metric("High-Risk Population", f"{high_risk_pct:.1f}%")
        
        st.divider()
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Global Risk Profile")
            fig = px.pie(all_df, names='ddi_risk_category', hole=0.4, 
                         color='ddi_risk_category',
                         color_discrete_map={'Low': '#74A8A4', 'Medium': '#B6D9E0', 'High': '#7F543D'},
                         template="plotly_white",
                         title="Distribution of Patient Risk Levels")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("📉 Chief Clinical Findings")
            # Dynamic insights generated from data
            most_delayed_disease = all_df.groupby('disease_type')['ddi'].mean().idxmax()
            max_delay_val = all_df.groupby('disease_type')['ddi'].mean().max()
            
            st.info(f"**Highest Delay Observed**: {most_delayed_disease.capitalize()} (Mean DDI: {max_delay_val:.1f})")
            
            if gaps is not None:
                max_gap_row = gaps.loc[gaps['max_gender_gap'].idxmax()]
                st.warning(f"**Largest Gender Inequity**: Found in {max_gap_row['disease'].capitalize()} (Gap: {max_gap_row['max_gender_gap']:.1f} pts)")
                
                max_age_row = gaps.loc[gaps['max_age_gap'].idxmax()]
                st.warning(f"**Critical Age Disparity**: {max_age_row['highest_delay_age_group']}+ individuals with {max_age_row['disease'].capitalize()} show peak delay.")
    else:
        st.warning("No data available. Please ensure main.py has been executed successfully.")

def disease_explorer(data):
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🔬 Clinical Disease Explorer")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if data:
        disease = st.sidebar.selectbox("Select Disease To Analyze", ["Cancer", "Diabetes", "Heart"]).lower()
        df = data[disease]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"DDI Distribution - {disease.title()}")
            fig = px.histogram(df, x="ddi", marginal="box", 
                               template="plotly_white",
                               color_discrete_sequence=['#335765'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Gender-Related Delay Factors")
            fig = px.box(df, x="gender", y="ddi", color="gender", 
                         template="plotly_white",
                         color_discrete_sequence=['#74A8A4', '#7F543D'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("DDI by Age Group")
            age_fig = px.bar(df.groupby('age_group')['ddi'].mean().reset_index(), x='age_group', y='ddi', color='age_group', color_discrete_sequence=USER_PALETTE)
            st.plotly_chart(age_fig, use_container_width=True)
            
        with col4:
            st.subheader("Feature Correlation with DDI")
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()[['ddi']].sort_values(by='ddi', ascending=False).head(11).iloc[1:]
            fig_corr = px.bar(corr, x='ddi', orientation='h', 
                              template="plotly_white",
                              color_discrete_sequence=['#335765'])
            fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Processed data not found.")

def equity_gap_atlas(data, gaps):
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("⚖️ Equity Gap Analysis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if data:
        all_df = pd.concat(data.values())
        
        st.subheader("Strategic Delay Matrix (Demographic Inequity)")
        heatmap_data = all_df.groupby(['disease_type', 'gender', 'age_group'])['ddi'].mean().reset_index()
        heatmap_data['demographic'] = heatmap_data['gender'] + " " + heatmap_data['age_group'].astype(str)
        
        fig = px.density_heatmap(heatmap_data, x="demographic", y="disease_type", z="ddi", 
                                 text_auto=True, color_continuous_scale=['#DBE2DC', '#74A8A4', '#335765'],
                                 template="plotly_white")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        if gaps is not None:
            st.subheader("Statistical Equity Analysis")
            st.table(gaps)
        
        if 'diabetes' in data:
            st.subheader("Income vs DDI (Diabetes)")
            df_diab = data['diabetes']
            if 'income' in df_diab.columns:
                fig_inc = px.scatter(df_diab.groupby('income')['ddi'].mean().reset_index(), x='income', y='ddi', 
                                     trendline="ols", 
                                     template="plotly_white",
                                     color_discrete_sequence=['#7F543D'],
                                     title="Mean DDI Across Income Brackets")
                fig_inc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_inc, use_container_width=True)
    else:
        st.warning("Processed data not found.")

def ml_predictions(data):
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🚀 Clinical Risk Prediction Engine")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
        Deploying high-fidelity ML models to predict patient-specific diagnostic delay risk. 
        Adjust the physiological and behavioral markers below to see the real-time risk assessment.
    """)
    
    disease = st.sidebar.selectbox("Select Diagnostic Model", ["Cancer", "Diabetes", "Heart"]).lower()
    
    model_path = os.path.join(script_dir, 'outputs', 'models', f'{disease}_model.pkl')
    scaler_path = os.path.join(script_dir, 'outputs', 'models', f'{disease}_scaler.pkl')
    encoder_path = os.path.join(script_dir, 'outputs', 'models', f'{disease}_encoder.pkl')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        
        # 1. Feature Identification
        full_df = data[disease]
        target_cols = ['ddi', 'ddi_raw', 'ddi_risk_category', 'disease_type', 'age_group', 'gender', 'patient_id', 'id', 'index']
        feature_df = full_df.drop(columns=[c for c in target_cols if c in full_df.columns], errors='ignore')
        feature_df = feature_df.select_dtypes(include=[np.number])
        
        # Identify top features (by correlation with DDI) to prioritize in UI
        corr = full_df.corr(numeric_only=True)[['ddi']].abs().sort_values(by='ddi', ascending=False)
        top_features = [f for f in corr.index if f in feature_df.columns][:12]
        other_features = [f for f in feature_df.columns if f not in top_features]
        
        st.subheader("🛠️ Patient Profile Configuration")
        
        input_data = {}
        
        # Prioritize Top Features in a 3-column layout
        st.info("Primary Clinical Indicators (Highest Impact)")
        cols = st.columns(3)
        for i, col_name in enumerate(top_features):
            with cols[i % 3]:
                min_v = float(feature_df[col_name].min())
                max_v = float(feature_df[col_name].max())
                mean_v = float(feature_df[col_name].mean())
                # Pretty formatting for labels
                label = col_name.replace('_', ' ').title()
                input_data[col_name] = st.slider(label, min_v, max_v, mean_v, key=f"top_{col_name}")

        # Other features in an expander
        with st.expander("Secondary Indicators & Demographics"):
            cols_sub = st.columns(3)
            for i, col_name in enumerate(other_features):
                with cols_sub[i % 3]:
                    min_v = float(feature_df[col_name].min())
                    max_v = float(feature_df[col_name].max())
                    mean_v = float(feature_df[col_name].mean())
                    label = col_name.replace('_', ' ').title()
                    input_data[col_name] = st.slider(label, min_v, max_v, mean_v, key=f"other_{col_name}")

        st.markdown("---")
        
        # 2. Prediction Core
        if st.button("🚀 Execute Risk Analysis", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            # Ensure column order matches training
            input_df = input_df[feature_df.columns]
            
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)
            probs = model.predict_proba(scaled_input)[0]
            
            pred_class = encoder.inverse_transform(prediction)[0]
            
            # 3. Results Visualization
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.write("### Prediction Result")
                color = "red" if pred_class == "High" else "orange" if pred_class == "Medium" else "green"
                st.markdown(f"""
                    <div style="background-color: {color}; padding: 30px; border-radius: 15px; text-align: center;">
                        <h1 style="color: white; margin: 0;">{pred_class}</h1>
                        <p style="color: white; font-size: 1.2rem;">Diagnostic Delay Risk</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.write("### Probability Distribution")
                prob_df = pd.DataFrame({
                    'Risk Category': encoder.classes_,
                    'Probability (%)': probs * 100
                })
                fig = px.bar(prob_df, x='Risk Category', y='Probability (%)', 
                             color='Risk Category', 
                             color_discrete_map={'Low': '#74A8A4', 'Medium': '#B6D9E0', 'High': '#7F543D'},
                             text_auto='.1f',
                             template="plotly_white")
                fig.update_layout(yaxis_range=[0, 100], paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
            st.info("Analysis complete. This model utilizes XGBoost/Random Forest ensembles trained on clinical observation data.")
    else:
        st.error(f"Prediction aborted: System could not locate the trained model for **{disease.capitalize()}**.")
        st.info("Please run the full pipeline (main.py) to train models before using the predictor.")

def raw_data(data):
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📂 Clinical Data Repository")
    st.markdown('</div>', unsafe_allow_html=True)
    if data:
        disease = st.selectbox("Select Dataset", list(data.keys()))
        st.dataframe(data[disease])
        
        csv = data[disease].to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{disease}_ddi_data.csv", "text/csv")
    else:
        st.warning("Processed data not found.")

def main():
    data, gaps = load_data()
    page = sidebar_nav()
    
    if page == "Overview":
        overview_page(data, gaps)
    elif page == "Disease Explorer":
        disease_explorer(data)
    elif page == "Equity Gap Atlas":
        equity_gap_atlas(data, gaps)
    elif page == "ML Predictions":
        ml_predictions(data)
    elif page == "Raw Data":
        raw_data(data)

if __name__ == "__main__":
    main()
