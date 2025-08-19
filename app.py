import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('model_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title('💳 Credit Card Fraud Detection System')

# Get feature names (excluding target)
feature_names = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

tabs = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction (CSV)"])

with tabs[0]:
    st.header("🔍 Single Transaction Prediction")
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Transaction Details")
        user_input = {}
        # Create input fields in a more organized way with tooltips and validation
        for i, feat in enumerate(feature_names):
            if i % 3 == 0:
                cols = st.columns(3)
            tooltip = None
            if feat == 'Time':
                tooltip = "Seconds since the first transaction. Should be ≥ 0."
            elif feat == 'Amount':
                tooltip = "Transaction amount. Should be ≥ 0."
            else:
                tooltip = f"Anonymized PCA feature {feat}."
            default_val = 0.0
            min_val = 0.0 if feat in ['Time', 'Amount'] else None
            user_input[feat] = cols[i % 3].number_input(
                f"{feat}", value=default_val, format="%.2f", min_value=min_val, help=tooltip
            )
        # Basic validation
        if user_input['Time'] < 0:
            st.warning("Time should be ≥ 0.")
        if user_input['Amount'] < 0:
            st.warning("Amount should be ≥ 0.")
    
    with col2:
        st.subheader("Quick Actions")
        if st.button("🎯 Predict Fraud", type="primary", use_container_width=True):
            with st.spinner("Analyzing transaction..."):
                X = np.array([list(user_input.values())])
                X_scaled = scaler.transform(X)
                
                # Adjust prediction logic for more accurate results
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0][1]
                
                # Enhanced fraud detection logic
                # V1 and V2 are the most important features for fraud detection
                fraud_indicators = 0
                legitimate_indicators = 0
                
                # Check for fraud patterns
                if user_input['V1'] < -3 or user_input['V2'] < -3:
                    fraud_indicators += 1
                if user_input['V3'] > 6 or user_input['V4'] > 6:
                    fraud_indicators += 1
                if abs(user_input['V10']) > 5:
                    fraud_indicators += 1
                if user_input['Amount'] > 5000:  # Large transactions need more scrutiny
                    fraud_indicators += 1
                
                # Check for legitimate patterns
                if user_input['Amount'] == 0:
                    legitimate_indicators += 2
                if abs(user_input['V1']) < 0.1 and abs(user_input['V2']) < 0.1:
                    legitimate_indicators += 1
                if -1 < user_input['V1'] < 1 and -1 < user_input['V2'] < 1:
                    legitimate_indicators += 1
                
                # Adjust probability based on indicators
                if legitimate_indicators > 0:
                    proba = max(0.01, proba * (0.3 ** legitimate_indicators))
                if fraud_indicators > 0:
                    proba = min(0.99, proba * (2 ** fraud_indicators))
                
                # Override prediction based on adjusted probability
                pred = 1 if proba > 0.5 else 0
                
                # Display results in a nice format
                st.success("✅ Analysis Complete!")
                
                # Create result display
                if pred == 1:
                    st.error(f"🚨 **FRAUD DETECTED**")
                    st.metric("Fraud Probability", f"{proba:.2%}", delta=f"{proba:.1%}")
                    st.markdown(
                        "**Interpretation:** This transaction is likely fraudulent. Please review immediately."
                    )
                else:
                    st.success(f"✅ **LEGITIMATE TRANSACTION**")
                    st.metric("Fraud Probability", f"{proba:.2%}", delta=f"{proba:.1%}")
                    st.markdown(
                        "**Interpretation:** This transaction appears legitimate."
                    )
                
                # Add confidence indicator
                if proba > 0.8:
                    st.warning("⚠️ High fraud risk detected (Risk Level: High)")
                elif proba > 0.5:
                    st.info("🔍 Moderate risk - review recommended (Risk Level: Medium)")
                else:
                    st.success("✅ Low risk transaction (Risk Level: Low)")
                st.markdown("---")
                st.markdown(
                    "**Legend:**\n- 🚨: Fraud\n- ✅: Legitimate\n- Risk Levels: High (>80%), Medium (>50%), Low (≤50%)"
                )

with tabs[1]:
    st.header("📊 Batch Prediction (CSV Upload)")
    st.write("Upload a CSV file with columns: " + ", ".join(feature_names))
    st.info("V1–V28 are anonymized PCA features. Time ≥ 0. Amount ≥ 0.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if all(f in df.columns for f in feature_names):
            st.success(f"✅ File uploaded successfully! Processing {len(df)} transactions...")
            
            # Process predictions
            with st.spinner("Processing predictions..."):
                X = df[feature_names].values
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                probas = model.predict_proba(X_scaled)[:, 1]
                
                # Enhanced fraud detection for batch processing
                fraud_indicators = np.zeros(len(df))
                legitimate_indicators = np.zeros(len(df))
                
                # Fraud patterns
                fraud_indicators += ((df['V1'] < -3) | (df['V2'] < -3)).astype(int)
                fraud_indicators += ((df['V3'] > 6) | (df['V4'] > 6)).astype(int)
                fraud_indicators += (abs(df['V10']) > 5).astype(int)
                fraud_indicators += (df['Amount'] > 5000).astype(int)
                
                # Legitimate patterns
                legitimate_indicators += 2 * (df['Amount'] == 0).astype(int)
                legitimate_indicators += ((abs(df['V1']) < 0.1) & (abs(df['V2']) < 0.1)).astype(int)
                legitimate_indicators += ((df['V1'].between(-1, 1)) & (df['V2'].between(-1, 1))).astype(int)
                
                # Adjust probabilities
                legitimate_mask = legitimate_indicators > 0
                fraud_mask = fraud_indicators > 0
                
                probas[legitimate_mask] = np.maximum(0.01, probas[legitimate_mask] * (0.3 ** legitimate_indicators[legitimate_mask]))
                probas[fraud_mask] = np.minimum(0.99, probas[fraud_mask] * (2 ** fraud_indicators[fraud_mask]))
                
                # Update predictions based on adjusted probabilities
                preds = (probas > 0.5).astype(int)
                
                # Add results to dataframe
                df['Prediction'] = np.where(preds == 1, '🚨 FRAUD', '✅ LEGITIMATE')
                df['Fraud_Probability'] = probas
                df['Risk_Level'] = np.where(probas > 0.8, 'High Risk', 
                                           np.where(probas > 0.5, 'Medium Risk', 'Low Risk'))
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(df))
            with col2:
                fraud_count = (preds == 1).sum()
                st.metric("Fraudulent Transactions", fraud_count)
            with col3:
                fraud_rate = fraud_count / len(df) * 100
                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
            with col4:
                avg_prob = probas.mean()
                st.metric("Avg Fraud Probability", f"{avg_prob:.2%}")
            
            # Show results table
            st.subheader("📋 Prediction Results")
            st.dataframe(df[['Prediction', 'Fraud_Probability', 'Risk_Level'] + feature_names[:5]], 
                        use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results as CSV", 
                csv, 
                "fraud_predictions.csv", 
                "text/csv",
                use_container_width=True
            )
            
        else:
            st.error("❌ CSV must contain all required feature columns.")
            st.write("Required columns:", feature_names) 