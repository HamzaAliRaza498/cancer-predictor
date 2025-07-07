import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64

# ----- CONFIG -----
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# ----- LOAD BACKGROUND -----
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        .glass {{
            background: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 20px;
            margin: auto;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
        }}
        h1, h4, p, label {{
            text-align: center;
            color: #333;
        }}
        </style>
    """, unsafe_allow_html=True)

set_bg("cancer_background.jpg")  # Make sure this image is in the same folder

# ----- LOAD MODEL -----
model = joblib.load("cancer_prediction_model.pkl")

# ----- UI -----
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("<h1>🧬 Breast Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4>Check if tumor is likely malignant or benign</h4>", unsafe_allow_html=True)
st.markdown("<p><b>Made by HamzaAliRaza</b></p>", unsafe_allow_html=True)

# --- Option to choose input method ---
option = st.radio("Select input method:", ["🔢 Manual Input", "📁 Upload CSV File"])

# ----- Manual Input -----
if option == "🔢 Manual Input":
    mean_radius = st.slider("Mean Radius", 6.0, 30.0, 14.0)
    mean_texture = st.slider("Mean Texture", 9.0, 40.0, 20.0)
    mean_perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 85.0)
    mean_area = st.slider("Mean Area", 140.0, 2500.0, 500.0)
    mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.2, 0.1)

    area_per_radius = mean_area / mean_radius
    features = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, area_per_radius]])

    if st.button("🔍 Predict"):
        prediction = model.predict(features)[0]
        if prediction == 0:
            st.error("⚠️ Result: Tumor is likely **Malignant** (Dangerous)")
        else:
            st.success("✅ Result: Tumor is likely **Benign** (Non-cancerous)")

# ----- CSV Upload Option -----
elif option == "📁 Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file (max 200MB)", type=["csv"])

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 200:
            st.error("❌ File too large! Please upload a file less than 200MB.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")

                # Check required columns
                required_cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"❌ File must contain these columns:\n{required_cols}")
                else:
                    df['area_per_radius'] = df['mean area'] / df['mean radius']
                    feature_cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'area_per_radius']
                    preds = model.predict(df[feature_cols])

                    df['Prediction'] = np.where(preds == 0, "Malignant", "Benign")
                    st.dataframe(df[['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'Prediction']])

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Predictions", csv, "cancer_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"⚠️ Error processing file: {e}")

st.markdown("</div>", unsafe_allow_html=True)
