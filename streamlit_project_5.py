import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
import tensorflow as tf
import os
import pandas as pd

# --- Custom CSS / Theme ---
st.set_page_config(page_title="Brain Tumour Classifier", layout="wide")

st.markdown("""
<style>
/* base styling: dark background, soft text */
html, body, [class*="css"] {
    background-color: #0d1114;
    color: #e6e6e6;
    font-family: 'sans-serif';
}
/* Headers with subtle teal-green / blue gradient */
h1, h2, h3, h4, h5, h6 {
    background: linear-gradient(135deg, #2fa1bd, #36c5a1);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: 600;
}
/* Style bar-chart bars / dataframes to match color palette */
.stBarChart>div>div>svg>g>path {
    fill: #36c5a1 !important;
}
/* Tables (dataframes) dark background */
.stTable td, .stTable th {
    color: #e6e6e6 !important;
    background-color: #1a1f24 !important;
}
.stDataFrame > div[data-testid="stDataFrameContainer"] {
    background-color: #1a1f24 !important;
    color: #e6e6e6 !important;
}
/* Buttons — lighter teal-green gradient */
.stButton>button {
    background: linear-gradient(135deg, #2fa1bd, #36c5a1);
    color: #000;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    opacity: 0.9;
    transform: scale(1.02);
}
/* File uploader background tweak */
.stFileUploader, .stFileUploader>div {
    background-color: #1e2328 !important;
    border-radius: 6px !important;
}

/* Custom stat-card styles (update background to darker slate) */
.stat-card {
    background-color: #1e2328;
    color: #e6e6e6;
    padding: 20px;
    border: 1px solid #2a2f36;
    border-radius: 8px;
    text-align: center;
    width: 180px;
    display: inline-block;
    margin-right: 12px;
    margin-bottom: 12px;
}
.stat-card .stat-value {
    font-size: 28px;
    font-weight: bold;
}
.stat-card .stat-label {
    font-size: 14px;
    color: #8a8a8a;
}
</style>
""", unsafe_allow_html=True)



# --- Classes & Model path ---
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
MODEL_PATH ="C:/Users/Ajay/Downloads/model_mob (1).keras" # here using Mobilenet v2 model

# --- Dataset summary values ---
TRAIN_IMAGES = 564 + 358 + 335 + 438  # = 1695
VALID_IMAGES = 161 + 124 + 99 + 118   # = 502
TEST_IMAGES  = 80 + 63 + 49 + 54       # = 246
TOTAL_IMAGES = TRAIN_IMAGES + VALID_IMAGES + TEST_IMAGES
TOTAL_CLASSES = len(CLASSES)

# --- Example model‑comparison data ---
MODEL_COMPARISON = {
    "Custom Model": 0.91,
    "VGG16": 0.93,
    "ResNet50": 0.89,
    "MobileNetV2": 0.94,
    "InceptionV3": 0.90,
    "EfficientNetB0": 0.92,
}

# --- Sidebar info ---
with st.sidebar:
    st.title("🧠 Brain Tumour Detector")
    st.write("Upload a brain scan (MRI / image) and get a prediction of tumour type or no tumer")
    st.write("---")
    st.write("**Developed by:** thejj_theju")

# --- Load model (cached) ---
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    else:
        return None

model = load_model(MODEL_PATH)

# --- Tabs: Prediction & Summary ---
tab1, tab2 = st.tabs(["Predict Image", "Summary & Model Info"])

with tab1:
    st.header("Upload MRI / Brain-Scan Image")

    if model is None:
        st.error(f"Model not found at `{MODEL_PATH}`")
    else:
        uploaded = st.file_uploader(
            "Choose an image (jpg / jpeg / png)",
            type=["jpg","jpeg","png"]
        )
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption="Uploaded MRI", use_column_width=True)

            with col2:
                # Preprocess & predict
                img_resized = img.resize((224, 224))
                img_arr = np.array(img_resized) / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)

                with st.spinner("Predicting..."):
                    preds = model.predict(img_arr)[0]

                idx = int(np.argmax(preds))
                pred_class = CLASSES[idx]
                confidence = preds[idx]

                st.subheader("✅ Prediction Result")
                st.markdown(f"**{pred_class}** — **Confidence: {confidence*100:.2f}%**")
                if confidence < 0.6:
                    st.warning(
                        "⚠️ Confidence is relatively low. Consider expert review or uploading another scan."
                    )

                st.write("### Class probabilities (%)")
                for i, cls in enumerate(CLASSES):
                    p = preds[i] * 100
                    if cls == pred_class:
                        st.markdown(f"**➡️ {cls} → {p:.2f}% (predicted)**")
                    else:
                        st.markdown(f"{cls} → {p:.2f}%")

            st.write("---")

            df_report = pd.DataFrame({
                "Class": CLASSES,
                "Probability (%)": (preds * 100).round(2)
            })
    
            # Optionally show bar-chart
            df_chart = df_report.set_index("Class")
            st.bar_chart(df_chart)
            
            st.info(
                "ℹ️ **Note:** This prediction is for research-purpose only and does not constitute a medical diagnosis. "
                "Always consult a qualified healthcare professional for medical advice and interpretation of MRI scans."
            )


with tab2:
    st.header(" Dataset Summary")
    # --- stat-cards ---
    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-value">{TRAIN_IMAGES}</div>
      <div class="stat-label">Train Images</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{VALID_IMAGES}</div>
      <div class="stat-label">Validation Images</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{TEST_IMAGES}</div>
      <div class="stat-label">Test Images</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">{TOTAL_CLASSES}</div>
      <div class="stat-label">Total Classes</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("---")

        # Example model performance data / evaluation metrics
    model_data = {
        'Model':       ['Custom Model','VGG16', 'ResNet50', 'MobileNet', 'InceptionV3', 'EfficientNetB0'],
        'Val_Accuracy':[67,76, 74, 85, 82, 32],
        'Val_Precision':[50,58, 86, 87, 86, 00],
        'Val_Recall': [80, 89, 61, 84, 77, 00],
    }
    df_perf = pd.DataFrame(model_data)

    # Example dataset-split class counts 
        'Class':     ['glioma','meningioma','no_tumor','pituitary'],
        'Train':     [564, 358, 335, 438],
        'Valid':     [161, 124,  99, 118],
        'Test':      [ 80,  63,  49,  54],
    }).set_index('Class')

    # Selectbox to choose what to display
    view = st.selectbox(
        "Select to View the summary",
        ("Model performance table",
         "Model performance visualization",
         "Best performing model",
         "Dataset summary visualization")
    )

    if view == "Model performance table":
        st.subheader(" Model Performance Table")
        st.dataframe(df_perf, use_container_width=True)

    elif view == "Model performance visualization":
        st.subheader("Multi-Metric Performance Comparison (Accuracy / Precision / Recall)")

        # Melt the performance DataFrame to long format
        metrics_df = df_perf.melt(
            id_vars=['Model'],
            value_vars=['Val_Accuracy','Val_Precision','Val_Recall'],
            var_name='Metric',
            value_name='Score'
        )

        #  convert scores to percentages
        metrics_df['Score_pct'] = metrics_df['Score'] * 100

        fig_metrics = px.line(
            metrics_df,
            x='Model',
            y='Score_pct',
            color='Metric',
            markers=True,
            title='Model performance across metrics',
            labels={'Score_pct': 'Score (%)'}
        )
        fig_metrics.update_layout(height=400)
        st.plotly_chart(fig_metrics, use_container_width=True)

    elif view == "Best performing model":
        st.subheader("🏆 Best Performing Model")
        best_idx = df_perf['Val_Accuracy'].idxmax()
        best_model = df_perf.loc[best_idx, 'Model']
        best_acc   = df_perf.loc[best_idx, 'Val_Accuracy']
        best_prec  = df_perf.loc[best_idx, 'Val_Precision']
        best_rec   = df_perf.loc[best_idx, 'Val_Recall']
        st.write(f"**{best_model}** — Accuracy: **{best_acc*100:.2f}%**")
        st.write(f"Precision: {best_prec*100:.2f}%  |  Recall: {best_rec*100:.2f}% ")

    elif view == "Dataset summary visualization":
        st.subheader("📊 Dataset Distribution per Class (Train / Valid / Test)")
        st.bar_chart(df_counts)


    st.write("")  