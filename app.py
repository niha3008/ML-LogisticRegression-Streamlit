import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

saved = load_model()
model = saved["model"]
feature_names = saved["feature_names"]
sample_data = saved["sample_data"]

st.title("🔬 Breast Cancer Classification")
st.write("Logistic Regression Model Deployment using Streamlit")

st.info(
    "Educational demonstration only. "
    "This application is not a medical diagnostic tool."
)

st.subheader("Select Input Sample")

sample_choice = st.selectbox(
    "Choose a sample",
    [f"Sample {i+1}" for i in range(len(sample_data))]
)

sample_index = int(sample_choice.split()[-1]) - 1
sample = sample_data[sample_index]

st.subheader("Enter Tumor Features")
st.write("Modify the measurements to generate a prediction.")

input_values = []

cols = st.columns(3)

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        value = st.number_input(
            feature.title(),
            min_value=0.0,
            value=float(sample[i]),
            format="%.5f",
            key=feature
        )
        input_values.append(value)

if st.button("Predict", type="primary"):

    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    benign_prob = probabilities[0]
    malignant_prob = probabilities[1]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("Predicted Classification: BENIGN")
        confidence = benign_prob
    else:
        st.error("Predicted Classification: MALIGNANT")
        confidence = malignant_prob

    st.metric(
        "Model Probability for Predicted Class",
        f"{confidence * 100:.2f}%"
    )

    st.write("### Class Probabilities")

    st.write(f"Benign: {benign_prob * 100:.2f}%")
    st.write(f"Malignant: {malignant_prob * 100:.2f}%")

    chart_data = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Probability": [benign_prob, malignant_prob]
    }).set_index("Class")

    st.bar_chart(chart_data)

    st.caption(
        "Predictions are based on dataset measurements "
        "and should not be used for clinical decisions."
    )
