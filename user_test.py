import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from matplotlib_venn import venn2
import matplotlib.pyplot as plt


# function to calculate entropy
def calc_entropy(labels):
    probs = labels.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))


st.set_page_config(page_title="Medical AI Dashboard", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1a73e8; text-align: center; font-weight: 700; }
    h2, h3 { color: #202124; border-left: 5px solid #1a73e8; padding-left: 10px; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Analyzing Information Relationships in Diabetes Diagnosis")

# Sidebar Data Loading
st.sidebar.header("📂 Data Source")
data_option = st.sidebar.selectbox("Dataset", ["Pima Indians Diabetes (Real-world)", "Upload CSV"])

if data_option == "Pima Indians Diabetes (Real-world)":
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age',
             'Outcome']
    df = pd.read_csv(url, names=names)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# Calculations
X = df.drop('Outcome', axis=1)
y = df['Outcome']
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# 1. Theory Section
st.header("🔬 1. Theory: Uncertainty Analysis")
base_ent = calc_entropy(y)
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Total Entropy", f"{base_ent:.4f} Bits")
    st.write("**Information Gain (MI Scores):**")
    st.dataframe(mi_results, use_container_width=True)
with col2:
    st.bar_chart(mi_results)

st.divider()

# 2. Venn Diagram Section (Moved Up)
st.header("🎯 2. Information Relationship (Venn Diagram)")
col_v1, col_v2 = st.columns([1, 2])

with col_v1:
    st.write("Select a feature to see how much information it shares with the Diabetes Outcome.")
    selected_feat = st.selectbox("Choose Feature for Analysis", X.columns, key="venn_select")
    mi_val = mi_results[selected_feat]
    st.info(f"Mutual Information: **{mi_val:.3f} Bits**")

with col_v2:
    fig, ax = plt.subplots(figsize=(5, 3))
    v = venn2(subsets=(0.5, 0.5, mi_val),
              set_labels=(selected_feat, 'Outcome'),
              set_colors=('skyblue', 'orange'),
              alpha=0.6)

    # Labeling Venn areas
    if v.get_label_by_id('11'): v.get_label_by_id('11').set_text(f'I(X;Y)\n{mi_val:.3f}')
    if v.get_label_by_id('10'): v.get_label_by_id('10').set_text(f'H({selected_feat}|Y)')
    if v.get_label_by_id('01'): v.get_label_by_id('01').set_text(f'H(Y|{selected_feat})')


    st.pyplot(fig)

st.divider()
#
# # 3. Prediction Form Section (Moved Down)
# st.header("🤖 3. Manual Prediction Form")
# model = RandomForestClassifier(random_state=42)
# model.fit(X, y)
#
# input_cols = st.columns(3)
# user_inputs = []
# for i, col_name in enumerate(X.columns):
#     with input_cols[i % 3]:
#         if col_name == 'DiabetesPedigree':
#             val = st.number_input(f"{col_name}", value=float(df[col_name].mean()), format="%.3f")
#         else:
#             val = st.number_input(f"{col_name}", value=int(df[col_name].mean()), step=1)
#         user_inputs.append(val)
# ... (Keep previous imports and setup)

# Dictionary for Medical Units
units = {
    'Pregnancies': 'Count',
    'Glucose': 'mg/dL',
    'BloodPressure': 'mm Hg',
    'SkinThickness': 'mm',
    'Insulin': 'mu U/ml',
    'BMI': 'kg/m²',
    'DiabetesPedigree': 'Score',
    'Age': 'Years'
}

# 3. Prediction Form Section
st.header("🤖 3. Manual Prediction Form")
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

input_cols = st.columns(3)
user_inputs = []

for i, col_name in enumerate(X.columns):
    unit = units.get(col_name, "")  # Get unit from dictionary
    with input_cols[i % 3]:
        label = f"{col_name} ({unit})"  # Append unit to label

        if col_name == 'DiabetesPedigree':
            val = st.number_input(label, value=float(df[col_name].mean()), format="%.3f")
        elif col_name == 'BMI':
            val = st.number_input(label, value=float(df[col_name].mean()), format="%.1f")
        else:
            val = st.number_input(label, value=int(df[col_name].mean()), step=1)
        user_inputs.append(val)

# ... (Keep remaining prediction logic)
if st.button("Predict Outcome"):
    prediction = model.predict([user_inputs])
    prob = model.predict_proba([user_inputs])

    st.markdown("---")
    res_conf = prob[0][1] if prediction[0] == 1 else prob[0][0]
    if prediction[0] == 1:
        st.error(f"### ⚠️ ရလဒ်: ဆီးချိုဖြစ်နိုင်ခြေ ရှိပါသည်။ (Confidence: {res_conf:.2%})")
    else:
        st.success(f"### ✅ ရလဒ်: ဆီးချိုဖြစ်နိုင်ခြေ နည်းပါသည်။ (Confidence: {res_conf:.2%})")
