import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from matplotlib_venn import venn2
import matplotlib.pyplot as plt


# ၁။ Entropy တွက်ချက်သည့် function
def calc_entropy(labels):
    probs = labels.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))


st.set_page_config(page_title="Medical AI Dashboard", layout="wide")

# UI ပိုင်းကို လှပအောင် CSS ထည့်ခြင်း
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Headers Customization */
    h1 { color: #1a73e8; text-align: center; font-weight: 700; }
    h2, h3 { color: #202124; border-left: 5px solid #1a73e8; padding-left: 10px; }

    /* Card-like containers for metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Button Style */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1557b0;
        border-color: #1557b0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Medical Information Theory & AI Prediction")

# Session State Initialize (Venn Diagram မပျောက်သွားစေရန်)
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

# Sidebar Data Loading
st.sidebar.header("📂 Data Source")
data_option = st.sidebar.selectbox("စမ်းသပ်မည့် Dataset ကိုရွေးပါ",
                                   ["Pima Indians Diabetes (Real-world)", "ကိုယ်ပိုင် CSV တင်မည်"])

if data_option == "Pima Indians Diabetes (Real-world)":
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age',
             'Outcome']
    df = pd.read_csv(url, names=names)
else:
    uploaded_file = st.sidebar.file_uploader("CSV တင်ပါ", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# --- Main Logic ---
X = df.drop('Outcome', axis=1)
y = df['Outcome']
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# ၁။ Theory Section
st.header("🔬 1. Theory: Uncertainty Analysis")
base_ent = calc_entropy(y)
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("မူလ မသေချာမှု (Total Entropy)", f"{base_ent:.4f} Bits")
    st.write("**Information Gain (MI Scores):**")
    st.dataframe(mi_results, use_container_width=True)
with col2:
    st.bar_chart(mi_results)

# ၂။ Prediction Section
st.divider()
st.header("🤖 2. AI Prediction Form")
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

input_cols = st.columns(3)
user_inputs = []
# input_cols အပိုင်းတွင် ဤကဲ့သို့ ပြင်ဆင်ပါ
for i, col_name in enumerate(X.columns):
    with input_cols[i % 3]:
        # DiabetesPedigree အတွက်သာ Decimal ထားပြီး ကျန်တာကို ကိန်းပြည့်ပြောင်းခြင်း
        if col_name == 'DiabetesPedigree':
            val = st.number_input(f"{col_name}", value=float(df[col_name].mean()), format="%.3f")
        else:
            val = st.number_input(f"{col_name}", value=int(df[col_name].mean()), step=1)
        user_inputs.append(val)

if st.button("Predict & Analyze Information Connection"):
    prediction = model.predict([user_inputs])
    prob = model.predict_proba([user_inputs])
    st.session_state.res = {
        'pred': prediction[0],
        'conf': prob[0][1] if prediction[0] == 1 else prob[0][0]
    }
    st.session_state.show_results = True

# ရလဒ်နှင့် Venn Diagram ပြသခြင်း
if st.session_state.show_results:
    res = st.session_state.res
    st.markdown("---")
    if res['pred'] == 1:
        st.error(f"### ⚠️ ရလဒ်: ဆီးချိုဖြစ်နိုင်ခြေ ရှိပါသည်။ (Confidence: {res['conf']:.2%})")
    else:
        st.success(f"### ✅ ရလဒ်: ကျန်းမာရေး ကောင်းမွန်ပါသည်။ (Confidence: {res['conf']:.2%})")

    st.header("🎯 3. Information Relationship (Venn Diagram)")
    st.write("စမ်းသပ်ချက်တစ်ခုသည် ရောဂါရှာဖွေမှုအပေါ် မည်မျှလွှမ်းမိုးမှုရှိကြောင်း Venn Diagram ဖြင့် ကြည့်ရှုခြင်း။")

    # Feature Selectbox (Session State ကြောင့် ရွေးချယ်မှုပြုလုပ်လျှင်လည်း Prediction Result ပျောက်မသွားပါ)
    selected_feat = st.selectbox("လေ့လာမည့် အချက်အလက်ကို ရွေးပါ", X.columns, key="venn_select")
    mi_val = mi_results[selected_feat]

    fig, ax = plt.subplots(figsize=(8, 5))
    v = venn2(subsets=(0.6, 0.6, mi_val), set_labels=(selected_feat, 'Outcome'), set_colors=('skyblue', 'orange'),
              alpha=0.7)

    if v.get_label_by_id('11'): v.get_label_by_id('11').set_text(f'I(X;Y)\n{mi_val:.3f} Bits')
    if v.get_label_by_id('01'): v.get_label_by_id('01').set_text(f'H({selected_feat}|Y)')
    if v.get_label_by_id('10'): v.get_label_by_id('10').set_text(f'H(Y|{selected_feat})')

    plt.title(f"Information Relationship: {selected_feat} vs Outcome", fontsize=14)
    st.pyplot(fig)
    st.info(
        f"💡 **သီအိုရီအရ ရှင်းပြချက်:** AI ၏ Confidence ({res['conf']:.2%}) သည် အလယ်ရှိ Shared Information $I(X;Y)$ အပေါ်တွင် မူတည်ပါသည်။")