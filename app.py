import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Bank Marketing Predictor", layout="centered")

st.title("Bank Marketing Subscription Predictor")
st.markdown("Predict whether a client will subscribe to a term deposit.")

model = load("model.pkl")

st.subheader("Client Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
    marital = st.selectbox("Marital Status", ["divorced", "married", "single"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
with col2:
    default = st.selectbox("Has Credit in Default?", ["no", "yes"])
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])

st.subheader("Financial Information")
col1, col2 = st.columns(2)
with col1:
    balance = st.number_input("Account Balance (EUR)", min_value=-10000, max_value=100000, value=1000)
with col2:
    st.write("")

st.subheader("Campaign Information")
col1, col2 = st.columns(2)
with col1:
    day = st.slider("Day of Month", min_value=1, max_value=31, value=15)
    month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    campaign = st.number_input("Number of Contacts This Campaign", min_value=1, max_value=50, value=3)
with col2:
    pdays = st.number_input("Days Since Last Contact", min_value=0, max_value=1000, value=999)
    previous = st.number_input("Number of Previous Contacts", min_value=0, max_value=50, value=0)
    poutcome = st.selectbox("Previous Campaign Outcome", ["failure", "other", "success", "unknown"])

predict_btn = st.button("Predict Subscription", type="primary", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "campaign": campaign,
        "pdays": pdays, "previous": previous, "poutcome": poutcome
    }])

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.subheader("Prediction")
    if pred == 1:
        st.error(f"**Will Subscribe** — Probability: {proba[1]:.1%}")
    else:
        st.success(f"**Will Not Subscribe** — Probability: {proba[1]:.1%}")

    st.subheader("Top Drivers")
    st.markdown("""
Based on the best tree-based model, the top factors influencing subscription are:

1. **poutcome_success** — A successful previous campaign is the strongest indicator
2. **balance** — Higher account balances increase likelihood
3. **age** — Middle-aged clients show higher propensity
4. **pdays** — Fewer days since last contact signals a warmer lead
5. **campaign** — Over-contacting reduces likelihood
""")
else:
    st.info("Fill in the client details above and click **Predict Subscription**.")
