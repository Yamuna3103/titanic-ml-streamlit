import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import joblib
import sklearn
st.write("Sklearn version:", sklearn.__version__)

st.set_page_config(page_title="Titanic Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction")

# ✅ LOAD SAVED MODEL
model = joblib.load("best_titanic_model.pkl")

st.subheader("🧍 Passenger Details")

pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 30)
sibsp = st.number_input("Siblings/Spouses", 0, 5, 0)
parch = st.number_input("Parents/Children", 0, 5, 0)
fare = st.slider("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": embarked
}])


if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success(" Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
