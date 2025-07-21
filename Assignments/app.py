import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("logistic_model.pkl")

# Title
st.title("Titanic Survival Prediction")

st.markdown("### Please fill in the passenger details:")

# Input fields with no default values
passenger_id = st.number_input("Passenger ID", min_value=1, step=1, value=None, format="%d")
pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=None, key="pclass")
sex = st.selectbox("Sex", options=["male", "female"], index=None, key="sex")
age = st.number_input("Age", min_value=0.0, step=1.0, value=None, format="%.1f")
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, step=1, value=None, format="%d")
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, step=1, value=None, format="%d")
fare = st.number_input("Fare", min_value=0.0, step=1.0, value=None, format="%.2f")
embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"], index=None, key="embarked")

# Predict button
if st.button("Predict Survival"):
    # Ensure all values are filled
    if None in (passenger_id, pclass, sex, age, sibsp, parch, fare, embarked):
        st.warning("🚨 Please fill in all fields before predicting.")
    else:
        # Encode categorical values as per training
        sex = 0 if sex == "male" else 1
        embarked_map = {"S": 0, "C": 1, "Q": 2}
        embarked = embarked_map[embarked]

        # Create DataFrame
        features = pd.DataFrame([[passenger_id, pclass, sex, age, sibsp, parch, fare, embarked]],
                                columns=['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

        # Predict
        prediction = model.predict(features)[0]

        # Output
        if prediction == 1:
            st.success("✅ The passenger is likely to survive.")
        else:
            st.error("❌ The passenger is not likely to survive.")
