import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="anupam-roy123/tourism-package-prediction",
    filename="best_tourism_model.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Insurance Charges Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the **Tourism package potential buyers ** based on personal and occupational details.
Please enter the required information below to get a prediction.
""")

# User input
age = st.number_input("age", min_value=19, max_value=61, value=30, step=1)
Gender = st.selectbox("Gender", ["male", "female"])
Type_of_Contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
Number_Of_Children_Visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=3, value=0, step=1)
City_Tier = st.number_input("City Tier", min_value=1, max_value=3, value=2, step=1)
Duration_Of_Pitch = st.number_input("Duration Of Pitch", min_value=5, max_value=127, value=60, step=1)
Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed" , "Free Lancer" , "Large Business"])
Number_Of_Person_Visiting = st.number_input("Number Of Person Visiting", min_value=2, max_value=4, value=3, step=1)
Number_Of_Followups=st.number_input("Number Of Followups", min_value=1, max_value=6, value=3, step=1)
Product_Pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe"])
Preferred_Property_Star = st.number_input("Preferred Property Star", min_value=3, max_value=5, value=4, step=1)
Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced" , "Unmarried"])
Number_Of_Trips = st.number_input("Number Of Trips", min_value=1, max_value=8, value=4, step=1)
Passport = st.number_input("Passport", min_value=0, max_value=1, value=0, step=1)
Pitch_Satisfaction_Score = st.number_input("Pitch Satisfaction Score", min_value=2, max_value=5, value=3, step=1)
OwnCar = st.number_input("OwnCar", min_value=0, max_value=1, value=0, step=1)
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Senior Manager", "AVP" , "VP"])
Monthly_Income = st.number_input("Monthly Income", min_value=16052, max_value=34545, value=25299, step=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'Gender': Gender,
    'Type_of_Contact': Type_of_Contact,
    'Number_Of_Children_Visiting' : Number_Of_Children_Visiting ,
    'City_Tier' : City_Tier,
    'Duration_Of_Pitch' : Duration_Of_Pitch,
    'Occupation' : Occupation,
    'Number-Of_Person_Visiting' : Number_Of_Person_Visiting,
    'Number_Of_Followups' : Number_Of_Followups,
    'Product_Pitched' : Product_Pitched,
    'Preferred_Property_Star': Preferred_Property_Star,
    'Marital_Status' : Marital_Status,
    'Number_Of_Trips' : Number_Of_Trips,
    'Passport' : Passport,
    'Pitch_Satisfaction_Score' : Pitch_Satisfaction_Score,
    'OwnCar' : OwnCar,
    'Designation' : Designation,
    'Monthly_Income': Monthly_Income
}])

# Prediction
if st.button("Predict ProdTaken"):
    prediction = model.predict(input_data)[0]
    result = "Customer is likely to purchase the package." if prediction == 1 else "Customer is not likely to purchase the package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
