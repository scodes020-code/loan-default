
import streamlit as st
import joblib
import pandas as pd

#page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

#load model
model=joblib.load("loan_default_model.pkl")

#custom css
st.markdown('''
<style>
.main-header {font-size:36px; color:#4CAF50; text-align:center;}
.sub-header {font-size:24px; color:#2196F3; text-align:center;}
.stButton>button {width:100%; height:3em; font-size:18px;}
</style>
''',unsafe_allow_html=True)

#sidebar
with st.sidebar:
  st.image(
  "https://cdn-icons-png-flaticon.com/512/3135/3135715.png",
  width=150)
  st.title("Loan Default Prediction")
  st.markdown('''### Models Used
  - Logistic Regression
  - Random Forest
  - XGBoost
  ### Objective
  Predict whether a borrower is likely to default on a loan
  ### Features
  - Real-Time Prediction
  - Risk Probability
  - ML-powered Decision system
  ''')
st.divider()
st.info("Built with Streamlit, scikit-learn, and XGBoost.")

#Header
st.markdown("<h1 class='main-header'>Loan Default Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='sub-header'>Machine Learning Powered Credit Risk Assessment Dashboard</h2>", unsafe_allow_html=True)
st.divider()

st.write("Please input the borrower's details below to get a loan default prediction.")

#Input section
col1, col2, col3 = st.columns(3)

# Numerical inputs
with col1:
 Age=st.number_input(
"Age",
min_value=18,
max_value=100,
value=30)
 Income=st.number_input(
"Income",
min_value=1000,
value=50000)
 LoanAmount=st.number_input(
"Loan Amount",
min_value=1000,
value=100000)

with col2:
  CreditScore = st.number_input(
  "Credit Score",
  min_value=300,
  max_value=850,
  value=650)
  InterestRate = st.number_input(
  "Interest Rate (%)",
  min_value=1.0,
  max_value=40.0,
  value=10.0)
  LoanTerm = st.number_input(
  "Loan Term (Months)",
 min_value=6,
 max_value=360,
 value=36 )

# Categorical inputs
with col3:
  MonthsEmployed = st.number_input(
  "Months Employed",
  min_value=0,
  value=5)

  NumCreditLines = st.number_input(
  "Number of Credit Lines",
  min_value=0,
  value=5)

  DTIRatio = st.number_input(
 "Debt-to-Income Ratio",
  min_value=0.0,
  max_value=1.0,
  value=0.30)

  Education = st.selectbox(
      "Education Level",
      ('Bachelor's', 'High School', 'Master's', 'PhD'))

  EmploymentType = st.selectbox(
      "Employment Type",
      ('Full-time', 'Unemployed', 'Self-employed', 'Part-time'))

  MaritalStatus = st.selectbox(
      "Marital Status",
      ('Divorced', 'Married', 'Single'))

  HasMortgage = st.radio(
      "Has Mortgage?",
      ('No', 'Yes'))

  HasDependents = st.radio(
      "Has Dependents?",
      ('No', 'Yes'))

  LoanPurpose = st.selectbox(
      "Loan Purpose",
      ('Auto', 'Business', 'Education', 'Home', 'Other'))

  HasCoSigner = st.radio(
      "Has Co-Signer?",
      ('No', 'Yes'))

# Prediction button
st.markdown('''<div style='text-align:center;'>''', unsafe_allow_html=True)
if st.button("Predict Loan Default"): # This should be outside of the column context
    # Prepare features for the model
    # Ensure the order and names match the training data X.columns
    input_data = pd.DataFrame([[Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio,
                                  1 if Education == 'High School' else 0,
                                  1 if Education == 'Master's' else 0,
                                  1 if Education == 'PhD' else 0,
                                  1 if EmploymentType == 'Part-time' else 0,
                                  1 if EmploymentType == 'Self-employed' else 0,
                                  1 if EmploymentType == 'Unemployed' else 0,
                                  1 if MaritalStatus == 'Married' else 0,
                                  1 if MaritalStatus == 'Single' else 0,
                                  1 if HasMortgage == 'Yes' else 0,
                                  1 if HasDependents == 'Yes' else 0,
                                  1 if LoanPurpose == 'Business' else 0,
                                  1 if LoanPurpose == 'Education' else 0,
                                  1 if LoanPurpose == 'Home' else 0,
                                  1 if LoanPurpose == 'Other' else 0,
                                  1 if HasCoSigner == 'Yes' else 0]],
                                columns=
                                [
                                    'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                                    'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
                                    'Education_High School', 'Education_Master's', 'Education_PhD',
                                    'EmploymentType_Part-time', 'EmploymentType_Self-employed',
                                    'EmploymentType_Unemployed', 'MaritalStatus_Married', 'MaritalStatus_Single',
                                    'HasMortgage_Yes', 'HasDependents_Yes', 'LoanPurpose_Business',
                                    'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other',
                                    'HasCoSigner_Yes'
                                ])

    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1] # Probability of default

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Prediction Result</h3>", unsafe_allow_html=True)
    if prediction == 1:
        st.error(f"The model predicts that this borrower is LIKELY TO DEFAULT! (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"The model predicts that this borrower is UNLIKELY TO DEFAULT. (Probability: {prediction_proba:.2f})")
st.markdown('''</div>''', unsafe_allow_html=True)
