import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Loan Acceptance Predictor",
    layout="wide"
)

@st.cache_resource
def load_model(model_path='model.pkl'):
    model = None
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Error: The model file '{model_path}' was not found.")
        st.error("Please ensure 'model.pkl' is in the same directory as the app.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
    return model

model = load_model()

expected_cols_after_encoding = [
    'Dependents',
    'Applicant_Income',
    'Coapplicant_Income',
    'Loan_Amount',
    'Term',
    'Credit_History',
    'Gender_Male',
    'Married_Yes',
    'Education_Not Graduate',
    'Self_Employed_Yes',
    'Area_Semiurban',
    'Area_Urban'
]

st.sidebar.header("Enter Loan Application Details")
def get_user_input():
    gender = st.sidebar.selectbox("Gender", options=['Male', 'Female'], index=0)
    married = st.sidebar.selectbox("Married", options=['Yes', 'No'], index=0)
    dependents_input = st.sidebar.selectbox("Number of Dependents", options=['0', '1', '2', '3+'], index=0)
    education = st.sidebar.selectbox("Education", options=['Graduate', 'Not Graduate'], index=0)
    self_employed = st.sidebar.selectbox("Self Employed", options=['No', 'Yes'], index=0)
    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, value=6000)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0.0, value=2500.0, format="%.2f")
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=1.0, value=150000.0, format="%.2f")
    term = st.sidebar.number_input("Loan Term (Months)", min_value=12, max_value=480, value=360, step=12)
    credit_history_input = st.sidebar.selectbox("Credit History Available?", options=['Yes (Good History)', 'No (Bad History)'], index=0)
    area = st.sidebar.selectbox("Property Area", options=['Urban', 'Semiurban', 'Rural'], index=1)

    input_data = {
        'Gender': [gender], 'Married': [married], 'Dependents': [dependents_input],
        'Education': [education], 'Self_Employed': [self_employed],
        'Applicant_Income': [applicant_income], 'Coapplicant_Income': [coapplicant_income],
        'Loan_Amount': [loan_amount], 'Term': [float(term)],
        'Credit_History_Input': [credit_history_input],
        'Area': [area]
    }
    input_df = pd.DataFrame.from_dict(input_data)
    return input_df

user_input_df = get_user_input()

st.subheader("User Input Values (Raw):")
display_df = user_input_df.T.rename(columns={0: 'Value'})
display_df['Value'] = display_df['Value'].astype(str)
st.dataframe(display_df, use_container_width=True)


def preprocess_input_final(df, expected_cols):

    df_processed = df.copy()
    try:
        df_processed['Dependents'] = df_processed['Dependents'].replace('3+', '3')
        df_processed['Dependents'] = pd.to_numeric(df_processed['Dependents'])

        df_processed['Credit_History'] = 1.0 if df_processed['Credit_History_Input'].iloc[0] == 'Yes (Good History)' else 0.0
        df_processed = df_processed.drop(columns=['Credit_History_Input'])


        gender = df_processed['Gender'].iloc[0]
        married = df_processed['Married'].iloc[0]
        education = df_processed['Education'].iloc[0]
        self_employed = df_processed['Self_Employed'].iloc[0]
        area = df_processed['Area'].iloc[0]

        df_processed['Gender_Male'] = 1 if gender == 'Male' else 0
        df_processed['Married_Yes'] = 1 if married == 'Yes' else 0
        df_processed['Education_Not Graduate'] = 1 if education == 'Not Graduate' else 0
        df_processed['Self_Employed_Yes'] = 1 if self_employed == 'Yes' else 0
        df_processed['Area_Semiurban'] = 1 if area == 'Semiurban' else 0
        df_processed['Area_Urban'] = 1 if area == 'Urban' else 0

        cols_to_drop_after_encoding = ['Gender', 'Married', 'Education', 'Self_Employed', 'Area']
        df_processed = df_processed.drop(columns=cols_to_drop_after_encoding)

        if not set(expected_cols).issubset(set(df_processed.columns)):
             st.error(f"Error: Preprocessing failed to create all expected columns. Expected: {expected_cols}. Got: {df_processed.columns.tolist()}")
             return None

        df_processed = df_processed[expected_cols]

        return df_processed

    except Exception as e:
        st.error(f"An error occurred during preprocessing step: {e}")

        return None


if model is not None:
    processed_input_df = preprocess_input_final(user_input_df.copy(), expected_cols_after_encoding)

    if processed_input_df is not None:
        st.header("Prediction")
        if st.button("Predict Loan Acceptance"):
            try:
                prediction = model.predict(processed_input_df)
                probability = model.predict_proba(processed_input_df)

                st.subheader("Result:")
                if prediction[0] == 1:
                    st.success("🎉 Loan Likely Approved!")
                else:
                    st.error("😞 Loan Likely Rejected.")

                st.write("Prediction Probability:")
                try:
                    class_names = [f"Rejected ({model.classes_[0]})", f"Approved ({model.classes_[1]})"]

                    prob_rejected_raw = probability[0][0]
                    prob_approved_raw = probability[0][1]

                    percent_rejected_str = f"{prob_rejected_raw * 100:.1f}%"
                    percent_approved_str = f"{prob_approved_raw * 100:.1f}%"

                    formatted_data = [[percent_rejected_str, percent_approved_str]]

                    prob_df = pd.DataFrame(formatted_data, columns=class_names, index=['Probability (%)'])  # Use new index label
                    st.dataframe(prob_df, use_container_width=True)

                except Exception as prob_e:
                    st.warning(f"Could not display probability details: {prob_e}")
                    st.write(f"Raw probabilities (if formatting failed): {probability}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    else:
        st.warning("Preprocessing failed. Prediction unavailable.")

else:
    st.warning("Model could not be loaded. Prediction is unavailable.")
    st.info("Please ensure 'model.pkl' is in the same directory as the app.")

st.markdown("---")
st.caption("Loan Acceptance Prediction App")