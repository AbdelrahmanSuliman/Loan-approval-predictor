# Loan Approval Predictor Streamlit App

A simple web application built with Streamlit to predict loan approval status based on user input, using a model trained on historical loan data.

[Deployed application link](https://loan-approval-predictor-fe27tcurj59lm22dgl9pnv.streamlit.app/)
---

## Tech Stack

- **Python**
- **Scikit-learn**
- **Pandas**
- **Streamlit**
- **Joblib**

---

## Feature Insights

Feature significance based on p-values from initial statistical tests conducted during data analysis (p < 0.05 considered significant):

| Feature        | p-value  | Significance           |
|----------------|----------|--------------------------|
| Gender         | 0.73915  | ❌ Insignificant         |
| Married        | 0.02961  | ✅ Significant           |
| Dependents     | 0.36887  | ❌ Insignificant         |
| Education      | 0.04310  | ✅ Significant           |
| Self Employed  | 1.00000  | ❌ Insignificant         |
| Credit History | 0.00000  | ✅ Highly Significant    |
| Area           | 0.00214  | ✅ Significant           |

---

## Running Locally

1.  Clone the repository:
    ```bash
    git clone https://github.com/AbdelrahmanSuliman/Loan-approval-predictor.git
    cd Loan-approval-predictor
    ```
2.  Install requirements (Python 3.7+ recommended):
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

---

