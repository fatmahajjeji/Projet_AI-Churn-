📊 Customer Churn Prediction – Machine Learning Project
🔍 Project Overview

This project focuses on predicting customer churn (whether a client will leave a telecom company) using supervised Machine Learning techniques.
The goal is to build a robust, interpretable and well-validated ML pipeline, from raw data preprocessing to final model evaluation.

The project was developed using Python and scikit-learn, following best practices to avoid data leakage and ensure reproducibility.

🎯 Objectives

Master core Machine Learning techniques

Apply end-to-end ML pipeline using scikit-learn

Justify every preprocessing decision

Compare multiple ML models fairly

Interpret results from both technical and business perspectives

📂 Dataset Description

Name: Telco Customer Churn

Source: Kaggle

Size: 7,043 rows × 21 columns

Type: Supervised learning – Binary Classification

Target variable: Churn (Yes / No)

📌 Feature Types

Numerical:

tenure, MonthlyCharges, TotalCharges

Categorical:

Gender, Contract type, Internet service, Payment method, etc.

⚠️ The dataset is imbalanced (~26.5% churners), which is handled explicitly during modeling.

🧹 Data Preprocessing (Fully Justified)

All preprocessing steps are clearly explained and justified inside the notebook.

✔ Missing Values

TotalCharges converted to numeric

Missing values correspond to new customers (tenure = 0)

Imputed with 0 for business consistency

✔ Outlier Analysis

Detection using IQR method

Outliers correspond to high-value / long-tenure customers

❌ No removal to avoid business bias

✅ Solution: RobustScaler

✔ Encoding & Scaling

RobustScaler for numerical features

OneHotEncoder

drop='first' → avoids multicollinearity

handle_unknown='ignore' → production-safe

✔ Data Leakage Prevention

All preprocessing steps applied using scikit-learn Pipelines

fit() only on training data

⚙️ Feature Selection

OneHotEncoding resulted in 5,663 features

Risk of overfitting

Solution:

SelectKBest (ANOVA – f_classif)

Reduced to 30 most informative features

🤖 Machine Learning Models Used

Seven different models were trained and compared:

Logistic Regression

Random Forest

XGBoost

Support Vector Machine (SVC)

K-Nearest Neighbors (KNN)

Decision Tree

Naive Bayes

⚖️ Model Optimization

5-Fold Cross Validation

Metrics used:

Accuracy

Precision

Recall

F1-score

ROC-AUC

class_weight='balanced' applied when relevant

📈 Results

🏆 Best Model: Logistic Regression

Metric	Score
F1-score	≈ 0.61
ROC-AUC	≈ 0.78
Accuracy	≈ 82%

📌 Logistic Regression was selected for its:

Strong balance between precision and recall

Interpretability

Stability on unseen data

📊 Visualizations

The project includes:

Churn distribution plots

Model comparison charts

Confusion matrix

Final LinkedIn-style project infographic

📷 Example:

💡 Business Insights

Improved churn detection

Reduced false churn alerts

Better targeting of at-risk customers

Clear trade-off between recall and precision

🛠 Technologies Used

Python

scikit-learn

XGBoost

Pandas

NumPy

Matplotlib

Seaborn

Google Colab / Jupyter Notebook

▶️ How to Run the Project
pip install -r requirements.txt


Then open the notebook:

jupyter notebook

📌 Author

Fatma Hajjeji
📧 Email: fatmahajjeji9@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/fatma-hajjeji-29b1a8295

⭐ Conclusion

This project demonstrates a complete and professional Machine Learning workflow, combining strong technical foundations with clear business interpretation.
It reflects my ability to design, evaluate and explain ML solutions in a real-world context.
