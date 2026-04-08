# Student_Dropout_LogRegModel
End-to-end machine learning project for student dropout prediction using Logistic Regression, including data preprocessing, model training, and evaluation
# 🎓 Student Dropout Prediction — Logistic Regression

Binary classification model that predicts whether a university student will drop out based on demographic, academic, and behavioral features.

---

📁 Project Structure

```
├── LogRegModel.ipynb               # Main notebook
├── student_dropout_dataset_v3.csv  # Dataset
└── README.md
```

---
📊 Dataset

| Property | Value |
|---|---|
| Source | `student_dropout_dataset_v3.csv` |
| Total rows | ~10,000 |
| After cleaning | 9,020 |
| Target column | `Dropout` (0 = stayed, 1 = dropped out) |
| Features | 15 |

Features used:

| Feature | Type |
|---|---|
| Age, Family_Income, CGPA | Numerical |
| Study_Hours_per_Day, Attendance_Rate | Numerical |
| Stress_Index, Assignment_Delay_Days, Travel_Time_Minutes | Numerical |
| Gender, Internet_Access, Part_Time_Job, Scholarship | Categorical |
| Semester, Department, Parental_Education | Categorical |

> Note: `GPA` and `Semester_GPA` were removed due to high multicollinearity with `CGPA` (correlation > 0.96).

---

⚙️ Methodology

1. Data Cleaning — removed duplicates, dropped rows with missing values
2. Correlation Analysis — heatmap to identify redundant features
3. Feature Engineering — `OneHotEncoder` for categoricals, `StandardScaler` for numericals
4. Modeling — `LogisticRegression` inside a `sklearn Pipeline`
5. Evaluation — `classification_report` on 15% test split

Model config:

LogisticRegression(
    class_weight='balanced',
    max_iter=5000,
    solver='saga'
)

📈 Results
              precision    recall  f1-score   support

           0       0.89      0.74      0.81      1023
           1       0.47      0.72      0.57       330

    accuracy                           0.73      1353
   macro avg       0.68      0.73      0.69      1353
weighted avg       0.79      0.73      0.75      1353


- Overall accuracy: 73%
- Dropout class (1) recall: 72% — model catches most actual dropouts
- Dropout class (1) precision: 47% — relatively high false positive rate

---

🚀 Getting Started

```
bash
pip install pandas scikit-learn seaborn matplotlib
jupyter notebook LogRegModel.ipynb
```

🛠️ Tech Stack

- Python 3.x
- pandas, scikit-learn, seaborn, matplotlib
