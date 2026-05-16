# Mid-Term Project: End-to-End Machine Learning Pipeline

> **AI Programming 101 — Mid-Term Project**
> Submission deadline: **2026/05/22**

---

## 🎯 Objectives

By completing this project, students will demonstrate their ability to:

1. Load, clean, and explore a real-world dataset
2. Engineer informative features
3. Train and compare multiple classical ML models
4. Tune hyperparameters systematically
5. Evaluate and interpret model performance
6. Present findings clearly in a Jupyter notebook

---

## 📂 Dataset

Choose **one** of the following open datasets:

| # | Dataset | Task | Source |
|---|---------|------|--------|
| A | Titanic Passenger Survival | Binary Classification | [Kaggle](https://www.kaggle.com/c/titanic) |
| B | Heart Disease UCI | Binary Classification | [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| C | Student Performance | Regression | [UCI ML Repository](https://archive.ics.uci.edu/dataset/320/student+performance) |
| D | Bike Sharing | Regression | [UCI ML Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) |

If you wish to use a different dataset, obtain instructor approval first.

---

## 📋 Requirements

### 1 · Data Loading & Exploration (20 pts)

- [ ] Load the dataset into a Pandas DataFrame
- [ ] Show `shape`, `dtypes`, `head()`, and `describe()`
- [ ] Identify and visualise missing values
- [ ] Plot at least **3 meaningful charts** (distribution, correlations, class balance, etc.)
- [ ] Write a short paragraph summarising key observations

### 2 · Data Preprocessing & Feature Engineering (20 pts)

- [ ] Handle missing values (drop, impute, or justify leaving them)
- [ ] Encode categorical variables (one-hot, ordinal, or label encoding)
- [ ] Scale numerical features using an appropriate scaler
- [ ] Create at least **1 new feature** derived from existing ones
- [ ] Wrap preprocessing in a **scikit-learn Pipeline**

### 3 · Model Training & Comparison (30 pts)

Train and compare at least **3** of the following models:

- Logistic Regression / Linear Regression (baseline)
- K-Nearest Neighbours
- Decision Tree
- Random Forest
- Gradient Boosting (sklearn, XGBoost, or LightGBM)

For each model:
- [ ] Use 5-fold cross-validation
- [ ] Report the mean ± std of the primary metric

### 4 · Hyperparameter Tuning (15 pts)

- [ ] Apply `GridSearchCV` or `RandomizedSearchCV` to your **best model**
- [ ] Search at least **2 hyperparameters** over at least **3 values each**
- [ ] Report the best parameters found
- [ ] Compare performance before and after tuning

### 5 · Final Evaluation & Interpretation (15 pts)

- [ ] Evaluate the tuned model on the held-out test set
- [ ] For **classification**: report accuracy, F1, confusion matrix, ROC-AUC
- [ ] For **regression**: report MAE, RMSE, R²
- [ ] Plot feature importances (for tree-based models)
- [ ] Write a **200-word conclusion**: What did you learn? Where does the model fail?

---

## 🗂️ Deliverables

Submit via the course portal a single `.zip` archive containing:

```
<student_id>_midterm/
├── notebook.ipynb    # Complete, fully-executed notebook
├── report.pdf        # 2-page PDF summary (intro, methods, results, conclusion)
└── data/             # Dataset file(s) — or a README with download instructions
```

---

## 🏆 Grading Rubric

| Section | Points |
|---------|--------|
| Data Loading & EDA | 20 |
| Preprocessing & Feature Engineering | 20 |
| Model Training & Comparison | 30 |
| Hyperparameter Tuning | 15 |
| Final Evaluation & Interpretation | 15 |
| **Total** | **100** |

> **Bonus (+5 pts)**: Deploy the model as a simple web API using FastAPI or Flask and include a screenshot of a working prediction.

---

## 💡 Tips

- Commit your notebook to GitHub regularly so you don't lose work.
- Always fit your scaler/encoder on the **training set only**.
- A well-explained simple model beats a badly-explained complex one.
- Start early — hyperparameter search can take time!
