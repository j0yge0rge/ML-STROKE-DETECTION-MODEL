# Stroke Prediction Analysis

A comprehensive data analysis and machine learning project to predict stroke occurrence based on healthcare and demographic data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-latest-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![presentation]([https://img.shields.io/badge/license-MIT-green.svg](https://www.canva.com/design/DAGnDpMH-30/QtC9ZnsGbMfH5ntO3yFgvA/edit?utm_content=DAGnDpMH-30&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton))

---

## Project Overview

This project analyzes a healthcare dataset containing patient information to identify patterns and risk factors associated with stroke occurrence. Using exploratory data analysis (EDA) and machine learning techniques, we aim to build a predictive model that can help identify individuals at high risk of stroke.

Stroke is a leading cause of death and disability worldwide. Early prediction and intervention can significantly improve patient outcomes. This analysis explores various demographic, health, and lifestyle factors that contribute to stroke risk.

---

## Objectives

- Perform comprehensive exploratory data analysis (EDA) on stroke patient data
- Identify key risk factors and patterns associated with stroke occurrence
- Handle missing data and perform data preprocessing
- Build and evaluate machine learning models for stroke prediction
- Provide actionable insights for healthcare professionals

---

## Dataset Description

### Source
Healthcare Stroke Dataset

### Dataset Size
- **Rows:** 5,110 patients
- **Columns:** 12 features + 1 target variable

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `id` | Integer | Unique patient identifier |
| `gender` | Categorical | Patient gender (Male, Female, Other) |
| `age` | Numerical | Patient age in years |
| `hypertension` | Binary | 0 = No hypertension, 1 = Has hypertension |
| `heart_disease` | Binary | 0 = No heart disease, 1 = Has heart disease |
| `ever_married` | Categorical | Yes/No - Marital status |
| `work_type` | Categorical | Type of work (Private, Self-employed, Govt_job, children, Never_worked) |
| `Residence_type` | Categorical | Urban or Rural residence |
| `avg_glucose_level` | Numerical | Average glucose level in blood (mg/dL) |
| `bmi` | Numerical | Body Mass Index |
| `smoking_status` | Categorical | Smoking status (formerly smoked, never smoked, smokes, Unknown) |
| `stroke` | Binary | **Target Variable** - 0 = No stroke, 1 = Had stroke |

### Sample Data

| id | gender | age | hypertension | heart_disease | ever_married | work_type | Residence_type | avg_glucose_level | bmi | smoking_status | stroke |
|----|--------|-----|--------------|---------------|--------------|-----------|----------------|-------------------|-----|----------------|--------|
| 9046 | Male | 67.0 | 0 | 1 | Yes | Private | Urban | 228.69 | 36.6 | formerly smoked | 1 |
| 51676 | Female | 61.0 | 0 | 0 | Yes | Self-employed | Rural | 202.21 | NaN | never smoked | 1 |
| 31112 | Male | 80.0 | 0 | 1 | Yes | Private | Rural | 105.92 | 32.5 | never smoked | 1 |

---

## Analysis Performed

### 1. Exploratory Data Analysis (EDA)
- **Data structure analysis:** Understanding data types and dimensions
- **Missing value detection:** Identifying and handling null values (especially in BMI column)
- **Statistical summary:** Descriptive statistics for all features
- **Distribution analysis:** Examining the distribution of numerical features
- **Class imbalance check:** Analyzing the target variable distribution

### 2. Data Visualization
- Distribution plots for age, BMI, and glucose levels
- Count plots for categorical variables
- Correlation heatmap to identify feature relationships
- Box plots to detect outliers
- Stroke occurrence by different demographic groups

### 3. Data Preprocessing
- **Handling missing values:** Imputation strategies for BMI
- **Encoding categorical variables:** One-hot encoding or label encoding
- **Feature scaling:** Normalization/standardization for numerical features
- **Handling class imbalance:** SMOTE or class weights for imbalanced dataset
- **Feature engineering:** Creating new meaningful features if needed

### 4. Machine Learning Models
- **Logistic Regression:** Baseline model
- **Decision Trees:** Non-linear relationships
- **Random Forest:** Ensemble method for better accuracy
- **Gradient Boosting (XGBoost/LightGBM):** Advanced ensemble technique
- **Support Vector Machines (SVM):** For complex decision boundaries

### 5. Model Evaluation
- **Metrics:**
  - Accuracy
  - Precision
  - Recall (Sensitivity) - Important for medical diagnosis
  - F1-Score
  - ROC-AUC Score
  - Confusion Matrix
- **Cross-validation:** K-fold validation for robust performance estimation
- **Feature importance analysis:** Identifying the most influential factors

---

## Technologies Used

### Core Libraries
- **Python 3.8+** - Programming language
- **Jupyter Notebook** - Interactive development environment

### Data Analysis & Manipulation
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Data Visualization
- **Matplotlib** - Basic plotting
- **Seaborn** - Statistical data visualization

### Machine Learning
- **Scikit-learn** - Machine learning algorithms and tools
  - Preprocessing
  - Model selection
  - Metrics
  - Ensemble methods

### Optional Advanced Libraries
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Gradient boosting framework
- **Imbalanced-learn** - SMOTE for handling imbalanced data

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/stroke-prediction-analysis.git
cd stroke-prediction-analysis
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

Or for JupyterLab:
```bash
jupyter lab
```

### Step 5: Open the Notebook

Navigate to and open `stroke-analysis.ipynb` in your browser.

---

## Requirements

Create a `requirements.txt` file with the following content:

```txt
# Core Data Science Libraries
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.3.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0

# Optional - Advanced ML
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0

# Utilities
scipy>=1.11.0
```

---

## Usage

### Running the Full Analysis

1. **Open the notebook:**
   ```bash
   jupyter notebook stroke-analysis.ipynb
   ```

2. **Run all cells:**
   - In Jupyter: `Cell → Run All`
   - Or run cells sequentially with `Shift + Enter`

### Notebook Structure

```
1. Import Libraries & Load Data
2. Data Exploration
   - Basic information
   - Statistical summary
   - Missing values analysis
3. Data Visualization
   - Distribution plots
   - Correlation analysis
   - Feature relationships
4. Data Preprocessing
   - Handle missing values
   - Encode categorical variables
   - Feature scaling
   - Train-test split
5. Model Training
   - Train multiple models
   - Hyperparameter tuning
6. Model Evaluation
   - Performance metrics
   - Comparison of models
   - Feature importance
7. Results & Insights
8. Conclusions & Recommendations
```

---

## Key Findings

> **Note:** Add your actual findings after completing the analysis

### Risk Factors Identified

1. **Age:** Strong positive correlation with stroke occurrence
   - Elderly patients (65+) show significantly higher stroke rates

2. **Hypertension:** Patients with hypertension are at higher risk
   - XX% of stroke patients had hypertension

3. **Heart Disease:** Strong predictor of stroke
   - Comorbidity increases stroke likelihood

4. **Average Glucose Level:** Elevated glucose levels associated with stroke
   - Threshold: >XXX mg/dL shows increased risk

5. **BMI:** Both underweight and obese patients show elevated risk
   - Optimal BMI range: XX-XX

6. **Smoking Status:** Former and current smokers at higher risk
   - "Formerly smoked" category shows significant correlation

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | XX% | XX% | XX% | XX% | XX% |
| Decision Tree | XX% | XX% | XX% | XX% | XX% |
| Random Forest | XX% | XX% | XX% | XX% | XX% |
| XGBoost | XX% | XX% | XX% | XX% | XX% |

**Best Model:** [Model Name] with [Metric] of XX%

### Feature Importance

Top 5 most important features:
1. Age
2. Average Glucose Level
3. BMI
4. Hypertension
5. Heart Disease

---

## Visualizations

### Sample Outputs

**1. Age Distribution by Stroke Status**
- Shows age is a significant factor

**2. Correlation Heatmap**
- Identifies relationships between features

**3. Feature Importance Plot**
- Highlights the most predictive features

**4. ROC Curves**
- Compares model performance

**5. Confusion Matrix**
- Shows classification results

*Note: Actual visualizations are available in the notebook.*

---

## Insights & Recommendations

### For Healthcare Professionals

1. **High-Risk Profile:**
   - Age > 65 years
   - Hypertension present
   - History of heart disease
   - Elevated glucose levels (>XXX mg/dL)
   - BMI outside healthy range

2. **Screening Recommendations:**
   - Regular monitoring for high-risk individuals
   - Early intervention programs
   - Lifestyle modification counseling

3. **Preventive Measures:**
   - Blood pressure control
   - Glucose management
   - Weight management
   - Smoking cessation programs

### Model Deployment Considerations

- **Priority:** Maximize recall (sensitivity) to minimize false negatives
- **Trade-off:** Accept slightly lower precision to catch more potential stroke cases
- **Integration:** Model can be integrated into electronic health records (EHR)
- **Continuous Learning:** Regular retraining with new patient data

---

## Future Improvements

- [ ] Collect more data to address class imbalance
- [ ] Include additional features (family history, medication, diet)
- [ ] Implement deep learning models (Neural Networks)
- [ ] Create interactive dashboard for predictions
- [ ] Deploy model as web API
- [ ] Conduct A/B testing in clinical settings
- [ ] Add time-series analysis for longitudinal patient data
- [ ] Integrate with real-time patient monitoring systems

---

## Project Structure

```
stroke-prediction-analysis/
│
├── data/
│   ├── raw/
│   │   └── healthcare-dataset-stroke-data.csv
│   └── processed/
│       └── cleaned_data.csv
│
├── notebooks/
│   └── stroke-analysis.ipynb
│
├── images/
│   ├── age_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── roc_curves.png
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution
- Additional visualizations
- New machine learning models
- Hyperparameter optimization
- Code optimization
- Documentation improvements
- Bug fixes

---


## Acknowledgments

- Dataset source: [(https://www.kaggle.com/code/dmitryuarov/stroke-eda-prediction-with-6-models#Preprocessing)]
- Inspiration from various stroke prediction research papers
- Thanks to the open-source community for amazing tools
- Healthcare professionals who provided domain expertise

---



## Star This Repository

If you found this project helpful, please consider giving it a star 

It helps others discover the project and motivates continued development!

---

## Project Status

**Active Development** - This project is actively maintained and updated.

**Last Updated:** January 30, 2026

**Version:** 1.0.0

---

## Data Privacy & Ethics

This project uses anonymized healthcare data for research and educational purposes only. Patient privacy is of utmost importance:

- All patient identifiers have been removed
- Data used in accordance with ethical guidelines
- Not intended for clinical use without proper validation
- Predictions should be used alongside professional medical judgment

---


