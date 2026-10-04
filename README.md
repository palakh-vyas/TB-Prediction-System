# 🫁 Tuberculosis (TB) Risk Screening System

A machine learning web app that estimates a person's TB risk from their **symptoms, age and gender**, and returns a risk percentage with a low / medium / high band.

> ⚠️ **Disclaimer:** This is an academic screening aid, not a medical diagnosis. Always consult a doctor.

---

## 📌 Problem Statement

Tuberculosis is a serious infectious disease, and early screening helps patients reach treatment sooner. This project builds a simple symptom-based screening tool and, more importantly, evaluates it properly for a **highly imbalanced** problem, where only about 3% of patients have TB.

## ✨ Features

- Tick-box symptom selection, plus age and gender inputs (no typing errors)
- Risk **percentage** with a colour-coded band (low / medium / high)
- Three models compared: Logistic Regression, Decision Tree, Random Forest
- Imbalance handling with class weights and a stratified train/test split
- Evaluation with precision, recall, F1, ROC-AUC and a confusion matrix, not accuracy alone
- Model trained once and saved, so the app loads instantly
- Model Performance and About tabs inside the app

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · joblib · Streamlit

## 📁 Project Structure

```
TB-Prediction-System/
├── Dataset/
│   └── Healthcare.csv          # patient records (25,000 rows)
├── models/                     # saved trained model + helpers
│   ├── tb_model.pkl
│   ├── symptom_list.pkl
│   ├── scaler.pkl
│   ├── gender_columns.pkl
│   └── model_comparison.csv
├── app.py                      # Streamlit web app
├── tb_model.py                 # training + evaluation script
├── TB-Prediction-System.ipynb  # notebook version (Google Colab)
├── requirements.txt
└── README.md
```

## 🔄 How It Works

```
Healthcare.csv → clean + encode symptoms, age, gender
              → stratified 80/20 split
              → train 3 models (balanced class weights)
              → evaluate (precision, recall, F1, ROC-AUC)
              → save best model by recall
              → Streamlit app loads model → risk %
```

**Why recall?** In screening, missing a real TB patient is worse than a false alarm, so the model is chosen by how many TB cases it catches.

## 📊 Results

Test set: 5,000 patients (20% split), TB = 3.3% of the data.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.536 | 0.035 | 0.491 | 0.064 | 0.495 |
| Decision Tree | 0.554 | 0.036 | 0.485 | 0.066 | 0.521 |
| Random Forest | 0.907 | 0.019 | 0.037 | 0.025 | 0.524 |

Best model by recall: **Logistic Regression**.

### Why accuracy was not used
An earlier version of this project reported about 96.7% accuracy, but it simply predicted "not TB" for everyone and caught **0** TB cases. Accuracy hides this on imbalanced data, which is why recall, precision and ROC-AUC are reported here.

### ⚠️ Dataset limitation
The dataset appears to be **synthetic**: symptom patterns are nearly identical across all 30 diseases, and a 30-class model scores at chance level. ROC-AUC near 0.5 therefore reflects the data, not the code. This project demonstrates a complete, correctly evaluated ML pipeline; results on real clinical data would differ and should be validated by medical professionals.

## 🚀 How to Run

**Option 1: Locally**
```bash
git clone https://github.com/<your-username>/TB-Prediction-System.git
cd TB-Prediction-System
pip install -r requirements.txt
python tb_model.py          # trains and saves models/ (optional, already included)
streamlit run app.py
```

**Option 2: Google Colab**
Open `TB-Prediction-System.ipynb`, run the cells in order, then start the app with a cloudflared tunnel.

## 📸 Screenshots

| Prediction | Model Performance |
|---|---|
| _add screenshot_ | _add screenshot_ |

## 🔮 Future Scope

- Train on a real, clinically validated TB dataset
- Add chest X-ray analysis with a CNN
- Add SHAP explanations for each prediction
- Store prediction history in a database
- Deploy permanently on Streamlit Community Cloud

## 👤 Author

Your Name · Your College · Final Year Project, 2026
