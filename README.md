
# THE WOLVES – CNC Machining Intelligence

### Machine Learning-Based CNC Machining Quality Prediction and Process Analysis

An academic M.Tech project focused on analysing CNC machining parameters and predicting surface roughness using machine learning.

## Project Overview

This project uses experimental CNC end-milling data to study the relationship between machining parameters and surface roughness.

## Objectives

1. Predict surface roughness (Ra) using Linear Regression.
2. Group machining conditions using K-Means Clustering.
3. Classify relative surface quality using a Decision Tree.
4. Develop an interactive Streamlit dashboard.

## Dataset

- Material: AISI 1045
- Experimental runs: 41
- Input parameters: 5
- Target variable: Surface roughness (Ra)
- Additional response: Specific cutting energy (Ec)

## Machine Learning Algorithms

| Algorithm | Application |
|---|---|
| Linear Regression | Surface roughness prediction |
| K-Means | Machining-condition clustering |
| Decision Tree | Relative quality classification |

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit

## Project Structure

- `app.py` – Interactive dashboard
- `data/` – Experimental dataset
- `src/` – Machine learning modules
- `outputs/` – Model results and visualizations
- `models/` – Saved models

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

## Developer

**Aashu Prakash**

Website: https://aashuprakash.vercel.app/

## Academic Guide

Prof. MJ Davidson

---

Academic M.Tech CIM Project.
