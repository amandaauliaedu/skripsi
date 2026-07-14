# 📈 FORRISX
### Forecasting and Risk Analysis System using ARIMAX and Value-at-Risk (VaR)

<p align="center">
  <img width="512" height="512" alt="image" src="https://github.com/user-attachments/assets/b2c79ce5-1d4b-42c0-9719-bd769bbf2806" />
</p>

<p align="center">
Web-based Stock Forecasting & Investment Risk Analysis using <b>ARIMAX</b> and <b>Historical Simulation Value-at-Risk (VaR)</b>.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit)

</p>

---

# 📖 Overview

FORRISX (**Forecasting and Risk Analysis System using ARIMAX and VaR**) merupakan aplikasi berbasis **Streamlit** yang dikembangkan sebagai implementasi penelitian skripsi untuk membantu melakukan **prediksi harga saham** sekaligus **analisis risiko investasi** dalam satu platform.

Aplikasi menggabungkan metode:

- 📈 **ARIMAX (Autoregressive Integrated Moving Average with Exogenous Variables)** untuk forecasting harga saham.
- 💰 **Historical Simulation Value-at-Risk (VaR)** untuk mengukur potensi kerugian investasi.

Studi kasus pada penelitian ini menggunakan saham **PT Bank Central Asia Tbk (BBCA)** dengan variabel eksternal:

- USD/IDR Exchange Rate
- SGD/IDR Exchange Rate

---

# ✨ Features

## 📂 Upload Dataset

- Upload file CSV / Excel
- Dataset validation
- Data preview
- Missing value detection
- Duplicate checking
- Data information

---

## 📊 Exploratory Data Analysis

- Descriptive Statistics
- Data Type Information
- Key Metrics
- Dataset Summary

---

## 📈 Interactive Visualization

- Time Series Plot
- Correlation Heatmap
- Outlier Detection
- Interactive Charts

---

## 🤖 ARIMAX Modeling

- Train-Test Split
- Auto ARIMA
- Manual Parameter Input
- Model Summary
- AIC & BIC
- Statistical Diagnostics

---

## 📉 Forecasting

- Actual vs Prediction
- Forecast Plot
- Prediction Table
- MAPE Evaluation

---

## 💰 Risk Analysis

Historical Simulation Value-at-Risk

Features:

- Log Return
- Historical Distribution
- Confidence Level
- VaR Result
- Risk Visualization

---

# 🧠 Methodology

```text
Raw Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Visualization
      │
      ▼
ARIMAX Modeling
      │
      ▼
Forecasting
      │
      ▼
Model Evaluation (MAPE)
      │
      ▼
Historical Simulation
      │
      ▼
Value-at-Risk
```

---

# 📂 Project Structure

```bash
FORRISX
│
├── app.py
├── assets/
│
├── pages/
│   ├── Home.py
│   ├── Upload_Data.py
│   ├── Visualization.py
│   ├── ARIMAX_Model.py
│   ├── Forecasting.py
│   └── Value_at_Risk.py
│
├── utils/
├── data/
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Statistical Modeling | Statsmodels, pmdarima |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-Learn |
| Risk Analysis | Historical Simulation VaR |

---

# 📊 Dataset

Source:

- Yahoo Finance

Objects:

- BBCA.JK
- USD/IDR
- SGD/IDR

Observation Period

**January 2019 – September 2024**

---

# 📈 Research Result

| Evaluation | Result |
|------------|--------|
| Forecasting Model | ARIMAX |
| Forecast Accuracy (MAPE) | **2.19%** |
| Risk Method | Historical Simulation |
| Confidence Level | 95% |

The obtained MAPE indicates that the ARIMAX model provides highly accurate stock price forecasting while incorporating macroeconomic variables.

---

# 🚀 Installation

Clone repository

```bash
git clone https://github.com/amandaauliaedu/skripsi.git
```

Go to project directory

```bash
cd skripsi
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
streamlit run app.py
```

---

# 🖥️ Application Workflow

1. Upload Dataset
2. Explore Dataset (EDA)
3. Visualize Data
4. Build ARIMAX Model
5. Forecast Stock Price
6. Evaluate Model
7. Calculate Value-at-Risk
8. Analyze Investment Risk

---

# 🎓 Research

**Title**

Financial Sector Stock Price Forecasting and Loss Risk Using the ARIMAX and Value-at-Risk (VaR) Methods

This project was developed as an undergraduate thesis in the Data Science Department, Universitas Pembangunan Nasional "Veteran" Jawa Timur.

---

# 📄 Publication

This research has also been published in:

**Bit-Tech Journal (2025)** 
https://jurnal.kdi.or.id/index.php/bt/article/view/3219

Financial Sector Stock Price Forecasting and Loss Risk Using the ARIMAX and Value-at-Risk Methods.

---

# 📜 License

This repository is intended for academic, educational, and research purposes.
