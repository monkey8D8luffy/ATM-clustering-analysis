# ATM- 🏧 ATM Intelligence: Demand Forecasting & Behavioural Analytics
**FinTrust Bank Ltd. — Data Mining Project** **Formative Assessment 2 (FA-2): Building Actionable Insights and an Interactive Python Script**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E)
![IBCP](https://img.shields.io/badge/IBCP-Artificial%20Intelligence-8A2BE2)

Welcome to the **ATM OS (Glass Edition)**. This repository contains an interactive, end-to-end data mining and machine learning application designed to optimize cash replenishment strategies for a network of ATMs. 

Developed for the **IBCP Artificial Intelligence** course, this project transitions raw transaction logs into proactive, predictive intelligence using advanced clustering and anomaly detection algorithms.

---

## 🎯 Project Overview
Currently, ATM networks operate on reactive schedules, leading to costly stockouts or millions in idle cash. This application processes historical withdrawal data, identifies distinct behavioral patterns, and provides actionable insights for treasury and logistics teams.

This script directly fulfills the **FA-2 Rubric Requirements**:
1. **Exploratory Data Analysis (EDA):** Uncovering temporal rhythms and environmental friction.
2. **Clustering & Anomaly Detection:** Grouping ATM behaviors and flagging severe holiday/weather spikes.
3. **Interactive Python Script Development:** Delivering a smooth, reproducible, and highly visual end-user experience.

---

## ✨ Key Features (The Glass OS Dashboard)
* **💎 Custom Glassmorphism UI:** A sleek, fully custom CSS interface featuring frosted glass panels, liquid morphing animations, and an animated ATM idle loop.
* **📊 3D Exploratory Data Analysis:** Interactive Plotly visualizations, including 3D relationship explorers and correlation heatmaps to uncover volume and velocity trends.
* **🧩 K-Means Clustering:** Dynamically groups ATMs into actionable profiles (e.g., "High Demand", "Stable Daily") with an adjustable *K* slider and Silhouette/Elbow method validation.
* **🚨 Isolation Forest Anomaly Detection:** Automatically flags volatile withdrawal spikes caused by holidays or severe weather, allowing for preemptive cash loading.
* **📈 Time-Series Forecasting:** Utilizes Exponential Smoothing to project network-wide cash demand up to 30 days into the future.
* **💾 Export Engine:** Generates downloadable intelligence reports enriched with ML cluster IDs and anomaly scores for downstream logistics routing.

---

## 🛠️ Technology Stack
* **Frontend:** [Streamlit](https://streamlit.io/) (with aggressive custom CSS injection)
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (StandardScaler, LabelEncoder, KMeans, IsolationForest)
* **Visualization:** `plotly.express`, `plotly.graph_objects`

---

## 🚀 Installation & Usage

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You will need the following libraries:
```bash
pip install streamlit pandas numpy scikit-learn plotly
-analysis
