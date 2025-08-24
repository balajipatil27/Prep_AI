## 🚀 PrepAI – Your Smart Data Cleaning & Preprocessing Assistant

**PrepAI** is an AI-powered toolkit that acts as a:

* 🧹 **Data Preprocessing Assistant** – Automates cleaning, transformation, and preparation of datasets.
* 📊 **ML Visualizer** – Provides interactive model and dataset visualizations for better interpretability.
* 📈 **DataVista Dashboard** – A dynamic analytics dashboard to monitor and generate exportable insights.

PrepAI is built to **accelerate the ML pipeline**, reduce repetitive work, and provide **ready-to-use insights** for Data Scientists, Analysts, and ML Engineers.

---

## ✨ Key Features

### 🧹 1. Data Preprocessing Assistant
It is an intelligent web-based tool designed to automate and simplify the preprocessing of datasets for machine learning projects. Built using Python (Flask), Bootstrap, and pandas, it helps users upload datasets (CSV/Excel), automatically detect data issues, and perform cleaning steps like handling missing values, encoding, scaling, and more — all with just a few clicks.


* 📁 Upload CSV/Excel datasets
* 🔍 Intelligent missing value handling (mean/median/mode/imputation)
* 📏 Feature scaling (StandardScaler, MinMaxScaler)
* 🎯 Label encoding and one-hot encoding
* 🧼 Outlier detection and handling (Z-score/IQR based)
* 📊 Data summary with stats & correlation matrix
* 📥 Download preprocessed dataset
* 💡 Clean UI built with Bootstrap
* 🔄 Realtime feedback on preprocessing steps


### 📊 2. ML Visualizer
A comprehensive machine learning visualizer and educational platform. As a learner, I found myself constantly scrolling through different sites for ML tools. ML Visualizer consolidates everything in one place, saving time and making it easier for you to visualize, learn, and experiment with machine learning algorithms.


* **Exploratory Data Analysis (EDA)**

* **Model Evaluation**

  * Confusion Matrix (heatmap).
  * ROC & Precision-Recall curves.
  * Feature Importance plots (Tree-based, SHAP).
  * Learning curves for training/validation performance.

### 📈 3. DataVista Dashboard
* Interactive dashboard with KPIs.
* Export results as **PDF, CSV, PNG reports**.

---

## 🛠️ Tech Stack

**Core Languages & Frameworks**

* Python 3.10+
* Flask 

**Data & ML Libraries**

* NumPy, Pandas – Data handling
* Scikit-learn – Preprocessing, ML models, metrics
* Matplotlib, Seaborn, Plotly – Visualization
* SHAP – Model explainability

**Other Tools**

* Frontend: HTML5, CSS3, Bootstrap 5, JavaScript
* Render – Deployment

---

## 📥 Input Format

PrepAI accepts datasets in multiple formats:

* **CSV** (`.csv`)
* **Excel** (`.xlsx`)

👉 User must specify **target column** for supervised ML tasks.

---

## ⚙️ Installation & Setup

### 🔹 1. Clone the Repository

git clone https://github.com/yourusername/prepai.git
cd prepai

### 🔹 2. Create Virtual Environment

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

### 🔹 3. Install Dependencies

pip install -r requirements.txt

### 🔹 4. Run Applications

python app.py

---

## 🚀 Example Workflow

### 1️⃣ Upload Dataset

* Upload `.csv` or `.xlsx` file.

### 2️⃣ Preprocess

* Choose options (missing value strategy, encoding, scaling).
* Get summary of steps
* Download **cleaned dataset**.

### 3️⃣ Visualize ML Models
* Upload `.csv` or `.xlsx` file.
* Train model based on your target variable → Generate plots: ROC, Confusion Matrix, Feature importance and SHAP plots.

### 4️⃣ Dashboard Insights

* Upload `.csv` or `.xlsx` file.
* View **DataVista** dashboard.
* Export results for sharing.

---

## 📊 Demo Screenshots 

* Data Preprocessing Assistant UI
<img width="1861" height="925" alt="Screenshot 2025-08-24 224559" src="https://github.com/user-attachments/assets/66dafbd0-f6b6-48d2-b3ef-b22f5b97a9ff" />

* EDA Visualizations
<img width="1860" height="924" alt="Screenshot 2025-08-24 224614" src="https://github.com/user-attachments/assets/edab3880-310a-41e1-b729-f918f2f82c36" />
<img width="1834" height="891" alt="Screenshot 2025-08-24 224721" src="https://github.com/user-attachments/assets/79228099-5637-4db0-ad4b-b66dfb5f1ab2" />
<img width="1853" height="911" alt="Screenshot 2025-08-24 224732" src="https://github.com/user-attachments/assets/2e8f049b-f913-4ad4-8e31-9b2e89af256c" />
<img width="1859" height="926" alt="Screenshot 2025-08-24 224744" src="https://github.com/user-attachments/assets/fcaf14d9-629e-4895-80c2-035d0ee030ca" />

* Model Performance Plots
<img width="1860" height="929" alt="Screenshot 2025-08-24 224800" src="https://github.com/user-attachments/assets/902938c8-375c-44eb-a83f-a64f4b568758" />
<img width="1866" height="926" alt="Screenshot 2025-08-24 230337" src="https://github.com/user-attachments/assets/27cf28bc-53b8-4437-a66a-2ba9e39c5e41" />
<img width="1917" height="490" alt="Screenshot 2025-08-24 230408" src="https://github.com/user-attachments/assets/23107c11-5506-490b-86f4-97a0cb6b2470" />
<img width="1915" height="1072" alt="Screenshot 2025-08-24 230418" src="https://github.com/user-attachments/assets/73210f80-fee7-49a7-ab1e-64e82d9fc4df" />
<img width="1917" height="1079" alt="Screenshot 2025-08-24 230423" src="https://github.com/user-attachments/assets/bd5ab4e7-9a18-4709-8cfd-ca09cd0c998e" />
<img width="1919" height="1079" alt="Screenshot 2025-08-24 230428" src="https://github.com/user-attachments/assets/0c9ffda6-85f5-44d8-a728-43229de561d5" />

* DataVista Dashboard
<img width="1862" height="920" alt="Screenshot 2025-08-24 230445" src="https://github.com/user-attachments/assets/a87c11aa-dc03-4d89-8494-90102d3f92ca" />
<img width="2592" height="1680" alt="DataVista_Dashboard" src="https://github.com/user-attachments/assets/8f80398f-a5bf-4f37-bd92-4cd2f570d459" />

---

## 🤝 Contributing

Contributions are welcome! 🎉

1. Fork the repo.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---


