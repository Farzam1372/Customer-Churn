# Customer-Churn-Prediction (Coursera Challenge)
# 🎯 Customer Churn Prediction - Data Science Coding Challenge

Welcome to the **Customer Churn Prediction Challenge**, where we tackle one of the most industry-relevant machine learning problems—**predicting customer churn** for a **video streaming platform**.  

## 🚀 Challenge Overview
Subscription-based services are a **multi-billion dollar industry**, spanning fitness, streaming, and retail sectors. One of the key business objectives for these companies is **reducing churn**—ensuring that users remain **active subscribers**.  

### **🔍 The Task**
As a **data scientist** at a video streaming company, your goal is to **predict whether a user will continue their subscription next month** using **machine learning**.  

We use **customer activity, preferences, and subscription history** to train a model that can **detect potential churners**, allowing the company to **take action** before a user cancels.

---

## 📂 Dataset Overview

The challenge provides **two datasets**:

| Dataset  | Description |
|----------|------------|
| **train.csv** | 243,787 subscriptions (70% of total) with the target column **Churn** (1 = churned, 0 = stayed) |
| **test.csv** | 104,480 subscriptions (30% of total) **without Churn labels** (your goal is to predict these) |

Each row represents **a unique customer subscription** at a snapshot in time.

### **📝 Features in the Data**
The dataset includes:
- `CustomerID` – Unique identifier for each customer
- `AccountAge` – How long the user has had an account
- `ViewingHoursPerWeek` – Average time spent watching per week
- `ContentDownloadsPerMonth` – Number of downloads per month
- `MonthlyCharges` – Subscription fee
- `TotalCharges` – Total amount spent
- `SupportTicketsPerMonth` – Customer support interactions
- `WatchlistSize` – Number of saved content items
- `UserRating` – Customer satisfaction rating (1-5)
- **Target (`Churn`)** – **(1 = Churned, 0 = Stayed) [Only in `train.csv`]**

---

## 🏆 Model Approach

### **1️⃣ Data Preprocessing**
✔ **Handled missing values**  
✔ **Converted categorical features using One-Hot Encoding**  
✔ **Normalized numerical data (if needed)**  
✔ **Checked for class imbalance & applied techniques like SMOTE if required**

### **2️⃣ Model Selection**
After testing various models, we chose **Random Forest Classifier** because:
✔ It handles categorical & numerical data efficiently  
✔ It is robust against overfitting  
✔ It provides **feature importance insights**  

### **3️⃣ Hyperparameter Tuning**
We used **`GridSearchCV`** to optimize parameters like:
- `n_estimators` → Number of trees
- `max_depth` → Maximum depth of trees
- `min_samples_split` → Minimum samples required to split a node
- `min_samples_leaf` → Minimum samples per leaf

### **4️⃣ Performance Evaluation**
✔ **Trained the model on `train.csv`**  
✔ **Tested on `X_test` and calculated AUC Score**  
✔ **AUC Score Before Tuning: `0.7313`**  
✔ **AUC Score After Tuning: `0.XXXX`** (Update based on results)

---

## 🛠 How to Run the Model

### **🔹 Install Dependencies**
```bash
pip install -r requirements.txt

