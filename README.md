# Salary-Predictor
ML-Powered Salary Predictor using Linear Regression


# ML-Powered Salary Predictor

A Machine Learning web application that predicts salaries for tech professionals based on multiple factors. Built with Linear Regression and deployed as a Flask web app.

🔗 **Live Demo:** [https://nadia53.pythonanywhere.com](https://nadia53.pythonanywhere.com)

---

## ✨ Objective of the Project

To build a smart and interactive web application that predicts employee salaries based on key factors such as experience, job role, location, and company size. The app aims to assist HR teams, employers, and job seekers in making data-driven compensation decisions using ML models.

---

## ⚡ What I Built

An end-to-end Flask-based ML application featuring:

🔍 Real-time salary prediction using Linear Regression model  
🧼 Complete data preprocessing pipeline (cleaning, encoding with LabelEncoder)  
📈 Model evaluation: R² Score (0.75). 
📊 Feature analysis based on 8 key factors  
🧪 Statistical testing with train/test split (80/20) and K-Fold Cross Validation  
🎨 Glassmorphism UI design with animated background  
📱 Responsive design for mobile, tablet, and desktop  
☁️ Live deployment on PythonAnywhere

---

## 🤖 Machine Learning Models Used

Linear Regression (Primary Model)  
Random Forest Regressor (Future Enhancement)  
XGBoost Regressor (Future Enhancement)  
GridSearchCV (Planned for tuning)  
K-Fold Cross Validation (Used for robustness)

---

## 📁 Dataset Info

**Dataset Name:** ds_salaries.csv  
**Source:** Kaggle  
**Size:** 607 rows  
**Years Covered:** 2020-2022 (extrapolated to 2027)  
**Key Features:** Work Year, Experience Level, Employment Type, Job Title, Company Location, Employee Residence, Remote Ratio, Company Size  
**Target:** Salary in USD  
**Preprocessing:** Label Encoding, outlier removal, train/test split

---

## 🗂️ Tools & Technologies Used

**Frontend:** HTML5, CSS3, JavaScript, Glassmorphism UI  
**Backend/ML:** Python, Flask, scikit-learn, Pandas, NumPy, Joblib  
**Visualization:** Matplotlib, Seaborn (in notebooks)  
**Model Serialization:** Pickle, Joblib  
**Deployment:** PythonAnywhere, Gunicorn  
**Version Control:** Git, GitHub

---

## 🎯 Achievements

✅ Deployed a complete ML web app from scratch  
📋 Built interactive UI with glassmorphism design  
💡 Designed a clean and user-friendly interface  
🧠 Trained and validated Linear Regression model with proper evaluation  
☁️ Deployed the project to PythonAnywhere  
📊 Achieved R² Score: 0.75 (75%) with 8 features  
🎨 Implemented animated background with Canvas API  
📱 Made fully responsive for all devices  


---

## 🌍 Live Demo

🔗 [https://nadia53.pythonanywhere.com](https://nadia53.pythonanywhere.com)

---

## 👨‍💼 My Role

Designed the overall system and workflows  
Handled all data preprocessing and model training  
Developed frontend and backend logic  
Built interactive UI with CSS animations and glassmorphism  
Managed deployment on PythonAnywhere  
Performed model evaluation and feature analysis  
Debugged and fixed production issues

---

## 📚 What I Learned

Building and deploying ML apps end-to-end  
Real-world data preprocessing and feature engineering  
Handling categorical variables with LabelEncoder  
Flask web development and routing  
Creating responsive UI with CSS and glassmorphism  
Deployment on PythonAnywhere platform  
Debugging production issues  
Importance of user experience in AI tools  
Model evaluation metrics (R², MAE, RMSE)  
Train/test split and cross-validation techniques

---

## 📊 Model Performance Details

```python
# Model Coefficients
Intercept: -15,095,881.34
Coefficients:
- work_year: 7,481.17
- experience_level: 12,531.90
- employment_type: -14,824.30
- job_title: 1,252.23
- company_location: 489.71
- employee_residence: 1,148.63
- remote_ratio: 109.00
- company_size: -11,628.47


----

## 📸 Screenshots

<p align="center">
  <img src="Screenshot%202026-03-06%20120225.png" width="400">
  <img src="Screenshot%202026-03-06%20120255.png" width="400">
</p>

<p align="center">
  <img src="Screenshot%202026-03-06%20120320.png" width="400">
  <img src="Screenshot%202026-03-06%20120344.png" width="400">
</p>

<p align="center">
  <img src="Screenshot%202026-03-06%20120447.png" width="600">
</p>
