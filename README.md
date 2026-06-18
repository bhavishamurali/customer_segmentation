# 👥 Customer Segmentation & Cluster Analysis Dashboard

An interactive Machine Learning web application built using **Python**, **scikit-learn**, and **Streamlit** to segment customers based on behavioral attributes and demographic profiles.

## 🎯 Project Overview
This project applies unsupervised machine learning (**K-Means Clustering**) to group customers into distinct segments using their **Annual Income** and **Spending Score**. This allows businesses to identify high-value consumer groups, optimize targeting strategies, and gain data-driven demographic insights.

## 🚀 Key Features
* **Machine Learning Pipeline:** Automatically scales features using `StandardScaler` and applies the `K-Means` algorithm.
* **Dynamic Hyperparameter Control:** Sidebar slider allows evaluators to change the number of clusters ($K$) from 2 to 5 in real time.
* **Interactive 2D Cluster Visualization:** Responsive scatter plots built with `Plotly Express` where bubble sizes dynamically represent customer age.
* **Statistical Profiling:** Generates real-time matrices displaying the average Age, Income, and Spending Scores across all discovered segments.

## 🛠️ Tech Stack
* **Language:** Python
* **ML Libraries:** scikit-learn (KMeans, StandardScaler)
* **Data Processing:** Pandas, Openpyxl
* **Dashboard Framework:** Streamlit
* **Visualizations:** Plotly Express, Matplotlib

## 📦 How To Run This Project Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/bhavishamurali/customer_segmentation.git](https://github.com/bhavishamurali/customer_segmentation.git)
   cd customer_segmentation
