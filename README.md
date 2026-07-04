# 🛍️ Customer Segmentation using K-Means Clustering

## 📌 Project Overview

This project performs **Customer Segmentation** using the **K-Means Clustering** algorithm. The objective is to group customers with similar purchasing behavior, demographics, and online shopping habits into meaningful customer segments. These segments can help businesses develop targeted marketing strategies, improve customer retention, and maximize sales.

The project includes:

- Data Cleaning and Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Customer Segmentation using K-Means
- PCA Visualization
- Customer Segment Naming
- Model Saving using Joblib
- Streamlit Web Application for Customer Segment Prediction

---

## 📂 Dataset

The dataset contains customer demographic information, purchasing history, campaign responses, and website interaction data.

### Important Features

- Year of Birth
- Education
- Marital Status
- Income
- Number of Children
- Purchase Amounts
- Website Purchases
- Store Purchases
- Web Visits
- Recency
- Marketing Campaign Responses

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PCA
- K-Means Clustering
- Joblib
- Streamlit

---

## 📊 Feature Engineering

Several new features were created to improve clustering performance.

### Age

```python
Age = 2025 - Year_Birth
```

### Total Children

```python
Total_children = Kidhome + Teenhome
```

### Total Spending

Calculated as the sum of:

- Wine Purchases
- Fruit Purchases
- Meat Purchases
- Fish Purchases
- Sweet Purchases

### Customer Since

Calculated using the customer registration date.

---

## 📈 Exploratory Data Analysis

The project includes several visualizations such as:

- Age Distribution
- Income Distribution
- Spending Distribution
- Income by Education Level
- Spending by Marital Status
- Correlation Heatmap
- Income by Education and Marital Status
- Campaign Acceptance Rate
- Income by Age Group

---

## ⚙️ Data Preprocessing

The selected features were standardized using **StandardScaler** before applying K-Means.

Selected Features:

- Age
- Income
- Total Spending
- Number of Web Purchases
- Number of Store Purchases
- Number of Monthly Web Visits
- Recency

---

## 🤖 K-Means Clustering

The Elbow Method was used to determine an appropriate number of clusters.

The final model was trained using:

```python
KMeans(n_clusters=6)
```

Each customer was assigned to one of six customer segments.

---

## 📉 PCA Visualization

Principal Component Analysis (PCA) was used to reduce the feature space into two dimensions for visualization.

The clusters were visualized using a scatter plot.

---

## 🏷️ Customer Segments

The generated clusters were interpreted and assigned business-friendly names.

| Cluster | Customer Segment |
|----------|------------------|
| 0 | Premium Customers |
| 1 | Budget Customers |
| 2 | Regular Customers |
| 3 | Inactive Customers |
| 4 | Average Customers |
| 5 | VIP Customers |

---

## 💾 Model Saving

The trained K-Means model and StandardScaler were saved using Joblib.

```python
joblib.dump(kmeans,"kmeans_model.pkl")
joblib.dump(scaler,"scaler.pkl")
```

These files are later used in the Streamlit application.

---

## 🌐 Streamlit Application

A Streamlit web application was developed that allows users to:

- Enter customer information
- Predict customer segment
- Display the predicted customer category
- Provide business insights based on the predicted segment

Example Inputs:

- Age
- Income
- Total Spending
- Number of Web Purchases
- Number of Store Purchases
- Monthly Web Visits
- Recency

Example Output:

```
Predicted Segment

⭐ VIP Customer
```

---

## 📁 Project Structure

```
Customer-Segmentation/
│
├── customer_segmentation.csv
├── customer_segmentation.py
├── app.py
├── kmeans_model.pkl
├── scaler.pkl
├── README.md
├── requirements.txt
└── screenshots/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Customer-Segmentation.git
```

Move to the project directory

```bash
cd Customer-Segmentation
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 📷 Sample Outputs

The project generates:

- Distribution Plots
- Box Plots
- Correlation Heatmap
- Customer Segments
- PCA Scatter Plot
- Customer Segment Distribution
- Streamlit Dashboard

*(Add screenshots of your dashboard and plots in the `screenshots` folder and reference them here.)*
<img width="920" height="793" alt="Screenshot 2026-07-04 220035" src="https://github.com/user-attachments/assets/b483e840-6bb6-4db3-87e3-c83392291eff" />

<img width="938" height="545" alt="image" src="https://github.com/user-attachments/assets/e226a580-cc3d-4574-b205-a2d5056e8df2" />


---

## 💡 Business Insights

The customer segments can be used to:

- Identify high-value customers
- Develop personalized marketing campaigns
- Improve customer retention
- Recommend loyalty programs
- Target inactive customers with promotional offers
- Optimize marketing budgets

---

## 🔮 Future Improvements

- Determine the optimal number of clusters using the Silhouette Score.
- Add interactive filters and analytics to the Streamlit dashboard.
- Deploy the application on Streamlit Community Cloud.
- Integrate additional clustering algorithms such as DBSCAN and Hierarchical Clustering for comparison.
- Add customer recommendations based on segment characteristics.

---

## 👨‍💻 Author

**Ujjwal Mittal**

If you found this project useful, consider giving the repository a ⭐ on GitHub.
