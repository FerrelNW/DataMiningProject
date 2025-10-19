# **Segmentica: Customer Segmentation & Response Prediction**

https://github.com/FerrelNW/DataMiningProject.git
*<p align="center">Segmentica Analytics Dashboard</p>*

---

### 📖 **Overview**

**Segmentica** is a **data mining project** focused on customer segmentation and campaign response prediction.  
Built with the **Python data ecosystem**, it leverages libraries like **Pandas** and **Scikit-learn** to develop a predictive model that estimates the likelihood of a customer responding to a marketing campaign.  
All insights and predictions are displayed through an **interactive, responsive dashboard** designed for data-driven decision-making.

This full-stack application consists of two main components:
1. **Predictor Tool:** A real-time form where users can input customer details and instantly get a response probability score.
2. **Analytics Dashboard:** A visual dashboard with key performance metrics (KPIs) and dynamic data visualizations for exploring customer behavior patterns.

---

### ✨ **Key Features**

#### 📊 **Interactive Analytics Dashboard**
- **4 Main KPIs:** Total Customers, Average Income, Average Spending, and Campaign Response Rate.  
- **6 Data Visualizations:** Breakdown by Age, Family Structure, Purchase Channel, Product Spending, Marital Status, and an Income vs. Spending scatter plot.

#### 🔮 **Real-time Prediction**
- Enter customer data (e.g., Age, Income, Spending, etc.) through a simple input form.  
- Instantly receive a probability score (e.g., “16.4% Likelihood to Respond”).  
- Get color-coded feedback such as “High Income (Positive)” or “Inactive (Risk)” for quick interpretation.

#### 🤖 **Machine Learning Backend**
- Powered by a `RandomForestClassifier` built with **Scikit-learn** for accurate prediction results.  
- A **Flask API** handles data processing, prediction requests, and serves metrics for the dashboard using **Pandas**.

#### 🎨 **Responsive Frontend**
- Clean and modern UI built with **Tailwind CSS** for seamless experience across desktop and mobile devices.

---

### 🛠️ **Tech Stack**

**Backend & Data Mining**
- **Python**
- **Flask** – REST API for serving predictions and analytics data.  
- **Pandas** – For data transformation, filtering, and KPI computation.  
- **Scikit-learn** – For building and training the `RandomForestClassifier` model.

**Frontend & Visualization**
- **HTML5**
- **Tailwind CSS** – For clean and adaptive styling.  
- **JavaScript (ES6+)** – Handles form validation, API requests, and dynamic updates.  
- **Chart.js** – Renders six interactive charts in the analytics dashboard.

---

### 🚀 **How to Run Locally**

1. **Clone the repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2. **Create a Python virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    *(Make sure `requirements.txt` exists in your project folder)*  
    ```bash
    pip install -r requirements.txt
    ```

4. **Train the model (if not already provided)**
    ```bash
    python predictive_model.py
    ```
    This will generate a `.pkl` model file inside the `models/` directory.

5. **Run the Flask application**
    ```bash
    flask run
    # or
    python app.py
    ```

6. **Open the application**
    - Visit `http://127.0.0.1:5000` for the prediction page.  
    - Visit `http://127.0.0.1:5000/dashboard` to explore the analytics dashboard.

---

### 🧠 **About the Project**

Segmentica is designed as a practical demonstration of how data-driven marketing can be enhanced using **machine learning** and **visual analytics**.  
By combining predictive modeling with real-time insights, businesses can better identify target audiences, allocate marketing budgets efficiently, and maximize response rates.

---
