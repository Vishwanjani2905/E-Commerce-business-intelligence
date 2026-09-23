# E-Commerce Business Intelligence & Customer Analytics Dashboard

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Olist%20Brazilian%20E--Commerce-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Model-Logistic%20Regression-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview
This project is an end-to-end **E-Commerce Business Intelligence & Customer Analytics** application developed on the **Olist Brazilian E-Commerce Public Dataset**. It transforms over 100,000 real-world commercial transactions into actionable business intelligence through a complete pipeline:

$$\text{Raw Data} \longrightarrow \text{Data Cleaning} \longrightarrow \text{KPI Formulations} \longrightarrow \text{Exploratory Analysis} \longrightarrow \text{RFM Segmentation} \longrightarrow \text{Predictive Modeling} \longrightarrow \text{Streamlit Dashboard}$$

The solution is unified inside a single Python script (`ecommerce_business_dashboard.py`) designed for dual-execution:
1. **Interactive Notebook Workflow (CLI / VS Code):** Step-by-step exploratory analysis, table outputs, and data transformations.
2. **Executive Web Dashboard (Streamlit):** Interactive multi-tab decision support system with live KPI cards, dynamic filters, and risk simulators.

---

## 🔗 Dataset Link & Information

The project uses the authentic **Brazilian E-Commerce Public Dataset by Olist**, publicly available on Kaggle:

* **Direct Kaggle Dataset URL:** [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
* **Dataset Scope:** 100,000+ orders transacted between October 2016 and October 2018 across Brazilian states.
* **Schema Inventory:** 9 relational tables providing order lifecycles, item details, customer demographics, payments, customer reviews, seller origins, and product category translations.

---

## 📖 Project Description

### 1. Business Problem Statement
Modern e-commerce marketplaces manage thousands of daily transactions across diverse sellers, geographic territories, and logistics providers. Management faces three primary operational and commercial challenges:
1. **High Customer Churn & Low Retention:** Over 96% of shoppers purchase only once, placing extreme pressure on customer acquisition costs (CAC).
2. **Logistics & Delivery Bottlenecks:** Wide regional delivery disparities exist between core metropolitan areas (e.g., São Paulo with 8.8-day delivery) and remote northern states (25–30 day delivery).
3. **Severe Review Penalties for Shipping Delays:** Delivery delays beyond the promised estimated date cause customer satisfaction scores to collapse, damaging brand reputation.

### 2. Core Objectives
- Clean, translate, and integrate 9 relational tables at the order grain to eliminate item-level financial duplication.
- Formulate and compute core business KPIs (Total Revenue, AOV, Repeat Rate, Delivery Time, Review Score).
- Implement Recency, Frequency, and Monetary (RFM) customer segmentation into actionable cohorts (*High Value*, *Loyal*, *Potential*, *At Risk*).
- Empirically quantify the impact of logistics delays on customer reviews.
- Train an explainable machine learning model to predict poor customer ratings ($\le 2$ stars) with live risk simulation.
- Deploy a responsive, professional Streamlit executive dashboard.

---

## 🛠️ Technologies Used

| Domain | Technology / Library | Purpose in Project |
| :--- | :--- | :--- |
| **Language** | **Python (3.12 / 3.13)** | Core programming language for end-to-end data pipeline. |
| **Data Processing** | **Pandas (v2.2+)** | Data ingestion, date parsing, relational joins, grouping, and aggregations. |
| **Numerical Computing** | **NumPy (v1.26+ / v2.2+)** | Vectorized computations, array operations, and feature transformations. |
| **Machine Learning** | **Scikit-Learn (v1.7+)** | Balanced Logistic Regression, train/test split, ROC-AUC, and odds ratio evaluation. |
| **Data Visualization** | **Matplotlib & Seaborn** | Exploratory trend plots, distribution histograms, and category comparison bar charts. |
| **BI Web Dashboard** | **Streamlit (v1.45+ / v1.64+)** | Interactive dashboard UI, sidebar filters, KPI cards, and live risk calculator. |
| **Reporting & Export** | **Python-Docx** | Automated Microsoft Word (`.docx`) project report generation. |

---

## 📊 Key Project Information & Results

### 1. Verified Core Business KPIs
All metrics are computed from delivered orders in the dataset without hardcoded values:

| Business Metric | Value | Formulation / Interpretation |
| :--- | :--- | :--- |
| **Total Revenue (GMV + Freight)** | **R$ 15,419,773.75** | Total transacted gross merchandise volume. |
| **Product Merchandise Revenue** | **R$ 13,221,498.11** | Net item value excluding shipping charges. |
| **Total Delivered Orders** | **96,478** | Successfully fulfilled customer purchases. |
| **Total Unique Customers** | **93,358** | Distinct buyers identified via `customer_unique_id`. |
| **Average Order Value (AOV)** | **R$ 159.83** | Order-level revenue per transaction. |
| **Repeat Customer Rate** | **3.00%** | Only 2,801 customers purchased more than once (5.60% revenue share). |
| **Average Review Rating** | **4.16 / 5.00 ★** | Overall customer satisfaction across all delivered orders. |
| **Average Delivery Lead Time** | **12.56 Days** | Mean calendar duration from purchase to customer delivery. |

### 2. Analytical Findings & Drivers
* **Top Revenue Product Line:** `Health Beauty` (R$ 1.41M), followed by `Watches Gifts` (R$ 1.26M) and `Bed Bath Table` (R$ 1.22M).
* **Geographic Concentration:** `São Paulo (SP)` generates **37.4%** of total revenue (R$ 5.77M) with an average delivery time of **8.8 days**. The top 3 states (SP, RJ, MG) account for **62.5%** of all sales.
* **Delivery Delay Impact:**
  * **On-Time Deliveries:** **`4.29 / 5.00 ★`** average review score.
  * **Delayed Deliveries:** Plunge to **`2.57 / 5.00 ★`** (-1.72 star satisfaction drop).
  * 8.11% of all orders (7,826 orders) suffered delivery delays beyond the estimated arrival date.
* **Customer RFM Segments:**
  * *At Risk* (65,060 customers, 69.7%): Single purchase >90 days ago — primary retention win-back target.
  * *Potential* (16,114 customers, 17.3%): Recent purchase <=90 days ago — cross-sell opportunity.
  * *High Value* (10,335 customers, 11.1%): Spending $\ge$ R$ 300 — VIP service targets.
  * *Loyal* (1,849 customers, 2.0%): Repeat buyers with frequency >1.
* **Predictive ML Performance:**
  * Balanced Logistic Regression predicting Low Reviews ($\le 2$ stars): **79.6% accuracy**, **0.7127 ROC-AUC**.
  * **Delay Odds Ratio: 1.1844** (+18.4% probability increase of poor reviews for each additional day of shipping delay).

---

## 🚀 Setup & Run Instructions

### 1. Clone the Repository
```powershell
git clone https://github.com/Vishwanjani2905/E-Commerce-business-intelligence.git
cd E-Commerce-business-intelligence
```

### 2. Download and Place Dataset
1. Download the zip archive from [Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. Extract the CSV files directly into the local `data/` folder:
   ```text
   data/
   ├── olist_customers_dataset.csv
   ├── olist_geolocation_dataset.csv
   ├── olist_order_items_dataset.csv
   ├── olist_order_payments_dataset.csv
   ├── olist_order_reviews_dataset.csv
   ├── olist_orders_dataset.csv
   ├── olist_products_dataset.csv
   ├── olist_sellers_dataset.csv
   └── product_category_name_translation.csv
   ```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Run Analytical Workflow (CLI / VS Code)
Run the script to inspect the step-by-step data cleaning, joins, KPI outputs, and exploratory statistics:
```powershell
python VishwanjaniRathod_EcommerceBusinessIntelligence.py
```

### 5. Launch Interactive Streamlit Dashboard
```powershell
streamlit run VishwanjaniRathod_EcommerceBusinessIntelligence.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 📁 Project Structure

```text
E-Commerce-business-intelligence/
│
├── VishwanjaniRathod_EcommerceBusinessIntelligence.py # Unified analysis workflow + Streamlit dashboard (15 sections)
├── E-Commerce_Business_Intelligence_Project_Report.pdf # Comprehensive academic project report (PDF)
├── VishwanjaniRathod_ProjectReport.docx               # Fully editable project report (DOCX)
├── requirements.txt                                   # Minimal pinned dependencies
├── README.md                                          # Complete project documentation
├── .gitignore                                         # Excludes local data folder and cache
└── data/                                              # Official Olist CSV datasets (local)
    ├── olist_customers_dataset.csv
    ├── olist_geolocation_dataset.csv
    ├── olist_order_items_dataset.csv
    ├── olist_order_payments_dataset.csv
    ├── olist_order_reviews_dataset.csv
    ├── olist_orders_dataset.csv
    ├── olist_products_dataset.csv
    ├── olist_sellers_dataset.csv
    └── product_category_name_translation.csv
```

---

## 🖥️ Streamlit Dashboard Views

The web dashboard is organized into 8 interactive tabs:
1. **📈 Executive Overview:** Executive KPI metric cards, monthly revenue trajectory, and top category revenue rankings.
2. **💰 Sales Performance:** Monthly order volume progression, growth rates, and payment method share (Credit Card, Boleto, Voucher, Debit).
3. **👥 Customer Intelligence:** RFM customer segmentation breakdown, new vs. repeat customer contribution.
4. **📦 Product Intelligence:** Top volume product categories, lowest-performing product lines, and Average Order Value by category.
5. **🗺️ Geographic Performance:** State revenue ranking, order density, and logistics lead times across Brazil.
6. **🚚 Delivery & Customer Satisfaction:** Delivery duration distribution and on-time vs. delayed review rating disparity.
7. **💡 Business Insights:** Calculated key drivers, operational risks, and phased strategic growth recommendations.
8. **🤖 Predictive Risk Simulator:** Live interactive risk calculator to simulate low-rating probability based on delivery days and price sliders.

---

## 📚 References & Academic Bibliography

1. **Olist & Kaggle (2018)**. *Brazilian E-Commerce Public Dataset by Olist: 100k Orders with Product, Customer, Seller, and Review Details*. [Kaggle Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. **De Oliveira, L. K., & De Assis, T. F. (2020)**. *E-Commerce Last-Mile Logistics in Brazil: Analyzing Delivery Delays and Freight Costs Using Open Transaction Data*. *Journal of Transport Geography*, 88, 102842.
3. **Hughes, A. M. (1994)**. *Strategic Database Marketing: The Masterplan for Starting and Managing a Profitable, Customer-Based Marketing Program*. Probus Publishing Company.
4. **Fader, P. S., Hardie, B. G., & Lee, K. L. (2005)**. *RFM and CLV: Using Iso-Value Curves for Customer Base Analysis*. *Journal of Marketing Research*, 42(4), 415–430.
5. **Blattberg, R. C., Kim, B. D., & Neslin, S. A. (2008)**. *Database Marketing: Analyzing and Managing Customers*. Springer Science & Business Media.
6. **Kumar, V., & Reinartz, W. (2016)**. *Customer Relationship Management: Concept, Strategy, and Tools*. Springer-Verlag.
7. **Rao, S., Griffis, S. E., & Goldsby, T. J. (2011)**. *Failure to Deliver? Linking Online Order Fulfillment Glitches with Future Purchase Behavior*. *Journal of Operations Management*, 29(4), 293–303.
8. **Thirumalai, S., & Sinha, K. K. (2005)**. *Customer Satisfaction with Order Fulfillment in Retail Supply Chains: Implications of Product Type in Electronic B2C Sites*. *Journal of Operations Management*, 23(3–4), 291–303.
9. **Esper, T. L., et al. (2003)**. *The Impact of the Logistics Delivery Process on Customer Satisfaction and Future Purchase Intentions in B2C E-Commerce*. *Journal of Business Logistics*, 24(2), 177–203.
10. **Oliver, R. L. (2010)**. *Satisfaction: A Behavioral Perspective on the Consumer* (2nd ed.). M.E. Sharpe.
11. **Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013)**. *Applied Logistic Regression* (3rd ed.). John Wiley & Sons.
12. **Pedregosa, F., et al. (2011)**. *Scikit-learn: Machine Learning in Python*. *Journal of Machine Learning Research*, 12, 2825–2830.
13. **Few, S. (2006)**. *Information Dashboard Design: The Effective Visual Communication of Data*. O'Reilly Media.
14. **McKinney, W. (2010)**. *Data Structures for Statistical Computing in Python*. *Proceedings of the 9th Python in Science Conference (SciPy 2010)*, 51–56.
15. **Streamlit Inc. (2024)**. *Streamlit Documentation: The Fastest Way to Build and Share Data Apps*. Snowflake Inc. [docs.streamlit.io](https://docs.streamlit.io).
