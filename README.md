# E-Commerce Business Intelligence Dashboard

[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Olist%20Brazilian%20E--Commerce-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

## Project Overview
This project is an end-to-end **E-Commerce Business Intelligence & Customer Analytics** application built using Python and Streamlit on the **Olist Brazilian E-Commerce Public Dataset**. It transforms raw marketplace data across orders, customers, items, payments, reviews, and products into actionable executive KPIs, trend analyses, customer segmentations, and operational risk metrics.

The project is structured with a notebook-like learning workflow for in-depth exploratory data analysis and a complete interactive Streamlit dashboard for business decision-making.

---

## Dataset

This project uses the official, real-world **Brazilian E-Commerce Public Dataset by Olist**, publicly hosted on Kaggle:

🔗 **Direct Kaggle Dataset URL:**  
**[https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**

### Dataset Information:
- **Source:** Olist via Kaggle
- **Coverage:** 100,000+ real e-commerce orders from 2016 to 2018 across Brazilian marketplaces.
- **Structure:** 9 relational tables covering orders, customers, products, payments, reviews, merchants, and translations.
- **Usage:** Download the dataset directly from Kaggle and place the unzipped CSV files into the local `data/` folder:
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

---

## Problem Statement
Modern e-commerce marketplaces handle thousands of daily transactions across diverse geographic regions, sellers, and product categories. Without unified business intelligence, executives struggle with:
1. Identifying high-value vs. unprofitable product categories and customer segments.
2. Understanding geographical demand concentrations and regional delivery bottlenecks.
3. Measuring the direct impact of shipping delays on customer satisfaction and review scores.
4. Improving low repeat purchase rates through targeted customer retention strategies.

This project delivers a centralized data pipeline and decision dashboard to monitor sales performance, optimize operational logistics, and guide strategic commercial decisions.

---

## Objectives
- **Data Engineering & Cleaning**: Ingest, clean, translate, and merge 9 relational e-commerce tables into an analytical dataset.
- **Core Business KPIs**: Compute accurate order-level metrics including Total Revenue, Total Orders, Unique Customers, Average Order Value (AOV), Repeat Customer Rate, and Average Delivery Time.
- **Sales Trend Analysis**: Track monthly revenue trajectories, seasonal spikes (e.g., Black Friday), and order volume patterns.
- **Product & Category Intelligence**: Uncover top revenue-generating categories, volume leaders, lagging product lines, and category-level pricing dynamics.
- **Customer Segmentation (RFM)**: Segment customers into High Value, Loyal, Potential, and At Risk cohorts using Recency, Frequency, and Monetary scores.
- **Geographic Performance**: Identify top commercial hubs (e.g., São Paulo, Rio de Janeiro, Minas Gerais) and assess logistics performance by state.
- **Delivery & Customer Satisfaction**: Quantify the relationship between delivery delays and customer review scores.
- **Predictive Analytics**: Implement an explainable machine learning model to predict the likelihood of poor customer review ratings based on logistics delays and cost factors.
- **Interactive Streamlit Dashboard**: Provide executive filters (date, state, category, segment) and interactive visualizations for business stakeholders.

---

## Features
- **Sales Analysis**: Monthly revenue trends, order volume trajectory, and payment method distribution.
- **Customer Analytics**: New vs. repeat customer breakdown, lifetime value distribution, and order frequency.
- **Product Analysis**: Top 10 categories by revenue, order volume rankings, low-performing categories, and AOV by category.
- **Geographic Analysis**: State-wise revenue contribution, order density, and regional logistics efficiency.
- **Delivery Analysis**: Delivery lead time distribution, estimated vs. actual delivery gap, and delayed shipment rates.
- **Review Analysis**: Customer satisfaction distribution (scores 1 to 5) and correlation between shipping delays and low ratings.
- **Customer Segmentation**: Explainable RFM (Recency, Frequency, Monetary) segmentation.
- **Predictive Scoring**: Logistic regression model predicting low customer ratings ($\le 2$ stars) with explainable odds ratios.
- **Streamlit Dashboard**: Responsive UI with executive KPI cards, dynamic sidebar filters, and multi-tab analytical views.

---

## Project Structure
```text
ecommerce_business_dashboard.py
requirements.txt
README.md
data/
    olist_customers_dataset.csv
    olist_geolocation_dataset.csv
    olist_order_items_dataset.csv
    olist_order_payments_dataset.csv
    olist_order_reviews_dataset.csv
    olist_orders_dataset.csv
    olist_products_dataset.csv
    olist_sellers_dataset.csv
    product_category_name_translation.csv
```

---

## Installation
Clone the repository and install the minimal dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## Running the Project

### 1. Run Analytical Workflow (CLI / VS Code)
Run the script to inspect the step-by-step data cleaning, joins, KPI outputs, and exploratory statistics:

```powershell
python ecommerce_business_dashboard.py
```

### 2. Launch Interactive Streamlit Dashboard
Launch the web dashboard in your browser:

```powershell
streamlit run ecommerce_business_dashboard.py
```

---

## Dashboard Screenshots
*(Screenshots can be added here after launching the dashboard)*

- **Executive Overview & KPI Cards**
- **Sales & Category Trends**
- **Customer RFM Segmentation**
- **Delivery vs. Review Score Impact Analysis**

---

## Business Insights & Decisions Supported
- **Logistics SLA Optimization**: Identifies shipping routes and states with chronic delivery delays, enabling supply chain renegotiation to protect customer satisfaction.
- **Marketing Budget Allocation**: Directs ad spend toward top-performing categories (e.g., Bed, Bath & Table, Health & Beauty, Computers & Accessories) and states with high purchasing power.
- **Retention Campaigns**: Highlights the small but vital repeat customer cohort and dormant accounts for win-back email workflows and personalized loyalty rewards.
- **Inventory & Pricing Strategy**: Evaluates categories with high order volume but lower AOV to design effective product bundling and minimum order thresholds.

---

## References & Academic Bibliography

### 1. Dataset & Benchmark Studies
1. **Olist & Kaggle (2018)**. *Brazilian E-Commerce Public Dataset by Olist: 100k Orders with Product, Customer, Seller, and Review Details*. [Kaggle Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. **De Oliveira, L. K., & De Assis, T. F. (2020)**. *E-Commerce Last-Mile Logistics in Brazil: Analyzing Delivery Delays and Freight Costs Using Open Transaction Data*. Journal of Transport Geography, 88, 102842.

### 2. Customer Analytics & RFM Segmentation
3. **Hughes, A. M. (1994)**. *Strategic Database Marketing: The Masterplan for Starting and Managing a Profitable, Customer-Based Marketing Program*. Probus Publishing Company.
4. **Fader, P. S., Hardie, B. G., & Lee, K. L. (2005)**. *RFM and CLV: Using Iso-Value Curves for Customer Base Analysis*. Journal of Marketing Research, 42(4), 415–430.
5. **Blattberg, R. C., Kim, B. D., & Neslin, S. A. (2008)**. *Database Marketing: Analyzing and Managing Customers*. Springer Science & Business Media.
6. **Kumar, V., & Reinartz, W. (2016)**. *Customer Relationship Management: Concept, Strategy, and Tools*. Springer-Verlag.

### 3. Supply Chain, Delivery Lead Times & Customer Satisfaction
7. **Rao, S., Griffis, S. E., & Goldsby, T. J. (2011)**. *Failure to Deliver? Linking Online Order Fulfillment Glitches with Future Purchase Behavior*. Journal of Operations Management, 29(4), 293–303.
8. **Thirumalai, S., & Sinha, K. K. (2005)**. *Customer Satisfaction with Order Fulfillment in Retail Supply Chains: Implications of Product Type in Electronic B2C Sites*. Journal of Operations Management, 23(3–4), 291–303.
9. **Esper, T. L., Jensen, T. D., Turnipseed, F. L., & Burton, S. (2003)**. *The Impact of the Logistics Delivery Process on Customer Satisfaction and Future Purchase Intentions in B2C E-Commerce*. Journal of Business Logistics, 24(2), 177–203.
10. **Oliver, R. L. (2010)**. *Satisfaction: A Behavioral Perspective on the Consumer* (2nd ed.). M.E. Sharpe.

### 4. Predictive Modeling & Machine Learning
11. **Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013)**. *Applied Logistic Regression* (3rd ed.). John Wiley & Sons.
12. **Pedregosa, F., et al. (2011)**. *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
13. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

### 5. Business Intelligence & Interactive Systems
14. **Few, S. (2006)**. *Information Dashboard Design: The Effective Visual Communication of Data*. O'Reilly Media.
15. **McKinney, W. (2010)**. *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference (SciPy 2010), 51–56.
16. **Streamlit Inc. (2024)**. *Streamlit Documentation: The Fastest Way to Build and Share Data Apps*. Snowflake Inc. Available at: [docs.streamlit.io](https://docs.streamlit.io).
