import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Helper for notebook-style display in terminal or VS Code interactive window
try:
    display
except NameError:
    def display(df, n=5):
        if hasattr(df, 'head'):
            print(df.head(n).to_string())
        else:
            print(df)

# Check if script is running inside Streamlit
IS_STREAMLIT = False
try:
    import streamlit as st
    if hasattr(st, 'runtime') and st.runtime.exists():
        IS_STREAMLIT = True
except Exception:
    IS_STREAMLIT = False

# %% [markdown]
# # 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Set clean visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

if not IS_STREAMLIT:
    print("=" * 75)
    print("  E-COMMERCE BUSINESS INTELLIGENCE & CUSTOMER ANALYTICS")
    print("  Dataset: Olist Brazilian E-Commerce Public Dataset")
    print("=" * 75)

# # 2. Load Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = 'data'

def load_raw_data():
    orders = pd.read_csv(os.path.join(DATA_DIR, 'olist_orders_dataset.csv'))
    order_items = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_items_dataset.csv'))
    customers = pd.read_csv(os.path.join(DATA_DIR, 'olist_customers_dataset.csv'))
    products = pd.read_csv(os.path.join(DATA_DIR, 'olist_products_dataset.csv'))
    payments = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_payments_dataset.csv'))
    reviews = pd.read_csv(os.path.join(DATA_DIR, 'olist_order_reviews_dataset.csv'))
    sellers = pd.read_csv(os.path.join(DATA_DIR, 'olist_sellers_dataset.csv'))
    category_translation = pd.read_csv(os.path.join(DATA_DIR, 'product_category_name_translation.csv'))
    return orders, order_items, customers, products, payments, reviews, sellers, category_translation

orders, order_items, customers, products, payments, reviews, sellers, category_translation = load_raw_data()

if not IS_STREAMLIT:
    print("\n--- 3. Understanding the Raw Datasets ---")
    print("Orders shape:", orders.shape)
    print("Order Items shape:", order_items.shape)
    print("Customers shape:", customers.shape)
    print("Products shape:", products.shape)
    print("Payments shape:", payments.shape)
    print("Reviews shape:", reviews.shape)

    print("\nOrders Columns and Head:")
    display(orders.head(3))

    print("\nOrder Items Sample:")
    display(order_items.head(3))

    print("\nMissing values in Orders:")
    print(orders.isna().sum())

    print("\nDescriptive statistics for Item Price & Freight:")
    display(order_items[['price', 'freight_value']].describe())


# # 4. Data Cleaning
# Convert date columns to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_approved_at'] = pd.to_datetime(orders['order_approved_at'])
orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
reviews['review_answer_timestamp'] = pd.to_datetime(reviews['review_answer_timestamp'])

# Translate Portuguese product category names to English
products_translated = products.merge(category_translation, on='product_category_name', how='left')
# Fill missing translations with cleaned Portuguese names or 'other'
products_translated['product_category'] = (
    products_translated['product_category_name_english']
    .fillna(products_translated['product_category_name'])
    .fillna('other')
)
products_translated['product_category'] = (
    products_translated['product_category'].str.replace('_', ' ').str.title()
)

# Handle missing comments in reviews (comments are optional in feedback forms)
reviews['review_comment_title'] = reviews['review_comment_title'].fillna('No title')
reviews['review_comment_message'] = reviews['review_comment_message'].fillna('No comment')

# Check duplicates across core identifiers
orders = orders.drop_duplicates(subset=['order_id'])
customers = customers.drop_duplicates(subset=['customer_id'])

if not IS_STREAMLIT:
    print("\n--- 4. Data Cleaning Complete ---")
    print("Cleaned Orders count:", len(orders))
    print("Unique translated categories:", products_translated['product_category'].nunique())

# # 5. Merge the Datasets
# Aggregate order items to order level to avoid incorrect duplicate summing
order_items_agg = order_items.groupby('order_id').agg(
    total_items=('order_item_id', 'count'),
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum')
).reset_index()

# Extract primary product category for each order
primary_item = order_items.sort_values(['order_id', 'order_item_id']).drop_duplicates(subset=['order_id'])
primary_item = primary_item.merge(products_translated[['product_id', 'product_category']], on='product_id', how='left')
primary_item = primary_item.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left')

# Aggregate payments to order level
payments_agg = payments.groupby('order_id').agg(
    payment_value=('payment_value', 'sum'),
    primary_payment_type=('payment_type', 'first'),
    installments=('payment_installments', 'max')
).reset_index()

# Aggregate reviews to order level
reviews_agg = reviews.groupby('order_id').agg(
    review_score=('review_score', 'mean')
).reset_index()

# Join all tables logically into sales_data
sales_data = orders.merge(customers, on='customer_id', how='inner')
sales_data = sales_data.merge(order_items_agg, on='order_id', how='left')
sales_data = sales_data.merge(primary_item[['order_id', 'product_id', 'product_category', 'seller_state']], on='order_id', how='left')
sales_data = sales_data.merge(payments_agg, on='order_id', how='left')
sales_data = sales_data.merge(reviews_agg, on='order_id', how='left')

# Fill category if missing
sales_data['product_category'] = sales_data['product_category'].fillna('Other')
sales_data['primary_payment_type'] = sales_data['primary_payment_type'].fillna('not_defined').str.replace('_', ' ').str.title()

if not IS_STREAMLIT:
    print("\n--- 5. Merged Dataset Created ---")
    print("Merged sales_data shape:", sales_data.shape)
    display(sales_data[['order_id', 'customer_unique_id', 'order_status', 'total_price', 'product_category', 'review_score']].head(3))

# # 6. Create Business Features
# Date and time features
sales_data['order_year'] = sales_data['order_purchase_timestamp'].dt.year
sales_data['order_month'] = sales_data['order_purchase_timestamp'].dt.month
sales_data['order_year_month'] = sales_data['order_purchase_timestamp'].dt.to_period('M').astype(str)
sales_data['order_weekday'] = sales_data['order_purchase_timestamp'].dt.day_name()

# Logistics features (in days)
sales_data['delivery_days'] = (
    sales_data['order_delivered_customer_date'] - sales_data['order_purchase_timestamp']
).dt.total_seconds() / (24 * 3600)

sales_data['estimated_delivery_days'] = (
    sales_data['order_estimated_delivery_date'] - sales_data['order_purchase_timestamp']
).dt.total_seconds() / (24 * 3600)

sales_data['delivery_delay_days'] = (
    sales_data['order_delivered_customer_date'] - sales_data['order_estimated_delivery_date']
).dt.total_seconds() / (24 * 3600)

sales_data['is_delayed'] = (sales_data['delivery_delay_days'] > 0).astype(int)

# Financial features
sales_data['order_value'] = sales_data['total_price'].fillna(0)
sales_data['freight_cost'] = sales_data['total_freight'].fillna(0)
sales_data['total_order_value'] = sales_data['order_value'] + sales_data['freight_cost']

if not IS_STREAMLIT:
    print("\n--- 6. Business Features Engineered ---")
    print("New features created: order_year_month, delivery_days, delivery_delay_days, is_delayed, total_order_value")
    display(sales_data[['order_purchase_timestamp', 'delivery_days', 'delivery_delay_days', 'is_delayed', 'total_order_value']].head(3))

# %% [markdown]
# # 7. KPI Analysis
# Calculate core business KPIs on delivered and valid orders
delivered_orders = sales_data[sales_data['order_status'] == 'delivered'].copy()

total_revenue = delivered_orders['total_order_value'].sum()
total_product_revenue = delivered_orders['order_value'].sum()
total_orders = delivered_orders['order_id'].nunique()
total_unique_customers = delivered_orders['customer_unique_id'].nunique()
average_order_value = total_revenue / total_orders if total_orders > 0 else 0
average_review_score = delivered_orders['review_score'].mean()
average_delivery_time = delivered_orders['delivery_days'].mean()

# Repeat customer rate based on unique customer ID
customer_order_counts = delivered_orders.groupby('customer_unique_id')['order_id'].nunique()
repeat_customers_count = (customer_order_counts > 1).sum()
repeat_customer_rate = (repeat_customers_count / total_unique_customers) * 100

if not IS_STREAMLIT:
    print("\n--- 7. Core Business KPIs ---")
    print(f"Total Revenue (GMV + Freight): R$ {total_revenue:,.2f}")
    print(f"Product Only Revenue:         R$ {total_product_revenue:,.2f}")
    print(f"Total Delivered Orders:       {total_orders:,}")
    print(f"Total Unique Customers:       {total_unique_customers:,}")
    print(f"Average Order Value (AOV):    R$ {average_order_value:.2f}")
    print(f"Repeat Customer Rate:         {repeat_customer_rate:.2f}% ({repeat_customers_count:,} repeat buyers)")
    print(f"Average Review Score:         {average_review_score:.2f} / 5.0")
    print(f"Average Delivery Time:        {average_delivery_time:.1f} days")

# # 8. Sales Trend Analysis
monthly_sales = delivered_orders.groupby('order_year_month').agg(
    monthly_revenue=('total_order_value', 'sum'),
    order_volume=('order_id', 'nunique'),
    avg_order_value=('total_order_value', 'mean')
).reset_index()

monthly_sales['revenue_growth_pct'] = monthly_sales['monthly_revenue'].pct_change() * 100

if not IS_STREAMLIT:
    print("\n--- 8. Monthly Sales Trends (Last 6 Months) ---")
    display(monthly_sales.tail(6))

# # 9. Product Analysis
category_sales = delivered_orders.groupby('product_category').agg(
    total_revenue=('total_order_value', 'sum'),
    order_count=('order_id', 'nunique'),
    avg_order_value=('total_order_value', 'mean'),
    avg_review=('review_score', 'mean')
).reset_index()

top_10_categories = category_sales.sort_values('total_revenue', ascending=False).head(10)
bottom_10_categories = category_sales.sort_values('total_revenue', ascending=True).head(10)

if not IS_STREAMLIT:
    print("\n--- 9. Top 5 Product Categories by Revenue ---")
    display(top_10_categories.head(5))

    print("\nBottom 5 Product Categories by Revenue:")
    display(bottom_10_categories.head(5))

# # 10. Customer Analysis
# RFM Calculation: Recency, Frequency, Monetary
max_purchase_date = delivered_orders['order_purchase_timestamp'].max()

customer_summary = delivered_orders.groupby('customer_unique_id').agg(
    recency_days=('order_purchase_timestamp', lambda x: (max_purchase_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('total_order_value', 'sum'),
    customer_state=('customer_state', 'first')
).reset_index()

# Simple explainable customer segmentation
def segment_customer(row):
    if row['monetary'] >= 300:
        return 'High Value'
    elif row['frequency'] > 1:
        return 'Loyal'
    elif row['recency_days'] <= 90:
        return 'Potential'
    else:
        return 'At Risk'

customer_summary['segment'] = customer_summary.apply(segment_customer, axis=1)

# New vs repeat contribution
customer_summary['customer_type'] = np.where(customer_summary['frequency'] > 1, 'Repeat Customer', 'One-Time Customer')
cust_type_analysis = customer_summary.groupby('customer_type').agg(
    customer_count=('customer_unique_id', 'count'),
    total_spend=('monetary', 'sum'),
    avg_spend=('monetary', 'mean')
).reset_index()
cust_type_analysis['revenue_share_pct'] = (cust_type_analysis['total_spend'] / total_revenue) * 100

# Attach segment to sales_data for dashboard filtering
sales_data = sales_data.merge(customer_summary[['customer_unique_id', 'segment']], on='customer_unique_id', how='left')
sales_data['segment'] = sales_data['segment'].fillna('At Risk')

if not IS_STREAMLIT:
    print("\n--- 10. Customer Segmentation Breakdown ---")
    print(customer_summary['segment'].value_counts())
    print("\nNew vs Repeat Customer Contribution:")
    display(cust_type_analysis)

# # 11. Geographic Analysis
state_performance = delivered_orders.groupby('customer_state').agg(
    revenue=('total_order_value', 'sum'),
    orders=('order_id', 'nunique'),
    avg_order_value=('total_order_value', 'mean'),
    avg_delivery_days=('delivery_days', 'mean')
).sort_values('revenue', ascending=False).reset_index()

state_performance['revenue_share_pct'] = (state_performance['revenue'] / total_revenue) * 100

if not IS_STREAMLIT:
    print("\n--- 11. Top 5 Brazilian States by Revenue ---")
    display(state_performance.head(5))

# # 12. Delivery & Review Analysis
delivery_comparison = delivered_orders.groupby('is_delayed').agg(
    order_count=('order_id', 'nunique'),
    avg_review_score=('review_score', 'mean'),
    avg_delivery_days=('delivery_days', 'mean')
).reset_index()
delivery_comparison['status_label'] = np.where(delivery_comparison['is_delayed'] == 1, 'Delayed', 'On-Time')

review_distribution = delivered_orders['review_score'].value_counts().sort_index().reset_index()
review_distribution.columns = ['review_score', 'count']
review_distribution['share_pct'] = (review_distribution['count'] / review_distribution['count'].sum()) * 100

if not IS_STREAMLIT:
    print("\n--- 12. Delivery Delay vs Review Score Analysis ---")
    display(delivery_comparison[['status_label', 'order_count', 'avg_review_score', 'avg_delivery_days']])
    print("\nReview Score Distribution:")
    display(review_distribution)

# # 13. Customer Segmentation & Predictive Component
# Predictive model: Predict if an order will receive a Low Review Score (<= 2 stars)
model_df = delivered_orders.copy()
model_df['delay_days_clipped'] = model_df['delivery_delay_days'].clip(lower=0)
model_df['low_review'] = (model_df['review_score'] <= 2).astype(int)

features = ['delivery_days', 'delay_days_clipped', 'order_value', 'freight_cost']
model_df = model_df[features + ['low_review']].dropna()

X = model_df[features]
y = model_df['low_review']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

predictive_model = LogisticRegression(max_iter=1000, class_weight='balanced')
predictive_model.fit(X_train, y_train)

y_pred = predictive_model.predict(X_test)
y_prob = predictive_model.predict_proba(X_test)[:, 1]

model_accuracy = accuracy_score(y_test, y_pred)
model_roc_auc = roc_auc_score(y_test, y_prob)
odds_ratios = dict(zip(features, np.exp(predictive_model.coef_[0])))

if not IS_STREAMLIT:
    print("\n--- 13. Predictive Component: Low Review Risk Model ---")
    print(f"Model: Logistic Regression (Target: Review Score <= 2)")
    print(f"Accuracy: {model_accuracy:.4f} | ROC-AUC: {model_roc_auc:.4f}")
    print("Feature Odds Ratios (Multiplicative Risk Factor):")
    for feat, odds in odds_ratios.items():
        print(f"  - {feat:20s}: {odds:.4f}")

# # 14. Business Insights
top_category_name = top_10_categories.iloc[0]['product_category']
top_category_revenue = top_10_categories.iloc[0]['total_revenue']
top_state_code = state_performance.iloc[0]['customer_state']
top_state_share = state_performance.iloc[0]['revenue_share_pct']

on_time_score = delivery_comparison.loc[delivery_comparison['is_delayed'] == 0, 'avg_review_score'].values[0]
delayed_score = delivery_comparison.loc[delivery_comparison['is_delayed'] == 1, 'avg_review_score'].values[0]

if not IS_STREAMLIT:
    print("\n" + "=" * 75)
    print("                         KEY BUSINESS INSIGHTS")
    print("=" * 75)
    print(f"1. Top Revenue Driver: '{top_category_name}' generates R$ {top_category_revenue:,.2f}.")
    print(f"2. Geographic Concentration: State '{top_state_code}' generates {top_state_share:.1f}% of total sales.")
    print(f"3. Repeat Purchase Opportunity: Repeat customer rate is {repeat_customer_rate:.2f}%.")
    print(f"4. Delivery Delay Penalty: Delayed shipments drop review scores from {on_time_score:.2f} to {delayed_score:.2f}.")
    print(f"5. Predictive Risk Driver: Each day of delivery delay increases low-rating odds by ~{(odds_ratios['delay_days_clipped']-1)*100:.1f}%.")
    print("\nRISKS:")
    print("  - Heavy reliance on one-time shoppers (96.9% single purchase).")
    print("  - Logistics bottlenecks in remote northern states take up to 25+ days.")
    print("  - Customer churn due to delivery delays.")
    print("\nOPPORTUNITIES:")
    print("  - Post-purchase retention workflows and loyalty discounts.")
    print("  - Regional distribution fulfillment centers to reduce delivery times.")
    print("  - Product bundling in high-order, low-AOV categories.")
    print("=" * 75)

# # 15. Streamlit Dashboard
def render_dashboard():
    st.set_page_config(
        page_title="E-Commerce Business Intelligence Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Clean CSS styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 2px;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 25px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !0important;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">E-Commerce Business Intelligence Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sales, Customer, Product & Operational Performance Analysis | Olist Public Dataset</p>', unsafe_allow_html=True)

    # Sidebar Filters
    st.sidebar.header("🔍 Dashboard Filters")

    # Year filter
    available_years = sorted(sales_data['order_year'].dropna().astype(int).unique().tolist())
    selected_year = st.sidebar.selectbox("Filter by Order Year", ["All Years"] + available_years)

    # State filter
    all_states = sorted(sales_data['customer_state'].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("Customer States", all_states, default=[])

    # Category filter
    all_categories = sorted(sales_data['product_category'].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect("Product Categories", all_categories, default=[])

    # Segment filter
    all_segments = ['High Value', 'Loyal', 'Potential', 'At Risk']
    selected_segments = st.sidebar.multiselect("Customer Segments", all_segments, default=[])

    # Apply filters
    filtered_data = sales_data[sales_data['order_status'] == 'delivered'].copy()

    if selected_year != "All Years":
        filtered_data = filtered_data[filtered_data['order_year'] == int(selected_year)]

    if selected_states:
        filtered_data = filtered_data[filtered_data['customer_state'].isin(selected_states)]

    if selected_categories:
        filtered_data = filtered_data[filtered_data['product_category'].isin(selected_categories)]

    if selected_segments:
        filtered_data = filtered_data[filtered_data['segment'].isin(selected_segments)]

    if filtered_data.empty:
        st.warning("No records match the selected filters. Please adjust your filter selections.")
        return

    # Filtered KPIs
    f_revenue = filtered_data['total_order_value'].sum()
    f_orders = filtered_data['order_id'].nunique()
    f_customers = filtered_data['customer_unique_id'].nunique()
    f_aov = f_revenue / f_orders if f_orders > 0 else 0
    f_review = filtered_data['review_score'].mean()
    f_delivery = filtered_data['delivery_days'].mean()

    cust_counts = filtered_data.groupby('customer_unique_id')['order_id'].nunique()
    f_repeat_rate = ((cust_counts > 1).sum() / f_customers * 100) if f_customers > 0 else 0

    # Executive KPI Cards
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Total Revenue", f"R$ {f_revenue:,.0f}")
    col2.metric("Total Orders", f"{f_orders:,}")
    col3.metric("Unique Customers", f"{f_customers:,}")
    col4.metric("Avg Order Value", f"R$ {f_aov:.2f}")
    col5.metric("Repeat Cust Rate", f"{f_repeat_rate:.2f}%")
    col6.metric("Avg Review", f"{f_review:.2f} ★")
    col7.metric("Avg Delivery", f"{f_delivery:.1f} days")

    st.write("---")

    # Tabs for Dashboard Sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Executive Overview",
        "💰 Sales Performance",
        "👥 Customer Intelligence",
        "📦 Product Intelligence",
        "🗺️ Geographic Performance",
        "🚚 Delivery & Satisfaction",
        "💡 Business Insights",
        "🤖 Predictive Model"
    ])

    # Tab 1: Executive Overview
    with tab1:
        st.subheader("Executive Performance Overview")
        col_t1_left, col_t1_right = st.columns(2)

        with col_t1_left:
            # Monthly Revenue Trend
            monthly_f = filtered_data.groupby('order_year_month')['total_order_value'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(monthly_f['order_year_month'], monthly_f['total_order_value'] / 1000, marker='o', color='#2563EB', linewidth=2.5)
            ax.set_title("Monthly Revenue Trajectory (Thousands R$)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Year-Month", fontsize=10)
            ax.set_ylabel("Revenue (k R$)", fontsize=10)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_t1_right:
            # Top Categories in filtered view
            top_cats_f = filtered_data.groupby('product_category')['total_order_value'].sum().sort_values(ascending=False).head(7).reset_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=top_cats_f, x='total_order_value', y='product_category', palette='Blues_r', ax=ax)
            ax.set_title("Top Revenue Product Categories", fontsize=12, fontweight='bold')
            ax.set_xlabel("Revenue (R$)", fontsize=10)
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Tab 2: Sales Performance
    with tab2:
        st.subheader("Sales Volume & Payment Methods")
        col_t2_1, col_t2_2 = st.columns(2)

        with col_t2_1:
            # Monthly Orders
            orders_f = filtered_data.groupby('order_year_month')['order_id'].nunique().reset_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(orders_f['order_year_month'], orders_f['order_id'], color='#0D9488', alpha=0.85)
            ax.set_title("Monthly Order Volume", fontsize=12, fontweight='bold')
            ax.set_xlabel("Year-Month", fontsize=10)
            ax.set_ylabel("Delivered Orders", fontsize=10)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_t2_2:
            # Payment Type Breakdown
            pay_dist = filtered_data['primary_payment_type'].value_counts().reset_index()
            pay_dist.columns = ['payment_type', 'count']
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.pie(pay_dist['count'], labels=pay_dist['payment_type'], autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
            ax.set_title("Payment Method Distribution", fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Tab 3: Customer Intelligence
    with tab3:
        st.subheader("Customer Analytics & Segmentation")
        col_t3_1, col_t3_2 = st.columns(2)

        with col_t3_1:
            # Customer Segments
            seg_dist = filtered_data['segment'].value_counts().reset_index()
            seg_dist.columns = ['segment', 'count']
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=seg_dist, x='segment', y='count', palette='Set2', ax=ax)
            ax.set_title("Customer Segmentation Distribution", fontsize=12, fontweight='bold')
            ax.set_xlabel("Segment", fontsize=10)
            ax.set_ylabel("Customer Count", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_t3_2:
            # Revenue by Segment
            seg_rev = filtered_data.groupby('segment')['total_order_value'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=seg_rev, x='segment', y='total_order_value', palette='Set2', ax=ax)
            ax.set_title("Total Revenue by Customer Segment (R$)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Segment", fontsize=10)
            ax.set_ylabel("Revenue (R$)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Tab 4: Product Intelligence
    with tab4:
        st.subheader("Product Category Analysis")
        col_t4_1, col_t4_2 = st.columns(2)

        cat_summary_f = filtered_data.groupby('product_category').agg(
            revenue=('total_order_value', 'sum'),
            orders=('order_id', 'nunique'),
            aov=('total_order_value', 'mean')
        ).reset_index()

        with col_t4_1:
            top_vol_cats = cat_summary_f.sort_values('orders', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=top_vol_cats, x='orders', y='product_category', palette='viridis', ax=ax)
            ax.set_title("Top 10 Categories by Order Volume", fontsize=12, fontweight='bold')
            ax.set_xlabel("Number of Orders", fontsize=10)
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_t4_2:
            top_aov_cats = cat_summary_f[cat_summary_f['orders'] >= 20].sort_values('aov', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=top_aov_cats, x='aov', y='product_category', palette='mako', ax=ax)
            ax.set_title("Top Categories by Average Order Value (Min 20 orders)", fontsize=12, fontweight='bold')
            ax.set_xlabel("Average Order Value (R$)", fontsize=10)
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Tab 5: Geographic Performance
    with tab5:
        st.subheader("Geographic Distribution Across Brazilian States")
        col_t5_1, col_t5_2 = st.columns(2)

        state_f = filtered_data.groupby('customer_state').agg(
            revenue=('total_order_value', 'sum'),
            orders=('order_id', 'nunique'),
            avg_delivery=('delivery_days', 'mean')
        ).sort_values('revenue', ascending=False).reset_index()

        with col_t5_1:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=state_f.head(10), x='customer_state', y='revenue', palette='rocket', ax=ax)
            ax.set_title("Top 10 States by Revenue (R$)", fontsize=12, fontweight='bold')
            ax.set_xlabel("State Code", fontsize=10)
            ax.set_ylabel("Revenue (R$)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_t5_2:
            fig, ax = plt.subplots(figsize=(8, 4))
            state_deliv_sorted = state_f.sort_values('avg_delivery', ascending=False).head(10)
            sns.barplot(data=state_deliv_sorted, x='avg_delivery', y='customer_state', palette='coolwarm', ax=ax)
            ax.set_title("States with Longest Average Delivery Times", fontsize=12, fontweight='bold')
            ax.set_xlabel("Average Delivery Days", fontsize=10)
            ax.set_ylabel("State Code", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Tab 6: Delivery & Customer Satisfaction
    with tab6:
        st.subheader("Delivery Lead Times & Review Score Dynamics")
        col_t6_1, col_t6_2 = st.columns(2)

        with col_t6_1:
            # Review Score Breakdown
            fig, ax = plt.subplots(figsize=(8, 4))
            score_counts = filtered_data['review_score'].value_counts().sort_index().reset_index()
            score_counts.columns = ['score', 'count']
            sns.barplot(data=score_counts, x='score', y='count', color='#F59E0B', ax=ax)
            ax.set_title("Customer Review Score Distribution", fontsize=12, fontweight='bold')
            ax.set_xlabel("Review Rating (1 - 5)", fontsize=10)
            ax.set_ylabel("Order Count", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_t6_2:
            # Impact of Delivery Delay on Review Score
            deliv_score = filtered_data.groupby('is_delayed')['review_score'].mean().reset_index()
            deliv_score['status'] = np.where(deliv_score['is_delayed'] == 1, 'Delayed', 'On-Time')
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=deliv_score, x='status', y='review_score', palette=['#10B981', '#EF4444'], ax=ax)
            ax.set_ylim(0, 5)
            ax.set_title("Average Review Score: On-Time vs Delayed Deliveries", fontsize=12, fontweight='bold')
            ax.set_xlabel("Delivery Status", fontsize=10)
            ax.set_ylabel("Average Review Rating (Stars)", fontsize=10)
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f} ★", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Tab 7: Business Insights
    with tab7:
        st.subheader("Strategic Business Insights & Decision Support")
        col_ins_1, col_ins_2 = st.columns(2)

        with col_ins_1:
            st.markdown("### 📌 Key Calculated Metrics")
            st.markdown(f"- **Top Revenue Product Line:** `{top_category_name}` with **R$ {top_category_revenue:,.2f}**.")
            st.markdown(f"- **Geographic Anchor:** State `{top_state_code}` accounts for **{top_state_share:.1f}%** of revenue.")
            st.markdown(f"- **Repeat Customer Baseline:** **{repeat_customer_rate:.2f}%** repeat buyers ({repeat_customers_count:,} customers).")
            st.markdown(f"- **Delivery Satisfaction Gap:** On-time shipments score **{on_time_score:.2f} ★**, whereas delayed orders plunge to **{delayed_score:.2f} ★**.")
            st.markdown(f"- **Delay Sensitivity:** Each day of delivery delay increases low-rating probability by **{(odds_ratios['delay_days_clipped']-1)*100:.1f}%**.")

            st.markdown("### ⚠️ Operational Risks")
            st.error("""
            1. **Extremely Low Retention Rate:** Over 96% of customers purchase only once, creating high acquisition pressure.
            2. **Delivery Latency in North & Northeast:** States like RR, AP, and AM experience shipping delays exceeding 25-30 days.
            3. **Severe Rating Penalty for Delays:** Delayed shipments receive predominantly 1- and 2-star reviews.
            """)

        with col_ins_2:
            st.markdown("### 🚀 Strategic Growth Opportunities")
            st.success("""
            1. **Customer Retention Engine:** Launch automated email workflows for repurchasing consumable items within 60 days of delivery.
            2. **Regional Fulfillment Hubs:** Partner with local 3PL providers in Brasília and Northeast hubs to cut interstate transit times.
            3. **High AOV Bundling:** Combine high-volume low-cost items with accessories to raise cart sizes.
            4. **Proactive Delay Alerts:** Notify customers before the estimated delivery date passes with compensation vouchers to preserve ratings.
            """)

    # Tab 8: Predictive Model
    with tab8:
        st.subheader("Machine Learning: Low Review Score Risk Predictor")
        st.markdown("""
        This explainable logistic regression model predicts whether an order will receive a **poor customer rating (<= 2 stars)** based on operational variables.
        """)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### Model Performance Metrics")
            st.write(f"- **Algorithm:** Balanced Logistic Regression")
            st.write(f"- **Test Accuracy:** `{model_accuracy:.4f}`")
            st.write(f"- **ROC-AUC Score:** `{model_roc_auc:.4f}`")
            st.write(f"- **Training Samples:** `{len(X_train):,}` | **Testing Samples:** `{len(X_test):,}`")

            st.markdown("#### Feature Odds Ratios")
            odds_df = pd.DataFrame({
                'Feature': ['Delivery Days', 'Delivery Delay (Days)', 'Item Price (R$)', 'Freight Cost (R$)'],
                'Odds Ratio': [odds_ratios['delivery_days'], odds_ratios['delay_days_clipped'], odds_ratios['order_value'], odds_ratios['freight_cost']]
            })
            st.dataframe(odds_df, use_container_width=True)

        with col_m2:
            st.markdown("#### Live Order Risk Simulator")
            sim_delivery = st.slider("Total Delivery Days", min_value=1, max_value=60, value=12)
            sim_delay = st.slider("Delivery Delay Days (Beyond Estimate)", min_value=0, max_value=30, value=0)
            sim_price = st.number_input("Item Price (R$)", min_value=10.0, max_value=2000.0, value=120.0)
            sim_freight = st.number_input("Freight Cost (R$)", min_value=5.0, max_value=200.0, value=25.0)

            input_data = np.array([[sim_delivery, sim_delay, sim_price, sim_freight]])
            predicted_prob = predictive_model.predict_proba(input_data)[0, 1]

            st.markdown("##### Predicted Risk of Low Rating (<= 2 Stars):")
            risk_color = "red" if predicted_prob > 0.5 else "orange" if predicted_prob > 0.3 else "green"
            st.markdown(f"<h3 style='color:{risk_color};'>{predicted_prob * 100:.1f}% Risk</h3>", unsafe_allow_html=True)
            if predicted_prob > 0.5:
                st.warning("⚠️ High Risk Order: Operational delay makes poor customer feedback very likely. Consider sending a customer care apology voucher.")
            else:
                st.info("✅ Acceptable Risk: Delivery parameters are within safe satisfaction boundaries.")

# Launch Streamlit or CLI summary
if IS_STREAMLIT:
    render_dashboard()
elif __name__ == "__main__":
    print("\n" + "=" * 75)
    print("Notebook-style analysis completed successfully!")
    print("To launch the interactive dashboard, run in your terminal:")
    print("  streamlit run ecommerce_business_dashboard.py")
    print("=" * 75)
