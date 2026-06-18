import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Page Layout Configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Segmentation & Cluster Analysis")
st.markdown("Segmenting customers based on behavior and demographics using K-Means Clustering.")
st.markdown("---")

# 2. Load Dataset safely
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("customer_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading customer_data.csv: {e}")
        st.stop()

df = load_data()

# 3. Model Controls in Sidebar
st.sidebar.header("⚙️ K-Means Configuration")
num_clusters = st.sidebar.slider(
    "Select Number of Clusters (K):", 
    min_value=2, 
    max_value=5, 
    value=3
)

# 4. Machine Learning Clustering Pipeline
# Selecting metrics for clustering (Income vs Spending Behavior)
features = ['AnnualIncome', 'SpendingScore']
X = df[features]

# Standardize features (Mean=0, Variance=1) for balanced distance math
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the K-Means algorithm
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster'] = df['Cluster'].astype(str)  # Convert to string for discrete color mappings

# 5. Key Metrics Panels
st.subheader("📊 Segmentation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers Processed", f"{len(df)} users")
col2.metric("Target Features", "Income & Spending Score")
col3.metric("Generated Groups (Clusters)", f"{num_clusters} Clusters")

st.markdown("---")

# 6. Interactive Visualizations
st.subheader("🎯 Customer Cluster Visualizations")
chart_col, metrics_col = st.columns([2, 1])

with chart_col:
    # Scatter plot mapping the identified clusters
    fig = px.scatter(
        df, 
        x='AnnualIncome', 
        y='SpendingScore', 
        color='Cluster',
        size='Age',
        hover_data=['CustomerID', 'Age'],
        title="Identified Customer Segments (Bubble size corresponds to Age)",
        labels={'AnnualIncome': 'Annual Income ($k)', 'SpendingScore': 'Spending Score (1-100)'},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with metrics_col:
    st.markdown("#### Cluster Profiles (Averages)")
    # Group by cluster to see characteristics of each segment
    summary_df = df.groupby('Cluster')[['Age', 'AnnualIncome', 'SpendingScore']].mean()
    st.dataframe(summary_df.style.background_gradient(cmap='Purples'), use_container_width=True)

# Raw Data Explorer
with st.expander("👀 View Segmented Raw Data Table"):
    st.dataframe(df, use_container_width=True)   
