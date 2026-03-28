import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Configure the Streamlit page
st.set_page_config(page_title="ATM Demand Forecasting", layout="wide")

# --- FA-1: DATA PREPROCESSING ---
@st.cache_data
def load_and_preprocess_data():
    # Load the dataset (ensure the CSV is in the same directory)
    df = pd.read_csv('atm_cash_management_dataset.csv')
    
    # 1. Datetime Formatting
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Handle Missing Values (Fill any blanks appropriately)
    df.fillna({'Holiday_Flag': 0, 'Special_Event_Flag': 0}, inplace=True)
    
    # 3. Categorical Encoding (Required for ML models later)
    # Creating a numeric copy of Location_Type for clustering
    df['Location_Encoded'] = df['Location_Type'].astype('category').cat.codes
    
    return df

df = load_and_preprocess_data()

st.title("🏦 ATM Intelligence Demand Forecasting")
st.markdown("Analyze withdrawal trends, cluster ATMs, and detect anomalous demand spikes.")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Module:", ["Exploratory Data Analysis (EDA)", "ATM Clustering", "Anomaly Detection"])

# --- FA-2: ACTIONABLE INSIGHTS ---

if page == "Exploratory Data Analysis (EDA)":
    st.header("📈 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Withdrawals by Weather Condition")
        fig_weather = px.box(df, x='Weather_Condition', y='Total_Withdrawals', color='Weather_Condition')
        st.plotly_chart(fig_weather, use_container_width=True)
        
    with col2:
        st.subheader("Cash Demand vs. Previous Day Level")
        fig_scatter = px.scatter(df, x='Previous_Day_Cash_Level', y='Cash_Demand_Next_Day', opacity=0.5)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.subheader("Feature Correlation Heatmap")
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

elif page == "ATM Clustering":
    st.header("🧩 ATM Clustering Analysis")
    st.markdown("Grouping ATMs based on withdrawal/deposit volumes and location data using K-Means.")
    
    # Select features for clustering
    features = ['Total_Withdrawals', 'Total_Deposits', 'Location_Encoded', 'Nearby_Competitor_ATMs']
    X = df[features]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # User selects number of clusters
    k = st.slider("Select number of clusters (K):", min_value=2, max_value=6, value=4)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster'] = df['Cluster'].astype(str) # For better Plotly discrete colors
    
    # Visualize Clusters
    fig_cluster = px.scatter(
        df, x='Total_Withdrawals', y='Total_Deposits', color='Cluster',
        hover_data=['ATM_ID', 'Location_Type'],
        title=f"K-Means Clustering (K={k})"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

elif page == "Anomaly Detection":
    st.header("🚨 Anomaly Detection on Holidays")
    st.markdown("Identifying unusual withdrawal spikes using Isolation Forest.")
    
    # Contamination parameter slider
    contamination = st.slider("Select contamination rate (expected % of anomalies):", 0.01, 0.10, 0.05)
    
    # Train Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(df[['Total_Withdrawals']])
    
    # Map -1 to 'Anomaly' and 1 to 'Normal'
    df['Status'] = df['Anomaly_Score'].map({-1: 'Anomaly', 1: 'Normal'})
    
    # Filter for holidays
    holidays_df = df[df['Holiday_Flag'] == 1]
    
    # Visualization
    fig_anomaly = px.histogram(
        holidays_df, x='Total_Withdrawals', color='Status',
        title="Distribution of Withdrawals on Holidays (Anomalies Highlighted)",
        barmode='overlay', color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'}
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Show data table of just the anomalies
    st.subheader("Detected Anomalies (Holiday Spikes)")
    anomalies_only = holidays_df[holidays_df['Status'] == 'Anomaly']
    st.dataframe(anomalies_only[['ATM_ID', 'Date', 'Location_Type', 'Total_Withdrawals', 'Weather_Condition']])
