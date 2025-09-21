import streamlit as st
import pandas as pd
import numpy as np
import boto3
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# AWS Configuration
AWS_BUCKET_NAME = 'ml-house-prediction-bucket-2024'
MODEL_KEY = 'models/singapore_hdb_model.pkl'
PREPROCESSOR_KEY = 'models/singapore_preprocessor.pkl'
FEATURE_COLUMNS_KEY = 'models/feature_columns.pkl'
DATA_KEY = 'data/housing_data.csv'

class AWSManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')

    def load_data_from_s3(self, bucket_name, key):
        """Load data from S3 bucket"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            return pd.read_csv(io.BytesIO(obj['Body'].read()))
        except Exception as e:
            st.error(f"Error loading data from S3: {str(e)}")
            return None

    def load_model_from_s3(self, bucket_name, key):
        """Load trained model from S3"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            return joblib.load(io.BytesIO(obj['Body'].read()))
        except Exception as e:
            st.error(f"Error loading model from S3: {str(e)}")
            return None

def preprocess_singapore_data(data):
    """Preprocess Singapore HDB data for display and analysis"""
    df = data.copy()

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Data type conversions
    if 'resale_price' in df.columns:
        df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
    if 'floor_area_sqm' in df.columns:
        df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
    if 'lease_commence_date' in df.columns:
        df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')

    # Feature engineering
    current_year = datetime.now().year
    if 'lease_commence_date' in df.columns:
        df['remaining_lease'] = 99 - (current_year - df['lease_commence_date'])
        df['remaining_lease'] = np.maximum(df['remaining_lease'], 0)

    if 'month' in df.columns:
        df['transaction_year'] = pd.to_datetime(df['month']).dt.year
        df['transaction_month'] = pd.to_datetime(df['month']).dt.month

    # Price per sqm
    if 'resale_price' in df.columns and 'floor_area_sqm' in df.columns:
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']

    return df

def predict_price(model, preprocessor, feature_columns, input_data):
    """Make price prediction for a single HDB flat"""

    # Singapore-specific mappings for feature engineering
    storey_ranges = {
        '01 TO 03': 2, '04 TO 06': 5, '07 TO 09': 8, '10 TO 12': 11,
        '13 TO 15': 14, '16 TO 18': 17, '19 TO 21': 20, '22 TO 24': 23,
        '25 TO 27': 26, '28 TO 30': 29, '31 TO 33': 32, '34 TO 36': 35,
        '37 TO 39': 38, '40 TO 42': 41, '43 TO 45': 44, '46 TO 48': 47,
        '49 TO 51': 50
    }

    town_tiers = {
        'CENTRAL AREA': 1, 'BISHAN': 1, 'BUKIT TIMAH': 1, 'QUEENSTOWN': 1,
        'TOA PAYOH': 2, 'KALLANG/WHAMPOA': 2, 'GEYLANG': 2, 'MARINE PARADE': 2,
        'CLEMENTI': 2, 'ANG MO KIO': 2, 'BEDOK': 2, 'TAMPINES': 2,
        'HOUGANG': 3, 'JURONG WEST': 3, 'WOODLANDS': 3, 'YISHUN': 3,
        'SENGKANG': 3, 'PUNGGOL': 3, 'PASIR RIS': 3, 'CHOA CHU KANG': 3,
        'BUKIT BATOK': 3, 'BUKIT MERAH': 3, 'BUKIT PANJANG': 3,
        'SERANGOON': 3, 'JURONG EAST': 3, 'SEMBAWANG': 4
    }

    flat_type_size = {
        '1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3,
        '4 ROOM': 4, '5 ROOM': 5, 'EXECUTIVE': 6
    }

    # Create DataFrame from input
    df = pd.DataFrame([input_data])

    # Feature engineering (same as training)
    current_year = datetime.now().year

    if 'remaining_lease' not in df.columns:
        df['remaining_lease'] = 99 - (current_year - df['lease_commence_date'])
        df['remaining_lease'] = np.maximum(df['remaining_lease'], 0)

    if 'month' in df.columns:
        df['transaction_year'] = pd.to_datetime(df['month']).dt.year
        df['transaction_month'] = pd.to_datetime(df['month']).dt.month
    else:
        df['transaction_year'] = current_year
        df['transaction_month'] = datetime.now().month

    if 'storey_range' in df.columns:
        df['storey_mid'] = df['storey_range'].map(storey_ranges).fillna(2)
        df['high_floor'] = (df['storey_mid'] >= 10).astype(int)

    if 'town' in df.columns:
        df['town_tier'] = df['town'].map(town_tiers).fillna(4)

    if 'flat_type' in df.columns:
        df['flat_type_size'] = df['flat_type'].map(flat_type_size).fillna(3)

    if 'transaction_year' in df.columns:
        df['lease_age_at_transaction'] = df['transaction_year'] - df['lease_commence_date']
        df['covid_period'] = ((df['transaction_year'] >= 2020) &
                            (df['transaction_year'] <= 2021)).astype(int)
        df['recent_transaction'] = (df['transaction_year'] >= 2022).astype(int)

    # Select only the features used in training
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]

    # Handle missing columns by adding them with default values
    for col in feature_columns:
        if col not in X.columns:
            if col in ['covid_period', 'recent_transaction', 'high_floor']:
                X[col] = 0
            elif col in ['town_tier']:
                X[col] = 3
            elif col in ['flat_type_size']:
                X[col] = 4
            elif col in ['storey_mid']:
                X[col] = 8
            else:
                X[col] = 0

    # Reorder columns to match training
    X = X[feature_columns]

    # Transform and predict
    X_processed = preprocessor.transform(X)
    prediction = model.predict(X_processed)[0]

    return prediction

def main():
    st.set_page_config(
        page_title="Singapore HDB Resale Price Prediction",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏠 Singapore HDB Resale Price Prediction")
    st.markdown("*Powered by Machine Learning | Data from HDB Official Records (2017 onwards)*")
    st.markdown("---")

    # Initialize AWS Manager
    aws_manager = AWSManager()

    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "🏡 Home Dashboard",
        "📊 Market Analytics",
        "🔮 Price Prediction",
        "📈 Trends Analysis",
        "🏘️ Town Comparison"
    ])

    # Load data and models once and cache
    @st.cache_data
    def load_singapore_data():
        data = aws_manager.load_data_from_s3(AWS_BUCKET_NAME, DATA_KEY)
        if data is None:
            st.warning("Could not load data from S3. Please ensure the data file exists.")
            return None
        return preprocess_singapore_data(data)

    @st.cache_resource
    def load_model_components():
        model = aws_manager.load_model_from_s3(AWS_BUCKET_NAME, MODEL_KEY)
        preprocessor = aws_manager.load_model_from_s3(AWS_BUCKET_NAME, PREPROCESSOR_KEY)
        feature_columns = aws_manager.load_model_from_s3(AWS_BUCKET_NAME, FEATURE_COLUMNS_KEY)
        return model, preprocessor, feature_columns

    # Load data
    data = load_singapore_data()
    model, preprocessor, feature_columns = load_model_components()

    if data is None:
        st.error("❌ Failed to load data. Please check your AWS configuration.")
        st.stop()

    if model is None or preprocessor is None or feature_columns is None:
        st.error("❌ Failed to load model components. Please train the model first.")
        st.info("Run: `python train_singapore_hdb_model.py` to train and save the model.")
        st.stop()

    if page == "🏡 Home Dashboard":
        st.header("Singapore HDB Resale Market Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_price = data['resale_price'].mean()
            st.metric("Average Price", f"${avg_price:,.0f}")

        with col2:
            median_price = data['resale_price'].median()
            st.metric("Median Price", f"${median_price:,.0f}")

        with col3:
            total_transactions = len(data)
            st.metric("Total Transactions", f"{total_transactions:,}")

        with col4:
            avg_psm = data['price_per_sqm'].mean() if 'price_per_sqm' in data.columns else 0
            st.metric("Avg Price/sqm", f"${avg_psm:,.0f}")

        # Price distribution
        st.subheader("📊 Price Distribution")
        fig_dist = px.histogram(data, x='resale_price', nbins=50,
                               title="Distribution of Resale Prices")
        fig_dist.update_layout(xaxis_title="Resale Price (SGD)", yaxis_title="Frequency")
        st.plotly_chart(fig_dist, use_container_width=True)

        # Recent trends
        if 'transaction_year' in data.columns:
            st.subheader("📈 Price Trends Over Time")
            yearly_avg = data.groupby('transaction_year')['resale_price'].mean().reset_index()
            fig_trend = px.line(yearly_avg, x='transaction_year', y='resale_price',
                               title="Average Resale Price by Year")
            fig_trend.update_layout(xaxis_title="Year", yaxis_title="Average Price (SGD)")
            st.plotly_chart(fig_trend, use_container_width=True)

    elif page == "📊 Market Analytics":
        st.header("Market Analytics Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            # Price by flat type
            st.subheader("💰 Average Price by Flat Type")
            if 'flat_type' in data.columns:
                price_by_type = data.groupby('flat_type')['resale_price'].mean().sort_values(ascending=True)
                fig_type = px.bar(x=price_by_type.values, y=price_by_type.index,
                                 orientation='h', title="Average Price by Flat Type")
                fig_type.update_layout(xaxis_title="Average Price (SGD)", yaxis_title="Flat Type")
                st.plotly_chart(fig_type, use_container_width=True)

        with col2:
            # Price by town (top 15)
            st.subheader("🏘️ Top 15 Most Expensive Towns")
            if 'town' in data.columns:
                price_by_town = data.groupby('town')['resale_price'].mean().sort_values(ascending=False).head(15)
                fig_town = px.bar(x=price_by_town.values, y=price_by_town.index,
                                 orientation='h', title="Average Price by Town (Top 15)")
                fig_town.update_layout(xaxis_title="Average Price (SGD)", yaxis_title="Town")
                st.plotly_chart(fig_town, use_container_width=True)

        # Correlation heatmap
        st.subheader("🔥 Feature Correlations")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                   title="Feature Correlation Matrix")
            st.plotly_chart(fig_heatmap, use_container_width=True)

    elif page == "🔮 Price Prediction":
        st.header("HDB Resale Price Prediction")
        st.markdown("Enter the details of your HDB flat to get a price prediction:")

        col1, col2 = st.columns(2)

        with col1:
            # Get unique values from data for dropdowns
            towns = sorted(data['town'].unique()) if 'town' in data.columns else ['TAMPINES']
            flat_types = sorted(data['flat_type'].unique()) if 'flat_type' in data.columns else ['4 ROOM']
            flat_models = sorted(data['flat_model'].unique()) if 'flat_model' in data.columns else ['Model A']
            storey_ranges = sorted(data['storey_range'].unique()) if 'storey_range' in data.columns else ['07 TO 09']

            # Input fields
            town = st.selectbox("🏘️ Town", towns)
            flat_type = st.selectbox("🏠 Flat Type", flat_types)
            flat_model = st.selectbox("🏗️ Flat Model", flat_models)
            floor_area = st.number_input("📐 Floor Area (sqm)", min_value=20, max_value=300, value=90)

        with col2:
            storey_range = st.selectbox("🏢 Storey Range", storey_ranges)
            lease_commence_date = st.number_input("📅 Lease Commence Date",
                                                 min_value=1960, max_value=2024, value=1990)
            # Generate month options for current and previous year
            current_year = datetime.now().year
            current_month = datetime.now().month

            month_options = []
            # Add current year months up to current month
            for i in range(1, current_month + 1):
                month_options.append(f"{current_year}-{i:02d}")
            # Add previous year months
            for i in range(1, 13):
                month_options.append(f"{current_year-1}-{i:02d}")

            # Sort in reverse order (most recent first)
            month_options.sort(reverse=True)

            transaction_month = st.selectbox("📆 Transaction Month",
                                           month_options, index=0)

        # Prediction button
        if st.button("🔮 Predict Price", type="primary"):
            try:
                # Prepare input data
                input_data = {
                    'town': town,
                    'flat_type': flat_type,
                    'flat_model': flat_model,
                    'floor_area_sqm': floor_area,
                    'storey_range': storey_range,
                    'lease_commence_date': lease_commence_date,
                    'month': transaction_month
                }

                # Make prediction
                predicted_price = predict_price(model, preprocessor, feature_columns, input_data)

                # Display results
                st.success("✅ Prediction Complete!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Price", f"${predicted_price:,.0f}")
                with col2:
                    price_per_sqm = predicted_price / floor_area
                    st.metric("Price per sqm", f"${price_per_sqm:,.0f}")
                with col3:
                    # Find similar properties for comparison
                    similar = data[(data['town'] == town) & (data['flat_type'] == flat_type)]
                    if len(similar) > 0:
                        market_avg = similar['resale_price'].mean()
                        difference = ((predicted_price - market_avg) / market_avg) * 100
                        st.metric("vs Market Avg", f"{difference:+.1f}%")

                # Additional insights
                st.subheader("📊 Price Insights")

                # Market comparison
                if len(similar) > 0:
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Histogram(x=similar['resale_price'],
                                                         name='Similar Properties',
                                                         opacity=0.7))
                    fig_comparison.add_vline(x=predicted_price, line_dash="dash",
                                           line_color="red", annotation_text="Predicted Price")
                    fig_comparison.update_layout(title=f"Price Comparison: {flat_type} in {town}",
                                               xaxis_title="Price (SGD)", yaxis_title="Frequency")
                    st.plotly_chart(fig_comparison, use_container_width=True)

                # Show input summary
                st.subheader("📝 Input Summary")
                summary_df = pd.DataFrame([{
                    'Property': f"{flat_type} in {town}",
                    'Floor Area': f"{floor_area} sqm",
                    'Storey': storey_range,
                    'Lease Start': lease_commence_date,
                    'Remaining Lease': f"{99 - (2024 - lease_commence_date)} years",
                    'Predicted Price': f"${predicted_price:,.0f}"
                }])
                st.dataframe(summary_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please check your inputs and try again.")

    elif page == "📈 Trends Analysis":
        st.header("Market Trends Analysis")

        if 'transaction_year' in data.columns:
            # Price trends by flat type
            st.subheader("📊 Price Trends by Flat Type")
            trend_data = data.groupby(['transaction_year', 'flat_type'])['resale_price'].mean().reset_index()
            fig_trends = px.line(trend_data, x='transaction_year', y='resale_price',
                               color='flat_type', title="Average Price Trends by Flat Type")
            fig_trends.update_layout(xaxis_title="Year", yaxis_title="Average Price (SGD)")
            st.plotly_chart(fig_trends, use_container_width=True)

            # Volume trends
            st.subheader("📈 Transaction Volume Trends")
            volume_data = data.groupby('transaction_year').size().reset_index(name='transactions')
            fig_volume = px.bar(volume_data, x='transaction_year', y='transactions',
                              title="Number of Transactions by Year")
            fig_volume.update_layout(xaxis_title="Year", yaxis_title="Number of Transactions")
            st.plotly_chart(fig_volume, use_container_width=True)

        # Price per sqm analysis
        if 'price_per_sqm' in data.columns:
            st.subheader("💰 Price per Square Meter Analysis")
            col1, col2 = st.columns(2)

            with col1:
                psm_by_type = data.groupby('flat_type')['price_per_sqm'].mean().sort_values(ascending=True)
                fig_psm_type = px.bar(x=psm_by_type.values, y=psm_by_type.index,
                                     orientation='h', title="Average Price/sqm by Flat Type")
                st.plotly_chart(fig_psm_type, use_container_width=True)

            with col2:
                if 'transaction_year' in data.columns:
                    psm_trend = data.groupby('transaction_year')['price_per_sqm'].mean().reset_index()
                    fig_psm_trend = px.line(psm_trend, x='transaction_year', y='price_per_sqm',
                                           title="Price/sqm Trend Over Time")
                    st.plotly_chart(fig_psm_trend, use_container_width=True)

    elif page == "🏘️ Town Comparison":
        st.header("Town-by-Town Comparison")

        # Select towns to compare
        available_towns = sorted(data['town'].unique()) if 'town' in data.columns else []
        selected_towns = st.multiselect("Select Towns to Compare",
                                       available_towns,
                                       default=available_towns[:5] if len(available_towns) >= 5 else available_towns)

        if selected_towns:
            comparison_data = data[data['town'].isin(selected_towns)]

            # Price comparison
            st.subheader("💰 Price Comparison")
            town_stats = comparison_data.groupby('town')['resale_price'].agg(['mean', 'median', 'std']).reset_index()
            town_stats.columns = ['Town', 'Average Price', 'Median Price', 'Price Std Dev']
            town_stats = town_stats.round(0)
            st.dataframe(town_stats, use_container_width=True)

            # Box plot comparison
            fig_box = px.box(comparison_data, x='town', y='resale_price',
                           title="Price Distribution by Town")
            fig_box.update_xaxis(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)

            # Transaction volume comparison
            st.subheader("📊 Transaction Volume")
            volume_comparison = comparison_data.groupby('town').size().reset_index(name='transactions')
            fig_volume_comp = px.bar(volume_comparison, x='town', y='transactions',
                                   title="Number of Transactions by Town")
            fig_volume_comp.update_xaxis(tickangle=45)
            st.plotly_chart(fig_volume_comp, use_container_width=True)
        else:
            st.info("Please select towns to compare.")

    # Footer
    st.markdown("---")
    st.markdown("*Data sourced from HDB official records. Predictions are for reference only.*")

if __name__ == "__main__":
    main()
