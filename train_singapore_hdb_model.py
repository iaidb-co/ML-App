#!/usr/bin/env python3
"""
Singapore HDB Resale Price Prediction Model Training Script
Run this script to train the model and save it to S3
"""

import pandas as pd
import numpy as np
import boto3
import joblib
import io
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SingaporeHDBPricePredictor:
    def __init__(self, bucket_name='ml-house-prediction-bucket-2024'):
        """
        Initialize Singapore HDB Price Predictor

        Args:
            bucket_name (str): AWS S3 bucket name
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.model = None
        self.preprocessor = None

        # Define feature columns based on typical Singapore HDB dataset
        self.categorical_features = ['town', 'flat_type', 'flat_model', 'storey_range']
        self.numerical_features = ['floor_area_sqm', 'lease_commence_date', 'remaining_lease']

        # Singapore-specific mappings
        self.flat_type_order = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']
        self.storey_ranges = {
            '01 TO 03': 2, '04 TO 06': 5, '07 TO 09': 8, '10 TO 12': 11,
            '13 TO 15': 14, '16 TO 18': 17, '19 TO 21': 20, '22 TO 24': 23,
            '25 TO 27': 26, '28 TO 30': 29, '31 TO 33': 32, '34 TO 36': 35,
            '37 TO 39': 38, '40 TO 42': 41, '43 TO 45': 44, '46 TO 48': 47,
            '49 TO 51': 50
        }

        # Town tiers based on location desirability (affects pricing)
        self.town_tiers = {
            'CENTRAL AREA': 1, 'BISHAN': 1, 'BUKIT TIMAH': 1, 'QUEENSTOWN': 1,
            'TOA PAYOH': 2, 'KALLANG/WHAMPOA': 2, 'GEYLANG': 2, 'MARINE PARADE': 2,
            'CLEMENTI': 2, 'ANG MO KIO': 2, 'BEDOK': 2, 'TAMPINES': 2,
            'HOUGANG': 3, 'JURONG WEST': 3, 'WOODLANDS': 3, 'YISHUN': 3,
            'SENGKANG': 3, 'PUNGGOL': 3, 'PASIR RIS': 3, 'CHOA CHU KANG': 3,
            'BUKIT BATOK': 3, 'BUKIT MERAH': 3, 'BUKIT PANJANG': 3,
            'SERANGOON': 3, 'JURONG EAST': 3, 'SEMBAWANG': 4
        }

    def load_data_from_s3(self, key='data/housing_data.csv'):
        """Load Singapore HDB dataset from S3"""
        try:
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = pd.read_csv(io.BytesIO(obj['Body'].read()))
            logger.info(f"Dataset loaded from S3. Shape: {data.shape}")
            logger.info(f"Columns: {list(data.columns)}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from S3: {str(e)}")
            return None

    def preprocess_singapore_data(self, data):
        """
        Preprocess Singapore HDB data with market-specific features

        Args:
            data (pd.DataFrame): Raw Singapore HDB dataset

        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        logger.info("Preprocessing Singapore HDB data...")
        df = data.copy()

        # Clean column names (remove spaces, standardize)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Handle missing values
        df = df.dropna(subset=['resale_price'])

        # Data type conversions
        df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
        df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
        df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')

        # Feature engineering for Singapore market
        current_year = datetime.now().year

        # 1. Parse remaining lease from string format "XX years YY months"
        def parse_remaining_lease(lease_str):
            """Parse remaining lease from string format like '78 years 02 months'"""
            if pd.isna(lease_str):
                return np.nan

            # If it's already numeric, return as is
            try:
                return float(lease_str)
            except (ValueError, TypeError):
                pass

            # Parse string format
            lease_str = str(lease_str).lower()
            years = 0
            months = 0

            # Extract years
            if 'years' in lease_str or 'year' in lease_str:
                year_part = lease_str.split('years')[0] if 'years' in lease_str else lease_str.split('year')[0]
                try:
                    years = int(year_part.strip())
                except (ValueError, AttributeError):
                    years = 0

            # Extract months
            if 'months' in lease_str or 'month' in lease_str:
                if 'years' in lease_str or 'year' in lease_str:
                    month_part = lease_str.split('years')[-1] if 'years' in lease_str else lease_str.split('year')[-1]
                else:
                    month_part = lease_str

                month_part = month_part.replace('months', '').replace('month', '').strip()
                try:
                    months = int(month_part)
                except (ValueError, AttributeError):
                    months = 0

            # Convert to years (decimal)
            total_years = years + (months / 12.0)
            return total_years

        # Apply remaining lease parsing
        if 'remaining_lease' in df.columns:
            df['remaining_lease_parsed'] = df['remaining_lease'].apply(parse_remaining_lease)
            # If parsing failed, calculate from lease_commence_date
            mask_missing = pd.isna(df['remaining_lease_parsed'])
            df.loc[mask_missing, 'remaining_lease_parsed'] = np.maximum(
                99 - (current_year - df.loc[mask_missing, 'lease_commence_date']), 0
            )
            df['remaining_lease'] = df['remaining_lease_parsed']
            df = df.drop('remaining_lease_parsed', axis=1)
        else:
            # Calculate remaining lease if column doesn't exist
            df['remaining_lease'] = 99 - (current_year - df['lease_commence_date'])
            df['remaining_lease'] = np.maximum(df['remaining_lease'], 0)

        # 2. Extract year and month from transaction date
        if 'month' in df.columns:
            df['transaction_year'] = pd.to_datetime(df['month']).dt.year
            df['transaction_month'] = pd.to_datetime(df['month']).dt.month

        # 3. Storey range conversion to numerical
        if 'storey_range' in df.columns:
            df['storey_mid'] = df['storey_range'].map(self.storey_ranges).fillna(2)
            df['high_floor'] = (df['storey_mid'] >= 10).astype(int)

        # 4. Town tier (location desirability)
        if 'town' in df.columns:
            df['town_tier'] = df['town'].map(self.town_tiers).fillna(4)

        # 5. Flat type size encoding
        if 'flat_type' in df.columns:
            flat_type_size = {
                '1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3,
                '4 ROOM': 4, '5 ROOM': 5, 'EXECUTIVE': 6
            }
            df['flat_type_size'] = df['flat_type'].map(flat_type_size).fillna(3)

        # 6. Price per sqm for analysis
        df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']

        # 7. Lease age at transaction
        if 'transaction_year' in df.columns:
            df['lease_age_at_transaction'] = df['transaction_year'] - df['lease_commence_date']

        # 8. Market timing features (Singapore-specific trends)
        if 'transaction_year' in df.columns:
            # COVID impact (2020-2021)
            df['covid_period'] = ((df['transaction_year'] >= 2020) &
                                (df['transaction_year'] <= 2021)).astype(int)

            # Recent years (higher prices)
            df['recent_transaction'] = (df['transaction_year'] >= 2022).astype(int)

        # Remove outliers and handle missing values
        # Drop rows with missing critical values
        critical_cols = ['resale_price', 'floor_area_sqm', 'lease_commence_date', 'remaining_lease']
        for col in critical_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])

        # Ensure remaining_lease is numeric
        df['remaining_lease'] = pd.to_numeric(df['remaining_lease'], errors='coerce')
        df = df.dropna(subset=['remaining_lease'])

        # Remove outliers (market-specific filters)
        # Price outliers (very cheap/expensive)
        price_q1, price_q99 = df['resale_price'].quantile([0.01, 0.99])
        df = df[(df['resale_price'] >= price_q1) & (df['resale_price'] <= price_q99)]

        # Floor area outliers
        area_q1, area_q99 = df['floor_area_sqm'].quantile([0.01, 0.99])
        df = df[(df['floor_area_sqm'] >= area_q1) & (df['floor_area_sqm'] <= area_q99)]

        logger.info(f"Data preprocessing complete. Final shape: {df.shape}")
        logger.info(f"Price range: ${df['resale_price'].min():,.0f} - ${df['resale_price'].max():,.0f}")

        return df

    def create_preprocessor(self, df):
        """Create preprocessing pipeline for features"""

        # Update feature lists based on available columns
        available_categorical = [col for col in self.categorical_features if col in df.columns]

        available_numerical = [col for col in self.numerical_features if col in df.columns]

        # Add engineered features to numerical
        engineered_numerical = ['storey_mid', 'town_tier', 'flat_type_size',
                              'lease_age_at_transaction', 'covid_period', 'recent_transaction']
        available_numerical.extend([col for col in engineered_numerical if col in df.columns])

        # Create preprocessing pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), available_numerical),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), available_categorical)
            ],
            remainder='drop'
        )

        return preprocessor, available_numerical, available_categorical

    def train_model(self, data, test_size=0.2):
        """
        Train Singapore HDB price prediction model

        Args:
            data (pd.DataFrame): Preprocessed dataset
            test_size (float): Test set proportion

        Returns:
            dict: Training results and metrics
        """
        logger.info("Training Singapore HDB price prediction model...")

        # Preprocess data
        df = self.preprocess_singapore_data(data)

        # Create preprocessor
        self.preprocessor, numerical_features, categorical_features = self.create_preprocessor(df)

        # Prepare features and target
        feature_columns = numerical_features + categorical_features
        available_features = [col for col in feature_columns if col in df.columns]

        X = df[available_features]
        y = df['resale_price']

        logger.info(f"Using features: {available_features}")
        logger.info(f"Target variable range: ${y.min():,.0f} - ${y.max():,.0f}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )

        # Preprocess features
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Train multiple models and select best
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        }

        best_model = None
        best_score = -np.inf
        model_results = {}

        for name, model in models.items():
            logger.info(f"Training {name}...")

            # Train model
            model.fit(X_train_processed, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train_processed)
            y_pred_test = model.predict(X_test_processed)

            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100

            model_results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae,
                'mape': test_mape
            }

            logger.info(f"{name} - Test R²: {test_r2:.4f}, RMSE: ${test_rmse:,.0f}, MAPE: {test_mape:.2f}%")

            if test_r2 > best_score:
                best_score = test_r2
                best_model = model
                self.model = model

        # Store feature columns for prediction
        self.feature_columns = available_features

        # Cross-validation on best model
        cv_scores = cross_val_score(best_model, X_train_processed, y_train, cv=5, scoring='r2')

        logger.info(f"Best model: {type(best_model).__name__}")
        logger.info(f"Cross-validation R² scores: {cv_scores}")
        logger.info(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Feature importance analysis
        if hasattr(best_model, 'feature_importances_'):
            feature_names = (numerical_features +
                           list(self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("Top 10 most important features:")
            logger.info(feature_importance.head(10))

        # Prepare results
        results = {
            'model': best_model,
            'preprocessor': self.preprocessor,
            'feature_columns': available_features,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': best_model.predict(X_test_processed),
            'model_results': model_results,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance if hasattr(best_model, 'feature_importances_') else None
        }

        return results

    def save_model_to_s3(self, model_key='models/singapore_hdb_model.pkl',
                        preprocessor_key='models/singapore_preprocessor.pkl',
                        feature_columns_key='models/feature_columns.pkl'):
        """Save trained model, preprocessor, and feature columns to S3"""
        try:
            # Save model
            if self.model is not None:
                model_buffer = io.BytesIO()
                joblib.dump(self.model, model_buffer)
                model_buffer.seek(0)

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=model_key,
                    Body=model_buffer.getvalue()
                )
                logger.info(f"Model saved to s3://{self.bucket_name}/{model_key}")

            # Save preprocessor
            if self.preprocessor is not None:
                preprocessor_buffer = io.BytesIO()
                joblib.dump(self.preprocessor, preprocessor_buffer)
                preprocessor_buffer.seek(0)

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=preprocessor_key,
                    Body=preprocessor_buffer.getvalue()
                )
                logger.info(f"Preprocessor saved to s3://{self.bucket_name}/{preprocessor_key}")

            # Save feature columns
            if hasattr(self, 'feature_columns'):
                feature_buffer = io.BytesIO()
                joblib.dump(self.feature_columns, feature_buffer)
                feature_buffer.seek(0)

                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=feature_columns_key,
                    Body=feature_buffer.getvalue()
                )
                logger.info(f"Feature columns saved to s3://{self.bucket_name}/{feature_columns_key}")

            return True

        except Exception as e:
            logger.error(f"Error saving model to S3: {str(e)}")
            return False

def main():
    """Main function to train and save Singapore HDB price prediction model"""

    print("🏠 Singapore HDB Price Prediction Model Training")
    print("=" * 50)

    # Initialize predictor
    predictor = SingaporeHDBPricePredictor()

    # Load data from S3
    print("📊 Loading data from S3...")
    data = predictor.load_data_from_s3()

    if data is not None:
        print(f"✅ Data loaded successfully! Shape: {data.shape}")
        print("\n🤖 Starting model training...")

        # Train model
        results = predictor.train_model(data)

        print("\n💾 Saving model to S3...")
        # Save model to S3
        success = predictor.save_model_to_s3()

        if success:
            print("✅ Model, preprocessor, and feature columns saved successfully to S3!")
            print("\n📈 Training Summary:")
            print("-" * 30)

            best_model_name = None
            best_r2 = -1

            for model_name, metrics in results['model_results'].items():
                print(f"{model_name}:")
                print(f"  - Test R²: {metrics['test_r2']:.4f}")
                print(f"  - RMSE: ${metrics['rmse']:,.0f}")
                print(f"  - MAE: ${metrics['mae']:,.0f}")
                print(f"  - MAPE: {metrics['mape']:.2f}%")

                if metrics['test_r2'] > best_r2:
                    best_r2 = metrics['test_r2']
                    best_model_name = model_name

            print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")
            print(f"📊 Cross-validation mean R²: {results['cv_scores'].mean():.4f}")

            print("\n🚀 Ready for predictions! You can now run the Streamlit app.")

        else:
            print("❌ Failed to save model to S3")

    else:
        print("❌ Failed to load data from S3")

if __name__ == "__main__":
    main()
