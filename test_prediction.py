#!/usr/bin/env python3
"""
Quick test script to verify the trained model can make predictions
"""

import boto3
import joblib
import io
import pandas as pd
import numpy as np
from datetime import datetime

# AWS Configuration
AWS_BUCKET_NAME = 'ml-house-prediction-bucket-2024'
MODEL_KEY = 'models/singapore_hdb_model.pkl'
PREPROCESSOR_KEY = 'models/singapore_preprocessor.pkl'
FEATURE_COLUMNS_KEY = 'models/feature_columns.pkl'

def load_model_components():
    """Load trained model components from S3"""
    s3_client = boto3.client('s3')

    try:
        # Load model
        model_obj = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=MODEL_KEY)
        model = joblib.load(io.BytesIO(model_obj['Body'].read()))

        # Load preprocessor
        preprocessor_obj = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=PREPROCESSOR_KEY)
        preprocessor = joblib.load(io.BytesIO(preprocessor_obj['Body'].read()))

        # Load feature columns
        feature_obj = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=FEATURE_COLUMNS_KEY)
        feature_columns = joblib.load(io.BytesIO(feature_obj['Body'].read()))

        return model, preprocessor, feature_columns

    except Exception as e:
        print(f"❌ Error loading model components: {str(e)}")
        return None, None, None

def make_test_prediction(model, preprocessor, feature_columns):
    """Make a test prediction"""

    # Test data - typical 4-room flat in Tampines
    test_data = {
        'town': 'TAMPINES',
        'flat_type': '4 ROOM',
        'flat_model': 'Model A',
        'floor_area_sqm': 90,
        'storey_range': '07 TO 09',
        'lease_commence_date': 1995,
        'month': '2024-01'
    }

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

    # Create DataFrame
    df = pd.DataFrame([test_data])

    # Feature engineering (same as training)
    current_year = datetime.now().year

    # Remaining lease
    df['remaining_lease'] = 99 - (current_year - df['lease_commence_date'])
    df['remaining_lease'] = np.maximum(df['remaining_lease'], 0)

    # Transaction date features
    df['transaction_year'] = pd.to_datetime(df['month']).dt.year
    df['transaction_month'] = pd.to_datetime(df['month']).dt.month

    # Storey features
    df['storey_mid'] = df['storey_range'].map(storey_ranges).fillna(2)
    df['high_floor'] = (df['storey_mid'] >= 10).astype(int)

    # Town tier
    df['town_tier'] = df['town'].map(town_tiers).fillna(4)

    # Flat type size
    df['flat_type_size'] = df['flat_type'].map(flat_type_size).fillna(3)

    # Lease age and market timing
    df['lease_age_at_transaction'] = df['transaction_year'] - df['lease_commence_date']
    df['covid_period'] = ((df['transaction_year'] >= 2020) &
                         (df['transaction_year'] <= 2021)).astype(int)
    df['recent_transaction'] = (df['transaction_year'] >= 2022).astype(int)

    # Select features and handle missing ones
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]

    # Add missing features with default values
    for col in feature_columns:
        if col not in X.columns:
            if col in ['covid_period', 'recent_transaction', 'high_floor']:
                X[col] = 0
            elif col == 'town_tier':
                X[col] = 3
            elif col == 'flat_type_size':
                X[col] = 4
            elif col == 'storey_mid':
                X[col] = 8
            else:
                X[col] = 0

    # Reorder columns to match training
    X = X[feature_columns]

    # Transform and predict
    X_processed = preprocessor.transform(X)
    prediction = model.predict(X_processed)[0]

    return prediction, test_data

def main():
    print("🧪 Testing Singapore HDB Price Prediction Model")
    print("=" * 50)

    # Load model components
    print("📦 Loading model components from S3...")
    model, preprocessor, feature_columns = load_model_components()

    if model is None:
        print("❌ Failed to load model. Please train the model first:")
        print("   python train_singapore_hdb_model.py")
        return

    print("✅ Model components loaded successfully!")
    print(f"📊 Model type: {type(model).__name__}")
    print(f"🔧 Feature columns: {len(feature_columns)}")

    # Make test prediction
    print("\n🔮 Making test prediction...")
    try:
        prediction, test_data = make_test_prediction(model, preprocessor, feature_columns)

        print("✅ Prediction successful!")
        print("\n📝 Test Property Details:")
        print("-" * 25)
        for key, value in test_data.items():
            print(f"  {key}: {value}")

        print(f"\n💰 Predicted Price: ${prediction:,.0f}")
        print(f"📐 Price per sqm: ${prediction/test_data['floor_area_sqm']:,.0f}")

        # Validate prediction range
        if 200000 <= prediction <= 2000000:  # Reasonable range for Singapore HDB
            print("✅ Prediction within reasonable range")
        else:
            print("⚠️ Prediction outside typical range - please check model")

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        return

    print("\n🎉 Model test completed successfully!")
    print("🚀 Ready to launch Streamlit app:")
    print("   streamlit run singapore_hdb_app.py")

if __name__ == "__main__":
    main()
