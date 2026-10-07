import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# 1. Setup Frontend Configuration
st.set_page_config(page_title="Advanced ML Price Predictor", layout="wide")

st.title("🏡 Advanced House Price Predictor")
st.write("Train and compare different machine learning models to predict real estate prices.")

# 2. Load the Dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
        # Clean data: Remove entries with zero price
        df = df[df['price'] > 0]
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("Error: 'data.csv' not found. Please place the dataset in the same directory.")
    st.stop()

# 3. Define Features and Target
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'yr_built', 'condition', 'waterfront', 'view']
X = df[features]
y = df['price']

# 4. Sidebar: ML Library Selection
st.sidebar.header("⚙️ ML Model Settings")
ml_library = st.sidebar.selectbox(
    "Select ML Algorithm",
    (
        "Scikit-Learn: Random Forest", 
        "Scikit-Learn: Gradient Boosting", 
        "Scikit-Learn: Ridge Regression",
        "XGBoost: XGBRegressor",
        "LightGBM: LGBMRegressor"
    )
)

# 5. Backend: Training Pipeline
@st.cache_resource
def train_model(algorithm_choice):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data scaling for linear models and overall stability
    preprocessor = StandardScaler()
    
    # Initialize the selected model
    if algorithm_choice == "Scikit-Learn: Random Forest":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    elif algorithm_choice == "Scikit-Learn: Gradient Boosting":
        regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif algorithm_choice == "Scikit-Learn: Ridge Regression":
        regressor = Ridge(alpha=1.0)
    elif algorithm_choice == "XGBoost: XGBRegressor":
        regressor = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif algorithm_choice == "LightGBM: LGBMRegressor":
        regressor = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        
    # Build complete ML pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate performance metrics
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions)
    }
    
    return model, metrics

# Train the model and get metrics based on dropdown selection
with st.spinner(f"Training {ml_library}..."):
    model, metrics = train_model(ml_library)

# Display real-time evaluation metrics in the sidebar
st.sidebar.subheader("📊 Model Performance")
st.sidebar.write(f"**R² Score:** {metrics['R2 Score']:.4f} (Higher is better)")
st.sidebar.write(f"**MAE:** ${metrics['MAE']:,.2f}")
st.sidebar.write(f"**RMSE:** ${metrics['RMSE']:,.2f}")

# 6. Frontend: Interactive UI
st.subheader("📝 Enter House Details")
col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
    sqft_living = st.number_input("Sqft Living Area", min_value=500, max_value=15000, value=1500)

with col2:
    sqft_lot = st.number_input("Sqft Lot Area", min_value=500, max_value=200000, value=4000)
    floors = st.number_input("Floors", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2026, value=1990)

with col3:
    condition = st.slider("Condition Rating", min_value=1, max_value=5, value=3)
    waterfront = st.selectbox("Waterfront Property?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    view = st.slider("View Quality Score", min_value=0, max_value=4, value=0)

# 7. Execution: Prediction
if st.button("🔮 Predict Price", type="primary"):
    # Format inputs into a Pandas DataFrame for inference
    input_data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "yr_built": [yr_built],
        "condition": [condition],
        "waterfront": [waterfront],
        "view": [view]
    })
    
    # Predict using the trained pipeline
    prediction = model.predict(input_data)[0]
    
    # Ensure prediction is positive
    prediction = max(prediction, 0)
    
    st.success(f"### Estimated Value: ${prediction:,.2f}")
    st.caption(f"Calculated using {ml_library}")