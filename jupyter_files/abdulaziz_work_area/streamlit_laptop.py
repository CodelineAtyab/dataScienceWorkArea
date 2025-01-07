import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

st.title('Laptop Price Prediction')

# Upload the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview:")
    st.write(data.head())

    # Selecting relevant features and target
    features = ['Company', 'TypeName', 'Ram', 'Weight', 'Cpu', 'Gpu', 'ScreenResolution']
    target = 'Price'
    
    data = data.dropna(subset=[target])  # Dropping rows with missing target values

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure all values in 'Cpu' column are strings and fill missing values
    X_train['Cpu'] = X_train['Cpu'].astype(str).fillna('Unknown')
    X_test['Cpu'] = X_test['Cpu'].astype(str).fillna('Unknown')

    # Extract CPU, GPU, and Screen features
    def extract_cpu_features(cpu_info):
        parts = cpu_info.split()
        if len(parts) < 3:
            return pd.Series(['Unknown', 0])
        brand = parts[0] + ' ' + parts[1]
        try:
            speed = float(parts[-1].replace('GHz', ''))
        except ValueError:
            speed = 0
        return pd.Series([brand, speed])

    def extract_gpu_brand(gpu_info):
        if not isinstance(gpu_info, str) or len(gpu_info.split()) < 1:
            return 'Unknown'
        return gpu_info.split()[0]

    def extract_screen_resolution(screen_info):
        if not isinstance(screen_info, str) or 'x' not in screen_info:
            return pd.Series([0, 0])
        resolution = screen_info.split()[-1]
        width, height = resolution.split('x')
        return pd.Series([int(width), int(height)])

    # Apply the functions to the respective columns
    cpu_features_train = X_train['Cpu'].apply(extract_cpu_features)
    cpu_features_train.columns = ['Cpu_Brand', 'Cpu_Speed']
    cpu_features_test = X_test['Cpu'].apply(extract_cpu_features)
    cpu_features_test.columns = ['Cpu_Brand', 'Cpu_Speed']

    gpu_features_train = X_train['Gpu'].apply(extract_gpu_brand).rename("Gpu_Brand")
    gpu_features_test = X_test['Gpu'].apply(extract_gpu_brand).rename("Gpu_Brand")

    screen_resolution_features_train = X_train['ScreenResolution'].apply(extract_screen_resolution)
    screen_resolution_features_train.columns = ['Screen_Width', 'Screen_Height']
    screen_resolution_features_test = X_test['ScreenResolution'].apply(extract_screen_resolution)
    screen_resolution_features_test.columns = ['Screen_Width', 'Screen_Height']

    # Concatenate the new features with the original data
    X_train = pd.concat([X_train, cpu_features_train, gpu_features_train, screen_resolution_features_train], axis=1)
    X_test = pd.concat([X_test, cpu_features_test, gpu_features_test, screen_resolution_features_test], axis=1)
    
    # Drop the original columns after feature extraction
    X_train = X_train.drop(columns=['Cpu', 'Gpu', 'ScreenResolution'])
    X_test = X_test.drop(columns=['Cpu', 'Gpu', 'ScreenResolution'])

    updated_features = ['Company', 'TypeName', 'Ram', 'Weight', 'Cpu_Brand', 'Cpu_Speed', 'Gpu_Brand', 'Screen_Width', 'Screen_Height']

    # Label encoding for categorical features
    label_encoders = {}
    for feature in updated_features:
        if X_train[feature].dtype == 'object':
            le = LabelEncoder()
            combined_data = pd.concat([X_train[feature], X_test[feature]], axis=0)
            le.fit(combined_data)
            X_train[feature] = le.transform(X_train[feature])
            X_test[feature] = le.transform(X_test[feature])
            label_encoders[feature] = le

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply Linear Regression
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred_linear = linear_reg.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    # Apply Ridge Regression
    ridge_reg = Ridge()
    ridge_reg.fit(X_train, y_train)
    y_pred_ridge = ridge_reg.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)

    st.write("Linear Regression: MSE =", mse_linear, ", R2 =", r2_linear)
    st.write("Ridge Regression: MSE =", mse_ridge, ", R2 =", r2_ridge)
