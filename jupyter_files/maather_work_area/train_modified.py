import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load and preprocess data
df = pd.read_csv("../../dataset/laptopData.csv")
df.drop(columns='Unnamed: 0', axis=1, inplace=True)

# Train-test split and handle missing values
df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
df_train = df_train.dropna()
df_test = df_test.dropna()

# Convert Memory feature
def convert_memory_size(size):
    if 'TB' in size:
        return float(size.replace('TB', '')) * 1024
    else:
        return float(size.replace('GB', ''))

memory_split_train = df_train['Memory'].str.extract(r'(\d+[GB|TB]+) ([A-Za-z ]+)')
df_train['Memory_Size'] = memory_split_train[0].apply(lambda x: convert_memory_size(x) if pd.notnull(x) else np.nan)
df_train['Memory_Type'] = memory_split_train[1]
df_train.drop('Memory', axis=1, inplace=True)

memory_split_test = df_test['Memory'].str.extract(r'(\d+[GB|TB]+) ([A-Za-z ]+)')
df_test['Memory_Size'] = memory_split_test[0].apply(lambda x: convert_memory_size(x) if pd.notnull(x) else np.nan)
df_test['Memory_Type'] = memory_split_test[1]
df_test.drop('Memory', axis=1, inplace=True)

# Process other features
df_train['Ram'] = df_train['Ram'].str.replace('GB', '').astype(int)
df_test['Ram'] = df_test['Ram'].str.replace('GB', '').astype(int)
df_train['Cpu_Speed'] = df_train['Cpu'].str.extract(r'(\d+\.\d+)GHz').astype(float)
df_test['Cpu_Speed'] = df_test['Cpu'].str.extract(r'(\d+\.\d+)GHz').astype(float)

# Extract more CPU details
df_train['Cpu_Brand'] = df_train['Cpu'].apply(lambda x: x.split()[0])
df_train['Cpu_Series'] = df_train['Cpu'].apply(lambda x: ' '.join(x.split()[1:3]))

df_test['Cpu_Brand'] = df_test['Cpu'].apply(lambda x: x.split()[0])
df_test['Cpu_Series'] = df_test['Cpu'].apply(lambda x: ' '.join(x.split()[1:3]))

# Create interaction terms
df_train['Ram_Memory'] = df_train['Ram'] * df_train['Memory_Size']
df_test['Ram_Memory'] = df_test['Ram'] * df_test['Memory_Size']

# OneHotEncode categorical features
categorical_features = ['Cpu_Brand', 'Cpu_Series', 'Memory_Type']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_train_cat = encoder.fit_transform(df_train[categorical_features])
X_test_cat = encoder.transform(df_test[categorical_features])

df_train_encoded = pd.DataFrame(X_train_cat, columns=encoder.get_feature_names_out(categorical_features))
df_test_encoded = pd.DataFrame(X_test_cat, columns=encoder.get_feature_names_out(categorical_features))

df_train = df_train.drop(categorical_features, axis=1).reset_index(drop=True)
df_test = df_test.drop(categorical_features, axis=1).reset_index(drop=True)

df_train = pd.concat([df_train, df_train_encoded], axis=1)
df_test = pd.concat([df_test, df_test_encoded], axis=1)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(df_train.drop(columns='Price').select_dtypes(include=[np.number]))
X_test_scaled = scaler.transform(df_test.drop(columns='Price').select_dtypes(include=[np.number]))

y_train = df_train['Price'].values
y_test = df_test['Price'].values

# Linear Regression
reg = LinearRegression()
reg.fit(X_train_scaled, y_train)
y_pred = reg.predict(X_test_scaled)
print("Linear Regression R2 Score:", r2_score(y_test, y_pred))

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)
print("Ridge R2 Score:", r2_score(y_test, ridge_pred))

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)
print("Lasso R2 Score:", r2_score(y_test, lasso_pred))

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

reg_poly = LinearRegression()
reg_poly.fit(X_train_poly, y_train)
poly_pred = reg_poly.predict(X_test_poly)
print("Polynomial Regression R2 Score:", r2_score(y_test, poly_pred))
