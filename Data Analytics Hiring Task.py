import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the dataset from Excel
df = pd.read_excel("demo_battery_data.xlsx")

# Inspect the dataset
print(df.head())
print(df.info())
print(df.describe())

# Correlation analysis
print(df.corr()['Effective SOC'].sort_values(ascending=False))

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Distribution of Effective SOC
sns.histplot(df['Effective SOC'], kde=True, bins=20)
plt.title('Distribution of Effective SOC')
plt.show()

# Scatter plot with regression line
sns.regplot(x='Fixed Battery Voltage', y='Effective SOC', data=df, line_kws={"color": "red"})
plt.title('Fixed Battery Voltage vs Effective SOC')
plt.show()

# Check for missing values
print(df.isnull().sum())

# Fill missing values with column means
df.fillna(df.mean(numeric_only=True), inplace=True)

# Create a new feature: Voltage Difference
df['Voltage Difference'] = df['Fixed Battery Voltage'] - df['Portable Battery Voltage']

# Normalize temperatures
df['Normalized Portable Temp'] = (
    df['Portable Battery Temperatures'] - df['Portable Battery Temperatures'].mean()
) / df['Portable Battery Temperatures'].std()

# Define features and target variable
X = df.drop(['Effective SOC'], axis=1)  # Features
y = df['Effective SOC']  # Target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR2 Score: {r2:.2f}")
print(f"Mean Squared Error: {mse:.2f}\n")

# Updated GridSearchCV without 'normalize' parameter
param_grid = {'fit_intercept': [True, False], 'positive': [True, False]}
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='r2')

# Run GridSearchCV
grid_search.fit(X_train_scaled, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

# Key Performance Indicators (KPIs)
print("\nKey Performance Indicators")

# KPI 1: Charge Cycles
charge_cycles = (df['Effective SOC'] < 10).sum()
print(f"Charge Cycles: {charge_cycles}")

# KPI 2: Average Effective SOC
avg_soc = df['Effective SOC'].mean()
print(f"Average Effective SOC: {avg_soc:.2f}")

# KPI 3: Impact of Temperature on SOC
temp_impact = df[['Portable Battery Temperatures', 'Effective SOC']].corr().iloc[0, 1]
print(f"Temperature Impact on SOC: {temp_impact:.2f}")

# KPI 4: Battery Range (Max - Min SOC)
max_soc = df['Effective SOC'].max()
min_soc = df['Effective SOC'].min()
battery_range = max_soc - min_soc
print(f"Battery Range: {battery_range:.2f}")

# KPI 5: Average Fixed Battery Voltage
avg_fixed_voltage = df['Fixed Battery Voltage'].mean()
print(f"Average Fixed Battery Voltage: {avg_fixed_voltage:.2f} V")

# KPI 6: Impact of Fixed Battery Temperature on SOC
fixed_temp_impact = df[['Fixed Battery Temperatures', 'Effective SOC']].corr().iloc[0, 1]
print(f"Fixed Battery Temperature Impact on SOC: {fixed_temp_impact:.2f}")

# KPI 7: BCM Battery Selected Percentage
bcm_selected_ratio = (df['BCM Battery Selected'].sum() / len(df)) * 100
print(f"BCM Battery Selected Percentage: {bcm_selected_ratio:.2f}%")

# KPI 8: Motor ON Percentage
motor_on_percentage = (df['Motor Status (On/Off)'].sum() / len(df)) * 100
print(f"Motor ON Percentage: {motor_on_percentage:.2f}%")
