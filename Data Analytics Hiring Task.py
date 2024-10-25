

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Load the dataset from Excel
df = pd.read_excel("Challenge_dataset.csv")

# Inspect the dataset

print(df.head())
print(df.info())
print(df.describe)

print(df.corr()['Effective SOC'].sort_values(ascending=False))

# Check for correlations using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plot the distribution of the target variable (Effective SOC)
sns.histplot(df['Effective SOC'], kde=True, bins=20)
plt.title('Distribution of Effective SOC')
plt.show()

# Scatter plot to see the relationship between voltage and Effective SOC
sns.scatterplot(x='Fixed Battery Voltage', y='Effective SOC', data=df)
plt.title('Fixed Battery Voltage vs Effective SOC')
plt.show()

# Check for missing values
print(df.isnull().sum())

# Fill any missing values with the mean of their respective columns
df.fillna(df.mean(), inplace=True)

# Create a new feature: Voltage Difference
df['Voltage Difference'] = df['Fixed Battery Voltage'] - df['Portable Battery Voltage']

# Normalize temperatures for better model performance (optional)
df['Normalized Portable Temp'] = (df['Portable Battery Temperatures'] - df['Portable Battery Temperatures'].mean()) / df['Portable Battery Temperatures'].std()

# Define features (X) and target (y)
X = df.drop(['Effective SOC'], axis=1)  # Drop target column
y = df['Effective SOC']  # Target variable

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using R2 Score and Mean Squared Error (MSE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR2 Score: {r2:.2f}")
print(f"\nMean Squared Error: {mse:.2f}\n")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Run GridSearch on scaled data
grid_search.fit(X_train_scaled, y_train)


print("\nKey Performance Indicators")

# KPI 1: Charge Cycle (Simulated by counting Effective SOC < 10%)
charge_cycles = (df['Effective SOC'] < 10).sum()
print(f"Charge Cycles: {charge_cycles}")

# KPI 2: Average Effective SOC across the dataset
avg_soc = df['Effective SOC'].mean()
print(f"Average Effective SOC: {avg_soc:.2f}")

# KPI 3: Impact of Temperature on SOC (Correlation)
temp_impact = df[['Portable Battery Temperatures', 'Effective SOC']].corr().iloc[0, 1]
print(f"Temperature Impact on SOC: {temp_impact:.2f}")

# KPI 4: Range Estimation - Maximum and Minimum SOC (proxy for usable battery range)
max_soc = df['Effective SOC'].max()
min_soc = df['Effective SOC'].min()
battery_range = max_soc - min_soc
print(f"Battery Range: {battery_range:.2f}")

# KPI 5: Performance Metric - Average Fixed Battery Voltage
avg_fixed_voltage = df['Fixed Battery Voltage'].mean()
print(f"Average Fixed Battery Voltage: {avg_fixed_voltage:.2f} V")

# KPI 6: Impact of Fixed Battery Temperature on SOC (Correlation)
fixed_temp_impact = df[['Fixed Battery Temperatures', 'Effective SOC']].corr().iloc[0, 1]
print(f"Fixed Battery Temperature Impact on SOC: {fixed_temp_impact:.2f}")

# KPI 7: BCM Battery Selected - Percentage of Times Used
bcm_selected_ratio = (df['BCM Battery Selected'].sum() / len(df)) * 100
print(f"BCM Battery Selected Percentage: {bcm_selected_ratio:.2f}%")

# KPI 8: Motor Usage Frequency - Percentage of Time Motor is ON
motor_on_percentage = (df['Motor Status (On/Off)'].sum() / len(df)) * 100
print(f"Motor ON Percentage: {motor_on_percentage:.2f}%")











