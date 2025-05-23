import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# 2. Load Data
df = pd.read_csv('ai_solutions_web_server_logs.csv')

# 3. Data Preprocessing
# Combine Date and Timestamp into a single datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], dayfirst=True)

# Normalize boolean field
df['Repeat Customer'] = df['Repeat Customer'].astype(bool)

# Convert to numeric
for col in ['Price (USD)', 'Cost (USD)', 'Profit (USD)', 'Quantity']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing or zero values if necessary
df.fillna(0, inplace=True)

# Feature Engineering
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month
df['Profit Margin (%)'] = np.where(df['Price (USD)'] > 0, df['Profit (USD)'] / df['Price (USD)'] * 100, 0)

def classify_request(path):
    if 'scheduledemo' in path:
        return 'Schedule Demo'
    elif 'cloud' in path:
        return 'Cloud Services'
    elif 'ai' in path:
        return 'AI Request'
    elif 'services' in path:
        return 'General Services'
    else:
        return 'Other'

# Categorize request paths
df['Request Category'] = df['URL Path'].apply(classify_request)

# 4. Exploratory Data Analysis (EDA)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Country')
plt.title('Requests by Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Job Type', y='Profit (USD)')
plt.title('Profit Distribution by Job Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Profit (USD)'], bins=20, kde=True)
plt.title('Profit Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Quantity', y='Profit (USD)', hue='Country')
plt.title('Quantity vs Profit by Country')
plt.tight_layout()
plt.show()

# Pie Chart: Request Category Distribution
category_counts = df['Request Category'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Request Category Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Pie Chart: Repeat vs New Customers
repeat_counts = df['Repeat Customer'].value_counts()
labels = ['New Customer', 'Repeat Customer']
plt.figure(figsize=(6, 6))
plt.pie(repeat_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightgreen'])
plt.title('Customer Type Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()

# 5. Modeling
# Encode categorical variables
df_encoded = pd.get_dummies(df[['Country', 'Job Type', 'Request Type', 'Request Category']], drop_first=True)

# Combine with numerical features
features = pd.concat([
    df_encoded,
    df[['Quantity', 'Price (USD)', 'Cost (USD)', 'Hour', 'DayOfWeek', 'Month', 'Repeat Customer']]
], axis=1)

# Define target variable
target = df['Profit (USD)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))

# 7. Feature Importance
importances = pd.Series(model.feature_importances_, index=features.columns)
plt.figure(figsize=(10, 6))
importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.show()

# 8. Export Trained Model
joblib.dump(model, 'trained_sales_model.pkl')
print("Model exported as 'trained_sales_model.pkl'")
