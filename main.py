
import pandas as pd

# Load dataset
df = pd.read_csv("Data/city_day.csv")

print("Original shape:", df.shape)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Remove duplicate rows
df = df.drop_duplicates()

# Sort data by City and Date
df = df.sort_values(["City", "Date"])

# Fill numeric missing values using median
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Remove rows where City or Date is missing
df = df.dropna(subset=["City", "Date"])

print("Cleaned shape:", df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("Data/cleaned_city_day.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Basic AQI analysis

print("\nAQI Statistics:")
print(df["AQI"].describe())

print("\nAverage AQI by City:")
print(df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10))

import matplotlib.pyplot as plt

# Average AQI by city
city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)

# Create bar chart
plt.figure(figsize=(10, 6))
city_aqi.plot(kind="bar")

plt.title("Top 10 Cities by Average AQI")
plt.xlabel("City")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()

# Display graph
plt.show()

# Create AQI categories

def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


df["AQI_Category"] = df["AQI"].apply(classify_aqi)

# Count each AQI category
print("\nAQI Category Distribution:")
print(df["AQI_Category"].value_counts())

# AQI Category Pie Chart

plt.figure(figsize=(7, 7))

category_counts = df["AQI_Category"].value_counts()

plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("AQI Category Distribution")
plt.tight_layout()
plt.show()

# AQI Trend Over Time

df["Date"] = pd.to_datetime(df["Date"])

daily_aqi = df.groupby("Date")["AQI"].mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_aqi.index, daily_aqi.values)

plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation Heatmap

plt.figure(figsize=(12, 8))

numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.imshow(correlation, cmap="coolwarm", aspect="auto")
plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Heatmap of Air Quality Parameters")

plt.tight_layout()
plt.show()

# Machine Learning - AQI Prediction

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Select input features
features = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene"
]

# Remove rows with missing values
ml_df = df[features + ["AQI"]].dropna()

X = ml_df[features]
y = ml_df["AQI"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Model trained successfully! ✅")

# Step 21 - Model Performance

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Step 22 - Actual vs Predicted AQI

plt.figure(figsize=(10, 5))

plt.plot(y_test.values[:100], label="Actual AQI")
plt.plot(y_pred[:100], label="Predicted AQI")

plt.title("Actual vs Predicted AQI")
plt.xlabel("Test Data Samples")
plt.ylabel("AQI")
plt.legend()

plt.tight_layout()
plt.show()

# Step 23 - Feature Importance

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Feature importance graph
plt.figure(figsize=(10, 6))
importance.sort_values().plot(kind="barh")

plt.title("Feature Importance for AQI Prediction")
plt.xlabel("Importance")
plt.ylabel("Air Quality Parameter")

plt.tight_layout()
plt.show()

# Step 24 - Save trained model

import os
import joblib

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained model
joblib.dump(model, "model/aqi_model.pkl")

# Save feature names
joblib.dump(features, "model/features.pkl")

print("\nModel saved successfully! ✅")

# Step 25 - Personalized Health Advisory

def get_health_advisory(aqi):
    if aqi <= 50:
        return {
            "category": "Good",
            "message": "Air quality is good. Normal outdoor activities are safe."
        }

    elif aqi <= 100:
        return {
            "category": "Satisfactory",
            "message": "Air quality is acceptable. Sensitive people should be a little careful."
        }

    elif aqi <= 200:
        return {
            "category": "Moderate",
            "message": "Air quality may affect sensitive people. Consider reducing prolonged outdoor activity."
        }

    elif aqi <= 300:
        return {
            "category": "Poor",
            "message": "Air quality is poor. Reduce prolonged outdoor exposure and consider using a mask outdoors."
        }

    elif aqi <= 400:
        return {
            "category": "Very Poor",
            "message": "Air quality is very poor. Avoid unnecessary outdoor activities and keep indoor air clean."
        }

    else:
        return {
            "category": "Severe",
            "message": "Air quality is severe. Avoid outdoor exposure as much as possible and follow local health guidance."
        }


# Test advisory using average AQI
average_aqi = df["AQI"].mean()

advisory = get_health_advisory(average_aqi)

print("\nPersonalized Health Advisory:")
print("AQI:", round(average_aqi, 2))
print("Category:", advisory["category"])

import pandas as pd

# Load dataset
df = pd.read_csv("Data/city_day.csv")

print("Original shape:", df.shape)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Remove duplicate rows
df = df.drop_duplicates()

# Sort data by City and Date
df = df.sort_values(["City", "Date"])

# Fill numeric missing values using median
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Remove rows where City or Date is missing
df = df.dropna(subset=["City", "Date"])

print("Cleaned shape:", df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv("Data/cleaned_city_day.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Basic AQI analysis

print("\nAQI Statistics:")
print(df["AQI"].describe())

print("\nAverage AQI by City:")
print(df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10))

import matplotlib.pyplot as plt

# Average AQI by city
city_aqi = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)

# Create bar chart
plt.figure(figsize=(10, 6))
city_aqi.plot(kind="bar")

plt.title("Top 10 Cities by Average AQI")
plt.xlabel("City")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()

# Display graph
plt.show()

# Create AQI categories

def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


df["AQI_Category"] = df["AQI"].apply(classify_aqi)

# Count each AQI category
print("\nAQI Category Distribution:")
print(df["AQI_Category"].value_counts())

# AQI Category Pie Chart

plt.figure(figsize=(7, 7))

category_counts = df["AQI_Category"].value_counts()

plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("AQI Category Distribution")
plt.tight_layout()
plt.show()

# AQI Trend Over Time

df["Date"] = pd.to_datetime(df["Date"])

daily_aqi = df.groupby("Date")["AQI"].mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_aqi.index, daily_aqi.values)

plt.title("AQI Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Correlation Heatmap

plt.figure(figsize=(12, 8))

numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.imshow(correlation, cmap="coolwarm", aspect="auto")
plt.colorbar()

plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)

plt.title("Correlation Heatmap of Air Quality Parameters")

plt.tight_layout()
plt.show()

# Machine Learning - AQI Prediction

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Select input features
features = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene"
]

# Remove rows with missing values
ml_df = df[features + ["AQI"]].dropna()

X = ml_df[features]
y = ml_df["AQI"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Model trained successfully! ✅")

# Step 21 - Model Performance

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Step 22 - Actual vs Predicted AQI

plt.figure(figsize=(10, 5))

plt.plot(y_test.values[:100], label="Actual AQI")
plt.plot(y_pred[:100], label="Predicted AQI")

plt.title("Actual vs Predicted AQI")
plt.xlabel("Test Data Samples")
plt.ylabel("AQI")
plt.legend()

plt.tight_layout()
plt.show()

# Step 23 - Feature Importance

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Feature importance graph
plt.figure(figsize=(10, 6))
importance.sort_values().plot(kind="barh")

plt.title("Feature Importance for AQI Prediction")
plt.xlabel("Importance")
plt.ylabel("Air Quality Parameter")

plt.tight_layout()
plt.show()

# Step 24 - Save trained model

import os
import joblib

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the trained model
joblib.dump(model, "model/aqi_model.pkl")

# Save feature names
joblib.dump(features, "model/features.pkl")

print("\nModel saved successfully! ✅")

# Step 25 - Personalized Health Advisory

def get_health_advisory(aqi):
    if aqi <= 50:
        return {
            "category": "Good",
            "message": "Air quality is good. Normal outdoor activities are safe."
        }

    elif aqi <= 100:
        return {
            "category": "Satisfactory",
            "message": "Air quality is acceptable. Sensitive people should be a little careful."
        }

    elif aqi <= 200:
        return {
            "category": "Moderate",
            "message": "Air quality may affect sensitive people. Consider reducing prolonged outdoor activity."
        }

    elif aqi <= 300:
        return {
            "category": "Poor",
            "message": "Air quality is poor. Reduce prolonged outdoor exposure and consider using a mask outdoors."
        }

    elif aqi <= 400:
        return {
            "category": "Very Poor",
            "message": "Air quality is very poor. Avoid unnecessary outdoor activities and keep indoor air clean."
        }

    else:
        return {
            "category": "Severe",
            "message": "Air quality is severe. Avoid outdoor exposure as much as possible and follow local health guidance."
        }


# Test advisory using average AQI
average_aqi = df["AQI"].mean()

advisory = get_health_advisory(average_aqi)

print("\nPersonalized Health Advisory:")
print("AQI:", round(average_aqi, 2))
print("Category:", advisory["category"])
print("Advice:", advisory["message"])