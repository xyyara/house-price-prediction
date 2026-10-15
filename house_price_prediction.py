# ==========================================
# House Price Prediction using Machine Learning
# ==========================================

# 1. Import Libraries
import pandas as pd
import joblib
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 2. Load Dataset
# ==========================================

file = filedialog.askopenfilename()
data = pd.read_csv(file)
print("First 5 rows:")
print(data.head())
print("\nDataset shape:")
print(data.shape)
print("\nDataset information:")
data.info()

# ==========================================
# 3. Select Features and Target
# ==========================================

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt"
]
X = data[features]
y = data["SalePrice"]
print("\nFeatures:")
print(X.head())
print("\nTarget:")
print(y.head())

# ==========================================
# 4. Check Missing Values
# ==========================================

print("\nMissing values:")
print(data[features + ["SalePrice"]].isnull().sum())

# ==========================================
# 5. Split the Data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining data shape:")
print(X_train.shape)
print("\nTesting data shape:")
print(X_test.shape)

# ==========================================
# 6. Create and Train the Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
print("\nModel training completed!")

# ==========================================
# 7. Make Predictions
# ==========================================

predictions = model.predict(X_test)
print("\nFirst 5 predictions:")
print(predictions[:5])
print("\nActual prices:")
print(y_test.head().values)

# ==========================================
# 8. Evaluate the Model
# ==========================================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# ==========================================
# 9. Save the Model
# ==========================================

joblib.dump(model, "house_price_model.pkl")
print("\nModel saved successfully!")

# ==========================================
# 10. Predict the Price of a New House
# ==========================================

model = joblib.load("house_price_model.pkl")
print("\n🏠 House Price Prediction")
quality = float(input("House quality (1-10): "))
area = float(input("Living area: "))
garage = float(input("Number of garage spaces: "))
basement = float(input("Basement area: "))
year = float(input("Year built: "))
new_house = pd.DataFrame([{
    "OverallQual": quality,
    "GrLivArea": area,
    "GarageCars": garage,
    "TotalBsmtSF": basement,
    "YearBuilt": year
}])

prediction = model.predict(new_house)

print("\nPredicted house price:", prediction[0])
