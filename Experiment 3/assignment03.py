import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(42)

data = {
    "Age": np.random.randint(22, 60, size=10).astype(float),
    "Salary": np.random.randint(40000, 120000, size=10).astype(float),
    "Department": np.random.choice(
        ["HR", "IT", "Marketing", "Sales"], size=10
    ),
    "Years_of_Experience": np.random.randint(1, 35, size=10).astype(float)
}

df = pd.DataFrame(data)

df.loc[2, "Age"] = np.nan
df.loc[5, "Salary"] = np.nan
df.loc[4, "Department"] = np.nan
df.loc[7, "Years_of_Experience"] = np.nan

numeric_features = ["Age", "Salary", "Years_of_Experience"]

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df[numeric_features])

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

X_standard = standard_scaler.fit_transform(X)
X_minmax = minmax_scaler.fit_transform(X)

print("--- StandardScaler ---")
print(X_standard)

print("\n--- MinMaxScaler ---")
print(X_minmax)

print("\n--- Range Comparison ---")
for i, col in enumerate(numeric_features):
    print(
        f"{col}: "
        f"Standard[min={X_standard[:, i].min():.2f}, "
        f"max={X_standard[:, i].max():.2f}]  "
        f"MinMax[min={X_minmax[:, i].min():.2f}, "
        f"max={X_minmax[:, i].max():.2f}]"
    )
