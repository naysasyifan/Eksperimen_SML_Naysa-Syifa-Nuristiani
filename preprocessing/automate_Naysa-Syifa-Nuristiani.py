# Import Library
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import os
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load Dataset
df = pd.read_csv("online_shoppers_intention_raw/online_shoppers_intention.csv")

# Data Preprocessing
df_clean = df.copy()
print("Shape awal:", df_clean.shape)

# Basic cleaning
df_clean.columns = [c.strip() for c in df_clean.columns]

# Drop duplicate
dup_before = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
dup_after = df_clean.duplicated().sum()

print(f"Duplikat sebelum: {dup_before}")
print(f"Duplikat sesudah: {dup_after}")
print("Shape setelah drop duplicates:", df_clean.shape)

print("Total missing values:", df_clean.isna().sum().sum())

# Split X & y
target_col = "Revenue"
if target_col not in df_clean.columns:
    raise ValueError("Kolom target 'Revenue' tidak ditemukan. Cek nama kolom dataset.")

X = df_clean.drop(columns=[target_col])
y = df_clean[target_col].astype(bool).map({True: 1, False: 0})

print("\nDistribusi target (0/1):")
print(y.value_counts())

# Define preprocessing pipeline
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print("\nKolom numerik :", num_cols)
print("Kolom kategorikal :", cat_cols)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\ntrain:", X_train.shape, y_train.shape)
print("test :", X_test.shape, y_test.shape)

# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("X_train_processed:", X_train_processed.shape)
print("X_test_processed :", X_test_processed.shape)

# Get feature names
feature_names = []
feature_names.extend(num_cols)

if len(cat_cols) > 0:
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names.extend(ohe_feature_names)

# Create DataFrames
X_train_df = pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed, columns=feature_names)

# Save preprocessed data
train_final = X_train_df.copy()
train_final[target_col] = y_train.values

test_final = X_test_df.copy()
test_final[target_col] = y_test.values

output_dir = "online_shoppers_intention_preprocessing"
os.makedirs(output_dir, exist_ok=True)

train_path = os.path.join(output_dir, "train_preprocessed.csv")
test_path = os.path.join(output_dir, "test_preprocessed.csv")

train_final.to_csv(train_path, index=False)
test_final.to_csv(test_path, index=False)

print("\nTrain data disimpan ke:", train_path)
print("Test data disimpan ke:", test_path)

# Save preprocessor
preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
joblib.dump(preprocessor, preprocessor_path)

print("\nPreprocessor disimpan ke:", preprocessor_path)
print("\nPreprocessing selesai!")