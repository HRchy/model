import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.stats import zscore

def data_cleaning(df):
    cols_to_drop = ['EmployeeID', 'EmployeeCount', 'Over18', 'StandardHours']

    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Encode categorical variables temporarily
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Apply multiple imputation
    imputer = IterativeImputer(random_state=0)
    df_imputed_array = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(df_imputed_array, columns=df_encoded.columns)

    # If needed, inverse transform categorical columns
    for col, le in label_encoders.items():
        df_imputed[col] = df_imputed[col].round().astype(int)
        df_imputed[col] = le.inverse_transform(df_imputed[col])


    # Apply Z-score only on numeric columns
    numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
    z_scores = np.abs(zscore(df_imputed[numeric_cols]))
    df_imputed = df_imputed[(z_scores < 3).all(axis=1)]


    for col in df_imputed.columns:
        if df_imputed[col].dtype == 'object':
            try:
                df_imputed[col] = df_imputed[col].astype(float)
            except:
                pass

    # Convert boolean-like strings to actual bools
    bool_map = {'Yes': True, 'No': False, 'True': True, 'False': False}
    for col in df_imputed.columns:
        if df_imputed[col].dtype == 'object':
            unique_vals = df_imputed[col].dropna().unique()
            if set(unique_vals).issubset(set(bool_map.keys())):
                df_imputed[col] = df_imputed[col].map(bool_map)

    return df_imputed


if __name__ == "__main__":
    df = pd.read_csv('dataset_HR.csv')
    df_cleaned = data_cleaning(df)
    print(df_cleaned, df)
   