import pandas as pd
import numpy as np
from collections import Counter

# Load the dataset
df = pd.read_csv('dataset_HR.csv')

print("=" * 80)
print("HR DATASET QUALITY ASSESSMENT REPORT")
print("=" * 80)

# Basic dataset information
print(f"\nDATASET OVERVIEW:")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

# Create comprehensive quality assessment table
quality_assessment = []

for column in df.columns:
    col_data = df[column]
    
    # Basic statistics
    total_count = len(col_data)
    missing_count = col_data.isnull().sum()
    missing_percentage = (missing_count / total_count) * 100
    unique_count = col_data.nunique()
    duplicate_count = total_count - unique_count
    
    # Data type analysis
    dtype = str(col_data.dtype)
    
    # For numeric columns
    if pd.api.types.is_numeric_dtype(col_data):
        min_val = col_data.min()
        max_val = col_data.max()
        mean_val = col_data.mean()
        std_val = col_data.std()
        
        # Outlier detection using IQR method
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        
        range_info = f"{min_val:.2f} to {max_val:.2f}"
        outlier_info = f"{outliers} ({(outliers/total_count)*100:.1f}%)"
        
    else:
        # For categorical columns
        range_info = f"{unique_count} categories"
        outlier_info = "N/A (categorical)"
        mean_val = "N/A"
        std_val = "N/A"
    
    # Data quality score (0-100)
    quality_score = 100
    if missing_percentage > 0:
        quality_score -= missing_percentage
    if pd.api.types.is_numeric_dtype(col_data) and outliers > 0:
        quality_score -= min((outliers/total_count)*100, 20)  # Cap outlier penalty at 20%
    
    quality_assessment.append({
        'Column': column,
        'Data_Type': dtype,
        'Total_Records': total_count,
        'Missing_Values': missing_count,
        'Missing_Percentage': f"{missing_percentage:.1f}%",
        'Unique_Values': unique_count,
        'Duplicates': duplicate_count,
        'Range/Categories': range_info,
        'Outliers': outlier_info,
        'Quality_Score': f"{quality_score:.1f}/100"
    })

# Convert to DataFrame for better display
quality_df = pd.DataFrame(quality_assessment)

print("\nQUALITY ASSESSMENT TABLE:")
print("=" * 120)
print(quality_df.to_string(index=False))

# Additional detailed analysis
print("\n\nDETAILED ANALYSIS:")
print("=" * 50)

# Missing values summary
print(f"\n1. MISSING VALUES SUMMARY:")
missing_summary = df.isnull().sum()
if missing_summary.sum() == 0:
    print("✓ No missing values found in the dataset")
else:
    print("Columns with missing values:")
    for col, count in missing_summary[missing_summary > 0].items():
        print(f"   - {col}: {count} ({(count/len(df))*100:.1f}%)")

# Duplicate rows analysis
print(f"\n2. DUPLICATE ANALYSIS:")
duplicate_rows = df.duplicated().sum()
print(f"Total duplicate rows: {duplicate_rows} ({(duplicate_rows/len(df))*100:.1f}%)")

# Data type consistency
print(f"\n3. DATA TYPE ANALYSIS:")
print("Expected vs Actual data types:")
expected_types = {
    'SatisfactionLevel': 'float',
    'LastEvaluation': 'float', 
    'NumberProjects': 'int/float',
    'AverageMonthlyHours': 'float',
    'TimeSpentCompany': 'int/float',
    'WorkAccident': 'int/float (0/1)',
    'PromotionLast5Years': 'int/float (0/1)',
    'Department': 'string/category',
    'Salary': 'string/category',
    'OverTime': 'string/category',
    'Attrition': 'string/category'
}

for col in df.columns:
    actual_type = str(df[col].dtype)
    expected = expected_types.get(col, 'unknown')
    status = "✓" if any(exp in actual_type.lower() for exp in expected.lower().split('/')) else "⚠"
    print(f"   {status} {col}: Expected({expected}) | Actual({actual_type})")

# Categorical variables analysis
print(f"\n4. CATEGORICAL VARIABLES ANALYSIS:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_vals = df[col].value_counts()
    print(f"\n{col} - {len(unique_vals)} unique values:")
    if len(unique_vals) <= 10:
        for val, count in unique_vals.items():
            print(f"   - {val}: {count} ({(count/len(df))*100:.1f}%)")
    else:
        print(f"   Top 5 values:")
        for val, count in unique_vals.head().items():
            print(f"   - {val}: {count} ({(count/len(df))*100:.1f}%)")

# Numeric variables analysis
print(f"\n5. NUMERIC VARIABLES ANALYSIS:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    print(df[numeric_cols].describe())

# Data consistency checks
print(f"\n6. DATA CONSISTENCY CHECKS:")

# Check for binary variables that should be 0/1
binary_expected = ['WorkAccident', 'PromotionLast5Years']
for col in binary_expected:
    if col in df.columns:
        unique_vals = sorted(df[col].unique())
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            print(f"✓ {col}: Binary values are consistent")
        else:
            print(f"⚠ {col}: Unexpected values - {unique_vals}")

# Check satisfaction level range (should be 0-1)
if 'SatisfactionLevel' in df.columns:
    sat_min, sat_max = df['SatisfactionLevel'].min(), df['SatisfactionLevel'].max()
    if 0 <= sat_min and sat_max <= 1:
        print(f"✓ SatisfactionLevel: Values in expected range [0,1]")
    else:
        print(f"⚠ SatisfactionLevel: Values outside expected range - [{sat_min:.2f}, {sat_max:.2f}]")

# Overall quality score
print(f"\n7. OVERALL DATASET QUALITY SCORE:")
avg_quality = np.mean([float(score.split('/')[0]) for score in quality_df['Quality_Score']])
print(f"Average Quality Score: {avg_quality:.1f}/100")

if avg_quality >= 90:
    print("✓ Excellent data quality")
elif avg_quality >= 80:
    print("✓ Good data quality")
elif avg_quality >= 70:
    print("⚠ Fair data quality - some issues need attention")
else:
    print("⚠ Poor data quality - significant issues need resolution")

print("\n" + "=" * 80)
print("END OF QUALITY ASSESSMENT REPORT")
print("=" * 80)