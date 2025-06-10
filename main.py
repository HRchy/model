import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def data_cleaning(df):
    cols_to_drop = ['EmployeeID', 'EmployeeCount', 'Over18', 'StandardHours']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    target_col = 'Attrition'
    if target_col in df.columns:
        if df[target_col].dtype == 'object':
            target_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'True': 1, 'False': 0}
            df[target_col] = df[target_col].map(target_map)
        elif df[target_col].dtype == 'bool':
            df[target_col] = df[target_col].astype(int)
        
        if df[target_col].isnull().any():
            df = df.dropna(subset=[target_col])

    df_encoded = df.copy()
    label_encoders = {}
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col != target_col]
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Improved imputation with better strategy
    imputer = IterativeImputer(random_state=42, max_iter=20, n_nearest_features=5)
    df_imputed_array = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(df_imputed_array, columns=df_encoded.columns)

    for col, le in label_encoders.items():
        if col != target_col:
            df_imputed[col] = df_imputed[col].round().astype(int)
            df_imputed[col] = le.inverse_transform(df_imputed[col])

    # Improved outlier detection using IQR method for better robustness
    numeric_cols = df_imputed.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) > 0:
        # Use IQR method instead of Z-score for better outlier detection
        for col in numeric_cols:
            Q1 = df_imputed[col].quantile(0.25)
            Q3 = df_imputed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.0 * IQR  # Slightly more conservative than 1.5
            upper_bound = Q3 + 2.0 * IQR
            df_imputed = df_imputed[(df_imputed[col] >= lower_bound) & (df_imputed[col] <= upper_bound)]

    for col in df_imputed.columns:
        if col != target_col and df_imputed[col].dtype == 'object':
            try:
                df_imputed[col] = df_imputed[col].astype(float)
            except:
                pass

    if target_col in df_imputed.columns:
        df_imputed[target_col] = df_imputed[target_col].round().astype(int)

    bool_map = {'Yes': True, 'No': False, 'True': True, 'False': False}
    for col in df_imputed.columns:
        if col != target_col and df_imputed[col].dtype == 'object':
            unique_vals = df_imputed[col].dropna().unique()
            if set(unique_vals).issubset(set(bool_map.keys())):
                df_imputed[col] = df_imputed[col].map(bool_map)

    return df_imputed

def prepare_features(X_train, X_test, scaler_type='robust'):
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    label_encoders = {}
    categorical_cols = X_train_processed.select_dtypes(include=['object', 'bool']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
        X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
        label_encoders[col] = le
    
    # Use RobustScaler for better handling of outliers
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    return X_train_processed, X_test_processed, X_train_scaled, X_test_scaled, label_encoders, scaler

def apply_data_augmentation(X_train, y_train, augmentation_factor=2):
    from sklearn.utils import resample
    from scipy.stats import norm
    import random
    
    X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train.copy()
    y_train_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train.copy()
    
    class_counts = y_train_series.value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    minority_indices = y_train_series[y_train_series == minority_class].index
    minority_X = X_train_df.loc[minority_indices]
    
    def generate_smote_samples(X_minority, n_samples):
        synthetic_samples = []
        n_samples = int(n_samples)  # Ensure integer
        for _ in range(n_samples):
            idx1, idx2 = random.sample(range(len(X_minority)), 2)
            sample1 = X_minority.iloc[idx1]
            sample2 = X_minority.iloc[idx2]
            alpha = random.random()
            synthetic_sample = alpha * sample1 + (1 - alpha) * sample2
            synthetic_samples.append(synthetic_sample)
        return pd.DataFrame(synthetic_samples, columns=X_minority.columns)
    
    def add_gaussian_noise(X_minority, n_samples, noise_factor=0.08):  # Reduced noise
        noisy_samples = []
        n_samples = int(n_samples)  # Ensure integer
        for _ in range(n_samples):
            idx = random.randint(0, len(X_minority) - 1)
            original_sample = X_minority.iloc[idx].copy()
            for col in X_minority.columns:
                if X_minority[col].dtype in ['float64', 'int64']:
                    feature_std = X_minority[col].std()
                    noise = np.random.normal(0, feature_std * noise_factor)
                    original_sample[col] += noise
                else:
                    if random.random() < 0.05:  # Reduced categorical noise
                        unique_vals = X_minority[col].unique()
                        if len(unique_vals) > 1:
                            original_sample[col] = random.choice(unique_vals)
            noisy_samples.append(original_sample)
        return pd.DataFrame(noisy_samples, columns=X_minority.columns)
    
    def perturb_features(X_minority, n_samples, perturbation_factor=0.03):  # Reduced perturbation
        perturbed_samples = []
        n_samples = int(n_samples)  # Ensure integer
        for _ in range(n_samples):
            idx = random.randint(0, len(X_minority) - 1)
            original_sample = X_minority.iloc[idx].copy()
            for col in X_minority.columns:
                if X_minority[col].dtype in ['float64', 'int64']:
                    feature_range = X_minority[col].max() - X_minority[col].min()
                    perturbation = np.random.uniform(-perturbation_factor, perturbation_factor) * feature_range
                    original_sample[col] += perturbation
            perturbed_samples.append(original_sample)
        return pd.DataFrame(perturbed_samples, columns=X_minority.columns)
    
    target_minority_count = int(min(class_counts[majority_class], len(minority_X) * augmentation_factor))
    n_synthetic = int(target_minority_count - len(minority_X))
    
    if n_synthetic <= 0:
        return X_train_df, y_train_series
    
    n_smote = int(n_synthetic // 3)
    n_noise = int(n_synthetic // 3)  
    n_perturbation = int(n_synthetic - n_smote - n_noise)
    
    augmented_samples = []
    augmented_labels = []
    
    if n_smote > 0:
        smote_samples = generate_smote_samples(minority_X, n_smote)
        augmented_samples.append(smote_samples)
        augmented_labels.extend([minority_class] * n_smote)
    
    if n_noise > 0:
        noise_samples = add_gaussian_noise(minority_X, n_noise)
        augmented_samples.append(noise_samples)
        augmented_labels.extend([minority_class] * n_noise)
    
    if n_perturbation > 0:
        perturbation_samples = perturb_features(minority_X, n_perturbation)
        augmented_samples.append(perturbation_samples)
        augmented_labels.extend([minority_class] * n_perturbation)
    
    if augmented_samples:
        all_augmented = pd.concat(augmented_samples, ignore_index=True)
        X_augmented = pd.concat([X_train_df, all_augmented], ignore_index=True)
        y_augmented = pd.concat([y_train_series, pd.Series(augmented_labels)], ignore_index=True)
        return X_augmented, y_augmented
    
    return X_train_df, y_train_series

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=150,  # Increased
        max_depth=12,      # Slightly increased
        min_samples_split=4,  # Reduced for more flexibility
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced_subsample'  # Better for imbalanced data
    )
    rf.fit(X_train, y_train)
    return rf

def train_random_forest_with_augmentation(X_train, y_train, use_augmentation=True):
    if use_augmentation:
        X_train_aug, y_train_aug = apply_data_augmentation(X_train, y_train, augmentation_factor=2.5)
    else:
        X_train_aug, y_train_aug = X_train, y_train
    
    rf = RandomForestClassifier(
        n_estimators=250,  # Increased
        max_depth=16,      # Slightly increased
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight='balanced_subsample',
        criterion='gini',   # Try different criteria
        min_impurity_decrease=0.0001,  # Prevent overfitting
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train_aug, y_train_aug)
    return rf

def train_xgboost(X_train, y_train):
    if not XGBOOST_AVAILABLE:
        return None
    
    # Improved XGBoost parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,  # Slightly reduced
        subsample=0.85,      # Increased
        colsample_bytree=0.85,  # Increased
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=0.1,      # L2 regularization
        scale_pos_weight=1,  # For class imbalance
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_lightgbm(X_train, y_train):
    if not LIGHTGBM_AVAILABLE:
        return None
    
    # Improved LightGBM parameters
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=8,         # Increased
        learning_rate=0.08,  # Slightly reduced
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        min_child_samples=20  # Prevent overfitting
    )
    lgb_model.fit(X_train, y_train)
    return lgb_model

def train_logistic_regression(X_train_scaled, y_train):
    # Improved logistic regression with regularization
    lr = LogisticRegression(
        max_iter=1500,  # Increased
        random_state=42,
        C=0.8,          # Slightly more regularization
        class_weight='balanced',
        solver='liblinear'  # Better for small datasets
    )
    lr.fit(X_train_scaled, y_train)
    return lr

def train_svm(X_train_scaled, y_train):
    # Improved SVM parameters
    svm = SVC(
        kernel='rbf',
        C=1.2,          # Slightly increased
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    svm.fit(X_train_scaled, y_train)
    return svm

def evaluate_model_with_cv(model, X, y, model_name, cv_folds=5):
    """Evaluate model with cross-validation for more robust results"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
    
    return {
        'model': model_name,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std()
    }

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    unique_classes = len(np.unique(y_test))
    
    accuracy = accuracy_score(y_test, y_pred)
    
    if unique_classes == 2:
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
    else:
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'predictions': y_pred,
        'num_classes': unique_classes
    }

def train_models_with_varying_sizes(df_cleaned):
    dataset_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    
    X_full = df_cleaned.drop('Attrition', axis=1)
    y_full = df_cleaned['Attrition']
    
    # Handle NaN values
    if y_full.isnull().any():
        mask = ~y_full.isnull()
        X_full = X_full[mask]
        y_full = y_full[mask]
    
    # Encode target variable
    if y_full.dtype == 'object' or y_full.dtype == 'bool':
        unique_vals = set(y_full.unique())
        if unique_vals <= {'Yes', 'No', 'yes', 'no'}:
            y_full = y_full.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        elif unique_vals <= {True, False}:
            y_full = y_full.astype(int)
        else:
            le = LabelEncoder()
            y_full = le.fit_transform(y_full.astype(str))
    
    # Handle remaining NaN values after encoding
    if pd.Series(y_full).isnull().any():
        mask = ~pd.Series(y_full).isnull()
        X_full = X_full[mask]
        y_full = y_full[mask]
    
    y_full = pd.Series(y_full).astype(int)
    
    # Stratified split for better evaluation
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.25, stratify=y_full, random_state=42  # Increased test size
    )
    
    X_train_processed, X_test_processed, X_train_scaled, X_test_scaled, _, _ = prepare_features(
        X_train_full, X_test_full, scaler_type='robust'
    )
    
    for size in dataset_sizes:
        print(f"Training with {size*100:.0f}% of data...")
        
        if size < 1.0:
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train_full, y_train_full, 
                train_size=size, 
                stratify=y_train_full, 
                random_state=42
            )
        else:
            X_train_sample, y_train_sample = X_train_full, y_train_full
        
        X_train_proc_sample, _, X_train_scaled_sample, _, _, _ = prepare_features(
            X_train_sample, X_test_full, scaler_type='robust'
        )
        
        # Train baseline Random Forest
        rf_baseline = train_random_forest(X_train_proc_sample, y_train_sample)
        rf_baseline_results = evaluate_model(rf_baseline, X_test_processed, y_test_full, 'Random Forest (Baseline)')
        
        # Train augmented Random Forest
        rf_augmented = train_random_forest_with_augmentation(X_train_proc_sample, y_train_sample, use_augmentation=True)
        rf_augmented_results = evaluate_model(rf_augmented, X_test_processed, y_test_full, 'Random Forest (Augmented)')
        
        # Store results
        for result in [rf_baseline_results, rf_augmented_results]:
            results.append({
                'dataset_size': size,
                'model': result['model'],
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score'],
                'precision': result['precision'],
                'recall': result['recall']
            })
        
        baseline_acc = rf_baseline_results['accuracy']
        augmented_acc = rf_augmented_results['accuracy']
        improvement = augmented_acc - baseline_acc
        
        print(f"  Baseline: {baseline_acc:.4f}, Augmented: {augmented_acc:.4f}, Improvement: {improvement:+.4f}")
    
    return pd.DataFrame(results)

def analyze_feature_importance(df_cleaned):
    X_full = df_cleaned.drop('Attrition', axis=1)
    y_full = df_cleaned['Attrition']
    
    # Clean target variable
    if y_full.isnull().any():
        mask = ~y_full.isnull()
        X_full = X_full[mask]
        y_full = y_full[mask]
    
    if y_full.dtype == 'object':
        y_full = y_full.map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    
    y_full = y_full.astype(int)
    
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )
    
    X_train_processed, _, _, _, _, _ = prepare_features(X_train_full, X_test_full)
    
    best_rf = train_random_forest_with_augmentation(X_train_processed, y_train_full, use_augmentation=True)
    
    feature_names = X_train_processed.columns if hasattr(X_train_processed, 'columns') else [f'feature_{i}' for i in range(X_train_processed.shape[1])]
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:20s}: {row['importance']:.4f}")
    
    return importance_df

def plot_results(results_df):
    if 'model' not in results_df.columns:
        return
        
    plt.figure(figsize=(15, 10))
    
    baseline_results = results_df[results_df['model'].str.contains('Baseline', na=False)]
    augmented_results = results_df[results_df['model'].str.contains('Augmented', na=False)]
    
    # Plot accuracy comparison
    plt.subplot(2, 2, 1)
    if not baseline_results.empty:
        plt.plot(baseline_results['dataset_size'], baseline_results['accuracy'], 
                marker='o', label='Baseline RF', linewidth=2)
    if not augmented_results.empty:
        plt.plot(augmented_results['dataset_size'], augmented_results['accuracy'], 
                marker='s', label='Augmented RF', linewidth=2)
    plt.xlabel('Dataset Size')
    plt.ylabel('Accuracy')
    plt.title('Random Forest: Baseline vs Augmented (Accuracy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot F1-score comparison
    plt.subplot(2, 2, 2)
    if not baseline_results.empty:
        plt.plot(baseline_results['dataset_size'], baseline_results['f1_score'], 
                marker='o', label='Baseline RF', linewidth=2)
    if not augmented_results.empty:
        plt.plot(augmented_results['dataset_size'], augmented_results['f1_score'], 
                marker='s', label='Augmented RF', linewidth=2)
    plt.xlabel('Dataset Size')
    plt.ylabel('F1-Score')
    plt.title('Random Forest: Baseline vs Augmented (F1-Score)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot precision comparison
    plt.subplot(2, 2, 3)
    if not baseline_results.empty:
        plt.plot(baseline_results['dataset_size'], baseline_results['precision'], 
                marker='o', label='Baseline RF', linewidth=2)
    if not augmented_results.empty:
        plt.plot(augmented_results['dataset_size'], augmented_results['precision'], 
                marker='s', label='Augmented RF', linewidth=2)
    plt.xlabel('Dataset Size')
    plt.ylabel('Precision')
    plt.title('Random Forest: Baseline vs Augmented (Precision)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot recall comparison
    plt.subplot(2, 2, 4)
    if not baseline_results.empty:
        plt.plot(baseline_results['dataset_size'], baseline_results['recall'], 
                marker='o', label='Baseline RF', linewidth=2)
    if not augmented_results.empty:
        plt.plot(augmented_results['dataset_size'], augmented_results['recall'], 
                marker='s', label='Augmented RF', linewidth=2)
    plt.xlabel('Dataset Size')
    plt.ylabel('Recall')
    plt.title('Random Forest: Baseline vs Augmented (Recall)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("HR Attrition Prediction with Random Forest and Data Augmentation")
    print("=" * 70)
    
    df = pd.read_csv('dataset_HR.csv')
    print(f"Dataset loaded: {df.shape}")
    
    df_cleaned = data_cleaning(df)
    print(f"Dataset cleaned: {df_cleaned.shape}")
    
    if 'Attrition' not in df_cleaned.columns:
        print("Error: Attrition column not found!")
        return None
    
    print(f"Class distribution: {df_cleaned['Attrition'].value_counts().to_dict()}")
    
    results_df = train_models_with_varying_sizes(df_cleaned)
    
    if results_df.empty:
        print("No results generated!")
        return None
    
    # Summary analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    baseline_results = results_df[results_df['model'].str.contains('Baseline', na=False)]
    augmented_results = results_df[results_df['model'].str.contains('Augmented', na=False)]
    
    if not baseline_results.empty:
        best_baseline = baseline_results.loc[baseline_results['accuracy'].idxmax()]
        print(f"Best Baseline: {best_baseline['accuracy']:.4f} accuracy at {best_baseline['dataset_size']*100:.0f}% data")
    
    if not augmented_results.empty:
        best_augmented = augmented_results.loc[augmented_results['accuracy'].idxmax()]
        print(f"Best Augmented: {best_augmented['accuracy']:.4f} accuracy at {best_augmented['dataset_size']*100:.0f}% data")
        
        if not baseline_results.empty:
            improvement = best_augmented['accuracy'] - best_baseline['accuracy']
            print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Feature importance analysis
    try:
        analyze_feature_importance(df_cleaned)
    except Exception as e:
        print(f"Feature importance analysis failed: {e}")
    
    # Generate plots
    try:
        plot_results(results_df)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    return results_df

if __name__ == "__main__":
    results = main()
    
    if results is not None:
        results.to_csv('hr_attrition_results_improved.csv', index=False)
        print(f"\nResults saved to 'hr_attrition_results_improved.csv'")
    else:
        print("Analysis failed!")