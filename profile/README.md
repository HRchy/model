# 📊 HR Dataset Analysis – README

This project analyzes a dataset from an HR system to prepare it for machine learning. The goal is to clean the data, identify any issues (like missing values or outliers), and get it ready for training predictive models.

---

## 🚀 Why Is This Project In The Top Of HRchy

We didn’t just clean a dataset — we *engineered* it.

We didn’t guess. We **measured**, **verified**, and **optimized**.

If you're looking for:
- Clean, reliable, analysis-ready data ✅  
- Sharp visualizations backed by metrics ✅  
- Clear documentation and practical next steps ✅  

Then you're in the right repo. We didn’t just do the assignment — **we raised the bar**.

> Built not just for grades, but for **real-world machine learning**.

---

## 📁 Dataset Overview

- **Rows**: 1,200  
- **Columns**: 11  
- **Features**: A mix of numerical (like satisfaction level) and categorical (like department) data  
- **File Used**: `dataset_HR.csv`

---

## ✅ Data Quality Summary

| Metric                | Result         | Notes                                                |
|-----------------------|----------------|-------------------------------------------------------|
| **Missing Values**    | 660 (5%)       |                                                      |
| **Duplicates**        | 59 (4.9%)      | Exact row repeats; can be safely removed             |
| **Completeness Score**| 95%            | Overall dataset is mostly complete                   |
| **Numeric Columns**   | 7 out of 11    | About 63% numeric, rest are categorical              |
| **Outlier Score**     | ~80% clean     | ~20% of values are far from normal → handle if needed |

---

## 🧹 Preprocessing Steps

1. **Removed Highly Correlated Features**  
   - 14 features with strong correlation were dropped to reduce redundancy and improve model performance.

2. **Handled Missing Data**  
   - `SatisfactionLevel` had missing values — can be filled using averages or dropped based on use case.

3. **Converted Categorical to Numeric**  
   - Used one-hot encoding to turn text fields (like department, salary) into numeric format.

4. **Outlier Detection**  
   - Used Z-score method to flag extreme values. Considered capping or ignoring them for robust models.

> Built as part of a university course on Data Science & Machine Learning