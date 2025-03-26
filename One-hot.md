## What is One-Hot Encoding?

**One-hot encoding** is a preprocessing technique used to convert categorical variables into a numerical format suitable for machine learning models. Since most algorithms require numerical inputs, categorical data (e.g., text labels) must be transformed. One-hot encoding creates binary columns (0 or 1) for each category, where only one column is "active" (set to 1) per data point.

### How It Works
- For a column with \( n \) unique categories, it generates \( n \) new binary columns (or \( n-1 \) if `drop_first=True` is used to avoid multicollinearity).
- Each row gets a 1 in the column matching its category and 0s elsewhere.

## Examples from the Medical Cost Dataset

The "Medical Cost Personal Datasets" (`insurance.csv`) includes categorical columns like `sex`, `smoker`, and `region`. Here’s how one-hot encoding applies:

### 1. Column: `sex`
- **Categories**: `female`, `male` (2 unique values).
- **Encoding** (with `drop_first=True`):
  - Original: `female`, `male`, `female`
  - Encoded: `sex_male` → `[0, 1, 0]`
  - Note: `sex_female` is dropped; 0 in `sex_male` implies female.
- **Result**: One new column (`sex_male`).

### 2. Column: `smoker`
- **Categories**: `yes`, `no` (2 unique values).
- **Encoding** (with `drop_first=True`):
  - Original: `yes`, `no`, `yes`
  - Encoded: `smoker_yes` → `[1, 0, 1]`
  - Note: `smoker_no` is dropped; 0 in `smoker_yes` implies no.
- **Result**: One new column (`smoker_yes`).

### 3. Column: `region`
- **Categories**: `southwest`, `southeast`, `northwest`, `northeast` (4 unique values).
- **Encoding** (with `drop_first=True`):
  - Original: `southwest`, `southeast`, `northeast`
  - Encoded:
    - `region_southeast` → `[0, 1, 0]`
    - `region_northwest` → `[0, 0, 0]`
    - `region_northeast` → `[0, 0, 1]`
  - Note: `region_southwest` is dropped; all 0s imply southwest.
- **Result**: Three new columns (`region_southeast`, `region_northwest`, `region_northeast`).

## Why Use It?
In this dataset, `sex`, `smoker`, and `region` are categorical, but the target (`charges`) is numerical. Models like Linear Regression or Random Forest need numeric inputs. One-hot encoding enables these features to be used, revealing how factors like smoking (`smoker_yes`) or region (`region_southeast`) influence insurance charges.

## Code Example
Using pandas in Python:
```python
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
```