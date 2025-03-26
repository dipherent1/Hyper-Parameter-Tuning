## What is Feature Scaling?

Feature scaling is a preprocessing step in machine learning that transforms numerical features to a common range or distribution. This ensures that features with different scales (e.g., age in years vs. BMI in kg/m²) contribute equally to a model’s predictions. Two common methods are Standardization (using mean and standard deviation) and Min-Max Scaling (using minimum and maximum values).

## 1. Standardization (Scaling Using Mean)

### How It Works
- Formula: \( x' = \frac{x - \mu}{\sigma} \)
  - \( x \): Original value
  - \( \mu \): Mean of the feature
  - \( \sigma \): Standard deviation of the feature
- Centers the data around 0 with a standard deviation of 1 (unit variance).
- Suitable for algorithms assuming normally distributed data (e.g., Linear Regression).

### Example from Medical Cost Dataset
- Feature: age (e.g., values: 19, 25, 30, 45)
  - Mean (\( \mu \)): 29.75
  - Std Dev (\( \sigma \)): 10.21 (hypothetical)
  - Original: 19 → Standardized: \( \frac{19 - 29.75}{10.21} \approx -1.05 \)
  - Original: 45 → Standardized: \( \frac{45 - 29.75}{10.21} \approx 1.49 \)
- Result: age values are transformed to a scale with mean 0 and variance 1 (e.g., -1.05, -0.47, 0.02, 1.49).

### Code Example
Using scikit-learn:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])
2. Min-Max Scaling
How It Works
Formula: 
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
x_{min}
: Minimum value of the feature
x_{max}
: Maximum value of the feature
Rescales the data to a fixed range, typically [0, 1].
Useful for algorithms sensitive to feature magnitudes (e.g., neural networks).
Example from Medical Cost Dataset
Feature: bmi (e.g., values: 15, 25, 30, 40)
Min (
x_{min}
): 15
Max (
x_{max}
): 40
Original: 15 → Min-Max: 
\frac{15 - 15}{40 - 15} = 0
Original: 30 → Min-Max: 
\frac{30 - 15}{40 - 15} = 0.6
Result: bmi values are scaled between 0 and 1 (e.g., 0, 0.4, 0.6, 1.0).
Code Example
Using scikit-learn:
python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['age', 'bmi', 'children']] = scaler.fit_transform(df[['age', 'bmi', 'children']])
Why Use Scaling in This Dataset?
In the Medical Cost Dataset, numerical features like age (18–64), bmi (15–53), and children (0–5) have different ranges. Without scaling:
Larger-range features (e.g., bmi) could dominate smaller-range ones (e.g., children) in models like Linear Regression.
Scaling ensures all features contribute fairly to predicting charges.
Key Differences
Standardization: No fixed range, mean = 0, std dev = 1; preserves distribution shape.
Min-Max: Fixed range [0, 1]; sensitive to outliers but intuitive for bounded data.
Both methods prepare the data for modeling, with the choice depending on the algorithm and data characteristics.

### How to Use This
1. Create the File:
   - Copy the content into a text editor.
   - Save as ScalingSummary.md.

2. Add to GitHub:
   - Upload it to your repository with your Colab notebook and other files.
   - Reference it in your 1-page report if needed.
