# Insurance Charges Prediction

## Overview
This project predicts insurance charges using the `insurance.csv` dataset, which contains 1,338 records with features such as `age`, `bmi`, `children`, `sex`, `smoker`, and `region`. The goal is to compare Linear Regression and Random Forest models, both untuned and tuned with GridSearchCV and RandomizedSearchCV, to identify the best model for accurate predictions. Evaluation metrics include Mean Squared Error (MSE) and R², with 5-fold cross-validation for robustness. Feature importance of the best model (Random Forest with GridSearchCV) is analyzed to understand key drivers of insurance charges.

### Key Findings
- **Best Model**: Random Forest (GridSearchCV) achieved the lowest Test MSE (20,666,560) and highest Test R² (0.867).
- **Feature Importance**: `smoker_yes` (importance ~0.619) is the primary driver of charges, followed by `bmi` (~0.211) and `age` (~0.133).
- **Practical Implication**: Smoking status is the most critical factor for insurance pricing, with BMI and age also significant.

For detailed results, see the [Model Comparison Report](docs/report.md). Additional references:
- [Google Docs Report](https://docs.google.com/document/d/1cj_b_HI_4j5r-962Tb0kKuc0-SVa5Hfn05tLpbQ6iXI/edit?usp=sharing)
- [Google Colab Notebook](https://colab.research.google.com/drive/1i3PZyLV0InEuxaax4wGQNY-5R3599u1_?usp=sharing)

## Repository Structure
- `insurance_prediction.py`: Python script containing the code for data preprocessing, model training, evaluation, and analysis.
- `data/insurance.csv`: Dataset used for the analysis.
- `docs/report.md`: Report summarizing the problem statement, model comparison, and key insights.

## Requirements
To run the code, you need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install them using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

### Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### Ensure the Dataset
The `insurance.csv` file should be in the `data/` directory. If not, place it there before running the script.

### Run the Script

If running in a local environment, modify the data loading section to read the file directly:
```python
df = pd.read_csv('data/insurance.csv')
```

### View Results
The script outputs:
- Model performance metrics
- Feature importance for the best model
- Visualizations (bar plots for MSE, R², and feature importance)

A detailed report is available in `docs/`.

## Results
The best model, Random Forest (GridSearchCV), achieved a Test MSE of 20,666,560 and Test R² of 0.867. Smoking status (`smoker_yes`) was the most important feature, contributing 61.9% to predictions, followed by BMI (21.1%) and age (~13.3%). See the report (`docs/report.md`) for more details. Additional references:
- [Google Docs Report](https://docs.google.com/document/d/1cj_b_HI_4j5r-962Tb0kKuc0-SVa5Hfn05tLpbQ6iXI/edit?usp=sharing)
- [Google Colab Notebook](https://colab.research.google.com/drive/1i3PZyLV0InEuxaax4wGQNY-5R3599u1_?usp=sharing)
