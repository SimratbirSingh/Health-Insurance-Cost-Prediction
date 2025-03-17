# Health-Insurance-Cost-Prediction

## Overview

The project uses a dataset (`insurance.csv`) containing information on age, sex, BMI, number of children, smoking status, region, and insurance charges.  It explores the data, preprocesses it, and trains several regression models (Linear Regression, Random Forest, Gradient Boosting, XGBoost) to predict insurance costs. The project compares model performance and includes an interactive prediction function.

## Files

*   **`insurance.csv`**: The dataset.
*   **`Health_insurance_cost_prediction.ipynb`**: Jupyter Notebook with the code for data analysis, modeling, and prediction.
*   **`linear_regression_model.pkl`**: Saved Linear Regression model (using joblib).
*   **`label_encoders.pkl`**: Saved label encoders (using joblib)

## Dependencies

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   plotly
*   scikit-learn
*   xgboost
*   joblib

Install dependencies using:  `pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost joblib`

## Usage

1.  **Run the Notebook:** Open and run `Health_insurance_cost_prediction.ipynb` to perform the analysis and train the models.  This notebook includes exploratory data analysis (EDA), data preprocessing, model training, evaluation, and comparison.
2.  **Make Predictions:** The notebook includes an interactive prediction function at the end.  Follow the prompts to enter individual details and get a predicted insurance charge.  The prediction uses the trained Linear Regression model.

## Models Used

*   Linear Regression
*   Random Forest
*   Gradient Boosting Machine (GBM)
*   XGBoost

## Results
The project evaluate and compare different machine learning models using R-squared, MSE, RMSE and MAPE. Cross-validation performed for reliable results.
