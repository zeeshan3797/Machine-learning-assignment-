Customer Spending Prediction using Linear Regression
This project explores the use of linear regression (simple and multiple) to analyze and predict customer spending scores based on demographic and financial attributes such as age and annual income. The project demonstrates the performance, challenges, and outcomes of linear regression models.

Features
Simple Linear Regression: Uses a single feature (age) to predict customer spending scores.
Multiple Linear Regression: Incorporates multiple features (age and annual income) to improve predictive power.
Performance Evaluation: Metrics include Mean Squared Error (MSE) and R-squared values.
Data Visualization: Scatter plots and regression lines for better interpretability of results.
Dataset
The dataset consists of customer demographic and spending data, available on Kaggle. It includes features such as:

Age
Gender
Annual Income
Spending Score

Preprocessing
Data Splitting: Divided into training (80%) and testing (20%).
Feature Scaling: Normalization applied to ensure model efficiency.

Models

Simple Linear Regression:
Predictor: Age
Slope and Intercept calculated to establish a linear relationship.
Moderate performance with low R-squared values.

Multiple Linear Regression:
Predictors: Age and Annual Income
Captures more complex relationships, but still limited in predictive power.
Results
Simple Linear Regression:
R-squared: 0.0518 (weak predictor)
MSE: 468.05

Multiple Linear Regression:
R-squared: 0.0196 (marginal improvement with multiple features)
MSE: High (indicating significant prediction errors)

Challenges
Simple Linear Regression:
Limited by using a single predictor (age).
Multiple Linear Regression:
Marginal improvement due to insufficient feature selection.
Model underperforms due to lack of stronger predictors.

Future Work
Incorporate additional features like gender, geographical location, or purchase history.
Experiment with advanced regression models or machine learning techniques.
Perform feature selection or engineering to improve predictions.

Installation
Clone the repository:
bash
git clone https://github.com/username/customer-spending-linear-regression.git
cd customer-spending-linear-regression
Install the required libraries:
bash

Run the analysis script:
bash
LinearReg (Single & Multiple).py
Usage
simple_linear_regression() to analyze using a single predictor.
multiple_linear_regression() to include multiple predictors.
Results include performance metrics, regression plots, and insights.

Requirements
Python 3.7+
pandas
scikit-learn
matplotlib
Project Structure

├── LinearReg (Single & Multiple).py      # Train and test script
├── data/                       # Dataset directory
├── results/                    # Output results (plots, metrics)
├── Report.pdf                  # Project documentation
└── README.md                   # Project documentation
License
This project is licensed under University of hertfordshire.

Author
Developed by Zeeshan Ali 23036973.

Feel free to customize this README and adapt it to your project's structure and details!