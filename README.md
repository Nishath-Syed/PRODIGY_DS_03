# Data Science Internship Task 3: Building a Decision Tree Classifier on Bank Marketing Data

## Overview

This repository contains a project that focuses on building a Decision Tree Classifier to predict whether a customer will purchase a product or service based on the Bank Marketing dataset. The goal is to develop a model that classifies customer purchase behavior and to visualize the decision-making process of the classifier.

## Dataset

The dataset used for this analysis is the Bank Marketing dataset, which includes various features about customers and their responses to a marketing campaign, specifically whether they subscribed to a term deposit.

## Task Description

The primary objectives of this task are:

### Data Preparation

- **Load the Dataset**: Import the dataset into a Pandas DataFrame to facilitate analysis.
- **Inspect the Data**: Review the dataset's structure and contents to understand its features and target variable.
- **Handle Categorical Variables**: Convert categorical variables into numerical formats using one-hot encoding to prepare them for analysis.
- **Split Data**: Divide the dataset into training and testing sets to evaluate model performance.

### Model Building

- **Initialize and Train the Model**: Train a Decision Tree Classifier on the training data with suitable hyperparameters to control model complexity and prevent overfitting.
- **Predict and Evaluate**: Use the trained model to make predictions on the test data and evaluate its performance.

### Visualization

- **Visualize the Decision Tree**: Create a graphical representation of the decision tree to illustrate how the model makes decisions based on different features.

## Steps Implemented

### 1. Import Necessary Libraries

Libraries for data manipulation (`pandas`), model building (`scikit-learn`), and visualization (`matplotlib`) are imported to support the analysis and model training.

### 2. Load and Inspect the Dataset

The dataset is loaded into a DataFrame, and initial data inspection is performed to understand the structure, including columns and the first few rows.

### 3. Prepare the Data

- **Define Features and Target**: The dataset is split into features (X) and the target variable (y).
- **Encode Categorical Variables**: Categorical variables are converted into numerical format using one-hot encoding.
- **Split Data**: The dataset is split into training and testing sets to enable model training and evaluation.

### 4. Initialize and Train the Decision Tree Classifier

- A Decision Tree Classifier is initialized with hyperparameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` to manage model complexity.
- The classifier is trained on the training data.

### 5. Predict and Evaluate the Model

- The trained model is used to make predictions on the test data.
- Model performance is evaluated using metrics such as accuracy and a detailed classification report, which provides insights into precision, recall, and F1-score.

### 6. Visualize the Decision Tree

- The decision tree is visualized using a plot to show the splits and decisions made by the model. This helps in understanding how the model uses different features to make predictions.

## Results

### Model Performance

- The performance of the model is summarized by accuracy and other metrics from the classification report, providing an understanding of how well the model predicts customer purchases.

### Decision Tree Visualization

- The visualization of the decision tree reveals how the model makes decisions, showing the feature splits and the conditions under which decisions are made.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **Scikit-Learn**: For building and evaluating the Decision Tree Classifier.
- **Matplotlib**: For visualizing the decision tree and other plots.

