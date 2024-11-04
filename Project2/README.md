# Machine Learning Project: Classification and Regression with FFNN

## Index
1. [Overview](#overview)
2. [Key Information](#key-information)
3. [Project Files](#project-files)

## Overview
This project is developed as part of the [FYS-STK3155/FYS4155 - Data Analysis and Machine Learning](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/index-eng.html) (Autumn 2024) course at the University of Oslo. The main objective is to explore both regression and classification problems by developing a custom feed-forward neural network (FFNN) code. For the regression task, we reused previously implemented OLS and Ridge regression code from earlier projects to benchmark the FFNN. For the classification task, we implemented logistic regression to compare its performance with that of the FFNN. Additionally, we utilized previously implemented algorithms and metrics, including cross-validation and bootstrap, to enhance our analysis and evaluation.

## Key Information
- **Goal**: Develop and analyze custom implementations for gradient descent, feed-forward neural networks, and logistic regression, focusing on both regression and classification problems. 
- **Datasets**: 
  - **Regression**: Various functions (e.g., a simple polynomial, the Franke function).
  - **Classification**: The [Wisconsin Breast Cancer dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data), used to classify tumors based on their characteristics as benign or malignant.

## Project Files

- **project2___Evaluation_of_Feed_Forward_Neural_Network_Performance.pdf**: Contains the full paper detailing our analysis process, including methodology, results, and discussions for the entire project.
  
- **Project2.pdf**: This document provides the assignment details and guidelines given by our professor for this project.

- **gd_analytic**: This file contains the implementation of the methods utilized in the `partA_analytical_function`.

- **gd_autograd**: This file contains the implementation of the methods utilized in the `partA_autograd`.

- **partA_analytical_function.ipynb**: This notebook includes the complete analysis of gradient descent methods, and evaluations of various optimizations.

- **partA_autograd.ipynb**:This notebook includes the complete analysis of gradient descent methods, with autograd and evaluations of various optimizations.

- **part_B_C_D.ipynb**: This file implements and evaluates the Feed Forward Neural Network (FFNN) for both regression and classification tasks.

- **part_E.ipynb**: This notebook implements logistic regression and compares its performance with the previously developed FFNN for classification.

- **FFNeuralNetwork.py**: Contains the class implementation of the neural network, including the primary functions used in the project.

- **packages.py**: Lists and imports all libraries used in this project for easy setup and configuration.

## Technical Details

### .gitignore
The `.gitignore` file is used to specify files and directories that Git should ignore. In this project, it prevents unnecessary files such as `__pycache__/`, `.ipynb_checkpoints/`, and compiled Python files (`*.pyc`) from being tracked, keeping the repository clean.

### requirements.txt
The `requirements.txt` file lists the Python dependencies needed to run this project. To install these dependencies, use the following command:
```bash
pip install -r requirements.txt
