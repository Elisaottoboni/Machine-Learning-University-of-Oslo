# Machine Learning Project: CNN-Based Detection of Venusian Volcanoes in Magellan SAR Data

## Index
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Objectives](#project-objectives)
4. [Relevance](#relevance)
5. [Project Files](#project-files)
6. [Technical Details](#technical-details)

## Overview
This project is developed as part of the [FYS-STK3155/FYS4155 - Data Analysis and Machine Learning](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/index-eng.html) (Autumn 2024) course at the University of Oslo. The main objective is to implement and evaluate different types of classifiers, including Logistic Regression, Random Forest, FFNN, and CNN, with a particular emphasis on CNNs, for the identification of volcanic formations in images of Venus's surface.

## Dataset
The [dataset](https://www.kaggle.com/datasets/fmena14/volcanoesvenus) used in this project originates from NASA's Magellan mission, which captured high-resolution radar images of the surface of Venus during its 1990-1994 mission. These images have been processed to a computationally feasible size of 110x110 pixels and contain binary labels indicating the presence or absence of volcanic features. Additional labels, such as volcano type, radius, and count, are included, although these have varying degrees of labeling confidence.

## Project Objectives
- Implement the following classifiers: Logistic Regression, Random Forest, FFNN, and CNN.
- Address challenges posed by the dataset, such as:
    - High dimensionality (addressed through PCA for dimensionality reduction).
    - Class imbalance (mitigated with oversampling techniques like SMOTE).
    - Noise, uncertainties in the labeling process, and corrupted images (identified and removed during preprocessing).
- Evaluate and compare the performance of the implemented classifiers using metrics like precision, recall, F1-score, accuracy, ROC curve, and AUC.

## Relevance
This project aligns with ongoing efforts in planetary science to develop automated systems capable of analyzing planetary imagery. It complements the objectives of NASA's upcoming VERITAS mission, which aims to further map Venus's surface using advanced radar and spectroscopic techniques.

## Project Files

[CNN_Based_Detection_Report.pdf](CNN_Based_Detection_Report.pdf): Final report detailing the methods, implementation, and results of the project. Includes analysis of classifiers (CNN, FFNN, Logistic Regression, Random Forest), preprocessing techniques (PCA, SMOTE), and key performance metrics.

[CNN.ipynb](CNN.ipynb): Implements a Convolutional Neural Network (CNN) for detecting volcanic formations in Venus's surface images. Includes both baseline (Model A) and improved (Model B) architectures, focusing on metrics such as accuracy, AUC, and loss dynamics. Model B incorporates dropout for enhanced generalization.

[FNN.ipynb](FNN.ipynb): Implements a Feedforward Neural Network (FFNN) using PyTorch and Keras frameworks. Optimized using Adam and RMSprop optimizers, achieving significant accuracy improvements through learning rate tuning and dropout layers.

[logistic_regression.ipynb](logistic_regression.ipynb): Contains the implementation of Logistic Regression for binary classification. Preprocessing includes PCA for dimensionality reduction and SMOTE to address class imbalance. Key metrics: precision (0.78), recall (0.79), and F1-score (0.78).

[random_forest.ipynb](random_forest.ipynb): Implements the Random Forest classifier, with parameter tuning via grid search. Evaluates performance using metrics like precision (0.55), recall (0.66), and F1-score (0.60). Includes analysis of overfitting and pruning strategies.

[methods.py](methods.py): A utility script providing shared functions for preprocessing, dimensionality reduction (PCA), and data balancing (SMOTE). Supports all classifier implementations.

## Technical Details

### .gitignore

The `.gitignore` file specifies files and directories to be excluded from version control. This includes virtual environment folders, trained model directories, cache files, notebook checkpoints, and generated images or PDFs, ensuring a clean and focused repository.

### requirements.txt
The `requirements.txt` file lists the Python dependencies needed to run this project. To install these dependencies, use the following command:
```bash
pip install -r requirements.txt
```

---

We welcome any feedback, suggestions, or contributions to improve the classifiers and analyses! Feel free to reach out at [lotopedro29@gmail.com](mailto:lotopedro29@gmail.com)