# Machine Learning Project: CNN-Based Detection of Venusian Volcanoes in Magellan SAR Data

## Index
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Objectives](#project-objectives)
4. [Relevance](#relevance)
5. [Project Files](#project-files)

## Overview
This project is developed as part of the [FYS-STK3155/FYS4155 - Data Analysis and Machine Learning](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/index-eng.html) (Autumn 2024) course at the University of Oslo. The main objective is to implement and evaluate different types of classifiers, including Logistic Regression, Random Forest, FFNN, and CNN, with a particular emphasis on CNNs, for the identification of volcanic formations in images of Venus's surface.

## Dataset
The [dataset](https://www.kaggle.com/datasets/fmena14/volcanoesvenus) used in this project originates from NASA's Magellan mission, which captured high-resolution radar images of the surface of Venus during its 1990-1994 mission. These images have been processed to a computationally feasible size of 110x110 pixels and contain binary labels indicating the presence or absence of volcanic features. Additional labels, such as volcano type, radius, and count, are included, although these have varying degrees of labeling confidence.

## Project Objectives
- Implement the following classifiers: Logistic Regression, Random Forest, FFNN, and CNN.
- Address challenges posed by the dataset, such as:
  - High dimensionality (addressed through PCA for dimensionality reduction).
  - Class imbalance (mitigated with oversampling techniques like SMOTE).
  - Noise and uncertainties in the labeling process.
- Evaluate and compare the performance of the implemented classifiers using metrics like precision, recall, F1-score, and accuracy.

## Relevance
This project aligns with ongoing efforts in planetary science to develop automated systems capable of analyzing planetary imagery. It complements the objectives of NASA's upcoming VERITAS mission, which aims to further map Venus's surface using advanced radar and spectroscopic techniques.

## Project Files
This section will provide an overview of the main files and scripts in the repository once the project is finalized.

---

We welcome any feedback, suggestions, or contributions to improve the classifiers and analyses!
