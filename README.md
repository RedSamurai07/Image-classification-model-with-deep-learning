# Image classification model with deep learning

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview

This project focuses on developing an automated image classification system using deep learning techniques. Utilizing the TensorFlow and Keras frameworks, the notebook outlines a complete pipeline for processing visual data, from initial preprocessing to model evaluation. The system is designed to handle multi-class classification, converting raw image input into meaningful categorical predictions.

### Executive Summary

The case study demonstrates the implementation of a neural network model aimed at accurately identifying objects within an image dataset. The methodology includes:

- Preprocessing: Implementation of Resizing and Rescaling layers to standardize image inputs for the model.

- Model Architecture: Construction of a Sequential model utilizing Flatten and Dense layers, along with regularizers to prevent overfitting.

- Reproducibility: Setting a specific random seed (111) to ensure consistent results across different training runs.

- Evaluation & Visualization: The project features a robust evaluation phase that randomly samples test images and compares predicted labels against true categories using a 5x3 visualization grid. This provides immediate visual confirmation of the model's accuracy, where correct predictions are typically highlighted in green and incorrect ones in red.
  
### Goal
The primary objectives for this case study are:

1. Develop a Scalable Classifier: Build a deep learning model capable of distinguishing between various categories within the dataset.

2. Optimize Model Performance: Use techniques such as rescaling and regularization to improve the model's ability to generalize to new, unseen images.

3. Establish Interpretability: Create visualization tools (like prediction plots) to analyze where the model succeeds and where it fails, allowing for targeted improvements.

4. Ensure Scientific Consistency: Implement rigorous data handling and seed-setting to allow for the validation and replication of findings.

### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/1GK4tnY4_YfX8ccNhVtedEoGpsUUGOwzlysEVEQr_xpA/edit?gid=1748548740#gid=1748548740)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| -------- | -------- | -------- | 
| -------- | -------- | -------- | 
| -------- | -------- | -------- | 
| -------- | -------- | -------- | 
| -------- | -------- | -------- | 
| -------- | -------- | -------- | 
| -------- | -------- | -------- | 
| -------- | -------- | -------- | 

### Tools
- Excel : Google Sheets - Check for data types, Table formatting
- SQL : Big QueryStudio - Querying, manipulating, and managing data in relational databases in 
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, inferential Statistics, Data manipulation and Analysis(Numpy, Pandas),Visualization (Matplotlib, Seaborn), Feature Engineering, Hypothesis Testing
  
### Analysis
1). Python

## Hypothesis testing:


### Insights

- Reproducibility & Consistency: The project implements a fixed random seed (111), which ensures that the results are reproducible across different training sessions, making the experimental findings scientifically valid.

- Standardized Data Pipeline: The use of Rescaling(1./255) and Resizing layers directly within the model architecture ensures that all input data is normalized to a common scale (0 to 1), which typically helps neural networks converge faster.

- Overfitting Mitigation: The model incorporates L2 regularization (regularizers.l2(0.01)) in its dense layers. This insight suggests that the initial models may have suffered from high variance or overfitting, necessitating penalty terms to keep the weights small.

- Visual Interpretability: The inclusion of a 15-sample visualization grid with color-coded labels (green for correct, red for incorrect) provides immediate qualitative feedback on where the model struggles, such as confused classes or poor lighting conditions.

### Recommendations

- Transition to Convolutional Neural Networks (CNNs): The current architecture relies on flattening image data into a single vector for Dense layers. To significantly improve performance, it is recommended to implement Conv2D and MaxPooling2D layers, which are specifically designed to capture spatial hierarchies and local patterns in images.

- Implement Data Augmentation: To improve the model's ability to generalize to new data, you should incorporate data augmentation layers (such as RandomFlip, RandomRotation, and RandomZoom) during the preprocessing phase. This helps the model become invariant to orientation and positioning.

- Leverage Transfer Learning: Instead of training from scratch, consider using a pre-trained model like MobileNetV2 or ResNet50 as a feature extractor. This would provide a more robust starting point, especially if the dataset size is limited.

- Optimize Hyperparameters: It is recommended to implement a Learning Rate Scheduler or use EarlyStopping callbacks. This would prevent the model from overshooting the global minimum and save computational resources by stopping training once the validation loss stops improving.

- Evaluate with a Confusion Matrix: Beyond visual sampling, generating a full confusion matrix would provide a quantitative insight into which specific classes are being misclassified as others, allowing for more targeted data collection or feature engineering for those specific categories.
