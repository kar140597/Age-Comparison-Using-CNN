# Age-Comparison-Using-CNN
The project is part of my course work for Deep learning for engineering applications. I have attempted to create CNN model to compare age of person in two images form UTK-Face. The aim is to compare the image and label who is younger and older not to precisely predict age.I have report and code attached for your reference.
## Overview
This project addresses the problem of **relative age estimation**, where the goal is
to determine **which of two face images represents a younger person**, rather than
predicting the exact chronological age.

The system combines:
- An **absolute age regression model**
- A **pairwise age comparison model** based on deep feature embeddings

This approach is robust to labeling noise and aligns well with human perception,
making it suitable for real-world age comparison tasks.

---

## Problem Statement
Given two face images:
- Estimate their individual ages
- Predict which image corresponds to the **younger person**

Instead of relying only on absolute age prediction, the project formulates
age estimation as a **relative ranking problem**, which improves generalization.

---

## Model Architecture

### 1. Absolute Age Estimation
- CNN-based regression model
- Input: Single face image
- Output: Predicted age (continuous value)

### 2. Feature Extraction
- Intermediate embeddings extracted from the trained age model
- Embeddings capture semantic facial age features

### 3. Relative Age Comparison
- Pairwise model takes concatenated embeddings:
- [embedding_1 || embedding_2]
- - Binary output:
- `1`: Left image is younger
- `0`: Right image is younger

---

## Dataset
- Face images with age annotations
- Preprocessing steps:
- Face resizing
- Normalization
- Data shuffling and splitting (train / validation / test)

*(Dataset name omitted if restricted by license.)*

---

## Training Strategy
- Mean Squared Error (MSE) loss for age regression
- Binary Cross-Entropy loss for relative age classification
- Adam optimizer
- Early stopping to avoid overfitting

---

## Evaluation
- Absolute age prediction error (MAE / RMSE)
- Relative age comparison accuracy
- Qualitative visualization of random image pairs with predictions

The relative age model demonstrates stronger consistency than standalone
absolute age regression.

---

## Results
- Stable convergence during training
- Improved robustness through pairwise comparison
- Correct relative ordering even when absolute age predictions differ slightly
