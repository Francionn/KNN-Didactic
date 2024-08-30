# K-Nearest Neighbors (KNN) Didactic Demonstration

This project provides a didactic demonstration of the five principal methods used in the mathematical concept of K-Nearest Neighbors (KNN) with scikit-learn.

## Methods Overview

### 1. `train_test_split()`

Before applying any methods, the data is split into **X** and **y**:

- **X (Features)**: Attributes of the data.
- **y (Labels)**: Values associated with each sample.

  - **Train**: The model learns patterns from the training data.
  - **Test**: The model's performance is evaluated using the test data after training.

### 2. `fit()`

This method calculates the Euclidean distance for each test sample relative to the training samples.

### 3. `KNeighborsClassifier()`

This method identifies the K-nearest neighbors for a given test sample.

### 4. `predict()`

This method determines the most common class among the K-nearest neighbors.

### 5. `accuracy_score()`

This method calculates the accuracy by dividing the number of correct predictions by the total number of predictions.
