# Cervical Cancer Detection through Machine Learning-Based Classification Models Integrated with CNN

A medical image classification project that combines **Convolutional Neural Networks (CNNs)** with traditional machine-learning classifiers to classify cervical-cell images from Pap smears as **Normal** or **Abnormal**.

## Authors

- **Samaher S. Alsharif**
- **Shahd H. Altalhi**

## Project Overview

Cervical cancer screening commonly relies on visual examination of Pap-smear cytology images. Manual screening can be time-consuming and is affected by variability in interpretation. This project investigates an automated image-classification pipeline using deep learning for feature extraction and machine learning for final classification.

The implemented workflow integrates:

- Exploratory Data Analysis (EDA)
- Image preprocessing and normalization
- Data augmentation
- CNN-based deep feature extraction
- Principal Component Analysis (PCA)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Accuracy, precision, recall, F1-score, classification reports, and confusion matrices

## Objectives

1. Develop machine-learning methods for classifying cervical cells into normal and abnormal groups using Pap-smear images.
2. Integrate CNN feature extraction with SVM and KNN classifiers.
3. Evaluate the models using standard classification metrics.
4. Compare the performance of the hybrid CNN + SVM and CNN + KNN approaches.

## Dataset — SIPaKMeD

The project uses the **SIPaKMeD** cervical-cell image dataset. The study documentation reports **4,049 cell images** extracted from **966 image clusters**.

The five original cervical-cell categories are:

| Category | Number of Cells |
|---|---:|
| Superficial / Intermediate | 831 |
| Parabasal | 787 |
| Koilocytotic | 825 |
| Metaplastic | 793 |
| Dyskeratotic | 813 |
| **Total** | **4,049** |

For the binary classification task, the project groups the classes as follows:

**Normal**
- Superficial-Intermediate
- Parabasal

**Abnormal**
- Koilocytotic
- Dyskeratotic
- Metaplastic

Dataset source: Kaggle — SIPaKMeD cervical cancer dataset.

## Methodology

### 1. Exploratory Data Analysis

The notebook first examines the dataset by:

- Counting images in each category
- Visualizing category counts with a bar chart
- Visualizing category proportions with a pie chart
- Displaying sample cervical-cell images from every class

### 2. Image Preprocessing

The preprocessing pipeline:

- Reads images using OpenCV
- Resizes every image to **64 × 64 pixels**
- Normalizes pixel values to the range **[0, 1]**
- Encodes Normal/Abnormal labels using `LabelEncoder`

### 3. Data Split

The dataset is divided into:

- **70% Training**
- **15% Validation**
- **15% Testing**

A fixed random state is used to support reproducibility.

### 4. Data Augmentation

Training images are augmented using `ImageDataGenerator` with:

- Rotation up to 30 degrees
- Zoom augmentation
- Horizontal flipping

The purpose is to increase training diversity and improve model generalization.

## CNN Architecture

The CNN used in the project contains:

1. Input layer: `64 × 64 × 3`
2. `Conv2D(32, 3×3, ReLU)`
3. `MaxPool2D(2×2)`
4. `Conv2D(64, 3×3, ReLU)`
5. `MaxPool2D(2×2)`
6. Flatten layer
7. Dense layer with 128 neurons and ReLU
8. Dropout = 0.5
9. Output layer with 2 neurons and Softmax

Training configuration:

- Optimizer: **Adam**
- Loss: **Sparse Categorical Cross-Entropy**
- Epochs: **10**
- Batch size: **32**

## CNN Feature Extraction

Instead of using only the CNN's final classification output, the project removes the final output layer and uses the remaining CNN as a **deep feature extractor**.

The extracted representations are generated for the training, validation, and test sets and passed to PCA and the classical ML classifiers.

## Principal Component Analysis (PCA)

PCA is applied to the CNN-extracted features to reduce their dimensionality to **50 principal components**.

This creates a compact feature representation for the SVM and KNN classifiers.

## CNN + SVM Pipeline

The first hybrid model uses:

`Pap-smear Image → Preprocessing → CNN Feature Extraction → PCA → Linear SVM → Prediction`

SVM configuration:

- Kernel: **Linear**
- `C = 1.0`

### Reported Result

**CNN + SVM Test Accuracy: 91.61%**

The notebook also produces a classification report and confusion matrix for detailed evaluation.

## CNN + KNN Pipeline

The second hybrid model uses:

`Pap-smear Image → Preprocessing → CNN Feature Extraction → PCA → KNN → Prediction`

KNN configuration:

- `n_neighbors = 5`

### Reported Result

**CNN + KNN Test Accuracy: 94.57%**

Among the two evaluated hybrid pipelines, **CNN + KNN achieved the higher test accuracy** in the submitted experiment.

## Model Comparison

| Model | Test Accuracy |
|---|---:|
| CNN + SVM | **91.61%** |
| CNN + KNN | **94.57%** |

## Evaluation Metrics

The project evaluates classification performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

These metrics provide a broader view of model behavior than accuracy alone.

## Single-Image Prediction

The notebook also includes helper functions for predicting a new cervical-cell image using either trained hybrid pipeline.

The inference sequence is:

1. Load the image
2. Resize to 64 × 64
3. Normalize pixel values
4. Extract CNN features
5. Apply the trained PCA transformation
6. Predict with SVM or KNN
7. Return the binary class: **Normal** or **Abnormal**

## Technologies and Libraries

- Python
- TensorFlow / Keras
- scikit-learn
- OpenCV
- NumPy
- pandas
- Matplotlib
- Seaborn
- Pillow
- Jupyter Notebook

## Repository Structure

```text
Cervical-Cancer-Detection-CNN-ML/
├── Cervical_Cancer_Detection_CNN_ML.ipynb
├── requirements.txt
└── README.md
```

## How to Run

1. Download the SIPaKMeD dataset.
2. Keep the original category folder structure used by the dataset.
3. Install the dependencies from `requirements.txt`.
4. Open `Cervical_Cancer_Detection_CNN_ML.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
5. Change the `base_dir` variable to your local SIPaKMeD dataset path.
6. Run the notebook cells in order.

```bash
pip install -r requirements.txt
```

## Key Findings

- CNNs can be used as feature extractors for cervical-cell images before applying classical machine-learning classifiers.
- PCA provides a lower-dimensional representation of the learned CNN features.
- Both SVM and KNN successfully classify the extracted features in the project experiment.
- CNN + KNN achieved the best test accuracy among the two evaluated hybrid models, with **94.57%**, compared with **91.61%** for CNN + SVM.

## Project Scope

This repository presents an academic machine-learning experiment and is intended for research and educational purposes. The model is **not a clinically validated diagnostic system** and should not be used for medical decision-making.

## Authors

**Samaher S. Alsharif**  
**Shahd H. Altalhi**
