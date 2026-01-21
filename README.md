
# 🚗 AI-Based Driver Drowsiness Detection System

## 📌 Project Overview
This project implements a deep learning-based Driver Drowsiness Detection system using Convolutional Neural Networks (CNN). The model is trained on a labeled facial image dataset to classify drivers into drowsy and non-drowsy states.

The system leverages image preprocessing, data augmentation, and CNN-based feature extraction to automatically learn visual patterns associated with fatigue, enabling accurate offline classification.

## 🎯 Project Objectives
- Detect driver drowsiness using facial image data  
- Build and train a CNN-based classification model  
- Apply data augmentation to improve generalization  
- Evaluate performance using validation metrics  
- Develop a scalable offline inference pipeline  

## 🛠 Tech Stack

### Programming & Frameworks
- Python  
- TensorFlow / Keras  
- OpenCV  

### Libraries
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- KaggleHub  

## 📂 Dataset Information

**Dataset Name:** Drowsy Detection Dataset  
**Source:** Kaggle  
**Author:** yasharjebraeily  

Dataset Link:  
https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset  

### Dataset Description
The dataset contains labeled facial images representing different driver alertness states. These images are used to train and validate the CNN classification model.

---

## ⚙️ Data Pipeline Workflow

1. Download dataset using KaggleHub API  
2. Load training and validation image directories  
3. Apply image normalization and rescaling  
4. Perform data augmentation  
5. Generate training batches  
6. Train CNN model  
7. Evaluate classification performance  

---

## 🧠 Model Architecture

The CNN model architecture includes:

- Convolutional layers for feature extraction  
- MaxPooling layers for spatial reduction  
- Flatten layer for vector conversion  
- Fully connected Dense layers  
- Softmax output layer for multi-class classification  

### Training Configuration

- Activation Function: ReLU  
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Input Normalization: Rescale = 1./255  

---

## 🏗 Data Augmentation Techniques

The following augmentation techniques were applied to improve generalization:

- Zoom augmentation  
- Shear transformation  
- Horizontal flipping  
- Image rescaling  

This reduces overfitting and improves model robustness.

---


