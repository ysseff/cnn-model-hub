# 🧠 CNN Model Hub

A PyQt5 desktop application for designing, training, and testing Convolutional Neural Networks (CNNs) using TensorFlow / Keras with a user-friendly GUI.

## 🚀 Features

- Upload Training and Validation datasets (RGB & Grayscale)
- Visual Model Builder (Input, Conv2D, BatchNorm, Pooling, Dropout, Dense, Flatten)
- Train with real-time Accuracy and Loss visualization
- Validate models and display results
- Classify images with predicted label & confidence
- Load pretrained .keras models

## 📦 Requirements

```bash
pip install PyQt5 tensorflow keras matplotlib
```

## 🛠️ How to Run

```bash
python cnn_model_hub.py
```

## 📂 Dataset Folder Structure

```
dataset/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

Each subfolder represents a class label.

## 🖼️ Application Screenshot

<img width="1268" alt="cnn_model_hub" src="https://github.com/user-attachments/assets/1755af82-b706-4f56-abdc-5c5bd952b093" />


## 🧠 Workflow

1. Upload Training & Validation data
2. Build CNN architecture visually
3. Train model & monitor metrics
4. Validate model performance
5. Classify test images & display results
