# Image Classifier – CNN vs ResNet (CIFAR-10)

A deep learning project comparing a custom Convolutional Neural Network (CNN) with a ResNet architecture on the CIFAR-10 dataset, deployed as an interactive web app using Streamlit.

---

## Demo

Upload an image and see how two different models classify it in real time
Compare predictions, confidence scores, and model behavior
Link to demo: https://imageclassifier-6rav7zmksm9mzqofllth4n.streamlit.app/

---

## Project Overview

This project explores:

* Building a CNN from scratch
* Improving architecture step-by-step
* Introducing a state-of-the-art model (ResNet)
* Comparing model performance on the same task
* Deploying the solution as an interactive application

---

## 🏗️ Models

### Custom CNN

* 3 convolutional layers
* Batch Normalization
* Dropout regularization
* Fully connected classifier

**Performance:**
~85% accuracy on CIFAR-10

---

### ResNet-18 (Transfer Learning)

* Pretrained architecture adapted to CIFAR-10
* Input resized to 224×224
* Fine-tuned on dataset

**Performance:**
~95–96% accuracy on CIFAR-10

---

## Model Comparison

| Feature        | CNN      | ResNet-18 |
| -------------- | -------- | --------- |
| Accuracy       | ~85%     | ~95%+     |
| Training time  | Fast     | Slower    |
| Complexity     | Low      | High      |
| Generalization | Moderate | Strong    |

The app highlights differences in predictions and confidence between models.

---

## Features

* Image classification (upload or example images)
* CNN vs ResNet comparison
* Top-3 predictions with confidence scores
* Model disagreement detection
* Model metadata (accuracy, epoch)
* Streamlit web interface

---

## Project Structure

```
image_classifier/
│
├── app/                # Streamlit app
│   ├── app.py
│   └── examples/
│
├── data/				# CIFAR 10 dataset
│
├── models/             # Saved models (dynamically downloaded for app)
│
├── notebooks/			# EDA & confusion matrix
│	├── EDA.ipynb
│	├── CNN_confusion_matrix.ipynb
│	└── ResNet_confusion_matrix.ipynb
│
├── src/                # Core code
│   ├── cnn_model.py
│   ├── resnet.py
│   ├── data.py
│	├── train_cnn.py
│	└── train_resnet.py
│
└── README.md
```

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* Streamlit
* DVC (for model versioning)
* Google Colab (training ResNet)

---

## Training Insights

* CNN required architectural tuning (BatchNorm, Dropout, deeper layers)
* Scheduler improved stability but not final accuracy significantly
* ResNet dramatically improved performance due to:

  * deeper architecture
  * residual connections
  * better feature extraction

---

## Deployment

The app is deployed using Streamlit.

Models are:

* stored externally (Google Drive)
* downloaded dynamically at runtime

---

## Example Use Case

Upload any image (e.g. dog, car, cat) and:

* compare predictions from both models
* observe confidence differences
* see when models disagree

---

## Future Improvements

* Add Grad-CAM (model explainability)
* Support more datasets
* Add model selection toggle
* Improve UI/UX further
* Deploy on a custom domain

---

## Author

Created as a deep learning portfolio project focused on:

* practical ML engineering
* model comparison
* deployment


