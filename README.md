# Kaggle Digit Recognizer – CNN (PyTorch)

This repository contains my solution to the **Kaggle Digit Recognizer** competition using a custom **Convolutional Neural Network (CNN)** implemented in **PyTorch**.

I achieved an approximate **rank of 381** on the public Kaggle leaderboard.

---

## 📌 Problem Statement

The Digit Recognizer competition is based on the classic **MNIST handwritten digit classification** task. The goal is to correctly classify grayscale images of handwritten digits (0–9) using supervised learning.

* Input: 28×28 grayscale images (provided as pixel values in CSV format)
* Output: Digit label from 0 to 9

---

## 🧠 Approach

I approached this problem using a deep learning pipeline built from scratch in PyTorch:

* Converted raw pixel values into tensors of shape `(N, 1, 28, 28)`
* Designed a multi-block CNN architecture for feature extraction
* Trained the model using modern optimization and regularization techniques

---

## 🏗️ Model Architecture

The CNN consists of:

* **3 Convolutional Blocks**

  * `Conv2D → ReLU → Conv2D → ReLU → MaxPool`
* **Fully Connected Classifier**

  * Flatten layer
  * Dense layer with ReLU
  * Dropout for regularization
  * Final output layer (10 classes)

This architecture balances performance and simplicity while avoiding overfitting.

---

## ⚙️ Training Details

* Framework: **PyTorch**
* Loss Function: `CrossEntropyLoss` (with label smoothing = 0.05)
* Optimizer: **AdamW**
* Learning Rate Scheduler: **CosineAnnealingLR**
* Batch Size: 32
* Epochs: 15

These choices were made to ensure stable training and good generalization.

---

## 📊 Results

* **Kaggle Digit Recognizer Leaderboard Rank:** ~381

The model performs competitively among thousands of submissions, validating the effectiveness of the CNN architecture and training strategy.

---

## 📁 Project Structure

```
├── train.py            # Training and evaluation script
├── model.py            # CNN model definition
├── submission.csv      # Kaggle submission file
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/ArunodayGupta/kaggle-digit-recognizer.git
   cd kaggle-digit-recognizer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model and generate submission:

   ```bash
   python train.py
   ```

The script will produce a `submission.csv` file compatible with Kaggle.

---

## 📚 What I Learned

* Designing and training CNNs from scratch in PyTorch
* Applying regularization techniques such as dropout and label smoothing
* Using learning rate scheduling to improve convergence
* End-to-end Kaggle workflow: training → evaluation → submission

---

## 🔗 Competition Link

* Kaggle Digit Recognizer: [https://www.kaggle.com/competitions/digit-recognizer](https://www.kaggle.com/competitions/digit-recognizer)

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use and learn from it.

---

If you have suggestions or would like to discuss improvements, feel free to open an issue or connect with me on LinkedIn.

