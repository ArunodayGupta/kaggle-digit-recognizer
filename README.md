# Kaggle Digit Recognizer – CNN (PyTorch)

This repository contains my solution to the **Kaggle Digit Recognizer** competition using a custom **Convolutional Neural Network (CNN)** implemented in **PyTorch**.

- Achieved a public leaderboard score of approximately **0.993**
  on the Kaggle Digit Recognizer competition.
- This placed the solution in the **top ~25% of submissions**
  at the time of evaluation.
-The result was obtained using a custom CNN trained with the **AdamW**
optimizer and a **cosine learning rate schedule**.

---

## 📌 Problem Statement

The Kaggle Digit Recognizer competition is based on the classic **MNIST handwritten digit classification** task.

Each sample in the dataset is provided as a **flattened vector of 784 pixel values (28 × 28)** stored in CSV format. The goal is to correctly classify each image into one of the **10 digit classes (0–9)**.

* Input: CSV rows with 784 pixel intensity values
* Output: Digit label (0–9)

---

## 🧠 Data Representation (Important Note)

Although the data is provided as CSV files, each row represents a **single grayscale image flattened into a 1D vector of length 784**.

For both training and visualization purposes, these vectors are **reshaped back into their original 2D form (28 × 28)**:

```
784 → 28 × 28
```

This reshaping step **reconstructs the original image structure** and is required for convolutional neural networks, which operate on spatial data.

---

## 🧠 Approach

I approached this problem using an end-to-end deep learning pipeline built from scratch in PyTorch:

* Loaded CSV data and reshaped flattened pixel vectors into 28×28 images
* Converted images into tensors of shape `(N, 1, 28, 28)`
* Designed a custom CNN architecture for feature extraction
* Trained the model using modern optimization and regularization techniques

---

## 🏗️ Model Architecture

The CNN consists of:

* **3 Convolutional Blocks**

  * `Conv2D → ReLU → Conv2D → ReLU → MaxPooling`
* **Classifier Head**

  * Flatten layer
  * Fully connected layer with ReLU
  * Dropout (p = 0.3) for regularization
  * Final linear layer with 10 outputs

This architecture balances performance and simplicity while avoiding overfitting.

---

## ⚙️ Training Details

* Framework: **PyTorch**
* Loss Function: `CrossEntropyLoss` (with label smoothing = 0.05)
* Optimizer: **AdamW**
* Learning Rate Scheduler: **CosineAnnealingLR**
* Batch Size: 32
* Epochs: 15

These choices were made to ensure stable training and better generalization.

---

## 📊 Results

* **Kaggle Digit Recognizer Leaderboard Rank:** ~381

The model performs competitively among thousands of submissions, validating the effectiveness of the CNN architecture and training strategy.

---

---

## 👁️ Data Visualization (Optional)

The dataset images are stored in CSV format, which can be unintuitive to inspect directly. For better understanding and exploration, an **optional visualization script** is included.

This script:

* Reshapes flattened 784-length vectors into 28×28 images
* Saves or displays reconstructed grayscale digit images

⚠️ **Note:** This step is **not used during training** and exists purely for visualization and learning purposes.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/kaggle-digit-recognizer.git
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

The script will generate a `submission.csv` file compatible with Kaggle.

---

## 📚 What I Learned

* How image datasets can be stored as flattened vectors and reconstructed
* Designing and training CNNs from scratch using PyTorch
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
