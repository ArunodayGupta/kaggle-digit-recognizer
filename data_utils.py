import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_csv_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    labels = train_df["label"].values
    train_images = train_df.drop("label", axis=1).values.reshape(-1, 28, 28)
    test_images = test_df.values.reshape(-1, 28, 28)

    return train_images, labels, test_images

def to_tensor(images, labels=None):
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    if labels is None:
        return x
    y = torch.tensor(labels, dtype=torch.long)
    return x, y

# NOTE:
# The following block reconstructs and saves images from the CSV files
# purely for visualization and understanding of the dataset.
# It is NOT required for training or for generating the Kaggle submission.

def save_images(images, labels=None, split="train"):
    os.makedirs(f"dataset/{split}", exist_ok=True)
    for i in tqdm(range(len(images))):
        name = f"{i}.png" if labels is None else f"{i}_{labels[i]}.png"
        plt.imsave(f"dataset/{split}/{name}", images[i], cmap="gray")
