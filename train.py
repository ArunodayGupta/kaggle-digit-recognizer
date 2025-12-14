# train.py

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn

from config import *
from data_utils import load_csv_data, save_images, to_tensor
from dataset import DigitDataset
from model import DigitCNN

from helper_functions import accuracy_fn

torch.manual_seed(RANDOM_SEED)

# Load data
train_images, labels, test_images = load_csv_data(
    "train[1].csv", "test[1].csv"
)

# Optional visualization (for understanding)
save_images(train_images, labels, split="train")
save_images(test_images, split="test")

x_train, y_train = to_tensor(train_images, labels)

dataset = DigitDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = DigitCNN(1, HIDDEN_UNITS, NUM_CLASSES)

loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_acc = 0, 0

    for x, y in loader:
        preds = model(x)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_fn(y, preds.argmax(dim=1))

    scheduler.step()

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss/len(loader):.4f} | "
        f"Train Acc: {train_acc/len(loader):.2f}%"
    )

torch.save(model.state_dict(), "model.pt")
