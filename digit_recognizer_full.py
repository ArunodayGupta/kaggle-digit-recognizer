import torch
from torch import nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T

train_df=pd.read_csv("train[1].csv")
test_df=pd.read_csv("test[1].csv")

train_df.head()

train_label=train_df["label"].values

train_pixels=train_df.drop("label",axis=1).values
test_pixels=test_df.values

base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
test_dir  = os.path.join(base_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_images = train_pixels.reshape(-1, 28, 28)

test_images = test_pixels.reshape(-1, 28, 28)

# NOTE:
# The following two for loops saves images from the CSV files
# purely for visualization and understanding of the dataset.
# It is NOT required for training or for generating the Kaggle submission.

for i in tqdm(range(len(train_images)), desc="Saving train images"):
    plt.imsave(f"{train_dir}/{i}_{train_label[i]}.png", train_images[i], cmap="gray")

for i in tqdm(range(len(test_images)), desc="Saving test images"):
    plt.imsave(f"{test_dir}/{i}.png", test_images[i], cmap="gray")

train_images_tensor = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)
train_label_tensor = torch.tensor(train_label, dtype=torch.long)

test_images_tensor=torch.tensor(test_images,dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(train_images_tensor, train_label_tensor)

train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True)

import requests

if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import accuracy_fn

batch = next(iter(train_dataloader))
print(type(batch))
try:
    print(len(batch))
except:
    print("Not iterable")
print(batch)

class digits_model(nn.Module):
  def __init__(self,input_shape:int,hidden_units:int,output_shape:int):
    super().__init__()
    self.block1=nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1,
                  stride=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1,
                  stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.block2=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1,
                  stride=1
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1,
                  stride=1
                  ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.block3=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1,
                  stride=1
                  ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  padding=1,
                  stride=1
                  ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)
    )
    self.classifier=nn.Sequential(
        nn.Flatten(),
        nn.Linear(hidden_units*3*3, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(in_features=128,out_features=output_shape)
    )

  def forward(self,x):
    x=self.block1(x)
    x=self.block2(x)
    x=self.block3(x)
    x=self.classifier(x)

    return x

torch.manual_seed(42)

model=digits_model(input_shape=1,hidden_units=20,output_shape=10)

loss_fn=nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer=torch.optim.AdamW(params=model.parameters(),lr=0.001,weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=15)

def train_model(model:torch.nn.Module,data_loader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,optimizer:torch.optim.Optimizer,accuracy_fn):
  train_loss,train_acc=0,0

  model.train()

  for batch,(x,y) in enumerate(data_loader):
    y_pred=model(x)

    loss=loss_fn(y_pred,y)
    train_loss+=loss.item()
    train_acc+=accuracy_fn(y,y_pred.argmax(dim=1))

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

  scheduler.step()

  train_loss/=len(data_loader)
  train_acc/=len(data_loader)
  print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

epochs=15

for epoch in range(epochs):
  print(f"Epoch: {epoch+1}")

  train_model(
      model=model,
      data_loader=train_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      accuracy_fn=accuracy_fn
  )

model.eval()
with torch.no_grad():
  predictions=model(test_images_tensor)

  submission=pd.DataFrame({"ImageId":range(1,len(predictions)+1),"Label":predictions.argmax(dim=1).numpy()})

  submission.to_csv("submission.csv",index=False)
