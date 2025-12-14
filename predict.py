import torch
import pandas as pd
from model import DigitCNN
from data_utils import load_csv_data, to_tensor
from config import *

_, _, test_images = load_csv_data("train[1].csv", "test[1].csv")
x_test = to_tensor(test_images)

model = DigitCNN(HIDDEN_UNITS)
model.load_state_dict(torch.load("model.pt"))
model.eval()

with torch.no_grad():
    preds = model(x_test).argmax(dim=1).numpy()

submission = pd.DataFrame({
    "ImageId": range(1, len(preds) + 1),
    "Label": preds
})
submission.to_csv("submission.csv", index=False)
