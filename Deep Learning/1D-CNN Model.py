import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
OUTPUT_DIR = "DL_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("Data_Processed/combined_daily.csv")

df["label"] = df["activity_level"].astype("category").cat.codes

data = df[["steps_total", "distance_total_km"]].values
labels = df["label"].values

SEQ_LEN = 7
X_seq, y_seq = [], []

for i in range(len(data) - SEQ_LEN):
    X_seq.append(data[i:i+SEQ_LEN])
    y_seq.append(labels[i+SEQ_LEN])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Reshape for CNN: (batch, channels, sequence_len)
X_seq = torch.FloatTensor(X_seq).permute(0, 2, 1)  # (N, 2 channels, 7 seq)
y_seq = torch.LongTensor(y_seq)

train_size = int(0.75 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# --------------------
# 1D CNN Model
# --------------------
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Linear(32 * 3, 3)  # 3 classes

    def forward(self, x):
        x = self.conv(x)             # (N, 32, 3)
        x = x.view(x.size(0), -1)    # flatten
        return self.fc(x)

model = CNN1D()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------
# Training + Tracking
# --------------------
loss_history = []
acc_history = []

for epoch in range(30):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    # Save loss
    loss_history.append(loss.item())

    # Save accuracy
    with torch.no_grad():
        preds = torch.argmax(pred, dim=1)
        acc = (preds == y_train).float().mean().item()
        acc_history.append(acc)

print("1D CNN training complete.")

# --------------------
# Plot Loss Curve
# --------------------
plt.figure(figsize=(6,4))
plt.plot(loss_history, label="Training Loss", color="#FF3F8E", linewidth=2)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

plt.figure(figsize=(6,4))
plt.plot(acc_history, label="Training Accuracy", color="#4F6BED", linewidth=2)
plt.title("Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
plt.close()

