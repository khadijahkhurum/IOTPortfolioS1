import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
OUTPUT_DIR = "DL_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("Data_Processed/combined_daily.csv")

# Encode labels
df["label"] = df["activity_level"].astype("category").cat.codes

# Use steps + distance
data = df[["steps_total", "distance_total_km"]].values
labels = df["label"].values

# Create sequences (7-day window)
SEQ_LEN = 7
X_seq, y_seq = [], []

for i in range(len(data) - SEQ_LEN):
    X_seq.append(data[i:i+SEQ_LEN])
    y_seq.append(labels[i+SEQ_LEN])

X_seq = torch.FloatTensor(X_seq)
y_seq = torch.LongTensor(y_seq)

# Train/test split
train_size = int(0.75 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

# --------------------
# LSTM Model
# --------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])

model = LSTMModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------
# Training + Tracking
# --------------------
loss_history = []
acc_history = []

for epoch in range(30):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Store loss
    loss_history.append(loss.item())

    # Accuracy
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_train).float().mean().item()
        acc_history.append(acc)

print("LSTM training complete.")

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
