import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
OUTPUT_DIR = "DL_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Load your data
# --------------------
df = pd.read_csv("Data_Processed/combined_daily.csv")

X = df[["steps_total", "distance_total_km"]].values
y = df["activity_level"].astype("category").cat.codes.values  # encode labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# --------------------
# Model
# --------------------
class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 3 classes
        )

    def forward(self, x):
        return self.net(x)

model = DNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------
# Training + Visual Tracking
# --------------------
loss_history = []
acc_history = []

for epoch in range(50):
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backprop
    loss.backward()
    optimizer.step()

    # Store loss
    loss_history.append(loss.item())

    # Compute accuracy
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_train).float().mean().item()
        acc_history.append(acc)

print("DNN training complete.")

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
