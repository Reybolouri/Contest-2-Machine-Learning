import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# 1. Load CSVs
train_url = 'https://raw.githubusercontent.com/Reybolouri/Contest-2-Machine-Learning/main/data/train.csv'
test_url  = 'https://raw.githubusercontent.com/Reybolouri/Contest-2-Machine-Learning/main/data/test.csv'
train_df  = pd.read_csv(train_url)
test_df   = pd.read_csv(test_url)

# 2. Extract numpy arrays
X = train_df.drop(['id','y'], axis=1).values.astype(np.float32)
y = (train_df['y'].values - 1).astype(np.int64)   # zero-based

X_test = test_df.drop('id', axis=1).values.astype(np.float32)

# 3. Dataset class
class ImageCSV(Dataset):
    def __init__(self, features, labels=None):
        self.X = features
        self.y = labels
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        img = self.X[idx].reshape(3, 32, 32)
        if self.y is None:
            return img
        return img, self.y[idx]

full_ds  = ImageCSV(X, y)
test_ds  = ImageCSV(X_test, None)

# 4. Train/val split and loaders
train_size = int(0.8 * len(full_ds))
val_size   = len(full_ds) - train_size
train_ds, val_ds = random_split(full_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2)

# 5. Your model (reuse TorchVisionCNN from above)
class TorchVisionCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 6. Setup
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model    = TorchVisionCNN().to(device)
criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr=1e-3)
epochs   = 10

# 7. Training loop
for epoch in range(1, epochs+1):
    # Train
    model.train()
    train_loss, train_corr, train_total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        train_loss  += loss.item() * imgs.size(0)
        train_corr  += (preds == labels).sum().item()
        train_total += imgs.size(0)

    # Validate
    model.eval()
    val_corr, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            preds= out.argmax(dim=1)
            val_corr  += (preds == labels).sum().item()
            val_total += imgs.size(0)

    print(f"Epoch {epoch:2d} — "
          f"Train Loss: {train_loss/train_total:.4f}, "
          f"Train Acc: {train_corr/train_total:.4f}, "
          f"Val Acc: {val_corr/val_total:.4f}")

# 8. (Optional) Quick test-set inference
model.eval()
all_preds = []
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(device)
        out  = model(imgs)
        all_preds.extend(out.argmax(dim=1).cpu().numpy() + 1)

# all_preds now holds your 1/2/3 predictions for test.csv
submission = pd.DataFrame({
    'id': test_df['id'],
    'y':  all_preds
})
submission.to_csv('cnn.csv', index=False)
print("Saved → cnn.csv")