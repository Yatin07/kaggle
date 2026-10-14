"""
RSNA Knee Abnormality Detection — Wave 4: PyTorch Multi-Plane CNN Training & Validation
Loads the cached 2.5D slices from Wave 3 and trains MultiPlaneKneeNet on Kaggle GPU T4.
Execution time: ~30-45 seconds.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

print("=" * 70)
print("RSNA Knee: Wave 4 — PyTorch Multi-Plane ResNet Training & Validation")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Active Training Device: {DEVICE}")

# 1. Load Pre-Cached 2.5D Data from Wave 3
cache_path = '/kaggle/working/knee_25d_cache.npz'
if not os.path.exists(cache_path):
    raise FileNotFoundError(f"Missing cache file: {cache_path}. Please run Wave 3 cell first!")

data = np.load(cache_path)
sag_tensor = torch.tensor(data['sag'], dtype=torch.float32)
cor_tensor = torch.tensor(data['cor'], dtype=torch.float32)
axi_tensor = torch.tensor(data['axi'], dtype=torch.float32)
labels_tensor = torch.tensor(data['labels'], dtype=torch.float32)

print(f"Loaded Cache from: {cache_path}")
print(f"Input Tensors: Sag={sag_tensor.shape}, Cor={cor_tensor.shape}, Axi={axi_tensor.shape}")
print(f"Labels Tensor: {labels_tensor.shape} (12 abnormality targets)")

# 2. Multi-Plane 2.5D Fusion Neural Network
class MultiPlaneKneeNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        # ResNet18 backbone natively supported in PyTorch torchvision
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feature_dim = 512
        
        # Classifier fuses features from all 3 anatomical planes (512 * 3 = 1536)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, sag, cor, axi):
        f_sag = self.backbone(sag).flatten(1)
        f_cor = self.backbone(cor).flatten(1)
        f_axi = self.backbone(axi).flatten(1)
        fused = torch.cat([f_sag, f_cor, f_axi], dim=1)
        return self.classifier(fused)

# 3. Setup Cross-Validation Split & DataLoaders
full_dataset = TensorDataset(sag_tensor, cor_tensor, axi_tensor, labels_tensor)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(kf.split(np.arange(len(labels_tensor))))

train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=8, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=8, shuffle=False)

print(f"Cross-Validation Split: {len(train_idx)} Train cases, {len(val_idx)} Validation cases")

# 4. Supervised Training Loop
model = MultiPlaneKneeNet(num_classes=12).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

EPOCHS = 15
best_auc = 0.0

print(f"\n[TRAINING] Starting 15 Epochs on {DEVICE}...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for sag, cor, axi, y in train_loader:
        sag, cor, axi, y = sag.to(DEVICE), cor.to(DEVICE), axi.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(sag, cor, axi)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)

    scheduler.step()
    train_loss = total_loss / len(train_idx)

    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for sag, cor, axi, y in val_loader:
            sag, cor, axi = sag.to(DEVICE), cor.to(DEVICE), axi.to(DEVICE)
            preds = torch.sigmoid(model(sag, cor, axi)).cpu().numpy()
            val_preds.append(preds)
            val_targets.append(y.numpy())

    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)

    # Macro ROC-AUC calculation
    aucs = []
    for c in range(12):
        if len(np.unique(val_targets[:, c])) > 1:
            aucs.append(roc_auc_score(val_targets[:, c], val_preds[:, c]))
    mean_auc = np.mean(aucs) if len(aucs) > 0 else 0.5

    print(f"Epoch {epoch:2d}/{EPOCHS} | Train BCE Loss: {train_loss:.4f} | Val Macro ROC-AUC: {mean_auc:.4f}")

    if mean_auc > best_auc:
        best_auc = mean_auc
        save_path = '/kaggle/working/best_multiplane_knee_model.pth'
        torch.save(model.state_dict(), save_path)
        print(f"  --> Saved new best checkpoint: AUC = {best_auc:.4f}")

print("\n" + "=" * 70)
print(f"[SUCCESS] Wave 4 Training Complete! Best Validation Macro ROC-AUC: {best_auc:.4f}")
print("Saved checkpoint: /kaggle/working/best_multiplane_knee_model.pth")
print("=" * 70)
