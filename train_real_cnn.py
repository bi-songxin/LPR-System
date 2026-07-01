"""在真实绿牌数字数据上训练 FastCNN"""
import sys, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==================== 模型定义 ====================

class FastCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==================== 数据集 ====================

class DigitDataset(Dataset):
    def __init__(self, X, y, augment=False, minority_threshold=1500):
        self.X = X
        self.y = y
        self.augment = augment
        self.samples = X.copy()
        self.labels = y.copy()
        if augment:
            self._augment_minority(minority_threshold)

    def _augment_minority(self, threshold):
        """对样本数不足的类别做简单增强"""
        from collections import Counter
        counts = Counter(self.labels)
        extra_X, extra_y = [], []
        for cls in range(10):
            if counts[cls] >= threshold:
                continue
            idx = np.where(self.y == cls)[0]
            needed = threshold - len(idx)
            for _ in range(needed):
                orig = self.X[np.random.choice(idx)].copy()
                # 随机平移 +/- 2px
                tx, ty = np.random.randint(-2, 3, 2)
                shifted = np.roll(orig, (ty, tx), axis=(0, 1))
                if tx > 0:
                    shifted[:, :tx] = 0
                elif tx < 0:
                    shifted[:, tx:] = 0
                if ty > 0:
                    shifted[:ty, :] = 0
                elif ty < 0:
                    shifted[ty:, :] = 0
                extra_X.append(shifted)
                extra_y.append(cls)
        if extra_X:
            self.samples = np.concatenate([self.samples, np.array(extra_X, dtype=np.float32)])
            self.labels = np.concatenate([self.labels, np.array(extra_y)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]
        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# ==================== 训练工具 ====================

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)

def train(model, train_loader, val_loader, device, epochs=50, lr=0.002):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, batches = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        scheduler.step()
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/digit_cnn_real.pth')

        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = "  Epoch %2d | loss=%.4f | train_acc=%.2f%% | val_acc=%.2f%% | best=%.2f%%" % (
                epoch + 1, total_loss / max(batches, 1), train_acc * 100, val_acc * 100, best_acc * 100)
            print(msg, flush=True)

    return best_acc

# ==================== 主流程 ====================

def main():
    print("Loading real digit data...", flush=True)
    with open('models/real_digit_data.pkl', 'rb') as f:
        data = pickle.load(f)
    X, y = data['X'], data['y']
    print("Loaded %d samples, %d classes" % (len(X), len(np.unique(y))))

    # 划分 train/val (80/20)
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print("Train: %d, Val: %d" % (len(X_train), len(X_val)))

    # 数据增强平衡少数类
    train_ds = DigitDataset(X_train, y_train, augment=True, minority_threshold=1500)
    val_ds = DigitDataset(X_val, y_val, augment=False)
    print("After augmentation: Train: %d" % len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # 设备选择
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    model = FastCNN(num_classes=10).to(device)
    params = sum(p.numel() for p in model.parameters())
    print("Model params: %d" % params)

    t0 = time.time()
    best = train(model, train_loader, val_loader, device, epochs=25)
    elapsed = time.time() - t0
    print("\nTraining complete in %.1f min | Best val acc: %.2f%%" % (elapsed / 60, best * 100), flush=True)
    print("Model saved to models/digit_cnn_real.pth", flush=True)

if __name__ == '__main__':
    main()
