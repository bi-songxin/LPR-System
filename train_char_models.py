"""训练省份和字母CNN（省/字母/数字 位置专用模型）"""
import sys, os, pickle, time, torch
import cv2, numpy as np
from collections import Counter
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_image
from plate_locator import locate_plate_from_filename
from benchmark import parse_plate_number, CHAR_WINDOWS_8

os.makedirs('models', exist_ok=True)

# ==================== 模型 ====================

class FastCNN(nn.Module):
    def __init__(self, num_classes):
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

class CharDataset(Dataset):
    def __init__(self, X, y, aug_factor=1, min_samples=200):
        self.X = X
        self.y = y
        counts = Counter(y)
        # 复制少数类样本
        all_x, all_y = list(X), list(y)
        for cls in set(y):
            n = counts[cls]
            if n >= min_samples:
                continue
            cls_idx = [j for j, label in enumerate(y) if label == cls]
            for _ in range(min_samples - n):
                orig = X[np.random.choice(cls_idx)].copy()
                # 随机平移 +/-2px
                tx, ty = np.random.randint(-2, 3, 2)
                shifted = np.roll(orig, (ty, tx), axis=(0, 1))
                if tx > 0: shifted[:, :tx] = 0
                elif tx < 0: shifted[:, tx:] = 0
                if ty > 0: shifted[:ty, :] = 0
                elif ty < 0: shifted[ty:, :] = 0
                all_x.append(shifted)
                all_y.append(cls)
        # 额外全局数据增强（旋转、缩放）
        for _ in range((aug_factor - 1) * len(all_x)):
            idx = np.random.randint(len(all_x))
            img = all_x[idx].copy()
            ang = np.random.uniform(-6, 6)
            M = cv2.getRotationMatrix2D((16, 24), ang, 1.0)
            img = cv2.warpAffine(img, M, (32, 48), borderValue=0)
            scale = np.random.uniform(0.9, 1.1)
            scaled = cv2.resize(img, None, fx=scale, fy=scale)
            if scaled.shape != (48, 32):
                pad_h = max(0, 48 - scaled.shape[0])
                pad_w = max(0, 32 - scaled.shape[1])
                scaled = cv2.copyMakeBorder(scaled, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                scaled = scaled[:48, :32]
            all_x.append(scaled)
            all_y.append(all_y[idx])
        self.samples = np.array(all_x, dtype=np.float32)
        self.labels = np.array(all_y)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples[idx]).unsqueeze(0),
                torch.tensor(self.labels[idx], dtype=torch.long))

# ==================== 训练 ====================

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

def train_model(model, train_loader, val_loader, device, epochs=25, name=''):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, batches = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()
        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/char_%s.pth' % name)
        if (epoch + 1) % 5 == 0:
            print("  %s Epoch %2d | val_acc=%.2f%% | best=%.2f%%" % (name, epoch + 1, val_acc * 100, best_acc * 100), flush=True)
    return best_acc

# ==================== 主流程 ====================

def extract_position_data(pos, max_images=3000):
    """提取指定位置的字符数据，返回 (X, y, labels_list)"""
    train_dir = 'CCPD2020/ccpd_green/train'
    images = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    X, y_raw = [], []
    for i, fname in enumerate(images):
        if max_images and i >= max_images:
            break
        try:
            gt = parse_plate_number(fname)
            if gt is None or len(gt) != 8:
                continue
        except Exception:
            continue
        try:
            img = load_image(os.path.join(train_dir, fname))
        except Exception:
            continue
        plate_img, _ = locate_plate_from_filename(img, fname)
        if plate_img is None:
            continue
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        h, w = gray.shape
        x1_raw, x2_raw = CHAR_WINDOWS_8[pos]
        scale_x = w / 440.0
        x1, x2 = int(x1_raw * scale_x), int(x2_raw * scale_x)
        y1, y2 = int(h * 0.08), int(h * 0.92)
        roi = gray[y1:y2, x1:x2]
        if roi.size < 20:
            continue
        char_img = cv2.resize(roi, (32, 48))
        _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
        char_img = char_img.astype(np.float32) / 255.0
        X.append(char_img)
        y_raw.append(gt[pos])

    # 过滤样本数 < 5 的类
    counts = Counter(y_raw)
    valid_labels = {c for c, n in counts.items() if n >= 5}
    valid_indices = [j for j, c in enumerate(y_raw) if c in valid_labels]
    X = X if not valid_indices else [X[j] for j in valid_indices]
    y_raw = [y_raw[j] for j in valid_indices]

    # 建立标签映射
    unique_labels = sorted(set(y_raw))
    label_to_idx = {c: i for i, c in enumerate(unique_labels)}
    y = np.array([label_to_idx[c] for c in y_raw])
    return X, y, unique_labels, label_to_idx

def train_position(pos, max_images=3000):
    """为某个位置训练模型"""
    print("\n=== Position %d ===" % pos, flush=True)
    X_list, y, labels, label_to_idx = extract_position_data(pos, max_images)
    if len(X_list) < 50:
        print("  Not enough data (%d samples), skip" % len(X_list))
        return None, None, 0

    X = np.array(X_list, dtype=np.float32)
    counts = Counter(y)
    print("  Classes: %d, Samples: %d" % (len(labels), len(X)), flush=True)
    for ci, label in enumerate(labels):
        print("    %s: %d" % (label, counts[ci]), flush=True)

    # 划分 train/val
    indices = np.arange(len(X))
    if len(X) > 50:
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    else:
        train_idx, val_idx = indices[:int(len(X)*0.8)], indices[int(len(X)*0.8):]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # 数据增强
    train_ds = CharDataset(X_train, y_train, aug_factor=3, min_samples=200)
    val_ds = CharDataset(X_val, y_val, aug_factor=1, min_samples=200)
    print("  After augmentation: Train=%d, Val=%d" % (len(train_ds), len(val_ds)), flush=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    device = torch.device('cpu')
    model = FastCNN(len(labels)).to(device)
    params = sum(p.numel() for p in model.parameters())
    print("  Model params: %d" % params, flush=True)

    best = train_model(model, train_loader, val_loader, device, epochs=25, name='pos%d' % pos)
    print("  Position %d best val_acc: %.2f%%" % (pos, best * 100), flush=True)
    return model, (labels, label_to_idx), best

if __name__ == '__main__':
    results = {}
    for pos in [0, 1, 2, 3]:
        model_data = train_position(pos, max_images=3000)
        if model_data[0] is not None:
            results[pos] = model_data
        else:
            print("  Skipped position %d (insufficient data)" % pos)

    print("\n=== Summary ===")
    for pos, (model, (labels, _), best) in results.items():
        print("  Pos %d: %d classes, val_acc=%.2f%%" % (pos, len(labels), best * 100))
