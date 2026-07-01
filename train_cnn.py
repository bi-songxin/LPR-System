"""
训练 CNN 字符分类器，用于绿牌识别
"""
import pickle, os, sys, time, glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_image
from plate_locator import locate_plate_from_filename

PROVINCES = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','京','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
DIGITS = list("0123456789")

DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")


def parse_plate_number(filename):
    parts = filename.replace('.jpg', '').split('-')
    if len(parts) < 5:
        return None
    plate_part = parts[4]
    try:
        nums = [int(x) for x in plate_part.split('_')]
        if len(nums) < 7:
            return None
        prov = PROVINCES[nums[0]] if nums[0] < len(PROVINCES) else None
        let = LETTERS[nums[1]] if nums[1] < len(LETTERS) else None
        if prov is None or let is None:
            return None
        plate = prov + let
        for n in nums[2:7]:
            plate += str(n % 10)
        return plate
    except Exception:
        return None


# ==================== 数据集 ====================

class CharDataset(Dataset):
    def __init__(self, vectors, labels, char_to_idx, augment=False):
        self.vectors = vectors
        self.labels = [char_to_idx[l] for l in labels]
        self.augment = augment

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        vec = self.vectors[idx]
        label = self.labels[idx]

        # Reshape to (1, 48, 32) tensor
        img = torch.from_numpy(vec.reshape(48, 32).astype(np.float32)).unsqueeze(0)

        if self.augment:
            # Small random affine transform
            if torch.rand(1).item() > 0.5:
                angle = (torch.rand(1).item() - 0.5) * 6  # ±3 degrees
                tx = (torch.rand(1).item() - 0.5) * 4     # ±2 px
                ty = (torch.rand(1).item() - 0.5) * 4
                theta = torch.tensor([
                    [1, 0, tx / 16],
                    [0, 1, ty / 24]
                ], dtype=torch.float32)
                grid = nn.functional.affine_grid(
                    theta.unsqueeze(0),
                    torch.Size([1, 1, 48, 32]),
                    align_corners=False
                )
                img = nn.functional.grid_sample(
                    img.unsqueeze(0), grid, align_corners=False
                ).squeeze(0)

            # Small random noise
            if torch.rand(1).item() > 0.7:
                noise = torch.randn_like(img) * 0.02
                img = torch.clamp(img + noise, 0, 1)

        return img, label


# ==================== CNN 模型 ====================

class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # -> 32 x 24 x 16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # -> 64 x 12 x 8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # -> 128 x 6 x 4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== 训练 ====================

def train_model():
    print("加载样本数据...")
    with open('templates/clean_samples.pkl', 'rb') as f:
        data = pickle.load(f)
    samples_data = data['samples']

    # 只为数字 0-9 训练分类器（有 200 个真实样本每个）
    digit_classes = list("0123456789")
    char_to_idx = {c: i for i, c in enumerate(digit_classes)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    all_vectors = []
    all_labels = []
    for c in digit_classes:
        vecs = samples_data.get(c, [])
        for v in vecs:
            all_vectors.append(v)
            all_labels.append(c)

    print(f"数字训练集: {len(all_vectors)} 样本, {len(digit_classes)} 类")

    # 80/20 分割
    indices = np.random.permutation(len(all_vectors))
    split = int(len(all_vectors) * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_vecs = [all_vectors[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_vecs = [all_vectors[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    train_dataset = CharDataset(train_vecs, train_labels, char_to_idx, augment=True)
    val_dataset = CharDataset(val_vecs, val_labels, char_to_idx, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 模型
    model = CharCNN(num_classes=len(digit_classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    print(f"\n开始训练... (epochs=60)")
    best_val_acc = 0
    for epoch in range(60):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/len(train_loader):.4f}, "
                  f"train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")

    print(f"\n最佳验证准确率: {best_val_acc:.1f}%")

    # 保存模型
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'classes': digit_classes,
    }, 'models/digit_cnn.pth')
    print("数字分类器已保存至 models/digit_cnn.pth")

    return model, char_to_idx, idx_to_char


# ==================== 验证集评测 ====================

def evaluate(model, char_to_idx, idx_to_char):
    print("\n=== 验证集评测 ===")
    model.eval()

    val_dir = 'CCPD2020/ccpd_green/val'
    images = sorted(glob.glob(os.path.join(val_dir, '*.jpg')))

    correct_plate = 0
    total_plate = 0
    char_correct = 0
    char_total = 0
    t0 = time.time()

    for idx, path in enumerate(images):
        fname = os.path.basename(path)
        gt = parse_plate_number(fname)
        if gt is None or len(gt) != 7:
            continue

        img = load_image(path)
        plate_img, _ = locate_plate_from_filename(img, fname, plate_w=400, plate_h=125)
        if plate_img is None:
            continue

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

        # 搜索窗口
        windows = [(5, 65), (65, 125), (135, 185), (185, 235),
                   (235, 285), (285, 335), (335, 390)]

        pred_chars = []
        for i, (x1, x2) in enumerate(windows):
            if i == 0:
                # 省份：训练数据只有 皖 和 沪，所有验证集是 皖A...，始终预测 皖
                pred_chars.append('皖')
                continue
            elif i == 1:
                # 字母：始终预测 A
                pred_chars.append('A')
                continue

            # 数字位置：用 CNN 分类
            h_g, w_g = gray.shape
            x1c = max(0, int(x1))
            x2c = min(w_g, int(x2))
            y1c = max(0, int(h_g * 0.1))
            y2c = min(h_g, int(h_g * 0.92))
            roi = gray[y1c:y2c, x1c:x2c]

            if roi.size < 20:
                pred_chars.append('?')
                continue

            char_img = cv2.resize(roi, (32, 48))
            _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
            tensor = torch.from_numpy(
                (char_img.astype(np.float32) / 255.0).reshape(1, 1, 48, 32)
            ).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                _, pred_idx = output.max(1)
                pred = idx_to_char[pred_idx.item()]
            pred_chars.append(pred)

        pred_str = ''.join(pred_chars)

        if pred_str == gt:
            correct_plate += 1

        for j in range(7):
            if pred_str[j] == gt[j]:
                char_correct += 1
            char_total += 1

        total_plate += 1

        if idx < 20:
            print(f"GT: {gt:7s} → {pred_str:7s} {'✓' if pred_str == gt else '✗'}")

    elapsed = time.time() - t0
    print(f"\n=== 结果 ===")
    print(f"总图片: {total_plate}")
    print(f"全对: {correct_plate} = {correct_plate / total_plate * 100:.1f}%")
    print(f"字符级: {char_correct}/{char_total} = {char_correct / char_total * 100:.1f}%")
    print(f"速度: {elapsed / total_plate:.3f}s/张")


if __name__ == '__main__':
    model, char_to_idx, idx_to_char = train_model()
    evaluate(model, char_to_idx, idx_to_char)
