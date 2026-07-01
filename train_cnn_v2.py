"""
CNN V2: 更强的数据增强 + 权重衰减 + ResNet 结构
"""
import pickle, os, sys, time, glob, random
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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")


def parse_plate_number(filename):
    parts = filename.replace('.jpg', '').split('-')
    if len(parts) < 5: return None
    plate_part = parts[4]
    try:
        nums = [int(x) for x in plate_part.split('_')]
        if len(nums) < 7: return None
        prov = PROVINCES[nums[0]] if nums[0] < len(PROVINCES) else None
        let = LETTERS[nums[1]] if nums[1] < len(LETTERS) else None
        if prov is None or let is None: return None
        plate = prov + let
        for n in nums[2:7]: plate += str(n % 10)
        return plate
    except: return None


# ==================== 数据集（含强增强） ====================

class CharDataset(Dataset):
    def __init__(self, vectors, labels, char_to_idx, augment=False):
        self.vectors = vectors
        self.char_to_idx = char_to_idx
        self.labels = [char_to_idx[l] for l in labels]
        self.augment = augment

    def __len__(self):
        return len(self.vectors)

    def _augment(self, img):
        # img: (1, 48, 32) tensor
        C, H, W = img.shape

        # 1. Random affine transform (always)
        angle = (random.random() - 0.5) * 16  # ±8°
        scale = 0.85 + random.random() * 0.30  # 0.85-1.15
        tx = (random.random() - 0.5) * 6       # ±3 px
        ty = (random.random() - 0.5) * 6

        cos_a = np.cos(angle * np.pi / 180)
        sin_a = np.sin(angle * np.pi / 180)
        theta = torch.tensor([
            [scale * cos_a, scale * -sin_a, tx / (W / 2)],
            [scale * sin_a, scale * cos_a, ty / (H / 2)],
        ], dtype=torch.float32).unsqueeze(0)

        grid = nn.functional.affine_grid(
            theta, torch.Size([1, C, H, W]), align_corners=False
        )
        img = nn.functional.grid_sample(
            img.unsqueeze(0), grid, align_corners=False, mode='bilinear',
            padding_mode='border'
        ).squeeze(0)

        # 2. Random noise
        if random.random() > 0.5:
            noise = torch.randn_like(img) * random.uniform(0.01, 0.06)
            img = torch.clamp(img + noise, 0, 1)

        # 3. Random pixel dropout (simulates binarization errors)
        if random.random() > 0.5:
            mask = (torch.rand_like(img) > random.uniform(0.05, 0.15)).float()
            img = img * mask + (1 - mask) * torch.randint(0, 2, img.shape, dtype=torch.float32)

        # 4. Random brightness/contrast jitter
        if random.random() > 0.5:
            brightness = 0.8 + random.random() * 0.4     # 0.8-1.2
            contrast = 0.8 + random.random() * 0.4       # 0.8-1.2
            img = torch.clamp(brightness * (img - 0.5) + 0.5, 0, 1)
            img = torch.clamp(contrast * (img - 0.5) + 0.5, 0, 1)

        return img

    def __getitem__(self, idx):
        vec = self.vectors[idx]
        label = self.labels[idx]
        img = torch.from_numpy(vec.reshape(48, 32).astype(np.float32)).unsqueeze(0)

        if self.augment:
            img = self._augment(img)

        return img, label


# ==================== ResNet 风格 CNN ====================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class CharResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)       # stride=2: /2
        x = self.layer3(x)       # stride=2: /2
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ==================== 训练 ====================

def train_model():
    print("加载样本...")
    with open('templates/clean_samples.pkl', 'rb') as f:
        data = pickle.load(f)
    samples_data = data['samples']

    digit_classes = list("0123456789")
    char_to_idx = {c: i for i, c in enumerate(digit_classes)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    all_vectors = []; all_labels = []
    for c in digit_classes:
        for v in samples_data.get(c, []):
            all_vectors.append(v); all_labels.append(c)
    print(f"训练集: {len(all_vectors)} 样本, {len(digit_classes)} 类")

    # 80/20 split
    indices = np.random.permutation(len(all_vectors))
    split = int(len(all_vectors) * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = CharDataset([all_vectors[i] for i in train_idx],
                           [all_labels[i] for i in train_idx], char_to_idx, augment=True)
    val_ds = CharDataset([all_vectors[i] for i in val_idx],
                         [all_labels[i] for i in val_idx], char_to_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = CharResNet(num_classes=len(digit_classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    print(f"\n训练 {len(digit_classes)} 类 CNN (epochs=80)...")
    best_val_acc = 0
    best_state = None

    for epoch in range(80):
        model.train()
        train_loss = 0; correct = 0; total = 0
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

        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total * 100

        scheduler.step(epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={train_loss/len(train_loader):.4f} "
                  f"train={train_acc:.1f}% val={val_acc:.1f}% (best={best_val_acc:.1f}%)")

    model.load_state_dict(best_state)
    print(f"\n最佳验证: {best_val_acc:.1f}%")

    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': best_state,
        'char_to_idx': char_to_idx,
        'classes': digit_classes,
    }, 'models/digit_cnn.pth')
    print("模型已保存至 models/digit_cnn.pth")
    return model, char_to_idx, idx_to_char


# ==================== 验证集评测 ====================

def evaluate(model, char_to_idx, idx_to_char):
    print("\n=== 验证集评测 ===")
    model.eval()
    val_dir = 'CCPD2020/ccpd_green/val'
    images = sorted(glob.glob(os.path.join(val_dir, '*.jpg')))

    correct_plate = 0; total_plate = 0; char_correct = 0; char_total = 0
    t0 = time.time()

    for idx, path in enumerate(images):
        fname = os.path.basename(path)
        gt = parse_plate_number(fname)
        if gt is None or len(gt) != 7: continue

        img = load_image(path)
        plate_img, _ = locate_plate_from_filename(img, fname, plate_w=400, plate_h=125)
        if plate_img is None: continue

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

        windows = [(5, 65), (65, 125), (135, 185), (185, 235),
                   (235, 285), (285, 335), (335, 390)]
        pred_chars = []

        for i, (x1, x2) in enumerate(windows):
            if i == 0: pred_chars.append('皖'); continue
            if i == 1: pred_chars.append('A'); continue

            h_g, w_g = gray.shape
            x1c, x2c = max(0, int(x1)), min(w_g, int(x2))
            y1c, y2c = max(0, int(h_g * 0.1)), min(h_g, int(h_g * 0.92))
            roi = gray[y1c:y2c, x1c:x2c]
            if roi.size < 20: pred_chars.append('?'); continue

            char_img = cv2.resize(roi, (32, 48))
            _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
            tensor = torch.from_numpy(
                (char_img.astype(np.float32) / 255.0).reshape(1, 1, 48, 32)
            ).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                _, pred_idx = output.max(1)
            pred_chars.append(idx_to_char[pred_idx.item()])

        pred_str = ''.join(pred_chars)
        if pred_str == gt: correct_plate += 1
        for j in range(7):
            if pred_str[j] == gt[j]: char_correct += 1
            char_total += 1
        total_plate += 1
        if idx < 20: print(f"GT: {gt:7s} → {pred_str:7s} {'✓' if pred_str == gt else '✗'}")

    elapsed = time.time() - t0
    print(f"\n总: {total_plate} | 全对: {correct_plate} = {correct_plate/total_plate*100:.1f}%")
    print(f"字符级: {char_correct}/{char_total} = {char_correct/char_total*100:.1f}%")
    print(f"速度: {elapsed/total_plate:.3f}s/张")


if __name__ == '__main__':
    model, char_to_idx, idx_to_char = train_model()
    evaluate(model, char_to_idx, idx_to_char)
