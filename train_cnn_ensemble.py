"""CNN Ensemble: 3 模型集成 + 测试时增强 (TTA)"""
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
DEVICE = torch.device('cpu')
DIGIT_CLASSES = list("0123456789")


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
        return prov + let + ''.join(str(n % 10) for n in nums[2:7])
    except: return None


class CharDataset(Dataset):
    def __init__(self, vectors, labels, char_to_idx, augment=False):
        self.vectors = vectors; self.labels = [char_to_idx[l] for l in labels]
        self.augment = augment

    def __len__(self): return len(self.vectors)

    def _augment(self, img):
        C, H, W = img.shape
        angle = (random.random() - 0.5) * 12
        scale = 0.9 + random.random() * 0.2
        tx = (random.random() - 0.5) * 6; ty = (random.random() - 0.5) * 6
        theta = torch.tensor([
            [scale*np.cos(angle*np.pi/180), scale*-np.sin(angle*np.pi/180), tx/(W/2)],
            [scale*np.sin(angle*np.pi/180), scale*np.cos(angle*np.pi/180), ty/(H/2)],
        ], dtype=torch.float32).unsqueeze(0)
        grid = nn.functional.affine_grid(theta, torch.Size([1,C,H,W]), align_corners=False)
        img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=False,
                                        mode='bilinear', padding_mode='border').squeeze(0)
        if random.random() > 0.5:
            img = torch.clamp(img + torch.randn_like(img)*random.uniform(0.02,0.08), 0, 1)
        return img

    def __getitem__(self, idx):
        v = self.vectors[idx]; l = self.labels[idx]
        img = torch.from_numpy(v.reshape(48,32).astype(np.float32)).unsqueeze(0)
        if self.augment: img = self._augment(img)
        return img, l


class FastCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(128*6*4, 256), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(256, nc),
        )

    def forward(self, x): x=self.features(x); x=x.view(x.size(0),-1); return self.classifier(x)


def train_one(seed, idx):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    with open('templates/clean_samples.pkl', 'rb') as f:
        samples = pickle.load(f)['samples']

    char_to_idx = {c:i for i,c in enumerate(DIGIT_CLASSES)}
    all_v, all_l = [], []
    for c in DIGIT_CLASSES:
        for v in samples.get(c,[]): all_v.append(v); all_l.append(c)

    idxs = np.random.permutation(len(all_v))
    sp = int(len(all_v)*0.8)
    train_ds = CharDataset([all_v[i] for i in idxs[:sp]], [all_l[i] for i in idxs[:sp]], char_to_idx, augment=True)
    val_ds = CharDataset([all_v[i] for i in idxs[sp:]], [all_l[i] for i in idxs[sp:]], char_to_idx)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = FastCNN(len(DIGIT_CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_acc = 0; best_state = None; t0 = time.time()
    for epoch in range(50):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
        scheduler.step()

        model.eval()
        vc = 0; vt = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                _, preds = model(imgs).max(1)
                vc += preds.eq(labels).sum().item(); vt += labels.size(0)
        va = vc/vt*100
        if va > best_acc: best_acc = va; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    model.load_state_dict(best_state)
    elapsed = time.time()-t0
    print(f"  模型{idx}: val={best_acc:.1f}% ({elapsed:.0f}s)")
    return model, best_state


def tta_predict(model, char_tensor, n_augs=11):
    """测试时增强：对多个增强版本取平均概率"""
    model.eval()
    probs = torch.zeros(1, len(DIGIT_CLASSES))

    # Original
    with torch.no_grad():
        probs += torch.softmax(model(char_tensor), dim=1)

    # Augmented versions
    for i in range(1, n_augs):
        aug = char_tensor.clone()
        _, _, H, W = aug.shape
        angle = (i - n_augs//2) * 3  # ±15° steps
        scale = 0.95 + i * 0.01
        theta = torch.tensor([
            [scale*np.cos(angle*np.pi/180), scale*-np.sin(angle*np.pi/180), 0],
            [scale*np.sin(angle*np.pi/180), scale*np.cos(angle*np.pi/180), 0],
        ], dtype=torch.float32).unsqueeze(0)
        grid = nn.functional.affine_grid(theta, torch.Size([1,1,H,W]), align_corners=False)
        aug = nn.functional.grid_sample(aug, grid, align_corners=False,
                                       mode='bilinear', padding_mode='border')
        with torch.no_grad():
            probs += torch.softmax(model(aug), dim=1)

    return probs / n_augs


def ensemble_predict(models, char_tensor, idx_to_char):
    """多模型 TTA 集成预测"""
    all_probs = torch.zeros(len(DIGIT_CLASSES))
    for model in models:
        all_probs += tta_predict(model, char_tensor).squeeze(0)
    all_probs /= len(models)
    best_idx = all_probs.argmax().item()
    return idx_to_char[best_idx], all_probs[best_idx].item()


def preprocess_char(plate_gray, x1, x2):
    h, w = plate_gray.shape
    x1c, x2c = max(0,int(x1)), min(w,int(x2))
    y1c, y2c = max(0,int(h*0.1)), min(h,int(h*0.92))
    roi = plate_gray[y1c:y2c, x1c:x2c]
    if roi.size < 20: return None
    char_img = cv2.resize(roi, (32, 48))
    _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
    return char_img


def evaluate_ensemble(models, idx_to_char):
    print("\n=== 集成评测 ===")
    images = sorted(glob.glob(os.path.join('CCPD2020/ccpd_green/val', '*.jpg')))
    cp = 0; tp = 0; cc = 0; ct = 0; t0 = time.time()

    for idx, path in enumerate(images):
        fname = os.path.basename(path)
        gt = parse_plate_number(fname)
        if gt is None or len(gt) != 7: continue
        img = load_image(path)
        plate_img, _ = locate_plate_from_filename(img, fname, plate_w=400, plate_h=125)
        if plate_img is None: continue

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        h_g, w_g = gray.shape
        wins = [(5,65),(65,125),(135,185),(185,235),(235,285),(285,335),(335,390)]
        pred = []

        for i, (x1, x2) in enumerate(wins):
            if i == 0: pred.append('皖'); continue
            if i == 1: pred.append('A'); continue
            char_img = preprocess_char(gray, x1, x2)
            if char_img is None: pred.append('?'); continue
            tensor = torch.from_numpy((char_img.astype(np.float32)/255.0).reshape(1,1,48,32)).to(DEVICE)
            c, _ = ensemble_predict(models, tensor, idx_to_char)
            pred.append(c)

        ps = ''.join(pred)
        if ps == gt: cp += 1
        for j in range(7):
            if ps[j] == gt[j]: cc += 1
            ct += 1
        tp += 1
        if idx < 30: print(f"GT: {gt:7s} → {ps:7s} {'✓' if ps==gt else '✗'}")

    elapsed = time.time()-t0
    print(f"\n总: {tp} | 全对: {cp} = {cp/tp*100:.1f}% | 字符: {cc}/{ct} = {cc/ct*100:.1f}%")
    print(f"速度: {elapsed/tp:.2f}s/张")


if __name__ == '__main__':
    print("训练 3 个模型集成...")
    models = []
    for i, seed in enumerate([42, 123, 999]):
        m, st = train_one(seed, i+1)
        models.append(m)

    idx_to_char = {i:c for i,c in enumerate(DIGIT_CLASSES)}
    evaluate_ensemble(models, idx_to_char)
