"""从CCPD绿牌训练集中提取真实数字字符，用于CNN训练"""
import sys, os, pickle, time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_image
from plate_locator import locate_plate_from_filename
from benchmark import parse_plate_number, CHAR_WINDOWS_8

def extract_digits(max_images=3000):
    """从训练集提取数字字符图像和标签"""
    train_dir = 'CCPD2020/ccpd_green/train'
    images = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])

    X, y = [], []
    total_processed = 0
    total_extracted = 0

    for i, fname in enumerate(images):
        if max_images and i >= max_images:
            break

        if (i + 1) % 500 == 0:
            print("  [%d/%d] extracted %d digits from %d plates..." % (
                i + 1, min(len(images), max_images or len(images)),
                total_extracted, total_processed))

        try:
            gt = parse_plate_number(fname)
            if gt is None or len(gt) != 8:
                continue
        except Exception:
            continue

        try:
            img = load_image(os.path.join(train_dir, fname))
            if img is None:
                continue
        except Exception:
            continue

        plate_img, _ = locate_plate_from_filename(img, fname)
        if plate_img is None:
            continue

        total_processed += 1
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        h, w = gray.shape

        # 提取数字位置（跳过字母位置）
        for pos in range(2, 8):
            # 只收集数字位置（0-9），跳过字母
            if not gt[pos].isdigit():
                continue

            x1_raw, x2_raw = CHAR_WINDOWS_8[pos]
            scale_x = w / 440.0
            x1, x2 = int(x1_raw * scale_x), int(x2_raw * scale_x)
            y1, y2 = int(h * 0.08), int(h * 0.92)

            roi = gray[y1:y2, x1:x2]
            if roi.size < 20:
                continue

            char_img = cv2.resize(roi, (32, 48))
            # Match recognizer preprocessing: THRESH_BINARY → /255
            # Characters are dark → 0.0, background is bright → 1.0
            _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
            char_img = char_img.astype(np.float32) / 255.0

            label = int(gt[pos])
            X.append(char_img)
            y.append(label)
            total_extracted += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print("\nExtracted %d digit samples from %d plates" % (len(X), total_processed))

    # Class distribution
    for c in range(10):
        count = np.sum(y == c)
        print("  Class %d: %d samples (%.1f%%)" % (c, count, count / len(y) * 100))

    # Save
    os.makedirs('models', exist_ok=True)
    with open('models/real_digit_data.pkl', 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    print("Saved to models/real_digit_data.pkl")
    return X, y

if __name__ == '__main__':
    extract_digits(max_images=3000)
