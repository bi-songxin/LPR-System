"""Benchmark plate recognition accuracy on CCPD validation set."""
import sys, os, time, glob
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_image
from plate_locator import locate_plate_from_filename
from char_recognizer import CharRecognizer

PROVINCES = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','京','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")

# 8位绿牌窗口位置（440×125 基准）
CHAR_WINDOWS_8 = [
    (12, 57), (65, 110), (118, 163), (171, 216),
    (224, 269), (277, 322), (330, 375), (383, 428),
]

FULL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_plate_number(filename):
    """解析CCPD文件名中的车牌号（支持绿色新能源8位车牌）

    CCPD编码: nums[0]=省份索引, nums[1]=字母索引,
              nums[2:8] 每个值: 0-23→字母, 24-33→数字(值-24)
    """
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
        for n in nums[2:8]:
            if 0 <= n < 24:
                plate += FULL_LETTERS[n]
            elif 24 <= n < 34:
                plate += str(n - 24)
            else:
                plate += '?'
        return plate
    except Exception:
        return None


def extract_chars_from_plate(plate_img):
    """从透视校正车牌中提取字符（固定窗口+预处理，自动适配7/8位）"""
    if plate_img is None or plate_img.size == 0:
        return None

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape

    ratio = w / max(h, 1)
    # 根据宽高比选择窗口布局
    if ratio > 3.35 and w >= 420:
        windows = CHAR_WINDOWS_8
        base_w = 440
        n_expected = 8
    else:
        # 8位车牌被挤压到400px → 用8位布局但缩放到实际宽度
        windows = CHAR_WINDOWS_8
        base_w = 440
        n_expected = 8

    scale_x = w / base_w

    chars = []
    for x1, x2 in windows:
        x1c, x2c = max(0, int(x1 * scale_x)), min(w, int(x2 * scale_x))
        y1c, y2c = max(0, int(h * 0.08)), min(h, int(h * 0.92))
        roi = gray[y1c:y2c, x1c:x2c]
        if roi.size < 20:
            chars.append(None)
            continue
        char_img = cv2.resize(roi, (32, 48))
        _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
        chars.append(char_img)

    return chars if len(chars) == n_expected else None


def run_benchmark(max_images=None):
    val_dir = 'CCPD2020/ccpd_green/val'
    images = sorted(glob.glob(os.path.join(val_dir, '*.jpg')))
    if not images:
        print("No validation images found!")
        return

    recognizer = CharRecognizer()
    recognizer.load_or_create_templates()

    total = 0
    full_match = 0
    total_chars_correct = 0
    total_chars = 0
    pos_correct = [0] * 8
    pos_total = [0] * 8
    no_plate = 0
    seg_fail = 0

    t0 = time.time()
    for i, path in enumerate(images):
        if max_images and i >= max_images:
            break

        fname = os.path.basename(path)
        gt = parse_plate_number(fname)
        if gt is None or len(gt) != 8:
            continue
        total += 1

        try:
            img = load_image(path)
        except Exception:
            no_plate += 1
            continue

        plate_img, _ = locate_plate_from_filename(img, fname)
        if plate_img is None:
            no_plate += 1
            continue

        chars = extract_chars_from_plate(plate_img)
        if chars is None or len(chars) != 8:
            seg_fail += 1
            continue

        # 使用CNN识别所有8个位置
        predicted_chars = []
        for ci in range(8):
            c, conf = recognizer.recognize_single(chars[ci], position=ci)
            predicted_chars.append(c if c else '?')
        pred = ''.join(predicted_chars)

        if len(pred) != 8:
            pred = (pred + '????????')[:8]

        if pred == gt:
            full_match += 1

        for j, (a, b) in enumerate(zip(pred, gt)):
            if a == b:
                total_chars_correct += 1
                pos_correct[j] += 1
            total_chars += 1
            pos_total[j] += 1

        if i < 30:
            status = 'Y' if pred == gt else 'N'
            print("GT: %s -> %s %s" % (gt, pred, status))

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print("  [%d/%d] full=%d/%d (%.1f%%), char=%d/%d (%.1f%%), %.1f img/s" % (
                i+1, min(len(images), max_images or len(images)),
                full_match, total, full_match/total*100,
                total_chars_correct, total_chars,
                total_chars_correct/total_chars*100,
                total/elapsed))

    elapsed = time.time() - t0
    print("\n" + "="*60)
    print("Benchmark Results on %d validation images (%.1fs)" % (total, elapsed))
    print("="*60)
    print("  Full match (8/8):  %4d  (%5.1f%%)" % (full_match, full_match/total*100))
    print("  Char accuracy:     %d/%d  (%5.1f%%)" % (total_chars_correct, total_chars, total_chars_correct/total_chars*100))
    for j in range(8):
        print("  Pos %d accuracy:    %d/%d  (%5.1f%%)" % (j, pos_correct[j], pos_total[j], pos_correct[j]/max(pos_total[j],1)*100))
    print("  No plate found:    %4d" % no_plate)
    print("  Segmentation fail: %4d" % seg_fail)
    print("  Speed:             %.2fs/img" % (elapsed/total))


if __name__ == '__main__':
    run_benchmark()
