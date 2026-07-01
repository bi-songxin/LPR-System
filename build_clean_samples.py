"""
通过模板匹配从精确透视变换的车牌中提取干净标注字符样本
绕过二值化问题，直接在灰度图上进行模板匹配
"""
import sys, cv2, numpy as np, os, glob, re, pickle, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_image
from plate_locator import locate_plate_from_filename

PROVINCES = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','京','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
DIGITS = list("0123456789")
ALL_CHARS = PROVINCES + LETTERS + DIGITS


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


def create_synthetic_templates():
    """为所有65个字符生成高清合成模板（白底黑字）"""
    templates = {}
    font_sizes = {
        'chinese': (1.4, 2),   # 中文字较大
        'letter': (1.2, 2),
        'digit': (1.2, 2),
    }
    # 使用多种字体生成平均模板以获得更好的泛化性
    font_list = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
                 cv2.FONT_HERSHEY_COMPLEX]

    for char in ALL_CHARS:
        accumulator = np.zeros((80, 60), dtype=np.float32)
        count = 0

        if '\u4e00' <= char <= '\u9fff':
            fs, th = font_sizes['chinese']
        elif char.isalpha():
            fs, th = font_sizes['letter']
        else:
            fs, th = font_sizes['digit']

        for font in font_list:
            for ox in [-2, 0, 2]:
                for oy in [-2, 0, 2]:
                    img = np.ones((80, 60), dtype=np.uint8) * 255
                    cv2.putText(img, char, (15 + ox, 55 + oy), font, fs, 0, th)
                    accumulator += img.astype(np.float32)
                    count += 1

        avg = (accumulator / count).astype(np.uint8)
        # 裁剪到内容范围
        coords = np.where(avg < 200)
        if coords[0].size > 0:
            y1, y2 = max(0, coords[0].min() - 3), min(80, coords[0].max() + 4)
            x1, x2 = max(0, coords[1].min() - 3), min(60, coords[1].max() + 4)
            roi = avg[y1:y2, x1:x2]
            if roi.size > 0:
                # 调整大小到32x48
                h, w = roi.shape
                scale = min(48.0 / h, 32.0 / w)
                new_h, new_w = int(h * scale), int(w * scale)
                roi = cv2.resize(roi, (new_w, new_h))
                # 居中放置到32x48画布
                canvas = np.ones((48, 32), dtype=np.uint8) * 255
                y_off = (48 - new_h) // 2
                x_off = (32 - new_w) // 2
                canvas[y_off:y_off+new_h, x_off:x_off+new_w] = roi
                templates[char] = canvas
            else:
                templates[char] = avg
        else:
            templates[char] = avg

    print(f"生成了 {len(templates)} 个合成模板")
    return templates


def extract_samples_with_templates(templates, max_per_char=500):
    """
    使用合成模板在灰度车牌上进行模板匹配，找出每个字符的精确位置
    然后从匹配位置提取真实字符样本
    """
    train_dir = 'CCPD2020/ccpd_green/train'
    images = glob.glob(os.path.join(train_dir, '*.jpg'))
    print(f"训练集共 {len(images)} 张图片")

    char_samples = {}
    processed = 0
    matched_count = 0

    t0 = time.time()
    for path in images:
        fname = os.path.basename(path)
        plate_num = parse_plate_number(fname)
        if plate_num is None or len(plate_num) != 7:
            continue

        try:
            img = load_image(path)
        except Exception:
            continue

        # 精确透视变换
        plate_img, _ = locate_plate_from_filename(img, fname, plate_w=400, plate_h=125)
        if plate_img is None:
            continue

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        # CLAHE 增强对比度
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
        h, w = gray.shape

        # 为每个字符在期望位置区域进行模板匹配
        # 7字符车牌布局（在400px宽的图像中）：
        # 省份(x=5~60), 字母(x=65~120), 数字1~5(x=130~370均匀分布)
        search_windows = [
            (5, 65),      # 省份
            (65, 125),    # 字母
            (135, 185),   # 数字1
            (185, 235),   # 数字2
            (235, 285),   # 数字3
            (285, 335),   # 数字4
            (335, 390),   # 数字5
        ]

        chars_this_plate = []
        for i, (x1, x2) in enumerate(search_windows):
            if i >= len(plate_num):
                break
            expected_char = plate_num[i]

            # 裁剪搜索区域
            x1_c = max(0, int(x1))
            x2_c = min(w, int(x2))
            y1_c = max(0, int(h * 0.1))
            y2_c = min(h, int(h * 0.92))
            roi = gray[y1_c:y2_c, x1_c:x2_c]

            if roi.size == 0 or roi.shape[1] < 10 or roi.shape[0] < 10:
                continue

            # 模板匹配
            template = templates.get(expected_char)
            if template is None:
                continue

            # 模板缩放以适应ROI（高度和宽度都限制）
            tmpl_h, tmpl_w = template.shape
            roi_h, roi_w = roi.shape
            scale_h = roi_h / tmpl_h * 0.85
            scale_w = roi_w / tmpl_w * 0.85
            scale_factor = min(scale_h, scale_w)
            if scale_factor <= 0.1:
                continue
            new_tw = max(5, int(tmpl_w * scale_factor))
            new_th = max(5, int(tmpl_h * scale_factor))
            if new_tw < 5 or new_th < 5 or new_tw > roi_w or new_th > roi_h:
                continue

            scaled_tmpl = cv2.resize(template, (new_tw, new_th))

            result = cv2.matchTemplate(roi, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # 提取字符（在灰度图像上）
            mx, my = max_loc
            # 在原图中的实际位置
            global_x = x1_c + mx
            global_y = y1_c + my

            # 提取并归一化到32x48
            char_roi = gray[global_y:global_y+new_th, global_x:global_x+new_tw]
            if char_roi.size < 20:
                continue
            char_img = cv2.resize(char_roi, (32, 48))
            _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

            if expected_char not in char_samples:
                char_samples[expected_char] = []
            if len(char_samples[expected_char]) < max_per_char:
                char_samples[expected_char].append(char_img)

            matched_count += 1

        processed += 1
        if processed % 200 == 0:
            elapsed = time.time() - t0
            print(f"  已处理 {processed} 张, 速度 {processed/elapsed:.1f} 张/秒")

    total = sum(len(v) for v in char_samples.values())
    print(f"\n从 {processed} 张车牌中提取了 {total} 个字符样本")
    print(f"覆盖 {len(char_samples)}/65 个字符类")

    for label, samples in sorted(char_samples.items(), key=lambda x: -len(x[1])):
        print(f"  '{label}': {len(samples)} 个样本")

    return char_samples


def save_samples_for_knn(char_samples, output_file='templates/clean_samples.pkl'):
    """将样本保存为KNN格式（使用32x48分辨率）"""
    samples_dict = {}
    all_chars = set(ALL_CHARS)
    missing = all_chars - set(char_samples.keys())

    for label, images in char_samples.items():
        vectors = []
        for img in images:
            # 使用完整32x48分辨率
            img_bin = img.copy()
            _, img_bin = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)
            vec = (img_bin.astype(np.float32) / 255.0).flatten()
            vectors.append(vec)
        samples_dict[label] = vectors

    # 为缺失类生成合成样本
    if missing:
        print(f"\n为 {len(missing)} 个缺失类生成合成样本...")
        for char in sorted(missing):
            vectors = []
            for _ in range(50):
                # 生成合成样本
                img = np.ones((80, 60), dtype=np.uint8) * 255
                fs = np.random.choice([0.8, 1.0, 1.2, 1.4])
                ft = np.random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX])
                th = np.random.choice([1, 2, 3])
                ox, oy = np.random.randint(-5, 6, 2)
                cv2.putText(img, char, (15 + ox, 55 + oy), ft, fs, 0, th)
                coords = np.where(img < 200)
                if coords[0].size > 0:
                    y1, y2 = max(0, coords[0].min() - 2), min(80, coords[0].max() + 3)
                    x1, x2 = max(0, coords[1].min() - 2), min(60, coords[1].max() + 3)
                    roi = img[y1:y2, x1:x2]
                    if roi.size > 0:
                        img = cv2.resize(roi, (32, 48))
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                vec = (img.astype(np.float32) / 255.0).flatten()
                vectors.append(vec)
            samples_dict[char] = vectors

    os.makedirs('templates', exist_ok=True)
    data = {'samples': samples_dict, 'labels': list(samples_dict.keys())}
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nKNN样本已保存至 {output_file} ({len(samples_dict)} 类, "
          f"{sum(len(v) for v in samples_dict.values())} 个样本)")


def main():
    templates = create_synthetic_templates()
    char_samples = extract_samples_with_templates(templates, max_per_char=200)
    save_samples_for_knn(char_samples)


if __name__ == '__main__':
    main()
