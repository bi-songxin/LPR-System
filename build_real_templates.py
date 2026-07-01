"""
从 CCPD 训练集提取真实字符模板
利用文件名中的标注数据，构建真实字符库
"""

import sys
import cv2
import numpy as np
import os
import glob
import re
import pickle
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_image
from plate_locator import locate_plate_from_filename
from char_segmenter import segment_characters

# CCPD 车牌号映射表
PROVINCES = ['皖','沪','津','渝','冀','晋','蒙','辽','吉','黑','苏','浙','京','闽','赣','鲁','豫','鄂','湘','粤','桂','琼','川','贵','云','藏','陕','甘','青','宁','新']
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
DIGITS = list("0123456789")


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


def build_templates(output_file='templates/real_templates.pkl',
                    max_per_char=50):
    train_dir = 'CCPD2020/ccpd_green/train'
    images = glob.glob(os.path.join(train_dir, '*.jpg'))
    print(f"训练集共 {len(images)} 张图片")

    # 按字符收集真实图像
    char_samples = {}  # char -> list of (32x48) binary images
    processed = 0
    found = 0

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

        # 精确车牌提取（透视变换）
        plate_img, _ = locate_plate_from_filename(img, fname, plate_w=400, plate_h=125)
        if plate_img is None:
            continue

        # 字符分割
        chars, _ = segment_characters(plate_img, debug=True)
        if chars is None or len(chars) < 4:
            continue

        processed += 1

        # 位置匹配：第0个字符=省份, 第1个=字母, 第2-6个=数字
        for i, char_img in enumerate(chars):
            if i >= len(plate_num):
                break
            label = plate_num[i]

            # 归一化：32x48 二值图
            if len(char_img.shape) == 3:
                char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            char_img = cv2.resize(char_img, (32, 48))
            _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

            if label not in char_samples:
                char_samples[label] = []
            char_samples[label].append(char_img)

        found += 1
        if processed % 100 == 0:
            elapsed = time.time() - t0
            print(f"  已处理 {processed} 张, 速度 {processed/elapsed:.1f} 张/秒")

    # 汇总统计
    total_chars = sum(len(v) for v in char_samples.values())
    chars_with_samples = len(char_samples)
    print(f"\n从 {processed} 张成功分割的车牌中提取了 {total_chars} 个字符样本")
    print(f"覆盖 {chars_with_samples}/65 个字符类")

    # 查找缺失的字符类
    all_chars = set(PROVINCES + LETTERS + DIGITS)
    missing = all_chars - set(char_samples.keys())
    if missing:
        print(f"缺失字符类: {sorted(missing)}")

    # 每类处理模板：≥3 个样本做均值，1-2 个样本直接使用，0 个样本用 fallback
    templates = {}
    for char, samples in char_samples.items():
        chosen = samples[:max_per_char]
        if len(chosen) >= 3:
            stacked = np.stack([s.astype(np.float32) for s in chosen])
            avg = np.mean(stacked, axis=0)
            avg = np.clip(avg, 0, 255).astype(np.uint8)
            _, avg = cv2.threshold(avg, 127, 255, cv2.THRESH_BINARY)
            templates[char] = avg
        else:
            # 样本太少，直接用第一张
            templates[char] = chosen[0]

    real_count = len(templates)
    print(f"生成 {real_count} 个真实模板")

    # 对于完全缺失的字符类，生成简单模板作为兜底
    for char in sorted(missing):
        templates[char] = _create_fallback_template(char)
    print(f"补全 {len(missing)} 个兜底模板，共 {len(templates)} 个模板")

    # 保存
    os.makedirs('templates', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump({'templates': templates}, f)
    print(f"真实模板已保存至 {output_file}")

    # 为所有模板（含兜底）计算 Hu 矩
    hu_dict = {}
    for char, samples in char_samples.items():
        chosen = samples[:max_per_char]
        hu_list = []
        for s in chosen:
            moments = cv2.moments(s)
            hu_raw = cv2.HuMoments(moments)
            hu_log = -np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-10)
            hu_list.append(hu_log.flatten())
        hu_dict[char] = np.mean(hu_list, axis=0) if len(hu_list) >= 3 else hu_list[0]

    # 对兜底模板也计算 Hu 矩
    for char, template_img in templates.items():
        if char not in hu_dict:
            moments = cv2.moments(template_img)
            hu_raw = cv2.HuMoments(moments)
            hu_log = -np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-10)
            hu_dict[char] = hu_log.flatten()

    with open('templates/real_templates_hu.pkl', 'wb') as f:
        pickle.dump(hu_dict, f)
    print(f"Hu 矩库已保存至 templates/real_templates_hu.pkl ({len(hu_dict)} 个)")

    return templates, char_samples


def _create_fallback_template(char):
    """为缺失字符创建简单模板"""
    img = np.ones((48, 32), dtype=np.uint8) * 255
    mid_x, mid_y = 16, 24
    cv2.rectangle(img, (8, 4), (24, 44), 0, 2)
    if '\u4e00' <= char <= '\u9fff':
        cv2.rectangle(img, (4, mid_y), (28, mid_y + 4), 0, -1)
    cv2.putText(img, char, (4, 36), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,), 2)
    return img


if __name__ == '__main__':
    build_templates()
