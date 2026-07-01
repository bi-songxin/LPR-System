"""
字符分割模块 - Character Segmenter
功能：车牌二值化和字符分割

支持蓝牌（白字蓝底）和绿牌（黑字绿底）
对透视校正板（400×125）优先使用固定窗口法
"""

import cv2
import numpy as np
from preprocessing import (
    to_grayscale, gaussian_blur, clahe,
    morphology_dilate, morphology_erode,
    morphology_open, morphology_close
)

# 透视校正板的标准字符窗口（基准: 400×125 比例，7字符CCPD格式）
# 窗口定义为 (x_start, x_end)，在400像素宽基准下的坐标
_WINDOW_LAYOUTS = {
    7: [(5, 65), (65, 125), (135, 185), (185, 235), (235, 285), (285, 335), (335, 390)],
    # 8字符新能源车牌：省份同宽，字符2+之间无间隙（首尾相连）
    8: [(12, 57), (65, 110), (118, 163), (163, 216),
        (216, 269), (269, 322), (322, 375), (375, 428)],
}


def _estimate_char_count(plate_w, plate_h):
    """根据透视校正板的尺寸估算字符数"""
    ratio = plate_w / max(plate_h, 1)
    if ratio > 3.35:
        return 8
    if ratio > 3.05:
        # 边界：可能7也8位，需要通过投影检测
        return 7  # 默认7，实际由 _detect_char_count 校正
    return 7


def _detect_char_peaks(plate_gray):
    """通过投影检测字符峰值位置，返回峰值 x 坐标列表

    策略：找所有候选峰 → 按位置排序 → 合并距离太近的峰（同字符多笔画）
    这样中文省份字符的多笔画会被自然合并为1个峰，不需要靠强度阈值判断。
    """
    h, w = plate_gray.shape
    y1, y2 = int(h * 0.1), int(h * 0.88)
    strip = plate_gray[y1:y2, :]

    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(strip)

    # 垂直投影反转为正（字符暗=高峰值）
    v_proj = np.mean(enhanced, axis=0).astype(float)
    v_proj = np.max(v_proj) - v_proj
    if np.max(v_proj) <= 0:
        return [], None

    # 背景减除
    bg_kernel = max(20, w // 20)
    bg = np.convolve(v_proj, np.ones(bg_kernel) / bg_kernel, mode='same')
    v_signal = v_proj - bg
    v_signal = np.clip(v_signal, 0, None)

    # 轻度平滑（主要用于去噪，峰的合并由后续聚类完成）
    smooth_kernel = max(5, w // 80)
    v_smooth = np.convolve(v_signal, np.ones(smooth_kernel) / smooth_kernel, mode='same')
    v_norm = v_smooth / (np.max(v_smooth) + 1e-6)

    # 找所有候选峰（阈值低，尽量捕获所有笔画）
    min_gap = max(8, w // 50)
    raw_peaks = []
    for i in range(min_gap, len(v_norm) - min_gap):
        if (v_norm[i] > 0.05 and
            all(v_norm[i] >= v_norm[j] for j in range(i - min_gap, i + min_gap + 1)
                if 0 <= j < len(v_norm))):
            raw_peaks.append((i, v_norm[i]))

    if not raw_peaks or len(raw_peaks) < 3:
        return [], v_norm

    # 按 x 坐标排序
    raw_peaks.sort(key=lambda x: x[0])

    # === 合并间距过近的峰（同字符内多笔画 → 单峰）===
    min_char_gap = w // 14  # ~28px for 400px → 半个字符宽
    merged = []
    cur_x = raw_peaks[0][0]
    cur_v = raw_peaks[0][1]
    for px, pv in raw_peaks[1:]:
        if px - cur_x < min_char_gap:
            # 同一字符：加权平均位置
            total_v = cur_v + max(pv, 0.01)
            cur_x = int((cur_x * cur_v + px * pv) / total_v)
            cur_v = max(cur_v, pv)
        else:
            merged.append(cur_x)
            cur_x = px
            cur_v = pv
    merged.append(cur_x)

    return merged, v_norm


def _is_perspective_corrected(plate_img):
    """检测是否为透视校正板（宽高比 ≈ 3.2:1，即400:125）"""
    if plate_img is None:
        return False
    h, w = plate_img.shape[:2]
    ratio = w / max(h, 1)
    return 2.8 <= ratio < 3.8


def _segment_fixed_windows(plate_img):
    """对透视校正板使用固定窗口提取字符（支持7位和8位车牌）

    根据车牌实际宽高比自动选择布局：
    - ratio <= 3.35: 7字符布局（标准蓝牌/绿牌）
    - ratio >  3.35: 8字符布局（新能源车牌）
    """
    if plate_img is None or plate_img.size == 0:
        return None, None

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape

    n_chars = _estimate_char_count(w, h)

    # 使用预设窗口 + 比例缩放
    base_windows = _WINDOW_LAYOUTS.get(n_chars, _WINDOW_LAYOUTS[7])
    base_w = 440 if n_chars == 8 else 400
    scale_x = w / base_w
    chars = []
    windows_used = []
    prev_x2c = None
    for i, (x1, x2) in enumerate(base_windows):
        x1c, x2c = max(0, int(x1 * scale_x)), min(w, int(x2 * scale_x))
        # 从第3个字符（index 2）起，窗口首尾相连，消除间隙
        if i >= 2 and prev_x2c is not None:
            x1c = prev_x2c
        y1c, y2c = max(0, int(h * 0.08)), min(h, int(h * 0.92))
        roi = gray[y1c:y2c, x1c:x2c]
        if roi.size < 20:
            chars.append(None)
            prev_x2c = x2c
            continue
        char_img = cv2.resize(roi, (32, 48))
        _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
        chars.append(char_img)
        windows_used.append((x1c, x2c))
        prev_x2c = x2c

    # 生成可视化
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x1c, x2c) in windows_used:
        y1c, y2c = max(0, int(h * 0.08)), min(h, int(h * 0.92))
        cv2.rectangle(vis, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)

    return chars, vis


def binarize_plate(plate_img):
    """
    车牌二值化：白字黑底

    蓝牌（白字蓝底）：B通道 + Otsu
    绿牌（黑字绿底）：G通道 + CLAHE + Otsu反转 / 固定窗口法
    自动选择最优结果
    """
    if plate_img is None or plate_img.size == 0:
        return None

    # 缩放到统一高度
    h_raw, w_raw = plate_img.shape[:2]
    target_h = 72
    scale = target_h / h_raw
    target_w = max(int(w_raw * scale), 30)
    resized = cv2.resize(plate_img, (target_w, target_h))
    h, w = target_h, target_w

    b, g, r = cv2.split(resized)
    gray = to_grayscale(resized)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]

    candidates = []

    # ===== 方法1: B通道Otsu（绿底最暗，白字最亮）=====
    b_enhanced = clahe(b, clip_limit=2.5)
    b_blur = gaussian_blur(b_enhanced, (3, 3))
    _, b_bin = cv2.threshold(b_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(('b_otsu', b_bin))
    _, b_bin_inv = cv2.threshold(b_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(('b_otsu_inv', b_bin_inv))

    # ===== 方法1b: B通道 mean+std 固定阈值 =====
    b_mean = np.mean(b_blur)
    b_std = np.std(b_blur)
    for offset in [0.5, 1.0, 1.5]:
        th_val = b_mean + offset * b_std
        if 10 < th_val < 245:
            _, fix_bin = cv2.threshold(b_blur, th_val, 255, cv2.THRESH_BINARY)
            candidates.append((f'b_fix_{offset:.1f}std', fix_bin))

    # ===== 方法2: 白度图 = min(R,G,B) × (1 - greenness) =====
    r_f = r.astype(np.float32)
    g_f = g.astype(np.float32)
    b_f = b.astype(np.float32)
    brightness = np.minimum(np.minimum(r_f, g_f), b_f)
    greenness = g_f - 0.5 * (r_f + b_f)
    greenness = np.clip(greenness, 0, 255)
    whiteness = brightness * (1.0 - greenness / 255.0)
    whiteness = np.clip(whiteness, 0, 255).astype(np.uint8)
    w_enhanced = clahe(whiteness, clip_limit=2.5)
    w_blur = gaussian_blur(w_enhanced, (3, 3))
    _, w_bin = cv2.threshold(w_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(('whiteness', w_bin))

    # ===== 方法3: RB平均 vs G的差值（反向后突出白字）=====
    avg_rb = (r_f + b_f) / 2.0
    diff_rbg = np.abs(avg_rb - g_f)
    diff_rbg_inv = 255.0 - diff_rbg
    diff_rbg_inv = np.clip(diff_rbg_inv, 0, 255).astype(np.uint8)
    d_enhanced = clahe(diff_rbg_inv, clip_limit=2.5)
    d_blur = gaussian_blur(d_enhanced, (3, 3))
    _, d_bin = cv2.threshold(d_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(('rb_g_diff', d_bin))

    # ===== 方法4: HSV白色范围 =====
    white_mask = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([180, 55, 255]))
    white_clean = morphology_open(white_mask, (2, 2))
    white_clean = morphology_close(white_clean, (3, 3))
    candidates.append(('hsv_white', white_clean))

    # ===== 方法5: 灰度Otsu =====
    gray_enhanced = clahe(gray, clip_limit=2.5)
    gray_blur = gaussian_blur(gray_enhanced, (3, 3))
    _, gray_otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(('gray_otsu', gray_otsu))
    _, gray_otsu_inv = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(('gray_otsu_inv', gray_otsu_inv))

    # ===== 方法6: 灰度自适应阈值 =====
    for bs, c in [(15, 5), (19, 7), (25, 9), (31, 12)]:
        adapt = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, bs, c
        )
        candidates.append((f'gray_adapt_{bs}', adapt))

    # ===== 方法7: LAB L通道Otsu =====
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    l_enhanced = clahe(l_ch, clip_limit=2.5)
    l_blur = gaussian_blur(l_enhanced, (3, 3))
    _, l_bin = cv2.threshold(l_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(('l_otsu', l_bin))

    # ===== 方法8: B/G比值图 = 白字B≈G≈R, 绿底B<<G =====
    g_safe = np.where(g_f < 1, 1.0, g_f)
    bg_ratio = b_f / g_safe  # 白字≈1, 绿底<<1
    bg_ratio = np.clip(bg_ratio * 255, 0, 255).astype(np.uint8)
    bg_enhanced = clahe(bg_ratio, clip_limit=3.0)
    bg_blur = gaussian_blur(bg_enhanced, (3, 3))
    _, bg_bin = cv2.threshold(bg_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(('bg_ratio', bg_bin))

    # ===== 方法9: B通道自适应阈值 =====
    for bs, c in [(15, 3), (21, 5), (31, 8)]:
        adapt = cv2.adaptiveThreshold(
            b_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, bs, c
        )
        candidates.append((f'b_adapt_{bs}', adapt))

    # ===== 方法10: (B-R)差值图 = 白字B≈R, 绿底B<<R =====
    b_r_diff = cv2.absdiff(b, r)
    br_inv = 255 - b_r_diff  # 差值小=白字=亮
    br_enhanced = clahe(br_inv, clip_limit=3.0)
    br_blur = gaussian_blur(br_enhanced, (3, 3))
    _, br_bin = cv2.threshold(br_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append(('br_inv', br_bin))

    # ===== 选择最优二值化 =====
    best = None
    best_score = -10
    best_info = ''

    for name, binary in candidates:
        white_ratio = cv2.countNonZero(binary) / binary.size

        # 过滤不合理比例
        if white_ratio < 0.005 or white_ratio > 0.50:
            continue

        # 水平投影
        h_proj = np.sum(binary == 255, axis=1)
        if np.max(h_proj) <= 0:
            continue
        h_proj_norm = h_proj / np.max(h_proj)

        # 找字符集中行
        char_rows = np.where(h_proj_norm > 0.08)[0]
        if len(char_rows) < 3:
            continue

        y_start, y_end = char_rows[0], char_rows[-1] + 1
        char_h = y_end - y_start

        if char_h < h * 0.10:
            continue

        # 在字符行范围内统计垂直投影峰值
        v_proj = np.sum(binary[y_start:y_end, :] == 255, axis=0)
        if np.max(v_proj) <= 0:
            continue
        v_proj_norm = v_proj / np.max(v_proj)

        # 找字符峰值数
        peaks = 0
        for i in range(1, len(v_proj_norm) - 1):
            if (v_proj_norm[i] > 0.18 and
                v_proj_norm[i] >= v_proj_norm[i - 1] and
                v_proj_norm[i] > v_proj_norm[i + 1]):
                peaks += 1

        if peaks < 2 or peaks > 16:
            continue

        # 评分：白像素比例接近理想值 + 峰值数接近7~8
        ideal_ratio = 0.15
        score = 1.0 - min(abs(white_ratio - ideal_ratio) / 0.25, 1.0)
        score += min(peaks / 8.0, 1.0) * 0.5

        if score > best_score:
            best_score = score
            best = binary
            best_info = name

    if best is None:
        # 回退1：放宽条件 — 只看white_ratio和水平投影
        for name, binary in candidates:
            white_ratio = cv2.countNonZero(binary) / binary.size
            if white_ratio < 0.01 or white_ratio > 0.55:
                continue
            h_proj = np.sum(binary == 255, axis=1)
            if np.max(h_proj) <= 0:
                continue
            h_proj_norm = h_proj / np.max(h_proj)
            char_rows = np.where(h_proj_norm > 0.06)[0]
            if len(char_rows) < 5:
                continue
            char_h = char_rows[-1] - char_rows[0] + 1
            if char_h < h * 0.08:
                continue
            best = binary
            best_info = f'fallback1_{name}'
            break

    if best is None:
        # 回退2：更宽松 — 只看white_ratio
        for name, binary in candidates:
            white_ratio = cv2.countNonZero(binary) / binary.size
            if 0.015 < white_ratio < 0.55:
                best = binary
                best_info = f'fallback2_{name}'
                break

    if best is None:
        # 回退3：默认阈值
        best = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )
        best_info = 'fallback3_default'

    # 确保黑底白字
    if cv2.countNonZero(best) > best.size * 0.5:
        best = cv2.bitwise_not(best)

    # 最终清理
    best = morphology_open(best, (2, 2))
    best = morphology_close(best, (3, 2))

    return best


def segment_characters(plate_img, debug=False):
    """
    从车牌图像分割单个字符

    透视校正板 → 优先固定窗口法
    非校正板   → 垂直投影法 + 轮廓法备选
    """
    intermediates = {}

    if plate_img is None or plate_img.size == 0:
        return ([], intermediates) if debug else []

    # 透视校正板：直接用固定窗口法
    if _is_perspective_corrected(plate_img):
        chars, vis = _segment_fixed_windows(plate_img)
        if chars is not None and len(chars) >= 4:
            intermediates['char_vis'] = vis
            intermediates['plate_binary'] = vis
            if debug:
                return chars, intermediates
            return chars

    binary = binarize_plate(plate_img)
    if binary is None:
        return ([], intermediates) if debug else []

    h, w = binary.shape
    intermediates['plate_binary'] = binary.copy()

    # ===== 垂直投影 =====
    v_proj = np.sum(binary == 255, axis=0).astype(np.float32)
    if np.max(v_proj) <= 0:
        return ([], intermediates) if debug else []
    v_norm = v_proj / np.max(v_proj)

    # 水平投影找字符垂直范围
    h_proj = np.sum(binary == 255, axis=1)
    if np.max(h_proj) > 0:
        h_norm = h_proj / np.max(h_proj)
        rows_idx = np.where(h_norm > 0.06)[0]
        y_top = rows_idx[0] if len(rows_idx) > 0 else 0
        y_bot = (rows_idx[-1] + 1) if len(rows_idx) > 0 else h
    else:
        y_top, y_bot = 0, h

    # 找字符列区间
    min_w = max(4, int(w * 0.018))
    max_w = int(w * 0.22)

    def find_regions(threshold):
        regions = []
        in_char = False
        start = 0
        for col in range(w):
            if v_norm[col] > threshold and not in_char:
                in_char = True
                start = col
            elif v_norm[col] <= threshold and in_char:
                in_char = False
                cw = col - start
                if min_w <= cw <= max_w:
                    regions.append((start, y_top, cw, y_bot - y_top))
        if in_char:
            cw = w - start
            if min_w <= cw <= max_w:
                regions.append((start, y_top, cw, y_bot - y_top))
        return regions

    # 尝试多个阈值
    char_regions = find_regions(0.04)
    if len(char_regions) < 3:
        char_regions = find_regions(0.025)
    if len(char_regions) < 3:
        char_regions = find_regions(0.015)

    # ===== 轮廓法备选 =====
    if len(char_regions) < 3:
        dilated = morphology_dilate(binary, (2, 3), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        c_boxes = []
        for cnt in contours:
            x, cy, cw, ch = cv2.boundingRect(cnt)
            if ch < h * 0.15 or ch > h * 0.95:
                continue
            if cw < min_w or cw > max_w:
                continue
            c_boxes.append((x, cy, cw, ch))

        if c_boxes:
            c_boxes.sort(key=lambda b: b[0])
            merged = _merge_overlapping(c_boxes)
            if len(merged) > len(char_regions):
                cy_t = min(b[1] for b in merged)
                cy_b = max(b[1] + b[3] for b in merged)
                char_regions = [(b[0], cy_t, b[2], cy_b - cy_t) for b in merged]

    # ===== 合并太近区域 =====
    final = _merge_close(char_regions)

    # ===== 提取字符 =====
    chars = []
    vis_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    prev_x2 = None

    for i, (rx, ry, rw, rh) in enumerate(final):
        x1, x2 = max(0, rx), min(w, rx + rw)
        y1, y2 = max(0, ry), min(h, ry + rh)

        # 从第3个字符（index 2）起，窗口首尾相连，消除间隙
        if i >= 2 and prev_x2 is not None:
            x1 = prev_x2

        roi = binary[y1:y2, x1:x2]
        if roi.size == 0:
            prev_x2 = x2
            continue

        rh_r, rw_r = roi.shape
        sz = max(rh_r, rw_r)
        pt = (sz - rh_r) // 2
        pb = sz - rh_r - pt
        pl = (sz - rw_r) // 2
        pr = sz - rw_r - pl

        roi = cv2.copyMakeBorder(
            roi, pt, pb, pl, pr,
            cv2.BORDER_CONSTANT, value=0
        )
        char_img = cv2.resize(roi, (32, 48))
        chars.append(char_img)

        cv2.rectangle(vis_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        prev_x2 = x2

    intermediates['char_vis'] = vis_img

    if debug:
        return chars, intermediates
    return chars


def _merge_overlapping(boxes):
    if len(boxes) < 2:
        return boxes
    merged = [list(boxes[0])]
    for bx, by, bw, bh in boxes[1:]:
        p = merged[-1]
        overlap = max(0, min(bx + bw, p[0] + p[2]) - max(bx, p[0]))
        if overlap > min(bw, p[2]) * 0.4:
            nx = min(bx, p[0])
            nw = max(bx + bw, p[0] + p[2]) - nx
            ny = min(by, p[1])
            nh = max(by + bh, p[1] + p[3]) - ny
            merged[-1] = [nx, ny, nw, nh]
        else:
            merged.append([bx, by, bw, bh])
    return merged


def _merge_close(regions):
    if len(regions) < 2:
        return regions
    merged = [list(regions[0])]
    for rx, ry, rw, rh in regions[1:]:
        p = merged[-1]
        gap = rx - (p[0] + p[2])
        if gap < 0 and abs(gap) < max(p[2], rw) * 0.3:
            nx, nw = p[0], rx + rw - p[0]
            ny, nh = min(p[1], ry), max(p[1] + p[3], ry + rh) - min(p[1], ry)
            merged[-1] = [nx, ny, nw, nh]
        else:
            merged.append([rx, ry, rw, rh])
    return merged
