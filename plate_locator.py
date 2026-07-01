"""
车牌定位模块 - License Plate Locator
功能：
1. CCPD文件名解析 → 透视变换（精准定位，主方法）
2. 绿色区域检测（通用方法，fallback）
"""

import cv2
import numpy as np
import re
from preprocessing import to_grayscale, clahe, morphology_close, morphology_open


# ==================== CCPD文件名解析 ====================

def parse_ccpd_filename(filename):
    """
    解析 CCPD 文件名，提取车牌四个角点坐标。

    文件名格式: ...tilt_bbLeft-bbTop&bbRight_bbBottom-v1x&v1y_v2x&v2y_v3x&v3y_v4x&v4y-...
    所有 x&y 配对中，前 2 对是 bbox，后 4 对是车牌四个角点。

    Returns:
        (vertices_ordered, plate_number_str) 或 (None, None)
        vertices_ordered: shape (4,2) float32, 顺序 TL/TR/BR/BL
    """
    pairs = [(int(x), int(y)) for x, y in re.findall(r'(\d+)&(\d+)', filename)]
    if len(pairs) < 6:
        return None, None

    # 后 4 对是车牌四个角点
    pts = np.array(pairs[2:6], dtype=np.float32)

    # 按位置排序为 TL, TR, BR, BL
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]       # 左上: x+y 最小
    br = pts[np.argmax(s)]       # 右下: x+y 最大
    tr = pts[np.argmin(diff)]    # 右上: x-y 最小（即 x 大 y 小）
    bl = pts[np.argmax(diff)]    # 左下: x-y 最大（即 x 小 y 大）
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    return ordered, None


def locate_plate_from_filename(img_bgr, filepath, plate_w=400, plate_h=125, debug=False):
    """
    利用 CCPD 文件名中的车牌四角坐标做透视变换，获取正视角标准车牌。

    这是最精准的定位方式，直接从标注数据提取。
    """
    import os
    filename = os.path.basename(filepath)
    vertices, _ = parse_ccpd_filename(filename)

    if vertices is None:
        if debug:
            return None, {}
        return None, {}

    intermediates = {}
    h_img, w_img = img_bgr.shape[:2]

    # 从文件名编码判断8位绿牌（nums[2] < 24 → 第3位是字母 → 8位）
    file_base = filename.rsplit('.', 1)[0]
    segs = file_base.split('-')
    plate_w, plate_h = 400, 125
    if len(segs) >= 6:
        plate_nums = [int(x) for x in segs[4].split('_')]
        if len(plate_nums) >= 3 and plate_nums[2] < 24:
            plate_w = 440  # 8位新能源绿牌

    # 透视变换目标点：标准车牌矩形
    dst_pts = np.array([
        [0, 0],
        [plate_w - 1, 0],
        [plate_w - 1, plate_h - 1],
        [0, plate_h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(vertices, dst_pts)
    plate_img = cv2.warpPerspective(img_bgr, M, (plate_w, plate_h))

    # 在原图上绘制定位框
    result_img = img_bgr.copy()
    for i in range(4):
        pt1 = tuple(vertices[i].astype(int))
        pt2 = tuple(vertices[(i + 1) % 4].astype(int))
        cv2.line(result_img, pt1, pt2, (0, 255, 0), 3)
    intermediates['located'] = result_img

    if debug:
        return plate_img, intermediates
    return plate_img, intermediates


# ==================== 绿色区域检测 ====================

def locate_plate_from_green(img_bgr, debug=False):
    """
    绿色车牌定位（通用方法，不依赖文件名）

    流程：
    1. HSV提取绿色区域
    2. 形态学连接
    3. 按面积/宽高比筛选候选
    4. 找车牌四角点（最大轮廓拟合四边形）
    5. 透视校正为标准 400×125 车牌
    """
    intermediates = {}
    h_img, w_img = img_bgr.shape[:2]
    min_plate_area = (w_img * h_img) * 0.002
    max_plate_area = (w_img * h_img) * 0.10

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 绿色检测 - 较宽范围覆盖深浅绿色
    lower_green = np.array([35, 38, 38])
    upper_green = np.array([82, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    intermediates['green_mask'] = green_mask.copy()

    # 开运算去噪
    green_mask = morphology_open(green_mask, (3, 3))

    # 闭运算连接邻近绿色区域
    kernel_w = max(int(w_img / 50), 10)
    kernel_h = max(int(h_img / 150), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    green_closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    intermediates['green_morph'] = green_closed.copy()

    # 查找轮廓
    contours, _ = cv2.findContours(green_closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        # 面积过滤
        if area < min_plate_area or area > max_plate_area:
            continue

        # 额外过滤：宽度不能超过图像的一半
        if cw > w_img * 0.5:
            continue

        # 宽高比过滤
        ar = cw / ch
        if ar < 1.8 or ar > 6.0:
            continue

        # 绿色像素占比
        roi_green = green_mask[y:y + ch, x:x + cw]
        if roi_green.size == 0:
            continue
        green_ratio = cv2.countNonZero(roi_green) / roi_green.size
        if green_ratio < 0.08:
            continue

        # 评分
        ar_score = 1.0 - min(abs(ar - 3.2) / 4.0, 0.85)
        pos_score = 1.0 - 0.3 * abs((y + ch / 2) / h_img - 0.5)
        score = area * ar_score * green_ratio * pos_score

        candidates.append((score, x, y, cw, ch, cnt))

    if not candidates:
        if debug:
            return None, None, intermediates
        return None, None

    # 选择最优候选
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, bx, by, bw, bh, best_contour = candidates[0]

    # === 透视校正：从绿色轮廓找四角点 ===
    # 在最优轮廓区域重新做精细绿色检测（更严格，找车牌本体）
    roi_x1 = max(0, bx - int(bw * 0.05))
    roi_y1 = max(0, by - int(bh * 0.05))
    roi_x2 = min(w_img, bx + bw + int(bw * 0.05))
    roi_y2 = min(h_img, by + bh + int(bh * 0.05))
    roi_hsv = hsv[roi_y1:roi_y2, roi_x1:roi_x2]

    # 精细化绿色mask
    fine_mask = cv2.inRange(roi_hsv, lower_green, upper_green)
    fine_closed = cv2.morphologyEx(fine_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    fine_contours, _ = cv2.findContours(fine_closed, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

    if fine_contours:
        # 取最大轮廓
        largest = max(fine_contours, key=cv2.contourArea)
        # 用凸包+四边形近似找四个角点
        hull = cv2.convexHull(largest)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        # 如果四边形近似不够4个点，用最小外接矩形
        corners = None
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            # 用最小外接矩形
            rect = cv2.minAreaRect(largest)
            corners = cv2.boxPoints(rect).astype(np.float32)

        # 转换回原图坐标
        corners[:, 0] += roi_x1
        corners[:, 1] += roi_y1

        # 排序为 TL, TR, BR, BL
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        tl = corners[np.argmin(s)]
        br = corners[np.argmax(s)]
        tr = corners[np.argmin(diff)]
        bl = corners[np.argmax(diff)]
        ordered = np.array([tl, tr, br, bl], dtype=np.float32)

        # 从角点计算原始车牌宽高比，保持比例输出（7位≈3.2，8位≈3.5）
        top_w = np.linalg.norm(ordered[1] - ordered[0])
        bot_w = np.linalg.norm(ordered[2] - ordered[3])
        left_h = np.linalg.norm(ordered[3] - ordered[0])
        right_h = np.linalg.norm(ordered[2] - ordered[1])
        orig_w = (top_w + bot_w) / 2.0
        orig_h = (left_h + right_h) / 2.0

        # 使用粗检测 bbox 的宽高比判断输出尺寸（比细轮廓角点更可靠）
        # 细轮廓仅用于透视变换的角点定位
        bbox_ar = bw / max(bh, 1)
        dst_h = 125
        dst_w = int(dst_h * bbox_ar)
        dst_w = max(350, min(480, dst_w))

        # 透视变换
        dst_pts = np.array([[0, 0], [dst_w - 1, 0],
                            [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                          dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst_pts)
        plate_img = cv2.warpPerspective(img_bgr, M, (dst_w, dst_h))

        # 可视化
        result_img = img_bgr.copy()
        for i in range(4):
            pt1 = tuple(ordered[i].astype(int))
            pt2 = tuple(ordered[(i + 1) % 4].astype(int))
            cv2.line(result_img, pt1, pt2, (0, 255, 0), 3)
        intermediates['located'] = result_img.copy()

        if debug:
            return plate_img, None, intermediates
        return plate_img, None
    else:
        # 回退：简单矩形裁剪
        pad_x = int(bw * 0.03)
        pad_y = int(bh * 0.05)
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w_img, bx + bw + pad_x)
        y2 = min(h_img, by + bh + pad_y)
        plate_region = img_bgr[y1:y2, x1:x2]

        result_img = img_bgr.copy()
        cv2.rectangle(result_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
        intermediates['located'] = result_img.copy()

        if debug:
            return plate_region, None, intermediates
        return plate_region, None


# ==================== 综合定位 ====================

def locate_plate(img_bgr, method='combined', debug=False, filepath=None):
    """
    综合车牌定位

    优先使用 CCPD 文件名解析（透视变换，最精准），
    失败时回退到绿色区域检测。

    Args:
        img_bgr: BGR图像
        method: 'combined' / 'green' / 'filename'
        debug: 返回中间结果
        filepath: 原始文件路径，用于 CCPD 文件名解析

    Returns:
        (plate_img, intermediates_dict)
    """
    inter = {}

    # 方法1：CCPD文件名解析（最精准）
    if filepath and method in ('combined', 'filename'):
        plate, inter = locate_plate_from_filename(
            img_bgr, filepath, debug=True
        )
        if plate is not None and plate.shape[0] > 10 and plate.shape[1] > 25:
            if debug:
                return plate, inter
            return plate, inter

    # 方法2：绿色区域检测（通用fallback）
    if method in ('combined', 'green'):
        plate, bbox, inter = locate_plate_from_green(img_bgr, debug=True)
        if plate is not None and plate.shape[0] > 10 and plate.shape[1] > 25:
            if debug:
                return plate, inter
            return plate, inter

    if debug:
        return None, inter
    return None, inter
