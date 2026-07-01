"""
字符识别模块 - Character Recognizer
基于 CNN 的字符识别 + Hu矩模板匹配（备选）
"""

import cv2
import numpy as np
import os
import pickle
import torch
import torch.nn as nn


# ==================== CNN 分类器 ====================

class FastCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(128 * 6 * 4, 256), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(256, nc),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ==================== CNN + 模板匹配识别器 ====================

CHINESE_PROVINCES = [
    '京', '津', '沪', '渝', '冀', '豫', '云', '辽', '黑',
    '湘', '皖', '鲁', '新', '苏', '浙', '赣', '鄂', '桂',
    '甘', '晋', '蒙', '陕', '吉', '闽', '贵', '粤', '青',
    '川', '藏', '琼', '宁'
]

DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'J', 'K', 'L', 'M', 'N',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

ALL_CHARS = CHINESE_PROVINCES + DIGITS + LETTERS


class CharRecognizer:
    """CNN 数字分类器 + Hu矩模板匹配（备选）"""

    def __init__(self, template_size=(32, 48)):
        self.template_size = template_size
        self.templates = {}
        self.template_hus = {}
        self.template_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'templates'
        )
        self.template_file = os.path.join(self.template_dir, 'templates.pkl')
        self.hu_file = os.path.join(self.template_dir, 'templates_hu.pkl')
        self.real_template_file = os.path.join(self.template_dir, 'real_templates.pkl')
        self.real_hu_file = os.path.join(self.template_dir, 'real_templates_hu.pkl')
        self.clean_samples_file = os.path.join(self.template_dir, 'clean_samples.pkl')

        # CNN 模型
        self.cnn_model = None
        self.cnn_classes = None
        self.cnn_idx_to_char = None
        self.cnn_loaded = False
        # 位置专用模型（pos 0=省份, 1=第1字母, 2=第2字母, 3=第3字母）
        self.pos_models = {}   # {pos: FastCNN}
        self.pos_labels = {}   # {pos: [label_list]}
        self.pos_idx_map = {}  # {pos: {idx: char}}
        self._load_cnn_model()
        self._load_position_models()

    def _load_cnn_model(self):
        """加载 CNN 数字分类模型（优先用真实数据训练的模型）"""
        base = os.path.dirname(os.path.abspath(__file__))
        # 优先尝试真实数据训练的模型
        candidates = [
            os.path.join(base, 'models', 'digit_cnn_real.pth'),
            os.path.join(base, 'models', 'digit_cnn.pth'),
        ]
        for model_path in candidates:
            if not os.path.exists(model_path):
                continue
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                # 兼容两种保存格式：checkpoint dict 或纯 state_dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    classes = checkpoint.get('classes', [str(i) for i in range(10)])
                    self.cnn_model = FastCNN(len(classes))
                    self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    classes = [str(i) for i in range(10)]
                    self.cnn_model = FastCNN(10)
                    self.cnn_model.load_state_dict(checkpoint)
                self.cnn_model.eval()
                self.cnn_classes = classes
                self.cnn_idx_to_char = {i: c for i, c in enumerate(classes)}
                self.cnn_loaded = True
                print("已加载CNN数字分类器 (%d类) [%s]" % (len(classes), os.path.basename(model_path)))
                return
            except Exception as e:
                print("加载CNN模型 %s 失败: %s" % (os.path.basename(model_path), e))
        else:
            print(f"CNN模型不存在: {model_path}")

    def _load_position_models(self):
        """加载位置专用 CNN 模型（省份 + 前3个字母/数字位置）"""
        base = os.path.dirname(os.path.abspath(__file__))
        for pos in range(4):
            model_path = os.path.join(base, 'models', 'char_pos%d.pth' % pos)
            label_path = os.path.join(base, 'models', 'char_pos%d_labels.pkl' % pos)
            if not os.path.exists(model_path) or not os.path.exists(label_path):
                continue
            try:
                with open(label_path, 'rb') as f:
                    data = pickle.load(f)
                labels = data['labels']
                model = FastCNN(len(labels))
                state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                self.pos_models[pos] = model
                self.pos_labels[pos] = labels
                self.pos_idx_map[pos] = {i: c for i, c in enumerate(labels)}
                print("已加载位置%d CNN (%d类): %s" % (pos, len(labels), labels))
            except Exception as e:
                print("加载位置%d CNN失败: %s" % (pos, e))

    def _cnn_recognize(self, char_img, pos=None):
        """CNN 识别，返回 (label, confidence)。
        pos 为 None 时使用默认数字CNN，pos 为 0-3 时使用位置专用模型。
        """
        # 选择模型
        if pos is not None and pos in self.pos_models:
            model = self.pos_models[pos]
            idx_map = self.pos_idx_map[pos]
        elif self.cnn_loaded and self.cnn_model is not None:
            model = self.cnn_model
            idx_map = self.cnn_idx_to_char
        else:
            return None, 0.0

        if len(char_img.shape) == 3:
            char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
        if char_img.shape != (48, 32):
            char_img = cv2.resize(char_img, (32, 48))
        _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

        tensor = torch.from_numpy(
            (char_img.astype(np.float32) / 255.0).reshape(1, 1, 48, 32)
        )
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            max_prob, pred_idx = probs.max(1)
        return idx_map[pred_idx.item()], max_prob.item()

    def load_or_create_templates(self):
        # 加载 Hu 矩模板（用于省份和字母的备选识别）
        if os.path.exists(self.real_template_file) and os.path.exists(self.real_hu_file):
            try:
                with open(self.real_template_file, 'rb') as f:
                    data = pickle.load(f)
                self.templates = data.get('templates', {})
                with open(self.real_hu_file, 'rb') as f:
                    self.template_hus = pickle.load(f)
                if len(self.templates) >= 50 and len(self.template_hus) >= 50:
                    print(f"已加载 {len(self.templates)} 个字符模板 + Hu矩")
                    return True
            except Exception:
                pass
        if os.path.exists(self.template_file) and os.path.exists(self.hu_file):
            try:
                with open(self.template_file, 'rb') as f:
                    data = pickle.load(f)
                self.templates = data.get('templates', {})
                with open(self.hu_file, 'rb') as f:
                    self.template_hus = pickle.load(f)
                if len(self.templates) >= 50 and len(self.template_hus) >= 50:
                    print(f"已加载 {len(self.templates)} 个字符模板 + Hu矩")
                    return True
            except Exception:
                pass
        # 未找到模板文件
        if not self.templates:
            print("正在生成字符模板...")
            self._create_templates()
            self._save_templates()
        print("正在计算 Hu 矩...")
        self._compute_all_hu()
        self._save_hu()
        return True

    def recognize(self, char_images):
        if not self.cnn_loaded and (not self.templates or not self.template_hus):
            self.load_or_create_templates()

        result = []
        confidences = []

        for i, char_img in enumerate(char_images):
            if char_img is None or char_img.size == 0:
                result.append('?')
                confidences.append(0)
                continue

            # 预处理灰度图
            if len(char_img.shape) == 3:
                gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = char_img.copy()

            # 位置 0-3 用位置专用CNN，位置 4-7 用数字CNN
            if i <= 3 and i in self.pos_models:
                cnn_char, cnn_conf = self._cnn_recognize(gray, pos=i)
            elif i >= 4 and self.cnn_loaded:
                cnn_char, cnn_conf = self._cnn_recognize(gray)
            else:
                cnn_char, cnn_conf = None, 0.0

            if cnn_char is not None:
                result.append(cnn_char)
                confidences.append(cnn_conf)
                continue

            # CNN不可用时 → Hu矩 + 模板匹配（备用）
            if gray.shape != self.template_size[::-1]:
                gray = cv2.resize(gray, self.template_size)
            _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            if i == 0:
                candidates = CHINESE_PROVINCES
            elif i == 1:
                candidates = LETTERS
            else:
                candidates = ALL_CHARS

            try:
                hu_input = self._compute_hu_moments(gray)
            except Exception:
                result.append('?')
                confidences.append(0)
                continue

            scored = []
            for char in candidates:
                if char not in self.template_hus:
                    continue
                dist = self._hu_distance(hu_input, self.template_hus[char])
                hu_score = 1.0 / (1.0 + dist)
                scored.append((char, hu_score))

            if not scored:
                result.append('?')
                confidences.append(0)
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored[:5]
            best_char = '?'
            best_score = -1

            for char, hu_score in top_candidates:
                template = self.templates[char]
                tm_score = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)[0][0]
                tm_score = (tm_score + 1.0) / 2.0
                overlap = np.sum((gray == 0) & (template == 0))
                overlap = overlap / max(1, np.sum(gray == 0) + np.sum(template == 0)) * 2.0
                overlap = min(overlap, 1.0)
                final_score = 0.35 * hu_score + 0.40 * tm_score + 0.25 * overlap
                if final_score > best_score:
                    best_score = final_score
                    best_char = char

            result.append(best_char)
            confidences.append(best_score)

        return result, confidences

    def recognize_single(self, char_img, position=-1):
        """识别单个字符。根据位置选择对应的CNN模型。"""
        if position >= 0 and position <= 3 and position in self.pos_models:
            cnn_char, cnn_conf = self._cnn_recognize(char_img, pos=position)
        elif position >= 4 and self.cnn_loaded:
            cnn_char, cnn_conf = self._cnn_recognize(char_img)
        else:
            cnn_char, cnn_conf = None, 0.0
        if cnn_char is not None:
            return cnn_char, cnn_conf
        chars, confs = self.recognize([char_img])
        return chars[0], confs[0]

    def _create_templates(self):
        tw, th = self.template_size
        self.templates = {}
        for char in ALL_CHARS:
            template = self._create_char_template(char, tw, th)
            if template is not None:
                self.templates[char] = template
        print(f"已创建 {len(self.templates)} 个字符模板")

    def _create_char_template(self, char, tw, th):
        img = np.ones((th, tw), dtype=np.uint8) * 255
        if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
            self._draw_chinese_char(img, char, tw, th)
        elif char.isdigit():
            self._draw_digit(img, char, tw, th)
        elif char.isalpha():
            self._draw_letter(img, char, tw, th)
        return img

    def _draw_chinese_char(self, img, char, tw, th):
        h, w = img.shape
        chinese_patterns = {
            '京': [(0.3,0.1,0.4,0.15),(0.3,0.1,0.4,0.85),(0.15,0.3,0.7,0.1)],
            '津': [(0.2,0.1,0.6,0.15),(0.2,0.1,0.6,0.85),(0.1,0.4,0.8,0.1)],
            '沪': [(0.2,0.1,0.6,0.15),(0.2,0.1,0.6,0.5),(0.15,0.5,0.7,0.12)],
            '渝': [(0.2,0.1,0.6,0.15),(0.2,0.1,0.6,0.85),(0.1,0.35,0.8,0.1),(0.1,0.6,0.8,0.1)],
            '冀': [(0.15,0.1,0.7,0.12),(0.5,0.1,0.1,0.85),(0.2,0.5,0.6,0.1)],
            '豫': [(0.15,0.1,0.7,0.12),(0.5,0.1,0.12,0.85),(0.2,0.5,0.6,0.1)],
            '云': [(0.15,0.1,0.7,0.12),(0.5,0.1,0.1,0.85),(0.2,0.4,0.6,0.1),(0.15,0.65,0.7,0.1)],
            '辽': [(0.15,0.15,0.7,0.1),(0.15,0.15,0.7,0.7),(0.15,0.5,0.6,0.1)],
            '黑': [(0.2,0.1,0.6,0.1),(0.5,0.1,0.1,0.85),(0.15,0.35,0.7,0.08),(0.15,0.7,0.7,0.08)],
            '湘': [(0.15,0.1,0.7,0.12),(0.15,0.1,0.7,0.85),(0.15,0.4,0.6,0.08),(0.15,0.65,0.6,0.08)],
            '皖': [(0.2,0.1,0.6,0.1),(0.5,0.1,0.1,0.85),(0.2,0.4,0.5,0.08)],
            '鲁': [(0.2,0.1,0.6,0.1),(0.5,0.1,0.1,0.85),(0.2,0.4,0.6,0.08),(0.2,0.6,0.6,0.08)],
            '新': [(0.2,0.1,0.6,0.08),(0.35,0.1,0.08,0.85),(0.5,0.1,0.08,0.85),(0.2,0.4,0.5,0.08)],
            '苏': [(0.15,0.1,0.7,0.1),(0.5,0.1,0.08,0.85),(0.2,0.4,0.6,0.08),(0.2,0.65,0.6,0.08)],
            '浙': [(0.15,0.1,0.7,0.12),(0.15,0.1,0.7,0.85),(0.15,0.4,0.5,0.08)],
            '赣': [(0.15,0.1,0.7,0.1),(0.5,0.1,0.1,0.85),(0.2,0.4,0.6,0.08),(0.2,0.65,0.6,0.08)],
            '鄂': [(0.2,0.1,0.6,0.08),(0.4,0.1,0.08,0.45),(0.6,0.1,0.1,0.45),(0.2,0.5,0.6,0.08)],
            '桂': [(0.15,0.1,0.7,0.1),(0.5,0.1,0.08,0.85),(0.2,0.4,0.6,0.08),(0.2,0.65,0.6,0.08)],
            '甘': [(0.15,0.1,0.7,0.1),(0.5,0.1,0.08,0.85),(0.25,0.5,0.5,0.08),(0.25,0.7,0.5,0.08)],
            '晋': [(0.15,0.1,0.7,0.1),(0.5,0.1,0.08,0.85),(0.2,0.4,0.6,0.08),(0.2,0.65,0.6,0.08)],
            '蒙': [(0.1,0.1,0.8,0.1),(0.5,0.1,0.08,0.85),(0.2,0.4,0.6,0.08),(0.2,0.65,0.6,0.08)],
            '陕': [(0.15,0.1,0.7,0.12),(0.15,0.1,0.7,0.85),(0.15,0.4,0.6,0.1)],
            '吉': [(0.15,0.1,0.7,0.1),(0.5,0.1,0.08,0.85),(0.2,0.5,0.6,0.08)],
            '闽': [(0.2,0.1,0.6,0.1),(0.5,0.1,0.1,0.85),(0.2,0.4,0.6,0.08),(0.2,0.65,0.6,0.08)],
            '贵': [(0.2,0.1,0.6,0.08),(0.5,0.1,0.08,0.85),(0.2,0.4,0.5,0.1)],
            '粤': [(0.2,0.1,0.6,0.08),(0.5,0.1,0.08,0.85),(0.2,0.4,0.5,0.08),(0.2,0.65,0.5,0.08)],
            '青': [(0.15,0.1,0.7,0.1),(0.5,0.1,0.08,0.85),(0.2,0.5,0.6,0.08)],
            '川': [(0.25,0.1,0.12,0.8),(0.4,0.1,0.12,0.55),(0.58,0.1,0.12,0.8)],
            '藏': [(0.1,0.1,0.8,0.08),(0.5,0.1,0.08,0.85),(0.2,0.4,0.6,0.08),(0.2,0.65,0.6,0.08)],
            '琼': [(0.15,0.1,0.7,0.1),(0.15,0.1,0.7,0.85),(0.15,0.4,0.6,0.1)],
            '宁': [(0.15,0.1,0.7,0.08),(0.5,0.1,0.08,0.85),(0.2,0.5,0.6,0.08)],
        }
        if char in chinese_patterns:
            for rx, ry, rw, rh in chinese_patterns[char]:
                x = int(rx*w); y = int(ry*h)
                bw = max(2, int(rw*w)); bh = max(2, int(rh*h))
                cv2.rectangle(img, (x,y), (x+bw, y+bh), 0, -1)
        else:
            cv2.rectangle(img, (int(0.15*w), int(0.1*h)),
                         (int(0.75*w), int(0.9*h)), 0, -1)

    def _draw_digit(self, img, digit, tw, th):
        h, w = img.shape
        thickness = max(2, int(w*0.12))
        patterns = {
            '0': [(0.2,0.1,0.6,0.8,False)],
            '1': [(0.45,0.1,0.1,0.8,True)],
            '2': [(0.15,0.1,0.7,0.08,True),(0.75,0.1,0.1,0.45,True),(0.15,0.45,0.7,0.08,True),(0.15,0.45,0.1,0.45,True),(0.15,0.8,0.7,0.1,True)],
            '3': [(0.15,0.1,0.7,0.08,True),(0.75,0.15,0.1,0.3,True),(0.15,0.45,0.7,0.08,True),(0.75,0.5,0.1,0.3,True),(0.15,0.8,0.7,0.1,True)],
            '4': [(0.15,0.1,0.1,0.55,True),(0.65,0.1,0.1,0.8,True),(0.2,0.55,0.65,0.08,True)],
            '5': [(0.7,0.1,0.08,0.4,True),(0.15,0.1,0.7,0.08,True),(0.15,0.4,0.6,0.08,True),(0.7,0.45,0.08,0.35,True),(0.15,0.75,0.65,0.1,True)],
            '6': [(0.2,0.1,0.55,0.8,True),(0.2,0.45,0.55,0.08,True),(0.15,0.1,0.7,0.08,True)],
            '7': [(0.15,0.1,0.7,0.08,True),(0.7,0.1,0.1,0.8,True)],
            '8': [(0.2,0.1,0.5,0.8,True),(0.2,0.45,0.5,0.08,True),(0.2,0.15,0.08,0.35,True),(0.63,0.15,0.08,0.35,True),(0.2,0.5,0.08,0.35,True),(0.63,0.5,0.08,0.35,True)],
            '9': [(0.15,0.15,0.65,0.7,True),(0.15,0.15,0.7,0.08,True),(0.7,0.15,0.08,0.4,True)],
        }
        if digit in patterns:
            for rx,ry,rw,rh,filled in patterns[digit]:
                x,y = int(rx*w), int(ry*h)
                bw,bh = max(2,int(rw*w)), max(2,int(rh*h))
                if filled: cv2.rectangle(img,(x,y),(x+bw,y+bh),0,-1)
                else: cv2.rectangle(img,(x,y),(x+bw,y+bh),0,thickness)

    def _draw_letter(self, img, letter, tw, th):
        h, w = img.shape
        patterns = {
            'A': [(0.2,0.1,0.1,0.8,True),(0.6,0.1,0.1,0.8,True),(0.2,0.1,0.55,0.1,True),(0.2,0.4,0.55,0.08,True)],
            'B': [(0.2,0.1,0.08,0.8,True),(0.2,0.1,0.5,0.08,True),(0.2,0.45,0.5,0.08,True),(0.2,0.8,0.5,0.1,True),(0.6,0.15,0.08,0.3,True),(0.6,0.5,0.08,0.3,True)],
            'C': [(0.2,0.1,0.6,0.8,True),(0.2,0.25,0.08,0.5,False)],
            'D': [(0.2,0.1,0.08,0.8,True),(0.2,0.1,0.5,0.08,True),(0.2,0.8,0.5,0.1,True),(0.6,0.15,0.08,0.65,True)],
            'E': [(0.2,0.1,0.08,0.8,True),(0.2,0.1,0.55,0.08,True),(0.2,0.45,0.45,0.08,True),(0.2,0.8,0.55,0.1,True)],
            'F': [(0.2,0.1,0.08,0.8,True),(0.2,0.1,0.55,0.08,True),(0.2,0.45,0.45,0.08,True)],
            'G': [(0.15,0.1,0.6,0.8,True),(0.15,0.1,0.6,0.08,True),(0.15,0.8,0.6,0.1,True),(0.18,0.45,0.08,0.3,False),(0.5,0.45,0.08,0.4,True)],
            'H': [(0.2,0.1,0.08,0.8,True),(0.6,0.1,0.08,0.8,True),(0.2,0.4,0.52,0.08,True)],
            'J': [(0.55,0.1,0.1,0.65,True),(0.2,0.6,0.4,0.1,True),(0.2,0.6,0.08,0.2,True)],
            'K': [(0.2,0.1,0.08,0.8,True),(0.6,0.25,0.08,0.6,True),(0.25,0.4,0.4,0.08,True)],
            'L': [(0.2,0.1,0.08,0.8,True),(0.2,0.75,0.55,0.1,True)],
            'M': [(0.2,0.1,0.08,0.8,True),(0.4,0.1,0.08,0.55,True),(0.6,0.1,0.08,0.8,True),(0.2,0.15,0.5,0.08,True)],
            'N': [(0.2,0.1,0.08,0.8,True),(0.6,0.1,0.08,0.8,True),(0.2,0.15,0.5,0.08,True)],
            'P': [(0.2,0.1,0.08,0.8,True),(0.2,0.1,0.5,0.08,True),(0.55,0.15,0.08,0.3,True),(0.2,0.45,0.5,0.08,True)],
            'Q': [(0.15,0.1,0.6,0.75,True),(0.15,0.1,0.55,0.08,True),(0.15,0.75,0.55,0.08,True),(0.6,0.65,0.1,0.2,True)],
            'R': [(0.2,0.1,0.08,0.8,True),(0.2,0.1,0.5,0.08,True),(0.55,0.15,0.08,0.3,True),(0.2,0.45,0.5,0.08,True),(0.55,0.45,0.1,0.4,True)],
            'S': [(0.15,0.1,0.6,0.08,True),(0.15,0.1,0.08,0.4,True),(0.15,0.45,0.55,0.08,True),(0.6,0.45,0.08,0.4,True),(0.15,0.75,0.55,0.1,True)],
            'T': [(0.15,0.1,0.6,0.08,True),(0.4,0.1,0.08,0.8,True)],
            'U': [(0.2,0.1,0.5,0.8,True),(0.2,0.75,0.08,0.15,False)],
            'V': [(0.25,0.15,0.08,0.7,True),(0.6,0.15,0.08,0.7,True),(0.2,0.7,0.5,0.08,True)],
            'W': [(0.15,0.15,0.08,0.7,True),(0.4,0.15,0.08,0.5,True),(0.65,0.15,0.08,0.7,True),(0.15,0.75,0.6,0.08,True)],
            'X': [(0.2,0.15,0.5,0.05,True),(0.6,0.15,0.08,0.7,True),(0.2,0.45,0.5,0.08,True),(0.2,0.15,0.08,0.7,True),(0.2,0.75,0.5,0.08,True)],
            'Y': [(0.2,0.15,0.08,0.5,True),(0.6,0.15,0.08,0.5,True),(0.2,0.55,0.55,0.08,True)],
            'Z': [(0.15,0.1,0.6,0.08,True),(0.6,0.15,0.08,0.7,True),(0.15,0.45,0.5,0.08,True),(0.2,0.15,0.08,0.7,True),(0.15,0.75,0.55,0.1,True)],
        }
        if letter in patterns:
            for rx,ry,rw,rh,filled in patterns[letter]:
                x,y = int(rx*w), int(ry*h)
                bw,bh = max(2,int(rw*w)), max(2,int(rh*h))
                if filled: cv2.rectangle(img,(x,y),(x+bw,y+bh),0,-1)
                else: cv2.rectangle(img,(x,y),(x+bw,y+bh),0,1)

    def _save_templates(self):
        os.makedirs(self.template_dir, exist_ok=True)
        with open(self.template_file, 'wb') as f:
            pickle.dump({'templates': self.templates}, f)
        print(f"模板已保存至 {self.template_file}")
