# LPR-System — 车牌识别系统

基于 OpenCV + PyTorch CNN 的中文车牌识别系统，支持蓝色标准车牌和绿色新能源车牌，配备 Tkinter 图形化界面。

## 项目架构

```
OpenCV（前段）                        PyTorch CNN（后段）
┌──────────────────────────────────┐  ┌──────────────────────┐
│ 原始图片 → 预处理 → 定位 → 分割  │→ │ 5个CNN模型 → 识别结果 │
│ cv2.Sobel  findContours 投影法   │  │ FastCNN + Hu矩备用    │
│ CLAHE增强  warpPerspective 固定窗│  │ 5个位置专用模型       │
└──────────────────────────────────┘  └──────────────────────┘
        纯 CV 手写管线                        深度学习分类
```

| 阶段 | 技术栈 | 核心方法 |
|------|--------|----------|
| 预处理 | OpenCV | 灰度化 → CLAHE 对比度增强 → 高斯滤波 → Sobel 边缘检测 |
| 车牌定位 | OpenCV | 边缘检测 + 轮廓筛选 + 颜色过滤 + 透视校正 |
| 字符分割 | OpenCV | 10 种二值化择优 + 垂直投影 + 7/8 字符固定窗口模板 |
| 字符识别 | PyTorch + OpenCV | 5 个位置专用 FastCNN + Hu 矩模板匹配（备用） |
| GUI | Tkinter | 6 面板实时可视化 + 置信度展示 |

## 目录结构

```
LPR-System/
├── main_app.py              # 主程序入口 (Tkinter GUI)
├── preprocessing.py         # 图像预处理 (灰度化/CLAHE/滤波/边缘)
├── plate_locator.py         # 车牌定位 (轮廓筛选/透视校正/颜色过滤)
├── char_segmenter.py        # 字符分割 (二值化/投影法/固定窗口)
├── char_recognizer.py       # 字符识别 (5个CNN + Hu矩备用)
├── benchmark.py             # 性能测试 (CCPD验证集)
│
├── models/                  # CNN 模型权重
│   ├── digit_cnn_real.pth   #   数字CNN (真实数据训练, 首选)
│   ├── digit_cnn.pth        #   数字CNN (合成数据训练, 备选)
│   ├── char_pos0.pth        #   位置0 CNN (省份, 5类)
│   ├── char_pos1.pth        #   位置1 CNN (字母1, 3类)
│   ├── char_pos2.pth        #   位置2 CNN (字符2, 3类)
│   ├── char_pos3.pth        #   位置3 CNN (字符3, 9类)
│   └── char_pos*_labels.pkl #   各位置标签映射表
│
├── templates/               # 模板匹配数据
│   ├── real_templates.pkl   #   真实字符模板 (优先)
│   ├── real_templates_hu.pkl#   真实字符Hu矩特征
│   └── clean_samples.pkl    #   清洗后训练样本
│
├── train_cnn_v3.py          # 数字CNN训练 (V3最终版, 当前使用)
├── train_cnn_v2.py          # 数字CNN训练 V2 (历史版本)
├── train_cnn.py             # 数字CNN训练 V1 (历史版本)
├── train_cnn_ensemble.py    # CNN集成训练 (实验版本, 不保存模型)
├── train_real_cnn.py        # 真实数据CNN训练
├── train_char_models.py     # 位置专用CNN训练
├── build_clean_samples.py   # 清洗样本构造
├── build_real_templates.py  # 真实模板构造
├── build_real_digit_data.py # 真实数字数据构造
│
├── CCPD2020/                # CCPD数据集 (可选, benchmark用)
└── debug/                   # 调试输出
```

## 快速开始

### 环境要求

- Python 3.7+
- OpenCV (`opencv-python`)
- PyTorch
- NumPy, Pillow (PIL)
- Tkinter (Python 内置)

### 安装依赖

```bash
pip install opencv-python torch numpy pillow
```

### 运行主程序

```bash
python main_app.py
```

点击"打开图片"选择车牌照片 → 点击"开始识别"即可看到四步流水线实时可视化。

### 运行性能测试

```bash
python benchmark.py
```

## CNN 模型说明

系统使用 **5 个 FastCNN 模型**，按字符位置分别识别：

| 模型 | 位置 | 类别数 | 候选集 | 输入尺寸 |
|------|------|--------|--------|----------|
| `char_pos0.pth` | 0 (省份) | 5 | 沪,浙,皖,粤,苏 | 48×32 灰度 |
| `char_pos1.pth` | 1 (字母1) | 3 | A,B,K | 48×32 灰度 |
| `char_pos2.pth` | 2 (字符2) | 3 | 0,D,F | 48×32 灰度 |
| `char_pos3.pth` | 3 (字符3) | 9 | 0-7,B,R | 48×32 灰度 |
| `digit_cnn_real.pth` | 4-7 (数字) | 10 | 0-9 | 48×32 灰度 |

网络结构 `FastCNN`：3 组 Conv+BN+ReLU+Pool → Dropout → FC → Dropout → FC

识别决策：位置 0-3 用对应位置模型 → 位置 4-7 用数字模型 → 失败则回退 Hu 矩 + 模板匹配。

## 测试结果

在 1001 张 CCPD 绿牌验证集上的表现：

| 指标 | 结果 |
|------|------|
| 整牌全对 (8/8) | **84.9%** |
| 单字准确率 | **96.8%** |
| 车牌定位成功率 | 100% |
| 字符分割成功率 | 100% |
| 速度 | 0.03s / 张 |

各位置准确率：

| Pos 0 | Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
|-------|-------|-------|-------|-------|-------|-------|-------|
| 98.3% | 98.6% | 94.3% | 96.6% | 95.7% | 96.6% | 97.5% | 97.0% |

## 训练流程

如需重新训练模型：

```bash
# 1. 构造训练数据
python build_clean_samples.py      # → templates/clean_samples.pkl
python build_real_templates.py     # → templates/real_templates.pkl
python build_real_digit_data.py    # → models/real_digit_data.pkl

# 2. 训练模型
python train_cnn_v3.py             # → models/digit_cnn.pth
python train_real_cnn.py           # → models/digit_cnn_real.pth
python train_char_models.py        # → models/char_pos{0..3}.pth
```
