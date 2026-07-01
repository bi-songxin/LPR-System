"""
图像预处理模块 - Image Preprocessing Module
功能：灰度化、滤波去噪、边缘检测、形态学运算
"""

import cv2
import numpy as np


def load_image(filepath):
    """加载图像文件"""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"无法加载图像: {filepath}")
    return img


def to_grayscale(img):
    """将BGR图像转换为灰度图像"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size=(5, 5)):
    """高斯滤波去噪"""
    return cv2.GaussianBlur(img, kernel_size, 0)


def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    """双边滤波 - 保边去噪，对车牌处理效果好"""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def sobel_edge(gray_img, dx=1, dy=0, ksize=3):
    """Sobel边缘检测 - 检测水平和垂直边缘"""
    return cv2.Sobel(gray_img, cv2.CV_64F, dx, dy, ksize=ksize)


def canny_edge(gray_img, low_threshold=50, high_threshold=150):
    """Canny边缘检测"""
    return cv2.Canny(gray_img, low_threshold, high_threshold)


def sobel_combined(gray_img, ksize=3):
    """
    Sobel算子检测水平和垂直边缘后合并
    适合检测车牌边框
    """
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)


def morphology_close(img, kernel_size=(17, 5), iterations=3):
    """形态学闭运算 - 连接车牌区域边缘"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morphology_open(img, kernel_size=(3, 3), iterations=1):
    """形态学开运算 - 去除小噪点"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morphology_dilate(img, kernel_size=(3, 3), iterations=1):
    """膨胀操作"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(img, kernel, iterations=iterations)


def morphology_erode(img, kernel_size=(3, 3), iterations=1):
    """腐蚀操作"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.erode(img, kernel, iterations=iterations)


def adaptive_threshold(gray_img, block_size=15, C=3):
    """自适应阈值二值化"""
    return cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )


def otsu_threshold(gray_img):
    """Otsu大津法阈值"""
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def resize_image(img, width=None, height=None):
    """缩放图像"""
    if width is not None:
        h, w = img.shape[:2]
        ratio = width / w
        return cv2.resize(img, (width, int(h * ratio)))
    if height is not None:
        h, w = img.shape[:2]
        ratio = height / h
        return cv2.resize(img, (int(w * ratio), height))
    return img


def equalize_histogram(gray_img):
    """直方图均衡化 - 增强对比度"""
    return cv2.equalizeHist(gray_img)


def clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE自适应直方图均衡化"""
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_obj.apply(gray_img)
