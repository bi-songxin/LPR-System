"""
车牌识别系统主程序 - License Plate Recognition System
基于 OpenCV + Tkinter 的图形化车牌识别工具

运行方式: python main_app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time
import os

# 导入自定义模块
from preprocessing import (
    load_image, to_grayscale, gaussian_blur,
    sobel_combined, clahe, equalize_histogram
)
from plate_locator import locate_plate
from char_segmenter import segment_characters
from char_recognizer import CharRecognizer


# ==================== 图像转换工具 ====================

def cv2_to_tk(image, target_size=None):
    """将OpenCV图像(BGR)转换为Tkinter可显示的PhotoImage"""
    if image is None:
        return None
    if len(image.shape) == 2:
        # 灰度图
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if target_size:
        h, w = image.shape[:2]
        tw, th = target_size
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    pil_img = Image.fromarray(image)
    return ImageTk.PhotoImage(pil_img)


# ==================== 主应用程序 ====================

class PlateRecognitionApp:
    """车牌识别系统主界面"""

    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别系统 - License Plate Recognition")
        self.root.geometry("1280x900")
        self.root.minsize(1000, 750)
        self.root.configure(bg='#f0f0f0')

        # 当前图像
        self.current_image = None
        self.current_filepath = None
        self.plate_image = None
        self.char_images = []
        self.recognized_text = ""
        self.processing = False

        # 识别器
        self.recognizer = CharRecognizer()

        # 设置UI
        self._setup_ui()

        # 异步加载模板
        threading.Thread(target=self._init_recognizer, daemon=True).start()

        # 窗口居中
        self.root.update_idletasks()
        self.root.geometry("1280x900")

    def _init_recognizer(self):
        """后台初始化识别器模板"""
        try:
            self.recognizer.load_or_create_templates()
            self._update_status("就绪 - 请打开车牌图片")
        except Exception as e:
            self._update_status(f"模板初始化失败: {e}")

    def _setup_ui(self):
        """设置UI布局"""
        # 主容器
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ===== 顶部控制栏 =====
        control_frame = tk.Frame(main_frame, bg='#ffffff', bd=1, relief=tk.SOLID)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(
            control_frame, text="车牌识别系统 V1.0",
            font=('Microsoft YaHei', 18, 'bold'),
            bg='#ffffff', fg='#2c3e50'
        )
        title_label.pack(side=tk.LEFT, padx=15, pady=10)

        # 按钮区域
        btn_frame = tk.Frame(control_frame, bg='#ffffff')
        btn_frame.pack(side=tk.RIGHT, padx=15, pady=10)

        self.open_btn = tk.Button(
            btn_frame, text="打开图片", command=self._open_image,
            width=12, height=1, font=('Microsoft YaHei', 10),
            bg='white', fg='black', cursor='hand2',
            activeforeground='black', bd=1, padx=10, pady=5
        )
        self.open_btn.pack(side=tk.LEFT, padx=5)
        self._bind_hover(self.open_btn)

        self.recognize_btn = tk.Button(
            btn_frame, text="开始识别", command=self._start_recognition,
            width=12, height=1, font=('Microsoft YaHei', 10),
            bg='white', fg='black', cursor='hand2',
            activeforeground='black', bd=1, padx=10, pady=5,
            state=tk.DISABLED
        )
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        self._bind_hover(self.recognize_btn)

        self.clear_btn = tk.Button(
            btn_frame, text="清空", command=self._clear,
            width=10, height=1, font=('Microsoft YaHei', 10),
            bg='white', fg='black', cursor='hand2',
            activeforeground='black', bd=1, padx=10, pady=5
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        self._bind_hover(self.clear_btn)

        # ===== 图像显示区域 (3列 x 2行网格) =====
        display_frame = tk.Frame(main_frame, bg='#f0f0f0')
        display_frame.pack(fill=tk.BOTH, expand=True)

        # 配置网格权重，行列均分
        display_frame.columnconfigure(0, weight=1, uniform='col')
        display_frame.columnconfigure(1, weight=1, uniform='col')
        display_frame.columnconfigure(2, weight=1, uniform='col')
        display_frame.rowconfigure(0, weight=1, uniform='row')
        display_frame.rowconfigure(1, weight=1, uniform='row')

        # Row 0, Col 0: 原始图片
        orig_frame = self._create_image_panel(display_frame, "原始图片")
        orig_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 3), pady=3)
        self.orig_label = orig_frame.img_label

        # Row 0, Col 1: 灰度预处理
        gray_frame = self._create_image_panel(display_frame, "灰度预处理")
        gray_frame.grid(row=0, column=1, sticky='nsew', padx=3, pady=3)
        self.gray_label = gray_frame.img_label

        # Row 0, Col 2: 边缘检测
        edge_frame = self._create_image_panel(display_frame, "边缘检测")
        edge_frame.grid(row=0, column=2, sticky='nsew', padx=(3, 0), pady=3)
        self.edge_label = edge_frame.img_label

        # Row 1, Col 0: 车牌定位
        plate_frame = self._create_image_panel(display_frame, "车牌定位")
        plate_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 3), pady=3)
        self.plate_label = plate_frame.img_label

        # Row 1, Col 1: 字符分割
        char_frame = self._create_image_panel(display_frame, "字符分割")
        char_frame.grid(row=1, column=1, sticky='nsew', padx=3, pady=3)
        self.char_label = char_frame.img_label

        # Row 1, Col 2: 识别结果（文本区域）
        result_frame = self._create_image_panel(display_frame, "识别结果")
        result_frame.grid(row=1, column=2, sticky='nsew', padx=(3, 0), pady=3)

        # 替换结果区域为文本框
        result_frame.img_label.destroy()
        result_text_frame = tk.Frame(result_frame.inner, bg='#2c3e50', bd=2, relief=tk.GROOVE)
        result_text_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.result_text = tk.Text(
            result_text_frame, font=('Microsoft YaHei', 32, 'bold'),
            bg='#2c3e50', fg='#2ecc71', bd=0,
            height=5, wrap=tk.WORD
        )
        self.result_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        self.result_text.insert('1.0', '等待识别...')
        self.result_text.config(state=tk.DISABLED)

        # ===== 底部状态栏 =====
        status_frame = tk.Frame(main_frame, bg='#ecf0f1', bd=1, relief=tk.SOLID)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = tk.Label(
            status_frame, text="就绪 - 请打开车牌图片",
            font=('Microsoft YaHei', 9), bg='#ecf0f1', fg='#7f8c8d',
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)

        self.info_label = tk.Label(
            status_frame, text="",
            font=('Microsoft YaHei', 9), bg='#ecf0f1', fg='#7f8c8d',
            anchor=tk.E
        )
        self.info_label.pack(side=tk.RIGHT, padx=10, pady=5)

    def _bind_hover(self, btn):
        """绑定按钮悬浮效果：鼠标进入时字体变蓝，离开时变黑"""
        btn.bind("<Enter>", lambda e: btn.config(fg='#2980b9'))
        btn.bind("<Leave>", lambda e: btn.config(fg='black'))

    def _create_image_panel(self, parent, title):
        """创建带标题的图像显示面板，标题显示在小框上方"""
        # 外层 wrapper
        wrapper = tk.Frame(parent, bg='#f0f0f0')

        # 小框上方的标题
        title_label = tk.Label(
            wrapper, text=title,
            font=('Microsoft YaHei', 10, 'bold'),
            bg='#f0f0f0', fg='#2c3e50', anchor=tk.W
        )
        title_label.pack(fill=tk.X, padx=(2, 0), pady=(0, 2))

        # 小框（图片容器）
        inner = tk.Frame(wrapper, bg='#ffffff', bd=1, relief=tk.SOLID)
        inner.pack(fill=tk.BOTH, expand=True)

        img_label = tk.Label(
            inner, bg='#ecf0f1', text='',
            font=('Microsoft YaHei', 10), fg='#bdc3c7'
        )
        img_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 把 img_label 和 inner panel 挂到 wrapper 上
        wrapper.img_label = img_label
        wrapper.inner = inner
        return wrapper

    def _update_display(self, label, image, target_size=(360, 240)):
        """更新图像显示标签"""
        if image is not None:
            tk_img = cv2_to_tk(image, target_size)
            label.config(image=tk_img, text='')
            label.image = tk_img  # 保持引用
        else:
            label.config(image='', text='无结果')

    def _update_status(self, text):
        """更新状态栏"""
        self.root.after(0, lambda: self.status_label.config(text=text))

    def _update_result(self, text, color='#2ecc71'):
        """更新识别结果"""
        self.root.after(0, lambda: self._set_result_text(text, color))

    def _set_result_text(self, text, color):
        """设置结果文本"""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', text)
        self.result_text.config(fg=color, state=tk.DISABLED)

    # ==================== 功能方法 ====================

    def _open_image(self):
        """打开本地图片文件"""
        filepath = filedialog.askopenfilename(
            title="选择车牌图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if not filepath:
            return

        try:
            self.current_image = load_image(filepath)
            self.current_filepath = filepath

            # 显示原图
            tk_img = cv2_to_tk(self.current_image, (360, 240))
            self.orig_label.config(image=tk_img, text='')
            self.orig_label.image = tk_img

            # 清空上次识别的中间结果面板
            self.gray_label.config(image='', text='')
            self.edge_label.config(image='', text='')
            self.plate_label.config(image='', text='')
            self.char_label.config(image='', text='')

            self._set_result_text('等待识别...', '#bdc3c7')

            # 启用识别按钮
            self.recognize_btn.config(state=tk.NORMAL)

            # 显示文件信息
            filename = os.path.basename(filepath)
            h, w = self.current_image.shape[:2]
            self.info_label.config(text=f"{filename} | {w}x{h}")
            self._update_status(f"已加载: {filename} ({w}x{h})")

            # 丢弃之前的处理结果
            self.plate_image = None
            self.char_images = []
            self.recognized_text = ""

        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片:\n{str(e)}")
            self._update_status(f"加载失败: {e}")

    def _start_recognition(self):
        """开始识别流程（后台线程）"""
        if self.current_image is None:
            messagebox.showwarning("提示", "请先打开一张车牌图片")
            return
        if self.processing:
            return

        self.processing = True
        self.recognize_btn.config(state=tk.DISABLED, text="处理中...")
        self._set_result_text('正在识别...', '#f39c12')
        self._update_status("正在处理...请稍候")

        # 后台执行识别
        thread = threading.Thread(target=self._recognition_pipeline, daemon=True)
        thread.start()

    def _recognition_pipeline(self):
        """完整的识别流水线"""
        try:
            img = self.current_image.copy()

            # ===== 步骤1: 图像预处理 =====
            self._update_status("步骤1/4: 图像预处理...")

            # 灰度化
            gray = to_grayscale(img)
            self.root.after(0, lambda: self._update_display(
                self.gray_label, gray, (360, 240)
            ))

            # CLAHE增强对比度
            enhanced = clahe(gray, clip_limit=2.0)

            # 高斯滤波
            blurred = gaussian_blur(enhanced, (5, 5))

            # Sobel边缘检测
            edges = sobel_combined(blurred, ksize=3)
            self.root.after(0, lambda: self._update_display(
                self.edge_label, edges, (360, 240)
            ))

            time.sleep(0.3)  # 让UI有时间更新

            # ===== 步骤2: 车牌定位 =====
            self._update_status("步骤2/4: 车牌定位...")

            plate_img, intermediates = locate_plate(
                img, method='combined', debug=True, filepath=self.current_filepath
            )

            if plate_img is None:
                self._update_status("车牌定位失败 - 尝试边缘检测方法...")
                plate_img, intermediates = locate_plate(
                    img, method='edge', debug=True, filepath=self.current_filepath
                )

            if plate_img is None:
                self._update_result('识别失败\n未找到车牌', '#e74c3c')
                self._update_status("识别失败: 未检测到车牌区域")
                self._enable_buttons()
                return

            self.plate_image = plate_img

            # 显示车牌定位结果
            if 'located' in intermediates:
                self.root.after(0, lambda: self._update_display(
                    self.plate_label, intermediates['located'], (360, 240)
                ))
            else:
                self.root.after(0, lambda: self._update_display(
                    self.plate_label, plate_img, (360, 240)
                ))

            time.sleep(0.3)

            # ===== 步骤3: 字符分割 =====
            self._update_status("步骤3/4: 字符分割...")

            chars, char_intermediates = segment_characters(plate_img, debug=True)
            self.char_images = chars

            if len(chars) < 3:
                self._update_result('分割失败\n字符不足', '#e74c3c')
                self._update_status(f"字符分割不完整: 仅分割出 {len(chars)} 个字符")
                self._enable_buttons()
                return

            # 显示字符分割结果
            if 'char_vis' in char_intermediates:
                self.root.after(0, lambda: self._update_display(
                    self.char_label, char_intermediates['char_vis'], (360, 240)
                ))
            elif 'plate_binary' in char_intermediates:
                self.root.after(0, lambda: self._update_display(
                    self.char_label, char_intermediates['plate_binary'], (360, 240)
                ))

            time.sleep(0.3)

            # ===== 步骤4: 字符识别 =====
            self._update_status("步骤4/4: 字符识别...")

            # 确保模板已加载
            if not self.recognizer.templates:
                self.recognizer.load_or_create_templates()

            result_chars, confidences = self.recognizer.recognize(chars)

            # 格式化结果
            plate_text = ''.join(result_chars)

            if len(plate_text) >= 8:
                formatted = f"{plate_text[0]}{plate_text[1]}·{plate_text[2]}{plate_text[3:]}\n"
            elif len(plate_text) == 7:
                formatted = f"{plate_text[0]}{plate_text[1]}·{plate_text[2:]}\n"
            else:
                formatted = plate_text

            formatted += f"\n(共{len(plate_text)}个字符)"

            self.recognized_text = plate_text

            avg_conf = np.mean(confidences) if confidences else 0
            if avg_conf > 0.6:
                color = '#2ecc71'
            elif avg_conf > 0.4:
                color = '#f39c12'
            else:
                color = '#e74c3c'

            self._update_result(formatted, color)

            # 状态栏显示每个字符的置信度
            char_conf_str = ' '.join(
                f"{r}({c:.0%})" for r, c in zip(result_chars, confidences)
            )
            self._update_status(
                f"识别完成: {plate_text} | 置信度: {char_conf_str}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._update_result(f'处理出错\n{str(e)[:30]}', '#e74c3c')
            self._update_status(f"处理出错: {str(e)[:50]}")

        finally:
            self._enable_buttons()

    def _enable_buttons(self):
        """恢复按钮状态"""
        self.root.after(0, lambda: self.recognize_btn.config(
            state=tk.NORMAL, text="开始识别"
        ))
        self.processing = False

    def _clear(self):
        """清空所有显示"""
        if self.processing:
            return

        self.current_image = None
        self.current_filepath = None
        self.plate_image = None
        self.char_images = []
        self.recognized_text = ""

        self.orig_label.config(image='', text='')
        self.gray_label.config(image='', text='')
        self.edge_label.config(image='', text='')
        self.plate_label.config(image='', text='')
        self.char_label.config(image='', text='')

        self._set_result_text('等待识别...', '#bdc3c7')
        self.recognize_btn.config(state=tk.DISABLED)
        self.info_label.config(text='')
        self._update_status("就绪 - 请打开车牌图片")


# ==================== 程序入口 ====================

def main():
    """主函数"""
    root = tk.Tk()

    # 尝试设置图标
    try:
        root.iconbitmap(default='')
    except Exception:
        pass

    app = PlateRecognitionApp(root)

    # 处理窗口关闭
    def on_closing():
        if app.processing:
            if messagebox.askokcancel("退出", "正在处理中，确定要退出吗？"):
                root.destroy()
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
