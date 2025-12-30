import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QRectF


class InteractiveCanvas(QWidget):
    """
    一个支持缩放、平移、Mask叠加显示的自定义画布控件。
    核心功能：
    1. 显示 OpenCV 格式的图片。
    2. 接收 Numpy 格式的 Mask 并叠加显示（半透明）。
    3. 处理鼠标点击，将 屏幕坐标 转换为 图像真实坐标。
    """

    # 信号：发射 (image_x, image_y, is_left_click)
    # is_left_click: 1 for Left (Add point), 0 for Right (Remove point)
    click_signal = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- 数据层 ---
        self.pixmap_image = None  # 底层原图
        self.pixmap_mask = None  # 顶层 Mask (带透明度)
        self._image_w = 0
        self._image_h = 0

        # --- 视图变换层 (View Transformation) ---
        self.scale = 1.0  # 缩放比例
        self.offset_x = 0.0  # 平移 X
        self.offset_y = 0.0  # 平移 Y

        # --- 交互状态 ---
        self.is_panning = False
        self.last_mouse_pos = QPoint()

        # --- 样式配置 ---
        self.setMouseTracking(True)  # 开启鼠标追踪
        self.setStyleSheet("background-color: #2b2b2b;")  # 深灰色背景

    # ============================
    # 1. 数据加载接口 (Public API)
    # ============================

    def set_image(self, img_np):
        """
        加载 OpenCV 图片 (BGR 或 Grayscale)
        """
        if img_np is None: return

        # 1. 格式转换: BGR -> RGB
        if len(img_np.shape) == 3:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            format_ = QImage.Format.Format_RGB888
        else:
            img_rgb = img_np
            h, w = img_rgb.shape
            bytes_per_line = w
            format_ = QImage.Format.Format_Grayscale8

        # 2. 创建 QPixmap
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, format_)
        self.pixmap_image = QPixmap.fromImage(q_img)
        self._image_w = w
        self._image_h = h

        # 3. 重置视图 (Fit Window)
        self.fit_to_window()
        self.pixmap_mask = None  # 清空旧 Mask
        self.update()  # 触发重绘

    def set_mask(self, mask_np):
        """
        加载 Mask (0/1 矩阵), 转换为半透明红色覆盖层
        """
        if mask_np is None or self.pixmap_image is None:
            self.pixmap_mask = None
            self.update()
            return

        # 确保 mask 尺寸匹配
        if mask_np.shape[:2] != (self._image_h, self._image_w):
            print("Error: Mask shape does not match image shape.")
            return

        # --- 制作带 Alpha 通道的 RGBA 图像 ---
        # 创建一个全黑的 RGBA 图像 (高度, 宽度, 4通道)
        # 技巧：我们不仅要在前景画红，还要让背景全透明

        # 1. 初始化 RGBA 矩阵
        # 通道顺序: R, G, B, A
        mask_rgba = np.zeros((self._image_h, self._image_w, 4), dtype=np.uint8)

        # 2. 找到前景区域 (mask == 1)
        foreground = (mask_np > 0)

        # 3. 填充颜色 (例如: 红色 [255, 0, 0] + 透明度 [100])
        mask_rgba[foreground, 0] = 255  # Red
        mask_rgba[foreground, 1] = 0  # Green
        mask_rgba[foreground, 2] = 0  # Blue
        mask_rgba[foreground, 3] = 100  # Alpha (0-255, 越大越不透明)

        # 4. 转换为 QPixmap
        q_img_mask = QImage(mask_rgba.data, self._image_w, self._image_h,
                            self._image_w * 4, QImage.Format.Format_RGBA8888)
        self.pixmap_mask = QPixmap.fromImage(q_img_mask)

        self.update()

    # ============================
    # 2. 核心绘图逻辑 (Rendering)
    # ============================

    def paintEvent(self, event):
        """
        PyQt 绘图循环。所有视觉元素都在这里被画出来。
        顺序：背景 -> 原图 -> Mask -> 其他UI元素
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)  # 平滑缩放

        if self.pixmap_image:
            # --- 变换坐标系 (Transform) ---
            # 这里的逻辑是：先移动画布原点，再缩放
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.scale, self.scale)

            # --- 1. 画原图 ---
            # 在 (0,0) 位置画图
            painter.drawPixmap(0, 0, self.pixmap_image)

            # --- 2. 画 Mask (如果有) ---
            if self.pixmap_mask:
                painter.drawPixmap(0, 0, self.pixmap_mask)

            # (可选) 你可以在这里画出用户刚刚点击的点，作为视觉反馈

        else:
            # 如果没有图片，显示提示文字
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")

    # ============================
    # 3. 交互逻辑 (Interaction)
    # ============================

    def wheelEvent(self, event):
        """
        鼠标滚轮缩放 (以鼠标指针为中心缩放)
        """
        if not self.pixmap_image: return

        # 缩放系数
        zoom_in_factor = 1.1
        zoom_out_factor = 0.9

        # 判断滚轮方向
        angle = event.angleDelta().y()
        if angle > 0:
            factor = zoom_in_factor
        else:
            factor = zoom_out_factor

        # --- 核心算法：以鼠标为中心缩放 ---
        # 1. 获取鼠标在 Widget 中的坐标
        mouse_pos = event.position()  # QPointF

        # 2. 将鼠标坐标反算回 变换前的坐标 (相对于图片的偏移)
        # 公式: (ViewPos - Offset) / Scale
        old_x = (mouse_pos.x() - self.offset_x) / self.scale
        old_y = (mouse_pos.y() - self.offset_y) / self.scale

        # 3. 更新缩放比例
        self.scale *= factor
        # 限制缩放范围 (避免无限大或无限小)
        self.scale = max(0.05, min(self.scale, 50.0))

        # 4. 调整 Offset，保持鼠标下的像素点位置不变
        # 新 Offset = ViewPos - (ImagePos * NewScale)
        self.offset_x = mouse_pos.x() - old_x * self.scale
        self.offset_y = mouse_pos.y() - old_y * self.scale

        self.update()

    def mousePressEvent(self, event):
        """
        鼠标点击事件
        左键/右键: 发送 SAM 点击信号
        中键: 开始拖拽平移
        """
        if not self.pixmap_image: return

        # 中键拖拽 (或者按住 Alt/Ctrl 拖拽，可自定义)
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # 左键或右键 -> SAM 点击
        # 1. 获取 Widget 坐标
        widget_x = event.position().x()
        widget_y = event.position().y()

        # 2. 映射回 真实图片坐标
        img_x = int((widget_x - self.offset_x) / self.scale)
        img_y = int((widget_y - self.offset_y) / self.scale)

        # 3. 边界检查 (点击在图片外不处理)
        if 0 <= img_x < self._image_w and 0 <= img_y < self._image_h:
            # 1: Positive(Left), 0: Negative(Right)
            is_left = 1 if event.button() == Qt.MouseButton.LeftButton else 0

            # 发射信号给 Controller
            print(f"[Canvas] Clicked: Image({img_x}, {img_y}), Type: {is_left}")
            self.click_signal.emit(img_x, img_y, is_left)

    def mouseMoveEvent(self, event):
        """处理拖拽平移"""
        if self.is_panning:
            delta = event.position() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ============================
    # 4. 辅助功能 (Helpers)
    # ============================

    def fit_to_window(self):
        """将图片自适应缩放到窗口大小并居中"""
        if not self.pixmap_image or self.width() == 0 or self.height() == 0:
            return

        # 计算宽和高的缩放比例，取较小值以保证完全显示
        scale_w = self.width() / self._image_w
        scale_h = self.height() / self._image_h
        self.scale = min(scale_w, scale_h) * 0.9  # 留一点边距

        # 居中计算
        new_w = self._image_w * self.scale
        new_h = self._image_h * self.scale
        self.offset_x = (self.width() - new_w) / 2
        self.offset_y = (self.height() - new_h) / 2

        self.update()