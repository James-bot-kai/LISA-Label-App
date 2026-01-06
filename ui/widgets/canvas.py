import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QRectF


class InteractiveCanvas(QWidget):
    """
    一个支持缩放、平移、双层 Mask (Base + Preview) 叠加显示的自定义画布控件。
    """

    # 信号：发射 (image_x, image_y, is_left_click)
    # is_left_click: 1 for Left (Add point), 0 for Right (Remove point)
    click_signal = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- 数据层 ---
        self.pixmap_image = None  # 底层原图
        self.pixmap_base = None  # 中层: 已确认的 Mask (红色)
        self.pixmap_preview = None  # 顶层: SAM 正在预测的 Mask (绿色)

        self._image_w = 0
        self._image_h = 0

        # --- 视图变换层 ---
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # --- 交互状态 ---
        self.is_panning = False
        self.last_mouse_pos = QPoint()

        # --- 样式配置 ---
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #2b2b2b;")

    # ============================
    # 1. 数据加载接口
    # ============================

    def set_image(self, img_np):
        """加载 OpenCV 图片"""
        if img_np is None:
            self.pixmap_image = None
            self.update()
            return

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

        self._image_w = w
        self._image_h = h

        # 2. 创建 QPixmap
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, format_)
        self.pixmap_image = QPixmap.fromImage(q_img)

        # 3. 换图时，必须重置所有 Mask
        self.pixmap_base = None
        self.pixmap_preview = None

        self.fit_to_window()
        self.update()

    def _make_colored_mask(self, mask_np, color_rgb):
        """
        [内部辅助] 将 0/1 Mask 转换为带颜色的透明 QPixmap
        color_rgb: tuple (R, G, B) 例如 (255, 0, 0)
        """
        if mask_np is None:
            return None

        # 尺寸对齐 (防止计算误差导致尺寸不匹配)
        if mask_np.shape[:2] != (self._image_h, self._image_w):
            mask_np = cv2.resize(mask_np, (self._image_w, self._image_h), interpolation=cv2.INTER_NEAREST)

        # 1. 初始化 RGBA 矩阵
        mask_rgba = np.zeros((self._image_h, self._image_w, 4), dtype=np.uint8)

        # 2. 找到前景区域
        foreground = (mask_np > 0)

        # 3. 填充颜色
        mask_rgba[foreground, 0] = color_rgb[0]  # R
        mask_rgba[foreground, 1] = color_rgb[1]  # G
        mask_rgba[foreground, 2] = color_rgb[2]  # B
        mask_rgba[foreground, 3] = 120  # Alpha (透明度)

        # 4. 转 QPixmap
        q_img = QImage(mask_rgba.data, self._image_w, self._image_h,
                       self._image_w * 4, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(q_img)

    def set_mask(self, mask_np):
        """设置底图 Mask (显示为红色)"""
        # 如果传入 None，清除底图
        if mask_np is None:
            self.pixmap_base = None
        else:
            self.pixmap_base = self._make_colored_mask(mask_np, (255, 0, 0))  # Red
        self.update()

    def set_preview_mask(self, mask_np):
        """【新】设置预览 Mask (显示为绿色)"""
        # 如果传入 None，清除预览
        if mask_np is None:
            self.pixmap_preview = None
        else:
            self.pixmap_preview = self._make_colored_mask(mask_np, (0, 255, 0))  # Green
        self.update()

    # ============================
    # 2. 核心绘图逻辑
    # ============================

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.pixmap_image:
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.scale, self.scale)

            # 1. 画原图
            painter.drawPixmap(0, 0, self.pixmap_image)

            # 2. 画红色底图 (Base)
            if self.pixmap_base:
                painter.drawPixmap(0, 0, self.pixmap_base)

            # 3. 画绿色预览 (Preview) - 这一层在最上面
            if self.pixmap_preview:
                painter.drawPixmap(0, 0, self.pixmap_preview)

        else:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")

    # ============================
    # 3. 交互逻辑 (保持不变)
    # ============================

    def wheelEvent(self, event):
        if not self.pixmap_image: return
        zoom_in_factor = 1.1
        zoom_out_factor = 0.9

        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
        else:
            factor = zoom_out_factor

        mouse_pos = event.position()
        old_x = (mouse_pos.x() - self.offset_x) / self.scale
        old_y = (mouse_pos.y() - self.offset_y) / self.scale

        self.scale *= factor
        self.scale = max(0.05, min(self.scale, 50.0))

        self.offset_x = mouse_pos.x() - old_x * self.scale
        self.offset_y = mouse_pos.y() - old_y * self.scale
        self.update()

    def mousePressEvent(self, event):
        if not self.pixmap_image: return

        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        widget_x = event.position().x()
        widget_y = event.position().y()
        img_x = int((widget_x - self.offset_x) / self.scale)
        img_y = int((widget_y - self.offset_y) / self.scale)

        if 0 <= img_x < self._image_w and 0 <= img_y < self._image_h:
            is_left = 1 if event.button() == Qt.MouseButton.LeftButton else 0
            print(f"[Canvas] Clicked: Image({img_x}, {img_y}), Type: {is_left}")
            self.click_signal.emit(img_x, img_y, is_left)

    def mouseMoveEvent(self, event):
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

    def fit_to_window(self):
        if not self.pixmap_image or self.width() == 0 or self.height() == 0:
            return
        scale_w = self.width() / self._image_w
        scale_h = self.height() / self._image_h
        self.scale = min(scale_w, scale_h) * 0.9
        new_w = self._image_w * self.scale
        new_h = self._image_h * self.scale
        self.offset_x = (self.width() - new_w) / 2
        self.offset_y = (self.height() - new_h) / 2
        self.update()