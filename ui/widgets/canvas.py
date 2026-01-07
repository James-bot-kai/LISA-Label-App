import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QCursor
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QRect


class InteractiveCanvas(QWidget):
    # 信号定义
    click_signal = pyqtSignal(int, int, int)  # SAM点击: x, y, is_left
    rect_erase_signal = pyqtSignal(int, int, int, int)  # 框选擦除: x, y, w, h
    brush_signal = pyqtSignal(int, int, int)  # 画笔: x, y, is_add (1=add, 0=sub)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- 数据层 ---
        self.pixmap_image = None
        self.pixmap_base = None  # 红色底图
        self.pixmap_preview = None  # 绿色预览
        self._image_w = 0
        self._image_h = 0

        # --- 视图变换 ---
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # --- 交互状态 ---
        self.mode = "sam"  # 模式: "sam", "eraser" (框选), "brush" (画笔)
        self.brush_size = 10  # 画笔半径

        self.is_panning = False
        self.is_drawing_rect = False  # 是否正在拉框
        self.is_brushing = False  # 是否正在涂抹
        self.last_mouse_pos = QPoint()

        # 框选时的临时矩形 (屏幕坐标)
        self.drag_start_pos = QPoint()
        self.drag_current_pos = QPoint()

        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #2b2b2b;")

    # ==========================
    # API
    # ==========================
    def set_mode(self, mode):
        """切换模式: 'sam', 'eraser', 'brush'"""
        self.mode = mode
        if mode == "sam":
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif mode == "eraser":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode == "brush":
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()

    def set_image(self, img_np):
        if img_np is None:
            self.pixmap_image = None
            self.update()
            return

        if len(img_np.shape) == 3:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            fmt = QImage.Format.Format_RGB888
        else:
            img_rgb = img_np
            h, w = img_rgb.shape
            ch = 1;
            fmt = QImage.Format.Format_Grayscale8

        self._image_w = w
        self._image_h = h
        q_img = QImage(img_rgb.data, w, h, w * (3 if ch == 3 else 1), fmt)
        self.pixmap_image = QPixmap.fromImage(q_img)

        self.pixmap_base = None
        self.pixmap_preview = None
        self.fit_to_window()
        self.update()

    def set_mask(self, mask_np):
        self.pixmap_base = self._make_colored_mask(mask_np, (255, 0, 0))
        self.update()

    def set_preview_mask(self, mask_np):
        self.pixmap_preview = self._make_colored_mask(mask_np, (0, 255, 0))
        self.update()

    def _make_colored_mask(self, mask_np, color):
        if mask_np is None: return None
        if mask_np.shape[:2] != (self._image_h, self._image_w):
            mask_np = cv2.resize(mask_np, (self._image_w, self._image_h), interpolation=cv2.INTER_NEAREST)

        mask_rgba = np.zeros((self._image_h, self._image_w, 4), dtype=np.uint8)
        mask_rgba[mask_np > 0] = [color[0], color[1], color[2], 120]
        return QPixmap.fromImage(
            QImage(mask_rgba.data, self._image_w, self._image_h, self._image_w * 4, QImage.Format.Format_RGBA8888))

    # ==========================
    # 绘图事件
    # ==========================
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.pixmap_image:
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.scale, self.scale)

            painter.drawPixmap(0, 0, self.pixmap_image)
            if self.pixmap_base: painter.drawPixmap(0, 0, self.pixmap_base)
            if self.pixmap_preview: painter.drawPixmap(0, 0, self.pixmap_preview)

            # 绘制交互元素 (框选时的红框)
            if self.mode == "eraser" and self.is_drawing_rect:
                # 逆变换回 图像坐标系 绘制? 不，直接在屏幕坐标系绘制更方便
                # 但由于 painter 已经 translate/scale 了，我们在这里画图就是画在图像上的
                # 转换屏幕坐标 -> 图像坐标
                x1 = (self.drag_start_pos.x() - self.offset_x) / self.scale
                y1 = (self.drag_start_pos.y() - self.offset_y) / self.scale
                x2 = (self.drag_current_pos.x() - self.offset_x) / self.scale
                y2 = (self.drag_current_pos.y() - self.offset_y) / self.scale

                pen = QPen(QColor(255, 255, 0), 2 / self.scale)  # 黄色框
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRect(QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2))))

            # 绘制画笔光标 (可选)
            if self.mode == "brush":
                # 这里略过光标绘制，直接用鼠标点击效果
                pass

        else:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")

    # ==========================
    # 鼠标交互逻辑
    # ==========================
    def mousePressEvent(self, event):
        if not self.pixmap_image: return

        # 中键平移 (优先级最高)
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # 坐标映射
        widget_pos = event.position()
        img_x = int((widget_pos.x() - self.offset_x) / self.scale)
        img_y = int((widget_pos.y() - self.offset_y) / self.scale)

        # 越界检查
        if not (0 <= img_x < self._image_w and 0 <= img_y < self._image_h):
            return

        # === 模式分发 ===
        if self.mode == "sam":
            is_left = 1 if event.button() == Qt.MouseButton.LeftButton else 0
            self.click_signal.emit(img_x, img_y, is_left)

        elif self.mode == "eraser":
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_drawing_rect = True
                self.drag_start_pos = widget_pos
                self.drag_current_pos = widget_pos

        elif self.mode == "brush":
            self.is_brushing = True
            is_add = 1 if event.button() == Qt.MouseButton.LeftButton else 0
            self.brush_signal.emit(img_x, img_y, is_add)

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.position() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.position()
            self.update()
            return

        widget_pos = event.position()

        if self.mode == "eraser" and self.is_drawing_rect:
            self.drag_current_pos = widget_pos
            self.update()  # 触发重绘以显示框

        elif self.mode == "brush" and self.is_brushing:
            img_x = int((widget_pos.x() - self.offset_x) / self.scale)
            img_y = int((widget_pos.y() - self.offset_y) / self.scale)
            # 只有在移动到新位置且在图内时才发射信号
            if 0 <= img_x < self._image_w and 0 <= img_y < self._image_h:
                is_add = 1 if event.buttons() & Qt.MouseButton.LeftButton else 0
                self.brush_signal.emit(img_x, img_y, is_add)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor if self.mode == "sam" else
                           (Qt.CursorShape.CrossCursor if self.mode == "eraser" else Qt.CursorShape.PointingHandCursor))
            return

        if self.mode == "eraser" and self.is_drawing_rect:
            self.is_drawing_rect = False
            # 计算最终框选区域 (图像坐标)
            p1 = self.drag_start_pos
            p2 = event.position()
            x1 = int((min(p1.x(), p2.x()) - self.offset_x) / self.scale)
            y1 = int((min(p1.y(), p2.y()) - self.offset_y) / self.scale)
            x2 = int((max(p1.x(), p2.x()) - self.offset_x) / self.scale)
            y2 = int((max(p1.y(), p2.y()) - self.offset_y) / self.scale)
            w = x2 - x1
            h = y2 - y1
            if w > 0 and h > 0:
                self.rect_erase_signal.emit(x1, y1, w, h)
            self.update()  # 清除黄框

        elif self.mode == "brush":
            self.is_brushing = False

    def wheelEvent(self, event):
        if not self.pixmap_image: return
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        mouse_pos = event.position()
        old_x = (mouse_pos.x() - self.offset_x) / self.scale
        old_y = (mouse_pos.y() - self.offset_y) / self.scale
        self.scale = max(0.05, min(self.scale, 50.0)) * factor
        self.offset_x = mouse_pos.x() - old_x * self.scale
        self.offset_y = mouse_pos.y() - old_y * self.scale
        self.update()

    def fit_to_window(self):
        if not self.pixmap_image: return
        scale_w = self.width() / self._image_w
        scale_h = self.height() / self._image_h
        self.scale = min(scale_w, scale_h) * 0.9
        self.offset_x = (self.width() - self._image_w * self.scale) / 2
        self.offset_y = (self.height() - self._image_h * self.scale) / 2
        self.update()