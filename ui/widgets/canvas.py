import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QPolygon
from PyQt6.QtCore import pyqtSignal, Qt, QPoint, QRect


class InteractiveCanvas(QWidget):
    # 信号定义
    click_signal = pyqtSignal(int, int, int)  # SAM点击: x, y, is_left
    rect_erase_signal = pyqtSignal(int, int, int, int)  # 框选擦除: x, y, w, h
    brush_signal = pyqtSignal(int, int, int)  # 画笔点涂: x, y, is_add
    # 【新增】多边形/套索填充信号: 发送点的列表 [(x1,y1), (x2,y2), ...]
    polygon_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- 数据层 ---
        self.pixmap_image = None
        self.pixmap_base = None
        self.pixmap_preview = None
        self._image_w = 0
        self._image_h = 0

        # --- 视图变换 ---
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        # --- 交互状态 ---
        self.mode = "sam"  # 模式: "sam", "eraser", "brush", "polygon"(新)
        self.is_panning = False
        self.is_drawing_rect = False
        self.is_brushing = False

        # 【新增】多边形绘制状态
        self.is_drawing_polygon = False
        self.polygon_points = []  # 存储绘制过程中的图像坐标点 (img_x, img_y)

        self.last_mouse_pos = QPoint()
        self.drag_start_pos = QPoint()
        self.drag_current_pos = QPoint()

        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #2b2b2b;")

    # ==========================
    # API
    # ==========================
    def set_mode(self, mode):
        """切换模式"""
        self.mode = mode
        # 重置所有交互状态
        self.is_drawing_rect = False
        self.is_brushing = False
        self.is_drawing_polygon = False
        self.polygon_points = []

        # 设置光标
        if mode == "sam":
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif mode == "eraser":
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif mode in ["brush", "polygon"]:
            # 画笔和套索都用手型光标，或者你可以自定义
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
    # 绘图事件 (Paint Event)
    # ==========================
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.pixmap_image:
            # 应用视图变换 (平移和缩放)
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.scale, self.scale)

            # 1. 画底图层
            painter.drawPixmap(0, 0, self.pixmap_image)
            if self.pixmap_base: painter.drawPixmap(0, 0, self.pixmap_base)
            if self.pixmap_preview: painter.drawPixmap(0, 0, self.pixmap_preview)

            # 2. 画交互元素 (不受缩放影响的线宽需要反向计算)

            # 框选橡皮擦的预览框
            if self.mode == "eraser" and self.is_drawing_rect:
                # 将屏幕坐标转回图像坐标
                x1 = (self.drag_start_pos.x() - self.offset_x) / self.scale
                y1 = (self.drag_start_pos.y() - self.offset_y) / self.scale
                x2 = (self.drag_current_pos.x() - self.offset_x) / self.scale
                y2 = (self.drag_current_pos.y() - self.offset_y) / self.scale

                pen = QPen(QColor(255, 255, 0), 2 / self.scale)  # 黄色虚线，线宽随缩放调整
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                # 在图像坐标系上画矩形
                painter.drawRect(QRect(QPoint(int(x1), int(y1)), QPoint(int(x2), int(y2))))

            # 【新增】多边形/套索的预览轨迹
            if self.mode == "polygon" and self.is_drawing_polygon and len(self.polygon_points) > 1:
                pen = QPen(QColor(0, 255, 255), 2 / self.scale)  # 青色实线
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                # 将记录的图像坐标点转换为 QPolygon 进行绘制
                qpoints = [QPoint(p[0], p[1]) for p in self.polygon_points]
                painter.drawPolyline(QPolygon(qpoints))

        else:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image Loaded")

    # ==========================
    # 鼠标交互逻辑
    # ==========================
    def mousePressEvent(self, event):
        if not self.pixmap_image: return

        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # 计算图像坐标
        widget_pos = event.position()
        img_x = int((widget_pos.x() - self.offset_x) / self.scale)
        img_y = int((widget_pos.y() - self.offset_y) / self.scale)

        is_inside = (0 <= img_x < self._image_w and 0 <= img_y < self._image_h)

        # === 模式分发 ===
        if self.mode == "sam" and is_inside:
            is_left = 1 if event.button() == Qt.MouseButton.LeftButton else 0
            self.click_signal.emit(img_x, img_y, is_left)

        elif self.mode == "eraser" and event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing_rect = True
            self.drag_start_pos = widget_pos
            self.drag_current_pos = widget_pos

        elif self.mode == "brush" and is_inside:
            self.is_brushing = True
            is_add = 1 if event.button() == Qt.MouseButton.LeftButton else 0
            self.brush_signal.emit(img_x, img_y, is_add)

        # 【新增】多边形模式开始绘制
        elif self.mode == "polygon" and event.button() == Qt.MouseButton.LeftButton and is_inside:
            self.is_drawing_polygon = True
            self.polygon_points = [(img_x, img_y)]  # 记录起点

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.position() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.position()
            self.update()
            return

        widget_pos = event.position()
        img_x = int((widget_pos.x() - self.offset_x) / self.scale)
        img_y = int((widget_pos.y() - self.offset_y) / self.scale)
        is_inside = (0 <= img_x < self._image_w and 0 <= img_y < self._image_h)

        if self.mode == "eraser" and self.is_drawing_rect:
            self.drag_current_pos = widget_pos
            self.update()  # 重绘以更新预览框

        elif self.mode == "brush" and self.is_brushing and is_inside:
            is_add = 1 if event.buttons() & Qt.MouseButton.LeftButton else 0
            self.brush_signal.emit(img_x, img_y, is_add)

        # 【新增】多边形模式：记录轨迹点
        elif self.mode == "polygon" and self.is_drawing_polygon and is_inside:
            # 避免记录过于密集的点，可以加一个简单的距离判断，这里暂且直接添加
            self.polygon_points.append((img_x, img_y))
            self.update()  # 重绘以更新预览线

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            # 恢复对应模式的光标
            cursor_map = {"sam": Qt.CursorShape.ArrowCursor, "eraser": Qt.CursorShape.CrossCursor}
            self.setCursor(cursor_map.get(self.mode, Qt.CursorShape.PointingHandCursor))
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
            self.update()

        elif self.mode == "brush":
            self.is_brushing = False

        # 【新增】多边形模式结束绘制
        elif self.mode == "polygon" and self.is_drawing_polygon:
            self.is_drawing_polygon = False
            # 只有点数足够构成多边形才发送信号
            if len(self.polygon_points) > 2:
                self.polygon_signal.emit(self.polygon_points)
            # 清空轨迹
            self.polygon_points = []
            self.update()

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