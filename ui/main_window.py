import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QListWidget, QPushButton, QTextEdit,
                             QLabel, QSplitter, QMessageBox, QFrame)
from PyQt6.QtCore import pyqtSlot, Qt

# 导入拆分好的模块
from ui.widgets.canvas import InteractiveCanvas
from core.sam_engine import SAMEngine
from core.data_manager import DataManager


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LISA Annotator (SAM)")
        self.resize(1400, 900)

        # 1. 初始化后端逻辑模块
        self.data_manager = DataManager()

        # 注意：实际使用时建议将模型加载放入独立线程，避免启动卡顿
        # 这里为了代码清晰，保持在主线程
        self.sam_engine = SAMEngine(checkpoint_path="checkpoints/sam_vit_b_01ec64.pth")

        # --- 交互状态缓存 (State) ---
        self.current_image = None  # 当前 OpenCV 图片 (BGR)
        self.input_points = []  # SAM 输入点集 [[x,y], [x,y]]
        self.input_labels = []  # SAM 输入标签集 [1, 0]
        self.current_mask = None  # 当前生成的 Mask

        # 2. 初始化 UI
        self.init_ui()

        # 3. 信号连接 (Wiring)
        self.canvas.click_signal.connect(self.handle_canvas_click)
        self.file_list_widget.currentRowChanged.connect(self.on_file_selected)

    def init_ui(self):
        """初始化界面布局"""
        # 主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 使用 QSplitter 让三栏宽度可调
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- 1. 左侧面板：文件导航 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.btn_load_dir = QPushButton("📂 Load Folder")
        self.btn_load_dir.clicked.connect(self.load_folder_action)
        self.btn_load_dir.setStyleSheet("height: 40px; font-weight: bold;")

        self.file_list_widget = QListWidget()

        left_layout.addWidget(self.btn_load_dir)
        left_layout.addWidget(self.file_list_widget)

        # --- 2. 中间面板：画布 ---
        # 实例化我们在上一步写的 Canvas
        self.canvas = InteractiveCanvas()

        # --- 3. 右侧面板：控制与文本 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 3.1 顶部说明与 SAM 控制
        info_label = QLabel(
            "<b>Instructions:</b><br>Left Click: Add Point<br>Right Click: Remove Area<br>Middle Drag: Pan Image")
        info_label.setTextFormat(Qt.TextFormat.RichText)

        self.btn_reset_mask = QPushButton("↺ Reset Mask")
        self.btn_reset_mask.clicked.connect(self.reset_sam_state)

        # 3.2 文本输入区域
        lbl_text = QLabel("Reasoning / Conversation:")
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("Enter reasoning text...")

        # === 布局添加顺序 (从上到下) ===
        right_layout.addWidget(info_label)
        right_layout.addWidget(self.btn_reset_mask)
        right_layout.addWidget(lbl_text)
        right_layout.addWidget(self.text_editor)

        # 3.3 弹簧 (把下面的按钮顶到底部)
        right_layout.addStretch()

        # === 3.4 导航按钮组 (放在 Trash 上面) ===
        nav_layout = QHBoxLayout()

        # 定义按钮样式：高度40像素，字体加粗，字号12pt
        nav_btn_style = """
                    QPushButton {
                        height: 40px; 
                        font-size: 14px; 
                        font-weight: bold;
                    }
                """

        self.btn_prev = QPushButton("<< Previous")
        self.btn_prev.setStyleSheet(nav_btn_style)  # 应用样式
        self.btn_prev.clicked.connect(self.navigate_prev)

        self.btn_next = QPushButton("Next >>")
        self.btn_next.setStyleSheet(nav_btn_style)  # 应用样式
        self.btn_next.clicked.connect(self.navigate_next)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)

        right_layout.addLayout(nav_layout)

        # 3.5 底部操作按钮
        self.btn_delete = QPushButton("🗑 Trash (Low Quality)")
        self.btn_delete.setStyleSheet("background-color: #d9534f; color: white;")
        self.btn_delete.clicked.connect(self.delete_current_image)

        self.btn_save = QPushButton("💾 Save & Next")
        self.btn_save.setStyleSheet("background-color: #5cb85c; color: white; height: 40px; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_and_next)

        right_layout.addWidget(self.btn_delete)
        right_layout.addWidget(self.btn_save)

        # 将三个面板加入 Splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)

        # 设置默认宽度比例 (1 : 4 : 1.5)
        splitter.setSizes([200, 800, 300])

        main_layout.addWidget(splitter)

    # ==========================
    # 逻辑处理槽函数 (Slots)
    # ==========================

    def load_folder_action(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Dataset Directory")
        if folder:
            # 调用 DataManager 获取文件列表
            files = self.data_manager.load_directory(folder)
            self.file_list_widget.clear()
            self.file_list_widget.addItems(files)

            if files:
                self.file_list_widget.setCurrentRow(0)  # 自动选中第一个

    def on_file_selected(self, index):
        """当用户在列表中点击某一行时触发"""
        if index < 0: return

        # 1. 获取图片路径
        self.data_manager.current_index = index
        img_path, json_path = self.data_manager.get_current_data()

        if not img_path: return

        # 2. 读取图片 (BGR)
        # 即使 canvas 会转 RGB，这里我们保留 BGR 给 OpenCV 处理保存
        img = cv2.imread(img_path)
        if img is None:
            QMessageBox.warning(self, "Error", f"Could not read image: {img_path}")
            return

        self.current_image = img

        # 3. 显示到画布
        self.canvas.set_image(img)

        # 4. 初始化 SAM 的 Image Embedding (这步比较耗时，约 0.5s - 1s)
        # 实际项目中建议加个 Loading 动画
        self.sam_engine.set_image(img)

        # 5. 重置交互状态
        self.reset_sam_state()

        # 6. 如果有已存在的 JSON 文本，加载它
        if json_path and os.path.exists(json_path):
            # 这里简单读取，具体看你的 JSON 结构
            # self.text_editor.setText(...)
            self.text_editor.clear()
        else:
            self.text_editor.clear()

    @pyqtSlot(int, int, int)
    def handle_canvas_click(self, x, y, is_left):
        """
        响应画布点击：
        UI (Canvas) -> Controller (Here) -> Model (SAM) -> Controller -> UI
        """
        if self.current_image is None: return

        # 1. 更新 Prompt 点集
        self.input_points.append([x, y])
        self.input_labels.append(is_left)  # 1: 前景, 0: 背景

        print(f"SAM Predicting... Points: {len(self.input_points)}")

        # 2. 调用 SAM 进行推理
        # 注意：SAM 支持传入所有历史点，这样效果最好
        mask = self.sam_engine.predict_mask(self.input_points, self.input_labels)

        if mask is not None:
            self.current_mask = mask
            # 3. 将结果显示回 Canvas
            self.canvas.set_mask(mask)

    def reset_sam_state(self):
        """清空当前的 Mask 和点击历史"""
        self.input_points = []
        self.input_labels = []
        self.current_mask = None
        self.canvas.set_mask(None)

    def save_and_next(self):
        """保存当前结果并自动跳转下一张"""
        if self.current_image is None: return

        # 1. 获取文本
        text_content = self.text_editor.toPlainText()

        # 2. 调用 DataManager 保存
        # 注意：需要把 current_mask 传进去，如果没有 mask 可能是 None
        if self.current_mask is not None:
            self.data_manager.save_annotation(self.current_mask, text_content)
            print("Saved.")
        else:
            print("No mask to save, skipping mask file.")

        # 3. 跳转下一张
        next_row = self.file_list_widget.currentRow() + 1
        if next_row < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(next_row)
        else:
            QMessageBox.information(self, "Finished", "All images in this folder processed!")

    def delete_current_image(self):
        """将当前低质量图片移入回收站"""
        reply = QMessageBox.question(self, 'Confirm Delete',
                                     "Are you sure you want to move this image to trash?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.data_manager.delete_current_file()

            # 刷新列表并跳到下一张
            # 简单做法：重新加载列表（或者在 ListWidget 中移除该行）
            current_row = self.file_list_widget.currentRow()
            self.file_list_widget.takeItem(current_row)

            # 尝试选中原来的行号（现在是下一张了）
            if current_row < self.file_list_widget.count():
                self.file_list_widget.setCurrentRow(current_row)

    def navigate_prev(self):
        """跳转到上一张"""
        current_row = self.file_list_widget.currentRow()
        if current_row > 0:
            self.file_list_widget.setCurrentRow(current_row - 1)
        else:
            # 可选：如果已经是第一张，提示一下
            # QMessageBox.information(self, "Info", "This is the first image.")
            pass

    def navigate_next(self):
        """跳转到下一张"""
        current_row = self.file_list_widget.currentRow()
        count = self.file_list_widget.count()
        if current_row < count - 1:
            self.file_list_widget.setCurrentRow(current_row + 1)
        else:
            # 可选：已经是最后一张
            # QMessageBox.information(self, "Info", "This is the last image.")
            pass