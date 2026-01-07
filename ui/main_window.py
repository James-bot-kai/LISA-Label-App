import os
import cv2
import json
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QListWidget, QPushButton, QTextEdit,
                             QLabel, QSplitter, QMessageBox, QFrame, QGroupBox,
                             QRadioButton, QButtonGroup, QSlider, QSpinBox,
                             QGridLayout)  # <--- 新增 QGridLayout
from PyQt6.QtCore import pyqtSlot, Qt
from pathlib import Path

# 确保引入的是修改过支持 set_preview_mask 的 Canvas
from ui.widgets.canvas import InteractiveCanvas
from core.sam_engine import SAMEngine
from core.data_manager import DataManager
# from utils.translate import BaiduTranslator # 根据实际情况取消注释
from utils.aiTranslate import BaiduTranslator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LISA Annotator (SAM)")
        self.resize(1400, 900)

        # 1. 初始化后端逻辑模块
        self.data_manager = DataManager()
        # 请确保路径正确，且文件已下载
        self.sam_engine = SAMEngine(checkpoint_path="checkpoints/sam_vit_b_01ec64.pth")

        # --- 交互状态缓存 (State) ---
        self.current_image = None
        self.base_mask = None  # 永久层：从文件加载或已合并的 Mask (显示为红色)
        self.sam_mask = None  # 临时层：SAM 当前预测的 Mask (显示为绿色)
        self.input_points = []
        self.input_labels = []
        self.current_mask = None

        # --- 画笔设置 ---
        self.brush_radius = 10  # 默认画笔半径

        # --- JSON 数据模式状态 ---
        self.json_data = []
        self.json_path = None
        self.json_current_index = -1
        self.current_mode = "folder"  # "folder" 或 "json"

        # 翻译器初始化
        self.translator = BaiduTranslator(
            appid='20260105002533609',
            api_key='8qBw_d5do3deol13gd3crgg7g'
        )

        # 2. 初始化 UI
        self.init_ui()

        # 3. 信号连接
        self.canvas.click_signal.connect(self.handle_canvas_click)
        self.file_list_widget.currentRowChanged.connect(self.on_file_selected)
        self.canvas.rect_erase_signal.connect(self.handle_rect_erase)
        self.canvas.brush_signal.connect(self.handle_brush_paint)
        self.canvas.polygon_signal.connect(self.handle_polygon_fill)

    def init_ui(self):
        """初始化界面布局"""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # === 左侧面板：文件导航 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 模式切换按钮组
        mode_group = QGroupBox("数据源")
        mode_layout = QHBoxLayout(mode_group)
        self.radio_folder = QRadioButton("📂 文件夹")
        self.radio_json = QRadioButton("📄 JSON")
        self.radio_folder.setChecked(True)
        self.radio_folder.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.radio_folder)
        mode_layout.addWidget(self.radio_json)
        left_layout.addWidget(mode_group)

        # 加载按钮
        self.btn_load_dir = QPushButton("📂 加载文件夹")
        self.btn_load_dir.clicked.connect(self.load_folder_action)
        self.btn_load_dir.setStyleSheet("height: 40px; font-weight: bold;")
        left_layout.addWidget(self.btn_load_dir)

        self.btn_load_json = QPushButton("📄 加载 JSON")
        self.btn_load_json.clicked.connect(self.load_json_action)
        self.btn_load_json.setStyleSheet("height: 40px; font-weight: bold;")
        self.btn_load_json.setVisible(False)
        left_layout.addWidget(self.btn_load_json)

        # 统计标签
        self.stats_label = QLabel("共 0 条数据")
        left_layout.addWidget(self.stats_label)

        # 文件/数据列表
        self.file_list_widget = QListWidget()
        left_layout.addWidget(self.file_list_widget)

        # === 中间面板：画布 ===
        self.canvas = InteractiveCanvas()

        # === 右侧面板：控制与信息 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 元信息显示
        meta_group = QGroupBox("元信息")
        meta_layout = QVBoxLayout(meta_group)
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setMaximumHeight(120)
        meta_layout.addWidget(self.meta_text)
        right_layout.addWidget(meta_group)

        # 操作说明
        info_label = QLabel(
            "<b>操作说明:</b><br>"
            "左键: 添加前景点 (预测)<br>"
            "右键: 添加背景点 (修正)<br>"
            "空格: 确认添加 (变红)<br>"
            "Del : 确认移除 (擦除)<br>"
            "Esc : 取消当前预览"
        )
        info_label.setTextFormat(Qt.TextFormat.RichText)
        right_layout.addWidget(info_label)

        # === 【修改】工具模式切换按钮组 ===
        tools_group = QGroupBox("工具箱")
        tools_layout = QVBoxLayout(tools_group)

        # 1. 按钮区域 (使用 Grid 布局，2行2列，节省宽度)
        btns_grid = QGridLayout()

        # 定义按钮
        self.btn_tool_sam = QPushButton("🎯 SAM (点选)")
        self.btn_tool_sam.setCheckable(True)
        self.btn_tool_sam.setChecked(True)
        self.btn_tool_sam.clicked.connect(lambda: self.switch_tool("sam"))

        self.btn_tool_erase = QPushButton("🔲 框选擦除")
        self.btn_tool_erase.setCheckable(True)
        self.btn_tool_erase.clicked.connect(lambda: self.switch_tool("eraser"))

        self.btn_tool_brush = QPushButton("🖌️ 画笔微调")
        self.btn_tool_brush.setCheckable(True)
        self.btn_tool_brush.clicked.connect(lambda: self.switch_tool("brush"))

        self.btn_tool_polygon = QPushButton("➰ 套索填充")
        self.btn_tool_polygon.setCheckable(True)
        self.btn_tool_polygon.clicked.connect(lambda: self.switch_tool("polygon"))

        # 互斥组
        self.tool_btn_group = QButtonGroup()
        self.tool_btn_group.addButton(self.btn_tool_sam)
        self.tool_btn_group.addButton(self.btn_tool_erase)
        self.tool_btn_group.addButton(self.btn_tool_brush)
        self.tool_btn_group.addButton(self.btn_tool_polygon)
        self.tool_btn_group.setExclusive(True)

        # 添加到网格 (行, 列)
        btns_grid.addWidget(self.btn_tool_sam, 0, 0)
        btns_grid.addWidget(self.btn_tool_erase, 0, 1)
        btns_grid.addWidget(self.btn_tool_brush, 1, 0)
        btns_grid.addWidget(self.btn_tool_polygon, 1, 1)

        tools_layout.addLayout(btns_grid)

        # 2. 画笔大小控制区 (默认隐藏容器)
        self.brush_control_widget = QWidget()
        size_layout = QHBoxLayout(self.brush_control_widget)
        size_layout.setContentsMargins(0, 5, 0, 0)

        size_layout.addWidget(QLabel("🖌️ 大小:"))
        self.slider_brush = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush.setRange(1, 100)
        self.slider_brush.setValue(self.brush_radius)

        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 100)
        self.spin_brush.setValue(self.brush_radius)

        self.slider_brush.valueChanged.connect(self.spin_brush.setValue)
        self.slider_brush.valueChanged.connect(self.set_brush_radius)
        self.spin_brush.valueChanged.connect(self.slider_brush.setValue)

        size_layout.addWidget(self.slider_brush)
        size_layout.addWidget(self.spin_brush)

        # 将大小控制容器加入布局，并默认隐藏
        tools_layout.addWidget(self.brush_control_widget)
        self.brush_control_widget.setVisible(False)

        right_layout.addWidget(tools_group)

        # SAM 重置按钮
        self.btn_reset_mask = QPushButton("↺ 取消当前 SAM 预览")
        self.btn_reset_mask.clicked.connect(self.reset_sam_interaction)
        right_layout.addWidget(self.btn_reset_mask)

        # === 增删改操作按钮组 ===
        action_layout = QHBoxLayout()
        self.btn_add_mask = QPushButton("➕ 确认 (Space)")
        self.btn_add_mask.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold;")
        self.btn_add_mask.clicked.connect(self.apply_sam_merge)

        self.btn_sub_mask = QPushButton("➖ 移除 (Del)")
        self.btn_sub_mask.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_sub_mask.clicked.connect(self.apply_sam_subtract)

        action_layout.addWidget(self.btn_add_mask)
        action_layout.addWidget(self.btn_sub_mask)
        right_layout.addLayout(action_layout)

        # 文本输入区域
        lbl_text = QLabel("对话/推理文本:")
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("输入推理文本...")
        right_layout.addWidget(lbl_text)
        right_layout.addWidget(self.text_editor)

        # 翻译按钮
        self.btn_translate = QPushButton("🌐 翻译为中文")
        self.btn_translate.setStyleSheet("height: 35px; font-weight: bold;")
        self.btn_translate.clicked.connect(self.translate_text)
        self.btn_translate.setVisible(False)
        right_layout.addWidget(self.btn_translate)

        # 翻译结果区域
        lbl_translated = QLabel("翻译结果:")
        self.translated_text = QTextEdit()
        self.translated_text.setReadOnly(True)
        self.translated_text.setPlaceholderText("翻译结果将显示在这里...")
        self.translated_text.setStyleSheet(
            "QTextEdit { background-color: #fffde7; color: #333333; font-size: 13px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }")
        self.translated_text.setMinimumHeight(120)
        right_layout.addWidget(lbl_translated)
        right_layout.addWidget(self.translated_text)

        right_layout.addStretch()

        # 导航按钮
        nav_layout = QHBoxLayout()
        nav_btn_style = "QPushButton { height: 40px; font-size: 14px; font-weight: bold; }"
        self.btn_prev = QPushButton("<< 上一条")
        self.btn_prev.setStyleSheet(nav_btn_style)
        self.btn_prev.clicked.connect(self.navigate_prev)
        self.btn_next = QPushButton("下一条 >>")
        self.btn_next.setStyleSheet(nav_btn_style)
        self.btn_next.clicked.connect(self.navigate_next)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        right_layout.addLayout(nav_layout)

        # 删除按钮
        self.btn_delete = QPushButton("🗑 删除当前条目")
        self.btn_delete.setStyleSheet("background-color: #d9534f; color: white; font-weight: bold;")
        self.btn_delete.clicked.connect(self.delete_current_item)
        right_layout.addWidget(self.btn_delete)

        # 保存按钮
        self.btn_save = QPushButton("💾 保存修改")
        self.btn_save.setStyleSheet("background-color: #5cb85c; color: white; height: 40px; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_current)
        right_layout.addWidget(self.btn_save)

        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 800, 350])
        main_layout.addWidget(splitter)

    # ==========================
    # 模式切换
    # ==========================

    def on_mode_changed(self):
        """切换文件夹/JSON模式"""
        if self.radio_folder.isChecked():
            self.current_mode = "folder"
            self.btn_load_dir.setVisible(True)
            self.btn_load_json.setVisible(False)
        else:
            self.current_mode = "json"
            self.btn_load_dir.setVisible(False)
            self.btn_load_json.setVisible(True)

        self.file_list_widget.clear()
        self.stats_label.setText("共 0 条数据")
        self.canvas.set_image(None)
        self.canvas.set_mask(None)
        self.canvas.set_preview_mask(None)
        self.meta_text.clear()
        self.text_editor.clear()
        self.translated_text.clear()
        self.base_mask = None
        self.sam_mask = None

    # ==========================
    # 文件夹模式
    # ==========================

    def load_folder_action(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if folder:
            files = self.data_manager.load_directory(folder)
            self.file_list_widget.clear()
            self.file_list_widget.addItems(files)
            self.stats_label.setText(f"共 {len(files)} 条数据")
            if files: self.file_list_widget.setCurrentRow(0)

    def on_file_selected(self, index):
        if index < 0: return
        if self.current_mode == "folder":
            self._load_folder_item(index)
        else:
            self._load_json_item(index)

    def _load_folder_item(self, index):
        self.data_manager.current_index = index
        img_path, json_path = self.data_manager.get_current_data()
        if not img_path: return
        img = cv2.imread(img_path)
        if img is None: return
        self.current_image = img
        self.canvas.set_image(img)
        self.sam_engine.set_image(img)
        h, w = img.shape[:2]
        self.base_mask = np.zeros((h, w), dtype=np.uint8)
        self.sam_mask = None
        self.input_points = []
        self.input_labels = []
        self.update_canvas_display()
        self.meta_text.setPlainText(f"文件: {img_path}")
        self.text_editor.clear()

    def load_json_action(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 JSON 文件", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.json_data = json.load(f)
                self.json_path = file_path
                self.file_list_widget.clear()
                for item in self.json_data:
                    item_id = item.get('id', 'Unknown')
                    category = item.get('category', '')
                    display = f"[{category}] {item_id}" if category else item_id
                    self.file_list_widget.addItem(display)
                self.stats_label.setText(f"共 {len(self.json_data)} 条数据")
                if self.json_data: self.file_list_widget.setCurrentRow(0)
                QMessageBox.information(self, "成功", f"已加载 {len(self.json_data)} 条数据")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {e}")

    def _load_json_item(self, index):
        if index < 0 or index >= len(self.json_data): return
        self.json_current_index = index
        item = self.json_data[index]
        rgb_path = item.get('image_path_rgb', '')
        img = None
        if rgb_path and Path(rgb_path).exists(): img = cv2.imread(rgb_path)
        if img is not None:
            self.current_image = img
            self.canvas.set_image(img)
            self.sam_engine.set_image(img)
        else:
            self.current_image = None
            self.canvas.set_image(None)
            print(f"图像不存在: {rgb_path}")
            return
        mask_path = item.get('mask_path', '') or item.get('training_mask_path', '')
        h, w = self.current_image.shape[:2]
        self.base_mask = np.zeros((h, w), dtype=np.uint8)
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, mask_binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                if mask_binary.shape == (h, w): self.base_mask = mask_binary
        self.sam_mask = None
        self.input_points = []
        self.input_labels = []
        self.update_canvas_display()
        self.meta_text.setPlainText(f"ID: {item.get('id', '')}\nImage: {rgb_path}\nMask: {mask_path}")
        conversations = item.get('conversations', [])
        if conversations:
            conv_text = ""
            for conv in conversations:
                role = conv.get('from', '')
                value = conv.get('value', '').replace('<image>\n', '')
                conv_text += f"{'👤 Human' if role == 'human' else '🤖 GPT'}:\n{value}\n\n"
            self.text_editor.setPlainText(conv_text)
            self._auto_translate(conv_text)
        else:
            self.text_editor.setPlainText("（无对话数据）")
            self.translated_text.clear()

    # ==========================
    # 核心：显示与合并逻辑
    # ==========================
    def update_canvas_display(self):
        if self.base_mask is not None:
            self.canvas.set_mask(self.base_mask)
        else:
            self.canvas.set_mask(None)

        if self.sam_mask is not None:
            self.canvas.set_preview_mask(self.sam_mask)
        else:
            self.canvas.set_preview_mask(None)

        if self.base_mask is not None:
            self.current_mask = self.base_mask.copy()
            if self.sam_mask is not None:
                try:
                    if self.base_mask.shape != self.sam_mask.shape:
                        h, w = self.base_mask.shape[:2]
                        self.sam_mask = cv2.resize(self.sam_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    self.current_mask = cv2.bitwise_or(self.current_mask, self.sam_mask)
                except:
                    pass
        else:
            self.current_mask = None

    def apply_sam_merge(self):
        if self.base_mask is None or self.sam_mask is None: return
        self.base_mask = np.bitwise_or(self.base_mask, self.sam_mask)
        print("操作：区域已添加")
        self.reset_sam_interaction()

    def apply_sam_subtract(self):
        if self.base_mask is None or self.sam_mask is None: return
        sam_inverted = 1 - self.sam_mask
        self.base_mask = np.bitwise_and(self.base_mask, sam_inverted)
        print("操作：区域已移除")
        self.reset_sam_interaction()

    def reset_sam_interaction(self):
        self.input_points = []
        self.input_labels = []
        self.sam_mask = None
        self.update_canvas_display()

    @pyqtSlot(int, int, int)
    def handle_canvas_click(self, x, y, is_left):
        if self.current_image is None: return
        self.input_points.append([x, y])
        self.input_labels.append(is_left)
        print(f"🖱️ 点击: ({x}, {y})")
        mask = self.sam_engine.predict_mask(self.input_points, self.input_labels)
        if mask is not None:
            self.sam_mask = mask
            self.update_canvas_display()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Enter):
            self.apply_sam_merge()
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.apply_sam_subtract()
        elif event.key() == Qt.Key.Key_Escape:
            self.reset_sam_interaction()
        elif event.key() == Qt.Key.Key_Left:
            self.navigate_prev()
        elif event.key() == Qt.Key.Key_Right:
            self.navigate_next()
        else:
            super().keyPressEvent(event)

    # ==========================
    # 保存与删除 (保持不变)
    # ==========================
    def save_current(self):
        if self.current_mode == "folder":
            self._save_folder_item()
        else:
            self._save_json_item()

    def _save_folder_item(self):
        if self.current_image is None: return
        if self.current_mask is not None:
            self.data_manager.save_annotation(self.current_mask, self.text_editor.toPlainText())
            print("已保存")
        self.navigate_next()

    def _save_json_item(self):
        if not self.json_path or self.json_current_index < 0: return
        item = self.json_data[self.json_current_index]
        convs = self._parse_conversations(self.text_editor.toPlainText())
        if convs: item['conversations'] = convs
        if self.current_mask is not None:
            mask_path = item.get('mask_path') or item.get('training_mask_path')
            if mask_path: cv2.imwrite(mask_path, (self.current_mask * 255).astype(np.uint8))
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "成功", f"已保存: {self.json_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def _parse_conversations(self, text: str) -> list:
        if not text.strip(): return []
        conversations = []
        parts = text.split('👤 Human:')
        for part in parts:
            if not part.strip(): continue
            if '🤖 GPT:' in part:
                human_gpt = part.split('🤖 GPT:')
                h_txt, g_txt = human_gpt[0].strip(), human_gpt[1].strip() if len(human_gpt) > 1 else ''
                if h_txt: conversations.append({'from': 'human', 'value': h_txt})
                if g_txt: conversations.append({'from': 'gpt', 'value': g_txt})
            else:
                h_txt = part.strip()
                if h_txt: conversations.append({'from': 'human', 'value': h_txt})
        return conversations

    def delete_current_item(self):
        if self.current_mode == "folder":
            self._delete_folder_item()
        else:
            self._delete_json_item()

    def _delete_folder_item(self):
        if QMessageBox.question(self, '确认删除', "确定要移入回收站吗？") == QMessageBox.StandardButton.Yes:
            self.data_manager.delete_current_file()
            row = self.file_list_widget.currentRow()
            self.file_list_widget.takeItem(row)
            self.stats_label.setText(f"共 {self.file_list_widget.count()} 条数据")
            if row < self.file_list_widget.count(): self.file_list_widget.setCurrentRow(row)

    def _delete_json_item(self):
        if self.json_current_index < 0: return
        if QMessageBox.question(self, '确认删除',
                                f"确定删除 ID: {self.json_data[self.json_current_index].get('id')}？") == QMessageBox.StandardButton.Yes:
            item = self.json_data[self.json_current_index]
            for k in ['visual_prompt_path', 'training_mask_path', 'mask_path']:
                if p := item.get(k):
                    try:
                        os.remove(p);
                        print(f"已删除: {p}")
                    except:
                        pass
            self.json_data.pop(self.json_current_index)
            self.file_list_widget.takeItem(self.json_current_index)
            self.stats_label.setText(f"共 {len(self.json_data)} 条数据")
            if self.json_data:
                self.file_list_widget.setCurrentRow(min(self.json_current_index, len(self.json_data) - 1))
            else:
                self.on_mode_changed()

    # ==========================
    # 辅助功能与导航
    # ==========================
    def navigate_prev(self):
        row = self.file_list_widget.currentRow()
        if row > 0:
            self._auto_save_current()
            self.file_list_widget.setCurrentRow(row - 1)

    def navigate_next(self):
        row = self.file_list_widget.currentRow()
        if row < self.file_list_widget.count() - 1:
            self._auto_save_current()
            self.file_list_widget.setCurrentRow(row + 1)

    def _auto_save_current(self):
        if self.current_mode == "folder":
            self._save_folder_item()
        else:
            if not self.json_path or self.json_current_index < 0: return
            item = self.json_data[self.json_current_index]
            convs = self._parse_conversations(self.text_editor.toPlainText())
            if convs: item['conversations'] = convs
            try:
                with open(self.json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.json_data, f, ensure_ascii=False, indent=4)
                print("文本已自动保存")
            except:
                pass

    def translate_text(self):
        text = self.text_editor.toPlainText().strip()
        if not text: return
        self.btn_translate.setEnabled(False);
        self.btn_translate.setText("翻译中...")
        try:
            self._auto_translate(text)
        finally:
            self.btn_translate.setEnabled(True);
            self.btn_translate.setText("🌐 翻译为中文")

    def _auto_translate(self, text):
        if not text.strip(): self.translated_text.clear(); return
        try:
            clean = text.replace('<image>\n', '').replace('[SEG]', '[分割]')
            self.translated_text.setPlainText(self.translator.translate(clean, from_lang='en', to_lang='zh'))
        except Exception as e:
            self.translated_text.setPlainText(f"翻译失败: {e}")

    # ==========================
    # 工具切换逻辑
    # ==========================
    def switch_tool(self, mode):
        """切换画布模式"""
        self.canvas.set_mode(mode)

        # 核心逻辑：只有在 brush 模式下，才显示大小控制容器
        if mode == "brush":
            self.brush_control_widget.setVisible(True)
        else:
            self.brush_control_widget.setVisible(False)

        tips = {
            "sam": "SAM模式：左键=前景点，右键=背景点。",
            "eraser": "擦除模式：拉框选中区域，该区域Mask将被清除。",
            "brush": "画笔模式：左键涂抹=添加，右键涂抹=擦除。",
            "polygon": "套索模式：按住左键拖拽画圈，松开后圈内区域将被填充为Mask。"
        }
        self.text_editor.setPlaceholderText(tips.get(mode, ""))

    def set_brush_radius(self, radius):
        self.brush_radius = radius

    # ==========================
    # 画笔与新工具功能实现
    # ==========================

    @pyqtSlot(int, int, int, int)
    def handle_rect_erase(self, x, y, w, h):
        if self.base_mask is None: return
        h_img, w_img = self.base_mask.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        if x2 > x1 and y2 > y1:
            self.base_mask[y1:y2, x1:x2] = 0
            if self.sam_mask is not None and self.sam_mask.shape == self.base_mask.shape:
                self.sam_mask[y1:y2, x1:x2] = 0
            self.update_canvas_display()

    @pyqtSlot(int, int, int)
    def handle_brush_paint(self, x, y, is_add):
        if self.base_mask is None:
            if self.current_image is None: return
            self.base_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)

        color = 1 if is_add else 0
        cv2.circle(self.base_mask, (x, y), self.brush_radius, color, -1)
        if not is_add and self.sam_mask is not None and self.sam_mask.shape == self.base_mask.shape:
            cv2.circle(self.sam_mask, (x, y), self.brush_radius, 0, -1)
        self.update_canvas_display()

    # 【新增】处理多边形填充信号
    @pyqtSlot(list)
    def handle_polygon_fill(self, points):
        """接收套索轨迹点，填充区域到 Base Mask"""
        if self.base_mask is None:
            if self.current_image is None: return
            # 如果没有 Mask，创建一个新的
            self.base_mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)

        # 将点列表转换为 OpenCV 需要的 NumPy 数组格式 (int32)
        # points 结构是 [(x1,y1), (x2,y2), ...]
        pts_np = np.array([points], dtype=np.int32)

        # 在 base_mask 上填充多边形，颜色为 1 (白色/前景)
        cv2.fillPoly(self.base_mask, pts_np, color=1)

        print(f"操作：已填充多边形区域，包含 {len(points)} 个点")
        # 刷新显示
        self.update_canvas_display()