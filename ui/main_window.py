import os
import cv2
import json
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QFileDialog, QListWidget, QPushButton, QTextEdit,
                             QLabel, QSplitter, QMessageBox, QFrame, QGroupBox,
                             QStackedWidget, QButtonGroup, QRadioButton)
from PyQt6.QtCore import pyqtSlot, Qt
from pathlib import Path

from ui.widgets.canvas import InteractiveCanvas
from core.sam_engine import SAMEngine
from core.data_manager import DataManager
#from utils.translate import BaiduTranslator
from utils.aiTranslate import BaiduTranslator


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LISA Annotator (SAM)")
        self.resize(1400, 900)

        # 1. 初始化后端逻辑模块
        self.data_manager = DataManager()
        self.sam_engine = SAMEngine(checkpoint_path="checkpoints/sam_vit_b_01ec64.pth")

        # --- 交互状态缓存 (State) ---
        self.current_image = None
        self.input_points = []
        self.input_labels = []
        self.current_mask = None

        # --- JSON 数据模式状态 ---
        self.json_data = []
        self.json_path = None
        self.json_current_index = -1
        self.current_mode = "folder"  # "folder" 或 "json"

        self.translator = BaiduTranslator(
            appid='20260105002533609',
            #appkey='fIFodJNEMlRAetRHM8Ec',
            api_key = '8qBw_d5do3deol13gd3crgg7g'
        )

        # 2. 初始化 UI
        self.init_ui()

        # 3. 信号连接
        self.canvas.click_signal.connect(self.handle_canvas_click)
        self.file_list_widget.currentRowChanged.connect(self.on_file_selected)

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

        # 加载按钮（文件夹模式）
        self.btn_load_dir = QPushButton("📂 加载文件夹")
        self.btn_load_dir.clicked.connect(self.load_folder_action)
        self.btn_load_dir.setStyleSheet("height: 40px; font-weight: bold;")
        left_layout.addWidget(self.btn_load_dir)

        # 加载按钮（JSON模式）
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
            "左键: 添加前景点<br>"
            "右键: 添加背景点<br>"
            "中键拖拽: 平移图像"
        )
        info_label.setTextFormat(Qt.TextFormat.RichText)
        right_layout.addWidget(info_label)

        # SAM 控制按钮
        self.btn_reset_mask = QPushButton("↺ 重置 Mask")
        self.btn_reset_mask.clicked.connect(self.reset_sam_state)
        right_layout.addWidget(self.btn_reset_mask)

        # 文本输入区域
        # 文本输入区域（替换原有的文本编辑器部分）
        lbl_text = QLabel("对话/推理文本:")
        self.text_editor = QTextEdit()
        self.text_editor.setPlaceholderText("输入推理文本...")
        right_layout.addWidget(lbl_text)
        right_layout.addWidget(self.text_editor)

        # 翻译按钮（现在隐藏了）
        self.btn_translate = QPushButton("🌐 翻译为中文")
        self.btn_translate.setStyleSheet("height: 35px; font-weight: bold;")
        self.btn_translate.clicked.connect(self.translate_text)
        self.btn_translate.setVisible(False)  # 隐藏手动翻译按钮
        right_layout.addWidget(self.btn_translate)

        # 翻译结果区域
        lbl_translated = QLabel("翻译结果:")
        self.translated_text = QTextEdit()
        self.translated_text.setReadOnly(True)
        self.translated_text.setPlaceholderText("翻译结果将显示在这里...")
        self.translated_text.setStyleSheet("""
            QTextEdit {
                background-color: #fffde7;
                color: #333333;
                font-size: 13px;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
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
        self.btn_delete.setStyleSheet(
            "background-color: #d9534f; color: white; height: 40px; font-weight: bold;"
        )
        self.btn_delete.clicked.connect(self.delete_current_item)
        right_layout.addWidget(self.btn_delete)

        # 保存按钮
        self.btn_save = QPushButton("💾 保存修改")
        self.btn_save.setStyleSheet(
            "background-color: #5cb85c; color: white; height: 40px; font-weight: bold;"
        )
        self.btn_save.clicked.connect(self.save_current)
        right_layout.addWidget(self.btn_save)

        # 组装 Splitter
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

        # 清空列表
        self.file_list_widget.clear()
        self.stats_label.setText("共 0 条数据")
        self.canvas.set_image(None)
        self.canvas.set_mask(None)
        self.meta_text.clear()
        self.text_editor.clear()
        self.translated_text.clear()

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
            if files:
                self.file_list_widget.setCurrentRow(0)

    def on_file_selected(self, index):
        """列表选中事件"""
        if index < 0:
            return

        if self.current_mode == "folder":
            self._load_folder_item(index)
        else:
            self._load_json_item(index)

    def _load_folder_item(self, index):
        """加载文件夹模式下的图片"""
        self.data_manager.current_index = index
        img_path, json_path = self.data_manager.get_current_data()

        if not img_path:
            return

        img = cv2.imread(img_path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法读取图片: {img_path}")
            return

        self.current_image = img
        self.canvas.set_image(img)
        self.sam_engine.set_image(img)
        self.reset_sam_state()

        self.meta_text.setPlainText(f"文件: {img_path}")

        if json_path and os.path.exists(json_path):
            self.text_editor.clear()
        else:
            self.text_editor.clear()

    # ==========================
    # JSON 模式
    # ==========================

    def load_json_action(self):
        """加载 JSON 文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 JSON 文件", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.json_data = json.load(f)
                self.json_path = file_path

                # 填充列表
                self.file_list_widget.clear()
                for item in self.json_data:
                    item_id = item.get('id', 'Unknown')
                    category = item.get('category', '')
                    display = f"[{category}] {item_id}" if category else item_id
                    self.file_list_widget.addItem(display)

                self.stats_label.setText(f"共 {len(self.json_data)} 条数据")

                if self.json_data:
                    self.file_list_widget.setCurrentRow(0)

                QMessageBox.information(self, "成功", f"已加载 {len(self.json_data)} 条数据")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {e}")

    def _load_json_item(self, index):
        """加载 JSON 模式下的数据项"""
        if index < 0 or index >= len(self.json_data):
            return

        self.json_current_index = index
        item = self.json_data[index]

        # 1. 加载 RGB 图像 (image_path_rgb)
        rgb_path = item.get('image_path_rgb', '')
        img = None

        if rgb_path and Path(rgb_path).exists():
            img = cv2.imread(rgb_path)
            print(f"加载图像: {rgb_path}")
        else:
            print(f"图像不存在: {rgb_path}")

        if img is not None:
            self.current_image = img
            self.canvas.set_image(img)
            self.sam_engine.set_image(img)
        else:
            self.current_image = None
            self.canvas.set_image(None)

        # 2. 加载 Mask (mask_path)
        mask_path = item.get('mask_path', '')
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, mask_binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                self.current_mask = mask_binary
                self.canvas.set_mask(mask_binary)
                print(f"加载Mask: {mask_path}")
            else:
                self.current_mask = None
                self.canvas.set_mask(None)
                print(f"Mask读取失败: {mask_path}")
        else:
            self.current_mask = None
            self.canvas.set_mask(None)
            print(f"Mask不存在: {mask_path}")

        # 3. 重置点击状态
        self.input_points = []
        self.input_labels = []

        # 4. 显示元信息
        bbox = item.get('bbox', [])
        meta_info = (
            f"ID: {item.get('id', '')}\n"
            f"BBox: {bbox}\n"
            f"─────────────────────\n"
            f"Image 4C: {item.get('image_path_4c', '')}\n"
            f"Image RGB: {rgb_path}\n"
            f"Mask: {mask_path}"
        )
        self.meta_text.setPlainText(meta_info)

        # 5. 显示对话内容并自动翻译
        conversations = item.get('conversations', [])
        if conversations:
            conv_text = ""
            for conv in conversations:
                role = conv.get('from', '')
                value = conv.get('value', '').replace('<image>\n', '')
                if role == 'human':
                    conv_text += f"👤 Human:\n{value}\n\n"
                else:
                    conv_text += f"🤖 GPT:\n{value}\n\n"
            self.text_editor.setPlainText(conv_text)

            # 自动翻译
            self._auto_translate(conv_text)
        else:
            self.text_editor.setPlainText("（无对话数据）")
            self.translated_text.clear()

    # ==========================
    # SAM 交互
    # ==========================

    @pyqtSlot(int, int, int)
    def handle_canvas_click(self, x, y, is_left):
        """响应画布点击"""
        if self.current_image is None:
            return

        self.input_points.append([x, y])
        self.input_labels.append(is_left)

        print(f"SAM Predicting... Points: {len(self.input_points)}")

        mask = self.sam_engine.predict_mask(self.input_points, self.input_labels)

        if mask is not None:
            self.current_mask = mask
            self.canvas.set_mask(mask)

    def reset_sam_state(self):
        """清空 Mask 和点击历史"""
        self.input_points = []
        self.input_labels = []
        self.current_mask = None
        self.canvas.set_mask(None)

    # ==========================
    # 删除操作
    # ==========================

    def delete_current_item(self):
        """删除当前条目"""
        if self.current_mode == "folder":
            self._delete_folder_item()
        else:
            self._delete_json_item()

    def _delete_folder_item(self):
        """删除文件夹模式下的图片"""
        reply = QMessageBox.question(
            self, '确认删除',
            "确定要将此图片移入回收站吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.data_manager.delete_current_file()
            current_row = self.file_list_widget.currentRow()
            self.file_list_widget.takeItem(current_row)
            self.stats_label.setText(f"共 {self.file_list_widget.count()} 条数据")

            if current_row < self.file_list_widget.count():
                self.file_list_widget.setCurrentRow(current_row)

    def _delete_json_item(self):
        """删除 JSON 模式下的条目"""
        if self.json_current_index < 0 or self.json_current_index >= len(self.json_data):
            QMessageBox.warning(self, "警告", "没有选中任何条目")
            return

        item = self.json_data[self.json_current_index]
        item_id = item.get('id', 'Unknown')

        reply = QMessageBox.question(
            self, '确认删除',
            f"确定要删除以下内容吗？\n\n"
            f"ID: {item_id}\n\n"
            f"这将删除：\n"
            f"• JSON 中的条目\n"
            f"• visual_prompt 图片\n"
            f"• training_mask 图片",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # 删除 visual_prompt 文件
        visual_path = item.get('visual_prompt_path', '')
        if visual_path and Path(visual_path).exists():
            try:
                os.remove(visual_path)
                print(f"已删除: {visual_path}")
            except Exception as e:
                print(f"删除失败: {e}")

        # 删除 mask 文件
        mask_path = item.get('training_mask_path', '')
        if mask_path and Path(mask_path).exists():
            try:
                os.remove(mask_path)
                print(f"已删除: {mask_path}")
            except Exception as e:
                print(f"删除失败: {e}")

        # 从数据列表移除
        self.json_data.pop(self.json_current_index)
        self.file_list_widget.takeItem(self.json_current_index)
        self.stats_label.setText(f"共 {len(self.json_data)} 条数据")

        # 选中下一条
        if len(self.json_data) > 0:
            new_index = min(self.json_current_index, len(self.json_data) - 1)
            self.file_list_widget.setCurrentRow(new_index)
        else:
            self.json_current_index = -1
            self.canvas.set_image(None)
            self.canvas.set_mask(None)
            self.meta_text.clear()
            self.text_editor.clear()

    # ==========================
    # 保存操作
    # ==========================

    def save_current(self):
        """保存当前修改"""
        if self.current_mode == "folder":
            self._save_folder_item()
        else:
            self._save_json_item()

    def _save_folder_item(self):
        """保存文件夹模式下的标注"""
        if self.current_image is None:
            return

        text_content = self.text_editor.toPlainText()

        if self.current_mask is not None:
            self.data_manager.save_annotation(self.current_mask, text_content)
            print("已保存")
        else:
            print("没有 mask 可保存")

        # 跳转下一张
        next_row = self.file_list_widget.currentRow() + 1
        if next_row < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(next_row)
        else:
            QMessageBox.information(self, "完成", "所有图片已处理完毕！")

    def _save_json_item(self):
        """保存 JSON 模式下的修改"""
        if not self.json_path:
            QMessageBox.warning(self, "警告", "没有加载 JSON 文件")
            return

        # 保存当前 mask 到文件
        if self.json_current_index >= 0 and self.current_mask is not None:
            item = self.json_data[self.json_current_index]
            mask_path = item.get('training_mask_path', '')
            if mask_path:
                mask_to_save = (self.current_mask * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask_to_save)
                print(f"Mask 已保存: {mask_path}")

        # 保存 JSON 文件
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "成功", f"已保存到:\n{self.json_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    # ==========================
    # 导航
    # ==========================

    def navigate_prev(self):
        """上一条"""
        current_row = self.file_list_widget.currentRow()
        if current_row > 0:
            self.file_list_widget.setCurrentRow(current_row - 1)

    def navigate_next(self):
        """下一条"""
        current_row = self.file_list_widget.currentRow()
        if current_row < self.file_list_widget.count() - 1:
            self._auto_save_current()
            self.file_list_widget.setCurrentRow(current_row + 1)

    def translate_text(self):
        """翻译当前对话文本"""
        text = self.text_editor.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "没有可翻译的文本")
            return

        self.btn_translate.setEnabled(False)
        self.btn_translate.setText("翻译中...")

        try:
            # 移除特殊标记后翻译
            clean_text = text.replace('<image>\n', '').replace('[SEG]', '[分割]')
            translated = self.translator.translate(clean_text, from_lang='en', to_lang='zh')
            self.translated_text.setPlainText(translated)
        except Exception as e:
            QMessageBox.warning(self, "翻译失败", str(e))
        finally:
            self.btn_translate.setEnabled(True)
            self.btn_translate.setText("🌐 翻译为中文")

    def _auto_translate(self, text: str):
        """自动翻译文本"""
        if not text.strip():
            self.translated_text.clear()
            return

        try:
            clean_text = text.replace('<image>\n', '').replace('[SEG]', '[分割]')
            translated = self.translator.translate(clean_text, from_lang='en', to_lang='zh')
            self.translated_text.setPlainText(translated)
        except Exception as e:
            self.translated_text.setPlainText(f"翻译失败: {e}")

    def _save_json_item(self):
        """保存 JSON 模式下的修改"""
        if not self.json_path:
            QMessageBox.warning(self, "警告", "没有加载 JSON 文件")
            return

        if self.json_current_index < 0 or self.json_current_index >= len(self.json_data):
            return

        item = self.json_data[self.json_current_index]

        # 1. 解析编辑器中的对话内容并更新 JSON
        text_content = self.text_editor.toPlainText()
        new_conversations = self._parse_conversations(text_content)
        if new_conversations:
            item['conversations'] = new_conversations

        # 2. 保存当前 mask 到文件
        if self.current_mask is not None:
            mask_path = item.get('mask_path', '') or item.get('training_mask_path', '')
            if mask_path:
                mask_to_save = (self.current_mask * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask_to_save)
                print(f"Mask 已保存: {mask_path}")

        # 3. 保存 JSON 文件
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "成功", f"已保存到:\n{self.json_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def _parse_conversations(self, text: str) -> list:
        """将编辑器文本解析回 conversations 格式"""
        if not text.strip():
            return []

        conversations = []
        # 按角色标记分割
        parts = text.split('👤 Human:')

        for part in parts:
            if not part.strip():
                continue

            # 检查是否包含 GPT 回复
            if '🤖 GPT:' in part:
                human_gpt = part.split('🤖 GPT:')
                human_text = human_gpt[0].strip()
                gpt_text = human_gpt[1].strip() if len(human_gpt) > 1 else ''

                if human_text:
                    conversations.append({
                        'from': 'human',
                        'value': human_text
                    })
                if gpt_text:
                    conversations.append({
                        'from': 'gpt',
                        'value': gpt_text
                    })
            else:
                # 只有 human 部分
                human_text = part.strip()
                if human_text:
                    conversations.append({
                        'from': 'human',
                        'value': human_text
                    })

        return conversations

    def _auto_save_current(self):
        """静默自动保存（不弹窗提示）"""
        if self.current_mode == "folder":
            self._auto_save_folder_item()
        else:
            self._auto_save_json_item()

    def _auto_save_folder_item(self):
        """自动保存文件夹模式"""
        if self.current_image is None or self.current_mask is None:
            return
        text_content = self.text_editor.toPlainText()
        self.data_manager.save_annotation(self.current_mask, text_content)
        print("已自动保存")

    def _auto_save_json_item(self):
        """自动保存 JSON 模式（无弹窗）"""
        if not self.json_path or self.json_current_index < 0:
            return

        item = self.json_data[self.json_current_index]

        # 1. 解析对话内容
        text_content = self.text_editor.toPlainText()
        new_conversations = self._parse_conversations(text_content)
        if new_conversations:
            item['conversations'] = new_conversations

        # 2. 保存 Mask
        if self.current_mask is not None:
            mask_path = item.get('mask_path', '') or item.get('training_mask_path', '')
            if mask_path:
                mask_to_save = (self.current_mask * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask_to_save)

        # 3. 保存 JSON
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, ensure_ascii=False, indent=4)
            print(f"已自动保存: {self.json_path}")
        except Exception as e:
            print(f"自动保存失败: {e}")