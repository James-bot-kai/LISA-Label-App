import os
import json
import shutil


class DataManager:
    def __init__(self):
        self.root_dir = ""
        self.file_list = []
        self.current_index = -1

    def load_directory(self, path):
        self.root_dir = path
        valid_ext = ('.jpg', '.png', '.tif')
        self.file_list = [f for f in os.listdir(path) if f.lower().endswith(valid_ext)]
        self.file_list.sort()
        return self.file_list

    def get_current_data(self):
        """获取当前图片路径、对应的JSON路径"""
        if 0 <= self.current_index < len(self.file_list):
            filename = self.file_list[self.current_index]
            img_path = os.path.join(self.root_dir, filename)
            # 假设 json 和图片同名
            json_path = os.path.join(self.root_dir, filename.replace(os.path.splitext(filename)[1], '.json'))
            return img_path, json_path
        return None, None

    def save_annotation(self, mask_np, text_data):
        # 实现保存 Mask 和 JSON 的逻辑
        pass

    def delete_current_file(self):
        # 实现移动到回收站逻辑
        pass