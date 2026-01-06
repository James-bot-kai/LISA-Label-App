import numpy as np
import torch
# 确保安装了 segment_anything: pip install segment-anything
from segment_anything import sam_model_registry, SamPredictor


class SAMEngine:
    def __init__(self, checkpoint_path, model_type="vit_b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing SAM Engine on {self.device}...")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        self.is_loaded = True

    def set_image(self, image_np):
        if not self.is_loaded: return
        # SAM 需要 RGB 格式
        if image_np.shape[-1] == 3:
            image_rgb = image_np  # 假设传入已经是RGB或者在外面转好了
            # 如果传入是BGR (OpenCV默认)，这里需要转:
            # image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.stack([image_np] * 3, axis=-1)

        self.predictor.set_image(image_rgb)
        print("SAM: Image embedding computed.")

    def predict_mask(self, points, labels):
        if not points or not self.is_loaded:
            return None

        points_np = np.array(points)
        labels_np = np.array(labels)

        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=False
            )
            # masks 原本是 [1, H, W] 的 bool 类型
            # 我们取 [0] 变成 [H, W]，然后转成 uint8 (0, 1)
            mask_result = masks[0].astype(np.uint8)
            return mask_result
        except Exception as e:
            print(f"SAM Predict Error: {e}")
            return None