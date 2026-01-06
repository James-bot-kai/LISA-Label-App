import os
import json
import base64
import time
import random
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = "sk-6e57ca6470284b42ace045432d98e6bd"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-max"

DATA_ROOT = "/Users/cuimingcan/Downloads/Potsdam/SegImage_Output_2"
INPUT_METADATA = os.path.join(DATA_ROOT, "step1_metadata.json")
OUTPUT_JSON = os.path.join(DATA_ROOT, "step2_dataset_qwen_varied.json")  # 改个名区分一下

TEST_MODE = True
TEST_COUNT = 50  # 测试5个看看效果
# ===========================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=120.0,
    max_retries=3
)

# ================= 定义多样化模板 (User & GPT) =================
# 这里的 {} 会被 Qwen 生成的名词短语替换 (例如 "the red car")
HUMAN_QUESTION_TEMPLATES = [
    "Please segment {}.",
    "Segment out {}.",
    "Can you segment {}?",
    "Please segment {} in the image.",
    "Help me find {}.",
    "Segment {}.",
    "{}",  # 只有名词短语本身，例如 "The red car."
    "Locate {}."
]

# 这里的回答尽量简洁，或者只有 [SEG]
GPT_ANSWER_TEMPLATES = [
    "Sure. [SEG]",
    "Sure, here you go. [SEG]",
    "Sure, this is the [SEG].",
    "Done. [SEG]",
    "Here is the segmentation. [SEG]",
    "[SEG]",  # 极其简洁
    "Certainly. [SEG]"
]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_caption_with_retry(image_path, category, max_retries=3):
    """
    句式的变化由 Python 代码来完成。
    """
    system_prompt = "You are an expert Remote Sensing Analyst. You answer strictly in JSON."

    user_prompt = f"""
    # Role
    You are a geospatial expert creating a training dataset for Referring Segmentation.

    # Input Status
    - **Target Category**: {category}
    - **Visual Cue**: Cyan Box (Ignore the box artifact in output).

    # Task
    Generate 3 distinct **Referring Expressions** (Target Noun Phrases) that uniquely identify the object.

    # CRITICAL GRAMMAR CONSTRAINTS
    1. **NOUN PHRASES ONLY**: e.g., "The red car", "The building next to the tree". NO complete sentences like "There is a car".
    2. **Start with 'The'**: Always start with "The".
    3. **No UI Mentions**: Do not mention the box/cyan color.

    # Output Format (JSON)
    {{
        "simple_instruction": "Short phrase (e.g., 'The red-roofed {category}').",
        "spatial_instruction": "Location phrase (e.g., 'The {category} adjacent to the parking lot').",
        "complex_instruction": "Detailed phrase (e.g., 'The rectangular {category} with a brown roof...').",
        "reasoning": "Explanation of why this object matches." 
    }}
    """

    base64_image = encode_image(image_path)

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,  # 稍微调高一点点温度，增加词汇丰富度
            )

            content = completion.choices[0].message.content
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
            return json.loads(content)

        except Exception as e:
            print(f"Error: {e}, Retrying...")
            time.sleep(2)
            if attempt == max_retries - 1: return None


def main():
    if not os.path.exists(INPUT_METADATA):
        print(f"找不到输入文件: {INPUT_METADATA}")
        return

    with open(INPUT_METADATA, 'r') as f:
        data_list = json.load(f)

    processed_ids = set()
    existing_data = []

    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                for item in existing_data:
                    processed_ids.add(item['id'])
        except:
            existing_data = []

    todos = [item for item in data_list if item['id'] not in processed_ids]
    if TEST_MODE: todos = todos[:TEST_COUNT]

    for i, item in enumerate(tqdm(todos)):
        vis_path = item['visual_prompt_path']
        if not os.path.exists(vis_path): continue

        result = generate_caption_with_retry(vis_path, item['category'])

        if result:
            # 1. 随机选择一种描述风格 (简单/空间/复杂)
            style = random.choice(["simple_instruction", "spatial_instruction", "complex_instruction"])
            noun_phrase = result.get(style, f"The {item['category']}")

            # 2. 处理首字母大小写，使其能融入句子
            # 如果名词短语是 "The red car"，变成 "the red car" 以便放入 "Please segment..."
            # 但如果模板就是 "{}" (只有短语)，则保持 "The red car"

            # 随机选一个人类提问模板
            human_tmpl = random.choice(HUMAN_QUESTION_TEMPLATES)

            # 简单的逻辑：如果模板开头不是 "{", 说明有前缀词，需要把名词首字母小写
            if not human_tmpl.startswith("{"):
                # 把 "The" 变成 "the"
                if noun_phrase.startswith("The "):
                    noun_phrase_formatted = "the " + noun_phrase[4:]
                else:
                    noun_phrase_formatted = noun_phrase.lower()  # 兜底
            else:
                # 如果模板只是 "{}"，保持首字母大写
                noun_phrase_formatted = noun_phrase

            # 组装 Question
            # 移除可能重复的句号
            final_question = human_tmpl.format(noun_phrase_formatted)
            if final_question.endswith(".."): final_question = final_question[:-1]

            # 3. 随机选一个 GPT 回答模板 (不再包含 Reasoning)
            gpt_response = random.choice(GPT_ANSWER_TEMPLATES)

            lisa_entry = {
                "id": item['id'],
                "image_path_4c": item['image_path_4c'],
                "image_path_rgb": item['visual_prompt_path'],
                "mask_path": item['training_mask_path'],
                "bbox": item['bbox'],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{final_question}"
                    },
                    {
                        "from": "gpt",
                        "value": gpt_response
                    }
                ],
                # 依然保存原始 Reasoning 供以后分析，但不在 conversations 里展示
                "raw_vlm_output": result
            }
            existing_data.append(lisa_entry)

            if (i + 1) % 5 == 0:
                with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 完成！生成数据已保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()