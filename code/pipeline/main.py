import os
import json
import random
import time
from typing import Dict, List, Any
import torch
from tqdm import tqdm
from qwen import QwenGenerator
from fg import FGClip
from nude import NudeChecker
from violence import ViolenceChecker
from sd import SDGenerator

# =========================================================
# 1. 参数配置
# =========================================================

class Config:
    """
    全局配置类：
    - 模型路径
    - 生成参数
    - 过滤阈值
    - 设备分配
    """

    # -------- 随机性控制 --------
    GLOBAL_SEED = 42

    # -------- Qwen 生成参数 --------
    QWEN_TEMPERATURE = 0.7
    QWEN_TOP_P = 0.9
    QWEN_DO_SAMPLE = True
    QWEN_MAX_NEW_TOKENS = 512

    # -------- 模型路径 --------
    QWEN_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-4B"
    QWEN_LORA = True
    QWEN_LORA_PATH = "/root/autodl-tmp/data_strengthen/add/qwen3/lora_rationale_caption_description/"

    SD35_MODEL_PATH = "/root/autodl-tmp/models/stable-diffusion-3.5-medium"
    SD35_LORA = True
    SD35_LORA_PATH = "/root/autodl-tmp/data_strengthen/add/sd35/caption_description"

    FG_CLIP_MODEL_PATH = "/root/autodl-tmp/models/fg-clip-base"

    # -------- 设备分配（显式指定，避免 device_map 混乱）--------
    QWEN_DEVICE = "cuda:0"
    SD_DEVICE = "cuda:1"
    FGCLIP_DEVICE = "cuda:0"

    # -------- 数据路径 --------
    INPUT_DATA_PATH = "/root/autodl-tmp/data_strengthen/Photochat/ori_data/dgwy/test.json"
    OUTPUT_SAVE_PATH = "/root/autodl-tmp/data_strengthen/Photochat/photochat+/dgwy/test.json"
    IMAGE_SAVE_DIR = "/root/autodl-tmp/data_strengthen/Photochat/photochat+/dgwy/images/test"
    # INPUT_DATA_PATH = "/root/autodl-tmp/data_strengthen/Photochat/ori_data/test.json"
    # OUTPUT_SAVE_PATH = "/root/autodl-tmp/data_strengthen/Photochat/aug_data/test.json"
    # IMAGE_SAVE_DIR = "/root/autodl-tmp/data_strengthen/Photochat/aug_data/images/test"

    # -------- 生成策略 --------
    MAX_CAPTION_RETRY = 2           # caption 重新生成次数
    SD_IMAGE_PER_PROMPT = 3         # 每个 caption 生成 3 张图
    CONTEXT_WINDOW = 2              # caption 上下文窗口

    # -------- 安全阈值 --------
    NUDE_THRESHOLD = 0.5
    VIOLENCE_THRESHOLD = 0.23
    FG_CLIP_THRESHOLD = 0.21        # 预留阈值（当前只记录，不过滤）

    DEBUG = True


# =========================================================
# 2. 数据加载
# =========================================================

def load_input_data(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 这里取样，如果不需要可删除
    sub_data = {k: v for i, (k, v) in enumerate(data.items()) if i % 1 == 0}
    return sub_data

# =========================================================
# 3. Qwen3 图片描述生成
# =========================================================

def generate_image_captions(generator: QwenGenerator, utterances: List[List[Any]], cfg: Config, seed: int) -> List[Dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    raw_captions = generator.generate_captions(utterances, seed)
    results = []
    for p in raw_captions:
        uid = p["utterance_id"]
        prev_context = [utterances[i][1] for i in range(max(0, uid - cfg.CONTEXT_WINDOW + 1), uid + 1) if utterances[i][0] == "txt"]
        next_context = [utterances[i][1] for i in range(uid + 1, min(len(utterances), uid + 1 + cfg.CONTEXT_WINDOW)) if utterances[i][0] == "txt"]
        results.append({
            "caption": generator.clean_description(p["caption"]),
            "utterance_id": uid,
            "context": {"prev": prev_context, "next": next_context}
        })
    return results

def compute_fg_clip(fg_clip_model: FGClip, image_path: str, text: str) -> float:
    return fg_clip_model.compute_score(image_path, text)

def check_nudity(nude_checker: NudeChecker, image_path: str, cfg: Config) -> bool:
    return nude_checker.check(image_path=image_path, nude_threshold=cfg.NUDE_THRESHOLD)

def check_violence(violence_checker: ViolenceChecker, image_path: str, cfg: Config) -> bool:
    result = violence_checker.predict(image_path)
    label, confidence = result["label"], result["confidence"]
    if label == "Unknown": return False
    if confidence >= cfg.VIOLENCE_THRESHOLD: return True
    return False

def generate_images_by_sd(sd_generator, dialogue_id: str, utterance_id: int, caption: str, cfg: Config, seed: int) -> List[str]:
    filename_prefix = f"{dialogue_id}_{utterance_id}"
    image_paths = sd_generator.generate(
        prompt=caption,
        negative_prompt="blurry, low quality, distorted, bad anatomy",
        num_images=cfg.SD_IMAGE_PER_PROMPT,
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=7.0,
        seed=seed,
        save_dir=cfg.IMAGE_SAVE_DIR,
        filename_prefix=filename_prefix,
    )
    return image_paths

# =========================================================
# 4. 单样本处理
# =========================================================

def process_single_sample(
    sample_id: str,
    sample: List[List[Any]],
    cfg: Config,
    qwen_generator: QwenGenerator,
    fg_clip: FGClip,
    nude_checker: NudeChecker,
    violence_checker: ViolenceChecker,
    sd_generator: SDGenerator,
    passed_fg_records: List[Dict[str, Any]],
    failed_caption_records: List[Dict[str, Any]]
) -> List[List[Any]]:
    """
    对单条对话样本进行图像增强：

    流程：
    1. 基于文本对话生成若干 caption
    2. 每个 caption 用 SD 生成 3 张图片
    3. 对图片进行：
       - 裸体检测
       - 暴力检测
       - fg-clip 图文相关性评分
    4. 若某 caption 的 3 张图全不安全 → 整轮失败，重新生成 caption
    5. 成功轮次中：
       - 插入安全且 fg-clip 分数最高的图片
       - 该 caption 对应的 3 张图片全部保留（用于分析）
    """

    dialogue = sample
    final_insert_map = None  # 保存最终插入结果

    # ---------- retry 机制 ----------
    for retry in range(cfg.MAX_CAPTION_RETRY):
        seed = cfg.GLOBAL_SEED + retry
        round_failed = False
        round_failed_info = []

        # 1️⃣ 生成图片描述
        captions = generate_image_captions(qwen_generator, dialogue, cfg, seed)
        insert_map = {}

        # 2️⃣ 针对每个 caption 生成图片
        for cap in captions:
            uid = cap["utterance_id"]
            caption_text = cap["caption"]

            # SD 生成 3 张图片
            image_paths = generate_images_by_sd(
                sd_generator, sample_id, uid, caption_text, cfg, seed
            )

            image_infos = []
            for img_path in image_paths:
                nudity = check_nudity(nude_checker, img_path, cfg)
                violence = check_violence(violence_checker, img_path, cfg)
                fg_score = compute_fg_clip(fg_clip, img_path, caption_text)

                image_infos.append({
                    "path": img_path,
                    "fg": fg_score,
                    "nudity": nudity,
                    "violence": violence
                })

            cap["images"] = image_infos

            # 3️⃣ 仅保留安全图片
            safe_imgs = [
                img for img in image_infos
                if not img["nudity"] and not img["violence"] and img["fg"] > cfg.FG_CLIP_THRESHOLD
            ]

            # ❌ 本 caption 全不安全 → 整轮失败
            if not safe_imgs and retry < cfg.MAX_CAPTION_RETRY - 1:
                round_failed = True
                round_failed_info.append({
                    "sample_id": sample_id,
                    "utterance_id": uid,
                    "caption": caption_text,
                    "images": image_infos
                })
                break

            # ✔ 至少一张安全图片
            if safe_imgs:
                best_img = max(safe_imgs, key=lambda x: x["fg"])
                insert_map[uid] = {
                    "best_img": best_img["path"],
                    "all_imgs": [img["path"] for img in image_infos],
                    "caption": caption_text,
                    "fg": best_img["fg"],
                }

                # 记录 fg-clip 分数，用于后续阈值分析
                passed_fg_records.append({
                    "sample_id": sample_id,
                    "utterance_id": uid,
                    "image_path": best_img["path"],
                    "fg_clip_score": best_img["fg"]
                })
            else:
                # 最后一轮允许失败，只记录不插入
                round_failed_info.append({
                    "sample_id": sample_id,
                    "utterance_id": uid,
                    "caption": caption_text,
                    "images": image_infos
                })

        # ---------- 失败轮次：删除图片 ----------
        if round_failed and retry < cfg.MAX_CAPTION_RETRY - 1:
            for cap in captions:
                for img in cap.get("images", []):
                    if os.path.exists(img["path"]):
                        os.remove(img["path"])

            failed_caption_records.clear()
            failed_caption_records.extend(round_failed_info)
            continue

        # ---------- 成功或最后一轮 ----------
        final_insert_map = insert_map
        failed_caption_records.clear()
        failed_caption_records.extend(round_failed_info)
        break

    # ---------- 构造最终对话 ----------
    new_dialogue = []
    for i, item in enumerate(dialogue):
        new_dialogue.append(item)
        if final_insert_map and i in final_insert_map:
            new_dialogue.append([
                "img",
                final_insert_map[i]["best_img"],
                final_insert_map[i]["caption"]
            ])

    return new_dialogue


# =========================================================
# 5. 主入口
# =========================================================

def main():
    cfg = Config()
    torch.manual_seed(cfg.GLOBAL_SEED)
    random.seed(cfg.GLOBAL_SEED)
    os.makedirs(cfg.IMAGE_SAVE_DIR, exist_ok=True)

    data = load_input_data(cfg.INPUT_DATA_PATH)
    total_samples = len(data)
    passed_fg_records = []
    failed_caption_records = []

    # 模型初始化
    print("[INIT] Qwen Generator")
    qwen_generator = QwenGenerator(base_model_path=cfg.QWEN_MODEL_PATH,
                                   lora_path=cfg.QWEN_LORA_PATH if cfg.QWEN_LORA else None,
                                   use_lora=cfg.QWEN_LORA,
                                   device=cfg.QWEN_DEVICE,
                                   max_input_tokens=1024,
                                   max_new_tokens=cfg.QWEN_MAX_NEW_TOKENS,
                                   do_sample=cfg.QWEN_DO_SAMPLE,
                                   temperature=cfg.QWEN_TEMPERATURE,
                                   top_p=cfg.QWEN_TOP_P)

    print("[INIT] FG-CLIP")
    fg_clip = FGClip(model_path=cfg.FG_CLIP_MODEL_PATH, device=cfg.FGCLIP_DEVICE)
    print("[INIT] Nude Checker")
    nude_checker = NudeChecker()
    print("[INIT] Violence Checker")
    violence_checker = ViolenceChecker(device="cuda:1", violence_threshold=cfg.VIOLENCE_THRESHOLD)
    print("[INIT] Stable Diffusion")
    sd_generator = SDGenerator(model_path=cfg.SD35_MODEL_PATH,
                               use_lora=cfg.SD35_LORA,
                               lora_path=cfg.SD35_LORA_PATH,
                               device=cfg.SD_DEVICE,)

    final_results = {}
    start_time = time.time()

    for idx, (sample_id, sample) in enumerate(tqdm(data.items(), desc="Processing samples")):
        new_dialogue = process_single_sample(sample_id, sample, cfg, qwen_generator, fg_clip,
                                             nude_checker, violence_checker, sd_generator,
                                             passed_fg_records, failed_caption_records)
        final_results[sample_id] = new_dialogue

        # 进度显示和预计剩余时间
        # elapsed = time.time() - start_time
        # avg_time = elapsed / (idx + 1)
        # remaining_time = avg_time * (total_samples - idx - 1)
        # print(f"[INFO] Sample {idx+1}/{total_samples} done. Elapsed: {elapsed:.1f}s, Est. remaining: {remaining_time:.1f}s", end="\r")

    # 保存结果
    with open(cfg.OUTPUT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    with open("passed_fg_records1.json", "w", encoding="utf-8") as f:
        json.dump(passed_fg_records, f, ensure_ascii=False, indent=2)
    with open("failed_caption_records1.json", "w", encoding="utf-8") as f:
        json.dump(failed_caption_records, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Saved to {cfg.OUTPUT_SAVE_PATH}")

if __name__ == "__main__":
    main()
