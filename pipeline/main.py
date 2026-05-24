import os
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from gemma import (load_gemma, gemma_generate_description, gemma_score)
from sd import (load_sd, generate_images,)
from safety import (load_nude_detector, load_violence_detector, is_nsfw_image, is_violence_image,)
from aesthetic_image import aesthetic_image_score
from aesthetic_text import (load_aesthetic_text_model, aesthetic_response_score,)
from novelty_image import (load_novelty_image_models, novelty_image_score,)
from novelty_description import (load_novelty_description_model, novelty_description_score,)
from aesthetic_mlp import (load_aesthetic_mlp, predict_aesthetic_score,)
from novelty_mlp import (load_novelty_mlp, predict_novelty_score,)
GEMMA_DEVICE = "cuda:1"
SD_DEVICE = "cuda:0"
OTHER_DEVICE = "cuda:0"
SEED = 42
NUM_GENERATE_IMAGES = 2
CONTEXT_WINDOW = 2
MAX_DESCRIPTION_REGENERATE = 3
USE_KEYWORDS = True
USE_SD_LORA = True
AESTHETIC_WEIGHT = 0.5
NOVELTY_WEIGHT = 0.5
DATASET = "photochat"
FILE = "train"
INPUT_JSON = "dataset/" + DATASET + "_aug/" + FILE + ".json"
CONFIG = "SEED" + str(SEED) + "_NUM_GENERATE_IMAGES" + str(NUM_GENERATE_IMAGES) + "_CONTEXT_WINDOW" + str(CONTEXT_WINDOW)
OUTPUT_DIR = "output/" + DATASET + "/" + CONFIG + "/" +FILE
RESULT_JSON = "output/" + DATASET + "/" + CONFIG + "/" + FILE + ".json"
GEMMA_BASE_MODEL = "/disk3/model/gemma-4-31B"
GEMMA_LORA_PATH = "./gemma_lora"
SD_MODEL_PATH = "/disk3/model/stable-diffusion-3.5-medium"
SD_MODEL_LORA_PATH = "./sd_lora/keywords_caption_description/pytorch_lora_weights.safetensors"
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nLoading Gemma...")
    print(f"BASE: {GEMMA_BASE_MODEL}")
    print(f"LORA: {GEMMA_LORA_PATH}")
    print(f"DEVICE: {GEMMA_DEVICE}")
    gemma_model, gemma_tokenizer = load_gemma(GEMMA_BASE_MODEL,GEMMA_LORA_PATH, device=GEMMA_DEVICE)
    print("Gemma Loaded.")
    print("\nLoading SD...")
    print(f"MODEL: {SD_MODEL_PATH}")
    print(f"USE_SD_LORA: {USE_SD_LORA}")
    if USE_SD_LORA:
        print(f"LORA: {SD_MODEL_LORA_PATH}")
    print(f"DEVICE: {SD_DEVICE}")
    sd_model = load_sd(SD_MODEL_PATH,SD_MODEL_LORA_PATH, USE_SD_LORA, device=SD_DEVICE)
    print("SD Loaded.")
    print("\nLoading Safety Models...")
    nude_detector, ocr_reader = (load_nude_detector())
    violence_checker = (load_violence_detector(device=OTHER_DEVICE))
    print("Safety Models Loaded.")
    print("\nLoading Aesthetic Models...")
    aesthetic_text_model = (load_aesthetic_text_model(device=OTHER_DEVICE))
    print("Aesthetic Models Loaded.")
    print("\nLoading Novelty Models...")
    novelty_image_models = (load_novelty_image_models(device=OTHER_DEVICE))
    novelty_description_model = (load_novelty_description_model())
    print("Novelty Models Loaded.")
    print("\nLoading MLP Models...")
    aesthetic_mlp_model, aesthetic_scaler = (load_aesthetic_mlp())
    novelty_mlp_model, novelty_scaler = (load_novelty_mlp())
    print("MLP Models Loaded.")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        # data = dict(list(data.items())[:3])  # 可调试
        print(f"\nLoaded Samples: {len(data)}")
    for sample_id, sample in tqdm(data.items(),total=len(data),desc="Processing Samples"):
        try:
            images_position = sample["predictions"]
            dialog = sample["new_dialog"]
            txt_turns = []
            for t in dialog:
                if t[0] == "txt":
                    txt_turns.append(t[1])
            for image_information in images_position:
                regenerate_count = 0
                best_score = -999999
                image_position = image_information["utt_id"]
                keywords = image_information["keywords"]
                description = image_information["description"]
                save_dir = OUTPUT_DIR + "/" + sample_id + "_" + str(image_position)
                os.makedirs(save_dir, exist_ok=True)
                context = txt_turns[:image_position]
                response = txt_turns[image_position]
                while regenerate_count < MAX_DESCRIPTION_REGENERATE:
                    images = generate_images(sd_model,keywords,description,NUM_GENERATE_IMAGES,save_dir,SEED,USE_KEYWORDS)
                    passed_images = []
                    for image_path in images:
                        is_nsfw = is_nsfw_image(detector=nude_detector,reader=ocr_reader,image_path=image_path,threshold=0.5 + regenerate_count * 0.1,)
                        is_violence = is_violence_image(checker=violence_checker,image_path=image_path,threshold=0.23 + regenerate_count * 0.1,)
                        if not is_nsfw and not is_violence:
                            passed_images.append(image_path)
                    if len(passed_images) == 0:
                        if regenerate_count < MAX_DESCRIPTION_REGENERATE -1:
                            print("All failed safety check.")
                            description = gemma_generate_description(model=gemma_model,tokenizer=gemma_tokenizer,context=context,
                            response=response,keywords=keywords,use_keywords=USE_KEYWORDS,)
                            image_information["description"] = description
                            txt = -1
                            for item in dialog:
                                if item[0] == "txt":
                                    txt += 1
                                elif item[0] == "img" and txt == image_position:
                                    item[2] = description
                                    break
                            regenerate_count += 1
                            continue
                        passed_images = images
                        print("*"*100)
                        print("ALL IMAGES FAILED!!!")
                        position = str(sample_id) + ": " + str(image_position)
                        print(position)
                        print("*" * 100)
                    # =================================================
                    # score
                    # =================================================
                    aesthetic_scores = aesthetic_image_score(image_paths=passed_images,batch_size=8,)
                    if len(passed_images) == 1:
                        novelty_image_scores = [5] * len(passed_images)
                    else:
                        novelty_image_scores = novelty_image_score(models=novelty_image_models,image_paths=passed_images,
                            image_description=description,dialogue=dialog,image_position=image_position,context_window=CONTEXT_WINDOW,)
                    aesthetic_gemma, novelty_gemma = gemma_score(gemma_model, gemma_tokenizer, txt_turns, image_position, description, CONTEXT_WINDOW)
                    result = []
                    for idx, image_path in enumerate(passed_images):
                        aesthetic_image = aesthetic_scores[idx]
                        aesthetic_text = aesthetic_response_score(model_dict=aesthetic_text_model,context=context,
                                response=response+ ". " + description,)
                        aesthetic_total = predict_aesthetic_score(model=aesthetic_mlp_model,scaler=aesthetic_scaler,image_score=aesthetic_image,
                                response_score=aesthetic_text,gemma_score=aesthetic_gemma)
                        novelty_image = novelty_image_scores[idx]
                        novelty_description = novelty_description_score(model=novelty_description_model,description=description)
                        novelty_total = predict_novelty_score(model=novelty_mlp_model,scaler=novelty_scaler,novelty_image_score=novelty_image,
                                novelty_description_score=novelty_description,gemma_novelty_score=novelty_gemma,)
                        final_score = AESTHETIC_WEIGHT * aesthetic_total + NOVELTY_WEIGHT * novelty_total
                        result.append({"image_path": image_path,
                                       "aesthetic_score": [aesthetic_image,aesthetic_text,aesthetic_gemma,aesthetic_total],
                                       "novelty_score": [novelty_image,novelty_description,novelty_gemma,novelty_total],
                                       "final_score": final_score})
                        if final_score > best_score:
                            image_information["best"] = image_path
                            best_score = final_score
                        image_information["result"] = result
                    break
        except Exception as e:
            print("=" * 80)
            print(f"[SAMPLE FAILED] {sample_id}")
            print(e)
            print("=" * 80)
            continue
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )
    print("Pipeline Finished.")

# =========================================================
# run
# =========================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Pipeline Start")
    print("=" * 80)
    print(f"DATASET: {DATASET}")
    print(f"FILE: {FILE}")
    print(f"INPUT_JSON: {INPUT_JSON}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"RESULT_JSON: {RESULT_JSON}")
    main()