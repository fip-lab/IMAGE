import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import gc
import traceback
import threading

import torch

from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

from mplug_owl2.assessor import Assessment

# =========================================================
# app
# =========================================================

app = FastAPI()

# =========================================================
# global gpu lock
# 防止多个请求同时进GPU
# =========================================================

gpu_lock = threading.Lock()

# =========================================================
# model
# =========================================================

assessment = Assessment(
    pretrained=(
        "/disk2/disk4/chenp/"
        "2024-chenpeng/data_strengthen/"
        "IMAGE/new_code/aesthetic/"
        "ROC4MLLM/ROC4MLLM_weights"
    )
)

print("Aesthetic model loaded.")

# =========================================================
# request
# =========================================================

class ScoreRequest(BaseModel):

    image_paths: list

    precision: int = 4

    batch_size: int = 8

# =========================================================
# api
# =========================================================

@app.post("/score_batch")
def score_batch(req: ScoreRequest):

    # =====================================================
    # 单请求互斥
    # 非常重要
    # =====================================================

    with gpu_lock:

        all_scores = []

        try:

            print(
                f"\nRequest images: "
                f"{len(req.image_paths)}"
            )

            batch_size = req.batch_size

            # =================================================
            # batch loop
            # =================================================

            for start in range(
                0,
                len(req.image_paths),
                batch_size,
            ):

                batch_paths = req.image_paths[
                    start:start + batch_size
                ]

                print(
                    f"\nBatch: "
                    f"{start} "
                    f"-> "
                    f"{start + len(batch_paths)}"
                )

                input_imgs = []

                valid_count = 0

                # =============================================
                # load images
                # =============================================

                for p in batch_paths:

                    try:

                        if not os.path.exists(p):

                            print(
                                f"Missing image: {p}"
                            )

                            continue

                        img = (
                            Image.open(p)
                            .convert("RGB")
                        )

                        input_imgs.append(img)

                        valid_count += 1

                    except Exception as e:

                        print(
                            f"Image load failed: {p}"
                        )

                        print(e)

                # =============================================
                # empty batch
                # =============================================

                if valid_count == 0:

                    continue

                # =============================================
                # inference
                # =============================================

                try:

                    with torch.inference_mode():

                        with torch.cuda.amp.autocast():

                            batch_scores = (
                                assessment(
                                    input_imgs,
                                    precision=req.precision,
                                )[-1]
                            )

                    batch_scores = [
                        float(x)
                        for x in batch_scores
                    ]

                    all_scores.extend(
                        batch_scores
                    )

                    print(
                        f"Batch success: "
                        f"{len(batch_scores)}"
                    )

                # =============================================
                # batch fail
                # =============================================

                except torch.cuda.OutOfMemoryError:

                    print(
                        "CUDA OOM"
                    )

                    traceback.print_exc()

                    torch.cuda.empty_cache()

                    gc.collect()

                    # fallback
                    all_scores.extend(
                        [0.0] * valid_count
                    )

                except Exception:

                    print(
                        "Batch inference failed"
                    )

                    traceback.print_exc()

                    # fallback
                    all_scores.extend(
                        [0.0] * valid_count
                    )

                # =============================================
                # release image handle
                # =============================================

                for img in input_imgs:

                    try:
                        img.close()
                    except:
                        pass

                # =============================================
                # gc
                # =============================================

                torch.cuda.empty_cache()

                gc.collect()

            # =================================================
            # done
            # =================================================

            print(
                f"\nFinished request "
                f"total={len(all_scores)}"
            )

            return {
                "scores": all_scores
            }

        # =====================================================
        # request fail
        # =====================================================

        except Exception as e:

            print("\nRequest failed")

            traceback.print_exc()

            torch.cuda.empty_cache()

            gc.collect()

            return {
                "error": str(e)
            }