import os
import gc
import torch
from diffusers import StableDiffusion3Pipeline

# =========================================================
# fixed path
# =========================================================

MODEL_PATH = "/disk3/model/stable-diffusion-3.5-medium"

LORA_PATH = "./sd_lora/keywords_caption_description/pytorch_lora_weights.safetensors"

# =========================================================
# generation config
# =========================================================

LORA_SCALE = 1.0

HEIGHT = 1024

WIDTH = 1024

GUIDANCE_SCALE = 7.0

NUM_INFERENCE_STEPS = 28

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, bad anatomy"
)

# =========================================================
# load sd
# =========================================================
def load_sd(
    MODEL_PATH,
    LORA_PATH=None,
    USE_LORA=True,
    LORA_SCALE=1.0,
    device="cuda:1",
):
    print("Loading SD3.5 Pipeline...")
    dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    if USE_LORA:
        print(f"Loading LoRA: {LORA_PATH}")
        pipe.load_lora_weights(
            LORA_PATH
        )
        adapter_names = (
            pipe.get_active_adapters()
        )
        pipe.set_adapters(
            adapter_names,
            adapter_weights=[
                LORA_SCALE
            ] * len(adapter_names)
        )
    else:
        print("Using Base SD3.5 Only")
    print("SD3.5 Loaded.")

    return pipe

# =========================================================
# build prompt
# =========================================================

def build_prompt(
    keywords,
    description,
):
    if keywords:
        prompt = f"""
    Keywords:
    {keywords}
    Description:
    {description}
    """
    else:
        prompt = f""""
    Description:
    {description}
    """

    return prompt.strip()

# =========================================================
# generate images
# =========================================================

def generate_images(
    model,
    keywords,
    description,
    num_images,
    save_dir,
    seed,
    use_keywords=True
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    if not use_keywords:
        keywords = ""

    prompt = build_prompt(
        keywords=keywords,
        description=description,
    )

    # =========================================================
    # remove old images
    # =========================================================

    for file_name in os.listdir(save_dir):

        if (
                file_name.endswith(".jpg")
                or file_name.endswith(".png")
                or file_name.endswith(".jpeg")
        ):

            try:

                os.remove(
                    os.path.join(
                        save_dir,
                        file_name
                    )
                )

            except Exception as e:

                print(
                    f"Delete Failed: {file_name}"
                )

                print(e)
    saved_paths = []

    for idx in range(num_images):

        current_seed = seed + idx

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(current_seed)

        save_path = os.path.join(
            save_dir,
            f"{idx}.jpg"
        )

        try:

            with torch.no_grad():

                image = model(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=HEIGHT,
                    width=WIDTH,
                    guidance_scale=GUIDANCE_SCALE,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    generator=generator,
                ).images[0]

            image.save(save_path)

            saved_paths.append(save_path)

        except Exception as e:

            print(f"Generate Failed: {save_path}")

            print(e)

        torch.cuda.empty_cache()

        gc.collect()

    return saved_paths