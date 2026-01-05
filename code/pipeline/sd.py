# sd.py
import os
import gc
from typing import List
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image


class SDGenerator:
    """
    Stable Diffusion 3.5 Generator (with optional LoRA)
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        use_lora: bool = False,
        lora_path: str = None,
        lora_scale: float = 1.0,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.use_lora = use_lora

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)

        if use_lora:
            if lora_path is None:
                raise ValueError("use_lora=True but lora_path is None")

            self.pipeline.load_lora_weights(lora_path)

            adapter_names = self.pipeline.get_active_adapters()
            self.pipeline.set_adapters(
                adapter_names,
                adapter_weights=[lora_scale] * len(adapter_names)
            )

        self.pipeline.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        num_images: int,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        save_dir: str,
        filename_prefix: str,
    ) -> List[str]:
        """
        生成多张图片并返回路径列表
        """
        os.makedirs(save_dir, exist_ok=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        image_paths = []

        for i in range(num_images):
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            image: Image.Image = output.images[0]

            image_id = i + 1
            save_path = os.path.join(
                save_dir,
                f"{filename_prefix}_{image_id}.png"
            )

            image.save(save_path)
            image_paths.append(save_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return image_paths
