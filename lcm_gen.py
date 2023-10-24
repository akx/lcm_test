import dataclasses
import os
import random
import sys
from contextlib import contextmanager
from functools import partial
from unittest.mock import patch

import torch
from PIL.Image import Image
from diffusers import DiffusionPipeline


def get_best_device() -> str:
    if dev := os.environ.get("LCM_DEVICE"):
        return dev
    if sys.platform == "darwin":
        try:
            torch.mps.current_allocated_memory()
            return "mps"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclasses.dataclass
class Result:
    seed: int
    images: list[Image]


@contextmanager
def default_progress_context(info: str):
    print("Starting", info)
    yield
    print("Finished", info)


class LCMGenerator:
    def __init__(
        self,
        *,
        device: str | None = None,
        fp16=False,
        progress_context=default_progress_context,
    ):
        if not device:
            device = get_best_device()
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        self._pipe = None
        self.progress_context = progress_context

    @property
    def pipe(self):
        if not self._pipe:
            with self.progress_context("Initializing pipeline..."):
                pipe = DiffusionPipeline.from_pretrained(
                    "SimianLuo/LCM_Dreamshaper_v7",
                    custom_pipeline="latent_consistency_txt2img",
                    custom_revision="main",
                )
                pipe.to(torch_device=self.device, torch_dtype=self.dtype)
                pipe.safety_checker = None  # we're all adults here
            self._pipe = pipe
        return self._pipe

    def generate(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        batch_size: int = 1,
        seed: int | None = None,
    ) -> Result:
        pipe: DiffusionPipeline = self.pipe
        with self.progress_context(
            f"Generating {batch_size} images ({self.device}, {self.dtype})...",
        ):
            if not seed or seed <= 0:
                seed = random.randint(0, 2**32)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            with patch("torch.randn", partial(torch.randn, generator=gen)):
                images = pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    lcm_origin_steps=50,
                    output_type="pil",
                    num_images_per_prompt=batch_size,
                ).images
            return Result(seed=seed, images=images)
