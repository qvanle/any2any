import os
import torch
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path

from diffusers import FluxTransformer2DModel, FluxPipeline, FluxInpaintPipeline, AutoencoderKL
from diffusers.hooks import apply_group_offloading
from transformers import T5EncoderModel, CLIPTextModel
from src.pipeline_tryon import FluxTryonPipeline, crop_to_multiple_of_16, resize_and_pad_to_size, resize_by_height

class Predictor(BasePredictor):
    def setup(self):
        model_path = "black-forest-labs/FLUX.1-dev"
        lora_name = "dev_lora_any2any_alltasks.safetensors"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs("weights", exist_ok=True)

        text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype, cache_dir="weights")
        text_encoder_2 = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=torch_dtype, cache_dir="weights")
        transformer = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=torch_dtype, cache_dir="weights")
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch_dtype, cache_dir="weights")

        pipe = FluxTryonPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            torch_dtype=torch_dtype,
            cache_dir="weights",
        )
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        pipe.enable_model_cpu_offload()
        pipe.load_lora_weights(
            "loooooong/Any2anyTryon",
            weight_name=lora_name,
            adapter_name="tryon",
        )
        pipe.remove_all_hooks()

        for module in [pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]:
            apply_group_offloading(
                module,
                offload_type="leaf_level",
                offload_device=torch.device("cpu"),
                onload_device=torch.device(device),
                use_stream=True,
            )

        pipe.to(device=device)
        self.pipe = pipe
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        model_image: Path = Input(description="Image of the person/model"),
        garment_image: Path = Input(description="Image of the garment"),
        prompt: str = Input(default="", description="Prompt to guide generation"),
        height: int = Input(default=576, description="Output height"),
        width: int = Input(default=576, description="Output width"),
        seed: int = Input(default=0, description="Random seed"),
        guidance_scale: float = Input(default=3.5, description="Guidance scale"),
        num_inference_steps: int = Input(default=30, description="Inference steps")
    ) -> Path:
        height, width = int(height), int(width)
        width = width - (width % 16)
        height = height - (height % 16)

        concat_image_list = [Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))]

        has_model_image = model_image is not None
        has_garment_image = garment_image is not None

        if has_model_image:
            model_image_pil = Image.open(model_image)
            if has_garment_image:
                input_height, input_width = model_image_pil.size[1], model_image_pil.size[0]
                model_image_pil, lp, tp, rp, bp = resize_and_pad_to_size(model_image_pil, width, height)
            else:
                model_image_pil = resize_by_height(model_image_pil, height)
            concat_image_list.append(model_image_pil)

        if has_garment_image:
            garment_image_pil = Image.open(garment_image)
            garment_image_pil = resize_by_height(garment_image_pil, height)
            concat_image_list.append(garment_image_pil)

        image = Image.fromarray(np.concatenate([np.array(img) for img in concat_image_list], axis=1))

        mask = np.zeros_like(np.array(image))
        mask[:, :width] = 255
        mask_image = Image.fromarray(mask)

        result = self.pipe(
            prompt,
            image=image,
            mask_image=mask_image,
            strength=1.0,
            height=height,
            width=image.width,
            target_width=width,
            tryon=has_model_image and has_garment_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="pil"
        ).images[0]

        if has_model_image and has_garment_image:
            result = result.crop((lp, tp, result.width - rp, result.height - bp)).resize((input_width, input_height))

        output_path = "/tmp/output.png"
        result.save(output_path)
        return Path(output_path)

