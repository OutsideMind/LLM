import gc
import uuid
from typing import List, Dict, Optional
import torch
from PIL import Image
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
import streamlit as st
from config import *

class ImageGenerator:
    """Main class for handling image generation pipelines with LoRA support"""
    def __init__(self):
        self.text2img_pipe = None # Text-to-image pipeline
        self.img2img_pipe = None # Image-to-image pipeline
        self.loaded_loras: List[Dict] = [] # Active LoRA adapters
        self.load_models() # Initialize pipelines on creation

    def load_models(self):
        """Initialize SDXL pipelines with LoRA adapters and memory optimizations"""
        if torch.cuda.is_available():
            try:
                # Cleanup previous pipeline instances to free memory
                if self.text2img_pipe is not None:
                    del self.text2img_pipe
                if self.img2img_pipe is not None:
                    del self.img2img_pipe

                # Clear GPU memory and perform garbage collection
                torch.cuda.empty_cache()
                gc.collect()

                # Core configuration for Stable Diffusion pipeline
                model_kwargs = {
                    "torch_dtype": torch.float16,  # Use half-precision for memory efficiency
                    "safety_checker": None,        # Disable NSFW filtering
                    "local_files_only": True,      # Ensure offline operation
                    "use_safetensors": True,       # Use safer tensor format
                    "device_map": "auto",          # Automatic GPU allocation
                    "variant": "fp16",             # Use fp16 model variant
                    "add_watermarker": False       # Disable automatic watermarking
                }

                # Initialize base text-to-image pipeline
                self.text2img_pipe = StableDiffusionXLPipeline.from_single_file(
                    SD_MODEL_DIR, **model_kwargs
                )

                # Reset existing LoRA adapters before loading new ones
                if hasattr(self.text2img_pipe, "unfuse_lora"):
                    self.text2img_pipe.unfuse_lora()

                # Load and register all active LoRA adapters
                adapter_names = []
                adapter_weights = []
                for lora in self.loaded_loras:
                    self.text2img_pipe.load_lora_weights(
                        lora["path"],
                        adapter_name=lora["name"]
                    )
                    adapter_names.append(lora["name"])
                    adapter_weights.append(lora["weight"])

                # Activate and fuse multiple LoRA adapters
                if adapter_names:
                    self.text2img_pipe.set_adapters(
                        adapter_names,
                        adapter_weights=adapter_weights
                    )
                    self.text2img_pipe.fuse_lora()  # Merge adapters for better performance

                # Apply memory optimization techniques
                self.text2img_pipe.enable_xformers_memory_efficient_attention()  # Optimize attention layers
                self.text2img_pipe.enable_attention_slicing()  # Reduce VRAM usage
                self.text2img_pipe.enable_model_cpu_offload()  # Automatic GPU-CPU swapping

                # Initialize image-to-image pipeline using shared components
                self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
                    **self.text2img_pipe.components
                )
                # Apply same optimizations to img2img pipeline
                self.img2img_pipe.enable_xformers_memory_efficient_attention()
                self.img2img_pipe.enable_model_cpu_offload()

                # Configure scheduler with quality improvements
                self.text2img_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                    self.text2img_pipe.scheduler.config,
                    use_karras_sigmas=True,       # Enable Karras noise schedule
                    timestep_spacing="trailing"   # Better for few-step generations
                )

            except Exception as e:
                st.error(f"Model loading error: {str(e)}")
                self.text2img_pipe = None
                torch.cuda.empty_cache()  # Cleanup after failure
        else:
            st.error("CUDA unavailable - Image generation requires NVIDIA GPU")

    def add_lora(self, lora_name: str, weight: float = 1.0):
        """Add a LoRA adapter to the pipeline and reload models

        Args:
            lora_name: Key from LORA_MODELS configuration
            weight: Influence strength (0.0-2.0, 1.0=default)

        Raises:
            ValueError: For unregistered LoRA names
            FileNotFoundError: If model file is missing
        """
        # Validate LoRA registration
        if lora_name not in LORA_MODELS:
            raise ValueError(f"LoRA {lora_name} not in config (Available: {', '.join(LORA_MODELS.keys())})")

        # Get model filename from config
        lora_filename = LORA_MODELS[lora_name]["filename"]
        lora_path = os.path.join(LORA_DIR, lora_filename)

        # Verify physical file existence
        if not os.path.exists(lora_path):
            raise FileNotFoundError(
                f"Missing LoRA file: {lora_path}\n"
                f"1. Place file in {LORA_DIR}\n"
                f"2. Verify filename matches config"
            )

        # Register adapter and trigger reload
        self.loaded_loras.append({
            "name": lora_name,  # Configuration key
            "path": lora_path,  # Full filesystem path
            "weight": max(0.0, min(weight, 2.0))  # Clamped weight value
        })
        self.load_models()  # Reinitialize pipeline with new adapter

    def remove_lora(self, lora_name: str):
        """Remove a LoRA adapter and reload the pipeline

        Args:
            lora_name: Registered LoRA identifier to remove
        """
        # Filter out the specified LoRA and maintain others
        self.loaded_loras = [lora for lora in self.loaded_loras
                             if lora["name"] != lora_name]
        self.load_models()  # Reinitialize with updated adapters

    def update_lora_weights(self):
        """Force reload models to apply updated LoRA weights

        Handles cases where dynamic weight changes aren't
        automatically detected by the pipeline"""
        try:
            self.load_models()  # Full pipeline reinitialization
            # Log updated weights to console
            weight_report = ", ".join(
                [f"{l['name']}:{l['weight']:.1f}"
                 for l in self.loaded_loras]
            )
            print(f"Weights updated: [{weight_report}]")
        except Exception as e:
            st.error(f"Weight update failed: {str(e)}")
            # Maintain previous working state on failure

    def _create_generator(self, seed=None) -> tuple[torch.Generator, int]:
        """Create reproducible torch generator with seed management

        Args:
            seed: Optional manual seed value
        Returns:
            tuple: (Configured generator, seed used)
        """
        # Initialize CUDA-based generator
        generator = torch.Generator(device="cuda")

        # Seed handling logic
        used_seed = seed if seed is not None else generator.seed()
        generator.manual_seed(used_seed)  # Ensure reproducibility

        return generator, used_seed

    def generate_image(self, **kwargs) -> tuple[Optional[str], Optional[int]]:
        """Generate image from text prompt with quality enhancements

        Args:
            **kwargs: Generation parameters including:
                prompt (str): Text description
                negative_prompt (str): Negative constraints
                seed (int): Optional reproducibility seed
                steps (int): Inference steps (35-50)
                guidance_scale (float): Prompt adherence (5.0-9.0)
                width/height (int): Output dimensions

        Returns:
            tuple: (image_path, seed) or (None, None) on failure
        """
        try:
            # Initialize generator with optional seed
            generator, used_seed = self._create_generator(kwargs.get("seed"))

            # Build generation config with quality boosters
            config = {
                "prompt": f"{kwargs.get('prompt', '')}, masterpiece, best quality, detailed",
                "negative_prompt": f"{kwargs.get('negative_prompt', '')}, low quality, worst quality, bad anatomy",
                "num_inference_steps": kwargs.get("steps", 35),  # Optimal for Euler a
                "guidance_scale": kwargs.get("guidance_scale", 6.0),  # Balanced creativity/accuracy
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024),
                "generator": generator
            }

            torch.cuda.empty_cache()  # Prevent VRAM fragmentation
            result = self.text2img_pipe(**config)
            return self._save_image(result.images[0]), used_seed

        except Exception as e:
            self._handle_error(f"Text-to-image error: {str(e)}")
            return None, None

    def generate_image_from_image(self, **kwargs) -> tuple[Optional[str], Optional[int]]:
        """Generate image variation from input image

        Args:
            **kwargs: Includes:
                init_image (file): Base image for variation
                strength (float): 0-1 modification strength
                Other params same as generate_image()

        Returns:
            tuple: (image_path, seed) or (None, None)
        """
        try:
            generator, used_seed = self._create_generator(kwargs.get("seed"))

            # Process input image
            init_image = Image.open(kwargs["init_image"]).convert("RGB")
            init_image = init_image.resize(
                (kwargs.get("width", 768), kwargs.get("height", 768)) # Default SDXL size
            )

            config = {
                "prompt": kwargs.get("prompt", ""),
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "image": init_image,
                "strength": kwargs.get("strength", 0.7),  # Balanced modification
                "num_inference_steps": kwargs.get("steps", 30),
                "guidance_scale": kwargs.get("guidance_scale", 11.0),  # Higher for img2img
                "generator": generator,
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1024)
            }

            torch.cuda.empty_cache()
            result = self.img2img_pipe(**config)
            return self._save_image(result.images[0]), used_seed

        except Exception as e:
            self._handle_error(f"Image-to-image error: {str(e)}")
            return None, None

    def _save_image(self, image) -> str:
        """Save generated image with UUID filename

        Args:
            image: PIL Image object

        Returns:
            str: Full path to saved PNG file
        """
        os.makedirs(SD_SAVE_PATH, exist_ok=True)  # Ensure output dir exists
        filename = os.path.join(SD_SAVE_PATH, f"{uuid.uuid4()}.png")  # Unique name
        try:
            image.save(filename, format="PNG", optimize=True)  # Lossless compression
            return filename
        except Exception as e:
            self._handle_error(f"Save failed: {str(e)}")
            return ""

    def _handle_error(self, message: str) -> None:
        """Central error handler with cleanup

        Args:
            message: Error description
        """
        torch.cuda.empty_cache()  # Free GPU memory
        st.error(message)  # User-facing alert
        print(f"[ERROR] {message}")  # Debug logging