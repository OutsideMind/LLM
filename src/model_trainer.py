"""A module for training LoRA (Low-Rank Adaptation) models for diffusion pipelines."""

import time
import torch
import streamlit as st
import os
from diffusers import DDPMScheduler
from peft import LoraConfig
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from safetensors.torch import save_file
from src.config import LORA_DIR, TRAINING_DATA_PATH
import gc
from typing import Optional, Dict, List, Callable
from pathlib import Path
from copy import deepcopy


class ModelTrainer:
    """A class for training LoRA models on diffusion pipelines.

    Attributes:
        training_in_progress (bool): Flag indicating if training is in progress.
        progress (float): Current training progress (0 to 1).
        current_epoch (int): Current epoch number during training.
        loss_history (List[float]): History of loss values during training.
        base_pipeline: The base diffusion pipeline for training.
        train_unet: The UNet model being trained with LoRA.
        placeholder_token (str): The token used as placeholder in prompts.
        current_lora_name (str): Name of the currently training LoRA model.
        callbacks (List[Callable]): List of callback functions for training updates.
    """

    def __init__(self, base_pipeline=None):
        """Initialize the ModelTrainer.

        Args:
            base_pipeline: The base diffusion pipeline to train on (optional).
        """
        self.training_in_progress = False
        self.progress = 0.0
        self.current_epoch = 0
        self.loss_history = []
        self.base_pipeline = base_pipeline
        self.train_unet = None
        self.placeholder_token = ""
        self.current_lora_name = ""
        self.callbacks = []

        # Ensure directories exist
        Path(LORA_DIR).mkdir(exist_ok=True)
        Path(TRAINING_DATA_PATH).mkdir(exist_ok=True)

    def is_ready(self) -> bool:
        """Check if the trainer is ready for training.

        Returns:
            bool: True if base pipeline is loaded and ready, False otherwise.
        """
        return self.base_pipeline is not None and self.base_pipeline.text_encoder is not None

    def set_base_pipeline(self, pipeline) -> bool:
        """Set the base diffusion pipeline for training.

        Args:
            pipeline: The diffusion pipeline to use as base.

        Returns:
            bool: True if pipeline was set successfully, False otherwise.
        """
        self.base_pipeline = pipeline
        return self.is_ready()

    def add_callback(self, callback: Callable):
        """Add a callback function for training updates.

        Args:
            callback (Callable): Function to call during training updates.
        """
        self.callbacks.append(callback)

    def setup_lora(
        self,
        placeholder_token: str,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        target_lora: Optional[str] = None
    ) -> bool:
        """Configure LoRA with improved architectural compatibility.

        Args:
            placeholder_token (str): Token to use as placeholder in prompts.
            lora_rank (int): Rank of LoRA matrices. Defaults to 8.
            lora_alpha (int): Alpha parameter for LoRA. Defaults to 32.
            target_lora (Optional[str]): Existing LoRA model to load. Defaults to None.

        Returns:
            bool: True if setup was successful, False otherwise.
        """
        if not self.is_ready():
            st.error("Base pipeline is not loaded or incomplete!")
            return False

        try:
            # Clean up previous resources
            self._cleanup_resources()

            # LoRA configuration
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=0.05,
                bias="none"
            )

            # Use a copy of the pipeline's UNet
            self.train_unet = deepcopy(self.base_pipeline.unet)

            # Apply LoRA
            self.train_unet.add_adapter(lora_config)

            # Load weights if existing model is specified
            if target_lora:
                lora_path = Path(LORA_DIR) / f"{target_lora}.safetensors"
                if lora_path.exists():
                    from safetensors.torch import load_file
                    lora_weights = load_file(lora_path)

                    # Apply only LoRA weights
                    for name, param in self.train_unet.named_parameters():
                        if name in lora_weights:
                            param.data.copy_(lora_weights[name])

            self.train_unet.train()
            self.train_unet.to("cuda")

            # Memory optimizations
            self.train_unet.enable_gradient_checkpointing()
            torch.backends.cuda.matmul.allow_tf32 = True

            self.placeholder_token = placeholder_token
            return True
        except Exception as e:
            st.error(f"LoRA setup error: {str(e)}")
            return False

    def create_lora_from_existing(
        self,
        source_lora: str,
        new_name: str,
        new_placeholder: str
    ) -> bool:
        """Create a new LoRA model based on an existing one.

        Args:
            source_lora (str): Name of the source LoRA model.
            new_name (str): Name for the new LoRA model.
            new_placeholder (str): Placeholder token for the new model.

        Returns:
            bool: True if creation was successful, False otherwise.
        """
        try:
            source_path = Path(LORA_DIR) / f"{source_lora}.safetensors"
            if not source_path.exists():
                st.error(f"Source LoRA file not found: {source_path}")
                return False

            new_filename = f"{new_name}.safetensors"
            dest_path = Path(LORA_DIR) / new_filename

            # Load and save with new metadata
            from safetensors.torch import load_file, save_file
            weights = load_file(source_path)
            save_file(weights, dest_path)

            # Register the new model in ImageGenerator
            self._register_new_lora(
                name=new_name,
                filename=new_filename,
                placeholder=new_placeholder,
                description=f"Cloned from {source_lora}"
            )
            return True
        except Exception as e:
            st.error(f"Failed to clone LoRA: {str(e)}")
            return False

    def train_model(
        self,
        dataset_path: str,
        placeholder_token: str,
        lora_name: str,
        epochs: int = 50,
        lr: float = 1e-4,
        batch_size: int = 1,
        resolution: int = 768,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        target_lora: Optional[str] = None
    ) -> bool:
        """Train a LoRA model with the given parameters.

        Args:
            dataset_path (str): Path to the training dataset.
            placeholder_token (str): Token to use as placeholder in prompts.
            lora_name (str): Name for the trained LoRA model.
            epochs (int): Number of training epochs. Defaults to 50.
            lr (float): Learning rate. Defaults to 1e-4.
            batch_size (int): Batch size for training. Defaults to 1.
            resolution (int): Resolution for input images. Defaults to 768.
            lora_rank (int): Rank of LoRA matrices. Defaults to 8.
            lora_alpha (int): Alpha parameter for LoRA. Defaults to 32.
            target_lora (Optional[str]): Existing LoRA model to fine-tune. Defaults to None.

        Returns:
            bool: True if training was successful, False otherwise.
        """
        try:
            # Check GPU availability
            if not torch.cuda.is_available():
                st.error("Training requires GPU! Please check your environment.")
                return False

            # Check pipeline readiness
            if not self.is_ready():
                st.error("Base pipeline is not ready for training!")
                return False

            self.current_lora_name = lora_name
            if not self.setup_lora(placeholder_token, lora_rank, lora_alpha, target_lora):
                return False

            self.training_in_progress = True
            self.current_epoch = 0
            self.loss_history = []

            # Prepare dataset
            dataset = self._create_dataset(dataset_path, size=resolution)
            if not dataset:
                st.error("Failed to create training dataset")
                return False

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )

            # Optimizer
            optimizer = torch.optim.AdamW(
                self.train_unet.parameters(),
                lr=lr
            )

            # Noise scheduler
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )

            # Training loop
            total_steps = epochs * len(dataloader)
            current_step = 0

            # Get required components
            device = self.train_unet.device
            dtype = self.train_unet.dtype
            vae = self.base_pipeline.vae
            tokenizers = [self.base_pipeline.tokenizer]
            text_encoders = [self.base_pipeline.text_encoder]

            # Check for second text encoder (for SDXL)
            if hasattr(self.base_pipeline, 'text_encoder_2'):
                tokenizers.append(self.base_pipeline.tokenizer_2)
                text_encoders.append(self.base_pipeline.text_encoder_2)

            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                epoch_loss = 0.0
                batch_count = 0

                for batch in dataloader:
                    if not self.training_in_progress:
                        return False

                    # Check for empty batch
                    if not batch or "pixel_values" not in batch or "input_ids" not in batch:
                        st.warning("Skipping empty batch")
                        continue

                    # Move data to device
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)

                    # Convert images to latent space
                    with torch.no_grad():
                        # Determine VAE data type
                        vae_dtype = next(vae.parameters()).dtype
                        pixel_values = pixel_values.to(vae_dtype)

                        # Encode images to latent space
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        # Prepare text inputs
                        text_inputs = []
                        for tokenizer in tokenizers:
                            text_input = tokenizer(
                                [f"a photo in the style of {placeholder_token}"] * pixel_values.shape[0],
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt"
                            )
                            text_inputs.append(text_input.input_ids.to(device))

                        # Get text embeddings
                        text_embeddings_list = []
                        for i, text_encoder in enumerate(text_encoders):
                            encoder_output = text_encoder(text_inputs[i])

                            # Handle different output formats
                            if hasattr(encoder_output, 'last_hidden_state'):
                                # Modern format: object with attributes
                                text_embedding = encoder_output.last_hidden_state
                            elif isinstance(encoder_output, tuple):
                                # Old format: tuple (last_hidden_state, ...)
                                text_embedding = encoder_output[0]
                            else:
                                # Unknown format
                                raise ValueError(f"Unsupported encoder output format: {type(encoder_output)}")

                            # Convert to 3D tensor if needed
                            if len(text_embedding.shape) == 2:
                                text_embedding = text_embedding.unsqueeze(1)

                            text_embeddings_list.append(text_embedding)

                        # Combine embeddings
                        if len(text_embeddings_list) > 1:
                            # For SDXL: concatenate along last dimension
                            text_embeddings = torch.cat(text_embeddings_list, dim=-1)
                        else:
                            text_embeddings = text_embeddings_list[0]

                        # Get pooled output
                        if len(text_encoders) > 1:
                            # For second encoder
                            if hasattr(encoder_output, 'pooler_output'):
                                pooled_output = encoder_output.pooler_output
                            elif isinstance(encoder_output, tuple) and len(encoder_output) > 1:
                                pooled_output = encoder_output[1]
                            else:
                                # Create dummy output
                                pooled_output = torch.zeros(
                                    (pixel_values.shape[0], 1280),
                                    device=device,
                                    dtype=vae_dtype
                                )
                        else:
                            pooled_output = torch.zeros(
                                (pixel_values.shape[0], 1280),
                                device=device,
                                dtype=vae_dtype
                            )

                    # Generate noise in latent space
                    noise = torch.randn_like(latents, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps,
                        (latents.shape[0],), device=device
                    ).long()

                    # Add noise to latents
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Forward pass
                    with torch.autocast(device.type, enabled=True):
                        # Prepare additional conditions for SDXL
                        original_size = (resolution, resolution)
                        crops_coords_top_left = (0, 0)
                        target_size = (resolution, resolution)

                        # Create add_time_ids
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
                        add_time_ids = add_time_ids.repeat(noisy_latents.shape[0], 1)

                        added_cond_kwargs = {
                            "text_embeds": pooled_output,
                            "time_ids": add_time_ids
                        }

                        # Call UNet
                        noise_pred = self.train_unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=text_embeddings,
                            added_cond_kwargs=added_cond_kwargs
                        ).sample

                        loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.train_unet.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    current_step += 1
                    batch_count += 1
                    self.progress = current_step / total_steps

                    # Call callbacks
                    for callback in self.callbacks:
                        callback(self)

                # Epoch statistics
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    self.loss_history.append(avg_loss)
                    print(f"Epoch {self.current_epoch} - Loss: {avg_loss:.4f}")
                else:
                    st.warning(f"Epoch {self.current_epoch} - No batches processed")

            # Save LoRA weights
            self._save_lora_weights(lora_name)
            return True

        except Exception as e:
            st.error(f"Training error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
        finally:
            # Clean up resources
            self.training_in_progress = False
            self._cleanup_resources()

    def _create_sdxl_dataset(self, image_folder: str, size: int = 768) -> Optional[Dataset]:
        """Create a dataset with image processing for SDXL.

        Args:
            image_folder (str): Path to folder containing training images.
            size (int): Target size for images. Defaults to 768.

        Returns:
            Optional[Dataset]: The created dataset or None if failed.
        """
        try:
            class ImageDataset(Dataset):
                def __init__(self, folder, tokenizer, placeholder, size):
                    self.image_paths = [
                        os.path.join(folder, f) for f in os.listdir(folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]
                    self.transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.CenterCrop(size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])
                    self.tokenizer = tokenizer
                    self.placeholder = placeholder

                def __len__(self):
                    return len(self.image_paths)

                def __getitem__(self, idx):
                    img = Image.open(self.image_paths[idx]).convert("RGB")
                    prompt = f"a photo in the style of {self.placeholder}"
                    inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt"
                    )
                    return {
                        "pixel_values": self.transform(img),
                        "input_ids": inputs.input_ids[0]
                    }

            return ImageDataset(
                image_folder,
                self.base_pipeline.tokenizer,
                self.placeholder_token,
                size
            )
        except Exception as e:
            st.error(f"Dataset creation error: {str(e)}")
            return None

    def _create_dataset(self, image_folder: str, size: int = 768) -> Optional[Dataset]:
        """Create a dataset with image processing.

        Args:
            image_folder (str): Path to folder containing training images.
            size (int): Target size for images. Defaults to 768.

        Returns:
            Optional[Dataset]: The created dataset or None if failed.
        """
        try:
            # Check if folder exists
            if not os.path.exists(image_folder):
                st.error(f"Data folder does not exist: {image_folder}")
                return None

            class ImageDataset(Dataset):
                def __init__(self, folder, tokenizer, placeholder, size):
                    self.image_paths = [
                        os.path.join(folder, f) for f in os.listdir(folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ]
                    self.transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.CenterCrop(size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5])
                    ])
                    self.tokenizer = tokenizer
                    self.placeholder = placeholder

                def __len__(self):
                    return len(self.image_paths)

                def __getitem__(self, idx):
                    img = Image.open(self.image_paths[idx]).convert("RGB")
                    prompt = f"a photo in the style of {self.placeholder}"
                    inputs = self.tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        return_tensors="pt"
                    )
                    return {
                        "pixel_values": self.transform(img),
                        "input_ids": inputs.input_ids[0]
                    }

            return ImageDataset(
                image_folder,
                self.base_pipeline.tokenizer,
                self.placeholder_token,
                size
            )
        except Exception as e:
            st.error(f"Dataset creation error: {str(e)}")
            return None

    def _save_lora_weights(self, lora_name: str):
        """Save LoRA weights to file.

        Args:
            lora_name (str): Name for the LoRA model.
        """
        try:
            lora_weights = {}
            for name, param in self.train_unet.named_parameters():
                if "lora" in name and param.requires_grad:
                    lora_weights[name] = param.detach().cpu()

            lora_filename = f"{lora_name}.safetensors"
            lora_path = Path(LORA_DIR) / lora_filename
            save_file(lora_weights, lora_path)

            # Register the new model
            self._register_new_lora(
                name=lora_name,
                filename=lora_filename,
                placeholder=self.placeholder_token,
                description=f"Trained on {time.strftime('%d.%m.%Y')}"
            )
        except Exception as e:
            st.error(f"Error saving weights: {str(e)}")

    def _register_new_lora(
        self,
        name: str,
        filename: str,
        placeholder: str,
        description: str = ""
    ) -> bool:
        """Register a new LoRA model in the system.

        Args:
            name (str): Name of the model.
            filename (str): Filename of the model weights.
            placeholder (str): Placeholder token for the model.
            description (str): Description of the model. Defaults to "".

        Returns:
            bool: True if registration was successful, False otherwise.
        """
        try:
            # Check if image_generator exists in session
            if "image_generator" not in st.session_state:
                st.error("ImageGenerator is not initialized!")
                return False

            # Add to ImageGenerator
            st.session_state.image_generator.add_dynamic_lora(
                name=name,
                filename=filename,
                placeholder=placeholder,
                description=description
            )
            return True
        except Exception as e:
            st.error(f"Model registration error: {str(e)}")
            return False

    def _cleanup_resources(self):
        """Clean up resources and free memory."""
        self.training_in_progress = False
        if self.train_unet is not None:
            del self.train_unet
            self.train_unet = None
        torch.cuda.empty_cache()
        gc.collect()

    def get_training_progress(self) -> Dict:
        """Get current training progress.

        Returns:
            Dict: Dictionary containing training progress information.
        """
        return {
            "progress": self.progress,
            "epoch": self.current_epoch,
            "loss_history": self.loss_history,
            "training": self.training_in_progress,
            "current_lora": self.current_lora_name
        }

    def stop_training(self):
        """Stop the current training process."""
        self.training_in_progress = False
        self._cleanup_resources()