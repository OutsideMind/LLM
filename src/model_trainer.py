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
from typing import Optional, Dict
from pathlib import Path
from copy import deepcopy


class ModelTrainer:
    def __init__(self, base_pipeline=None):
        self.training_in_progress = False
        self.progress = 0
        self.current_epoch = 0
        self.loss_history = []
        self.base_pipeline = base_pipeline
        self.train_unet = None
        self.placeholder_token = ""
        self.current_lora_name = ""
        self.callbacks = []

        Path(LORA_DIR).mkdir(exist_ok=True)
        Path(TRAINING_DATA_PATH).mkdir(exist_ok=True)

    def is_ready(self) -> bool:
        return self.base_pipeline is not None and self.base_pipeline.text_encoder is not None

    def set_base_pipeline(self, pipeline) -> bool:
        self.base_pipeline = pipeline
        return self.is_ready()

    def add_callback(self, callback: callable):
        self.callbacks.append(callback)

    def setup_lora(self, placeholder_token: str, lora_rank: int = 8,
                   lora_alpha: int = 32, target_lora: Optional[str] = None) -> bool:
        """Настройка LoRA с улучшенной совместимостью архитектур"""
        if not self.is_ready():
            st.error("Базовый пайплайн не загружен или неполный!")
            return False

        try:
            # Очистка предыдущих ресурсов
            self._cleanup_resources()

            # Конфигурация LoRA
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=0.05,
                bias="none"
            )

            # Используем копию UNet из пайплайна
            self.train_unet = deepcopy(self.base_pipeline.unet)

            # Применяем LoRA
            self.train_unet.add_adapter(lora_config)

            # Загрузка весов если есть существующая модель
            if target_lora:
                lora_path = Path(LORA_DIR) / f"{target_lora}.safetensors"
                if lora_path.exists():
                    from safetensors.torch import load_file
                    lora_weights = load_file(lora_path)

                    # Применяем только веса LoRA
                    for name, param in self.train_unet.named_parameters():
                        if name in lora_weights:
                            param.data.copy_(lora_weights[name])

            self.train_unet.train()
            self.train_unet.to("cuda")

            # Оптимизации памяти
            self.train_unet.enable_gradient_checkpointing()
            torch.backends.cuda.matmul.allow_tf32 = True

            self.placeholder_token = placeholder_token
            return True
        except Exception as e:
            st.error(f"LoRA setup error: {str(e)}")
            return False

    def create_lora_from_existing(self, source_lora: str, new_name: str, new_placeholder: str) -> bool:
        """Создание новой LoRA на основе существующей"""
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

            # Регистрируем новую модель в ImageGenerator
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

    def train_model(self, dataset_path: str, placeholder_token: str,
                    lora_name: str, epochs: int = 50, lr: float = 1e-4,
                    batch_size: int = 1, resolution: int = 768,
                    lora_rank: int = 8, lora_alpha: int = 32,
                    target_lora: Optional[str] = None) -> bool:
        """Финальная версия метода обучения с исправлением всех ошибок"""
        try:
            # Проверка доступности GPU
            if not torch.cuda.is_available():
                st.error("Обучение требует GPU! Проверьте ваше окружение.")
                return False

            # Проверка готовности пайплайна
            if not self.is_ready():
                st.error("Базовый пайплайн не готов к обучению!")
                return False

            self.current_lora_name = lora_name
            if not self.setup_lora(placeholder_token, lora_rank, lora_alpha, target_lora):
                return False

            self.training_in_progress = True
            self.current_epoch = 0
            self.loss_history = []

            # Подготовка данных
            dataset = self._create_dataset(dataset_path, size=resolution)
            if not dataset:
                st.error("Не удалось создать датасет для обучения")
                return False

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )

            # Оптимизатор
            optimizer = torch.optim.AdamW(
                self.train_unet.parameters(),
                lr=lr
            )

            # Шумовой scheduler
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear"
            )

            # Цикл обучения
            total_steps = epochs * len(dataloader)
            current_step = 0

            # Получим необходимые компоненты
            device = self.train_unet.device
            dtype = self.train_unet.dtype
            vae = self.base_pipeline.vae
            tokenizers = [self.base_pipeline.tokenizer]
            text_encoders = [self.base_pipeline.text_encoder]

            # Проверяем наличие второго текстового энкодера (для SDXL)
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

                    # Проверка наличия данных в батче
                    if not batch or "pixel_values" not in batch or "input_ids" not in batch:
                        st.warning("Пропущен пустой батч")
                        continue

                    # Перенос данных на устройство
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)

                    # Преобразование изображений в латентное пространство
                    with torch.no_grad():
                        # Определяем тип данных VAE
                        vae_dtype = next(vae.parameters()).dtype
                        pixel_values = pixel_values.to(vae_dtype)

                        # Кодируем изображения в латентное пространство
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        # Подготовка текстовых входов
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

                        # Получение текстовых эмбеддингов
                        text_embeddings_list = []
                        for i, text_encoder in enumerate(text_encoders):
                            encoder_output = text_encoder(text_inputs[i])

                            # Обработка разных форматов вывода
                            if hasattr(encoder_output, 'last_hidden_state'):
                                # Современный формат: объект с атрибутами
                                text_embedding = encoder_output.last_hidden_state
                            elif isinstance(encoder_output, tuple):
                                # Старый формат: кортеж (last_hidden_state, ...)
                                text_embedding = encoder_output[0]
                            else:
                                # Неизвестный формат
                                raise ValueError(f"Неподдерживаемый формат вывода энкодера: {type(encoder_output)}")

                            # Приведение к 3D тензору
                            if len(text_embedding.shape) == 2:
                                text_embedding = text_embedding.unsqueeze(1)

                            text_embeddings_list.append(text_embedding)

                        # Объединение эмбеддингов
                        if len(text_embeddings_list) > 1:
                            # Для SDXL: объединяем по последнему измерению
                            text_embeddings = torch.cat(text_embeddings_list, dim=-1)
                        else:
                            text_embeddings = text_embeddings_list[0]

                        # Получение pooled_output
                        if len(text_encoders) > 1:
                            # Для второго энкодера
                            if hasattr(encoder_output, 'pooler_output'):
                                pooled_output = encoder_output.pooler_output
                            elif isinstance(encoder_output, tuple) and len(encoder_output) > 1:
                                pooled_output = encoder_output[1]
                            else:
                                # Создаем заглушку
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

                    # Генерация шума в латентном пространстве
                    noise = torch.randn_like(latents, device=device)
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps,
                        (latents.shape[0],), device=device
                    ).long()

                    # Добавление шума к латентам
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Прямой проход
                    with torch.autocast(device.type, enabled=True):
                        # Подготовка additional conditions для SDXL
                        original_size = (resolution, resolution)
                        crops_coords_top_left = (0, 0)
                        target_size = (resolution, resolution)

                        # Создаем add_time_ids
                        add_time_ids = list(original_size + crops_coords_top_left + target_size)
                        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
                        add_time_ids = add_time_ids.repeat(noisy_latents.shape[0], 1)

                        added_cond_kwargs = {
                            "text_embeds": pooled_output,
                            "time_ids": add_time_ids
                        }

                        # Вызов UNet
                        noise_pred = self.train_unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=text_embeddings,
                            added_cond_kwargs=added_cond_kwargs
                        ).sample

                        loss = torch.nn.functional.mse_loss(noise_pred, noise)

                    # Обратное распространение
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.train_unet.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    current_step += 1
                    batch_count += 1
                    self.progress = current_step / total_steps

                    # Вызов callback-функций
                    for callback in self.callbacks:
                        callback(self)

                # Статистика эпохи
                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    self.loss_history.append(avg_loss)
                    print(f"Epoch {self.current_epoch} - Loss: {avg_loss:.4f}")
                else:
                    st.warning(f"Epoch {self.current_epoch} - No batches processed")

            # Сохранение весов LoRA
            self._save_lora_weights(lora_name)
            return True

        except Exception as e:
            st.error(f"Ошибка обучения: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
        finally:
            # Очистка ресурсов
            self.training_in_progress = False
            self._cleanup_resources()

    def _create_sdxl_dataset(self, image_folder: str, size: int = 768) -> Optional[Dataset]:
        """Создание датасета с обработкой изображений для SDXL"""
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
            st.error(f"Ошибка создания датасета: {str(e)}")
            return None

    def _create_dataset(self, image_folder: str, size: int = 768) -> Optional[Dataset]:
        """Создание датасета с обработкой изображений"""
        try:
            # Проверка существования папки
            if not os.path.exists(image_folder):
                st.error(f"Папка с данными не существует: {image_folder}")
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
            st.error(f"Ошибка создания датасета: {str(e)}")
            return None

    def _save_lora_weights(self, lora_name: str):
        """Сохранение весов LoRA"""
        try:
            lora_weights = {}
            for name, param in self.train_unet.named_parameters():
                if "lora" in name and param.requires_grad:
                    lora_weights[name] = param.detach().cpu()

            lora_filename = f"{lora_name}.safetensors"
            lora_path = Path(LORA_DIR) / lora_filename
            save_file(lora_weights, lora_path)

            # Регистрация новой модели
            self._register_new_lora(
                name=lora_name,
                filename=lora_filename,
                placeholder=self.placeholder_token,
                description=f"Trained on {time.strftime('%d.%m.%Y')}"
            )
        except Exception as e:
            st.error(f"Ошибка сохранения весов: {str(e)}")

    def _register_new_lora(self, name: str, filename: str, placeholder: str, description: str = ""):
        """Регистрирует новую LoRA модель в системе"""
        try:
            # Проверяем, что image_generator существует в сессии
            if "image_generator" not in st.session_state:
                st.error("ImageGenerator не инициализирован!")
                return False

            # Добавляем в ImageGenerator
            st.session_state.image_generator.add_dynamic_lora(
                name=name,
                filename=filename,
                placeholder=placeholder,
                description=description
            )
            return True
        except Exception as e:
            st.error(f"Ошибка регистрации модели: {str(e)}")
            return False

    def _cleanup_resources(self):
        """Очистка ресурсов и освобождение памяти"""
        self.training_in_progress = False
        if self.train_unet is not None:
            del self.train_unet
            self.train_unet = None
        torch.cuda.empty_cache()
        gc.collect()

    def get_training_progress(self) -> Dict:
        """Возвращает текущий прогресс обучения"""
        return {
            "progress": self.progress,
            "epoch": self.current_epoch,
            "loss_history": self.loss_history,
            "training": self.training_in_progress,
            "current_lora": self.current_lora_name
        }

    def stop_training(self):
        """Остановка текущего процесса обучения"""
        self.training_in_progress = False
        self._cleanup_resources()