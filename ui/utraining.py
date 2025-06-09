import streamlit as st
import time
import torch
from typing import Dict, List, Optional
from pathlib import Path
from src.config import TRAINING_DATA_PATH, SD_MODEL_NAME, LORA_MODELS
from src.model_trainer import ModelTrainer


def render_training_interface():
    """Основной интерфейс для создания и управления LoRA моделями"""
    st.header("🔄 Model Training Dashboard")

    # Проверка доступности GPU
    if not torch.cuda.is_available():
        st.error("⚠️ Требуется GPU с поддержкой CUDA")
        st.info("Обучение невозможно без NVIDIA GPU")
        return

    # Инициализация тренера
    if "model_trainer" not in st.session_state:
        if "image_generator" in st.session_state and st.session_state.image_generator.text2img_pipe:
            st.session_state.model_trainer = ModelTrainer(
                st.session_state.image_generator.text2img_pipe
            )
        else:
            st.warning("Сначала загрузите модель генерации изображений")
            return

    # Проверка готовности пайплайна
    if not st.session_state.model_trainer.is_ready():
        st.warning("Модель изображений не загружена!")
        if st.button("Загрузить модель по умолчанию"):
            st.session_state.image_generator.load_models()
            st.rerun()
        return

    tab_train, tab_clone, tab_manage = st.tabs([
        "🏋️ Train New LoRA",
        "🧬 Clone Model",
        "🗂️ Manage Models"
    ])

    with tab_train:
        render_training_tab()

    with tab_clone:
        render_cloning_tab()

    with tab_manage:
        render_management_tab()


def render_training_tab():
    """Вкладка для обучения новых моделей"""
    st.subheader("Create New LoRA Model")

    with st.form("training_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            lora_name = st.text_input(
                "Model Name",
                value="my_style",
                help="Уникальное имя для вашей модели"
            )
            placeholder = st.text_input(
                "Trigger Phrase",
                value="<my-style>",
                help="Токен для активации стиля в промпте"
            )

            # Выбор существующей модели для дообучения
            available_loras = get_available_loras()
            target_lora = st.selectbox(
                "Continue Training (optional)",
                options=[""] + list(available_loras.keys()),
                format_func=lambda x: x if x else "-- Create New --",
                help="Выберите существующую LoRA для дообучения"
            )

        with col2:
            # Расширенные параметры обучения
            with st.expander("⚙️ Advanced Settings", expanded=True):
                epochs = st.slider("Epochs", 10, 200, 50)
                lr = st.slider(
                    "Learning Rate",
                    1e-6, 1e-3, 1e-4,
                    format="%.0e"
                )
                lora_rank = st.slider("LoRA Rank", 4, 64, 8)
                lora_alpha = st.slider("LoRA Alpha", 8, 128, 32)
                batch_size = st.selectbox("Batch Size", [1, 2, 4], index=0)
                resolution = st.select_slider(
                    "Resolution",
                    options=[512, 640, 768, 896, 1024],
                    value=768
                )

            uploaded_images = st.file_uploader(
                "Training Images (5-20 images)",
                type=["jpg", "png", "jpeg"],
                accept_multiple_files=True,
                help="Изображения в одном стиле для обучения"
            )

        # Кнопка запуска обучения
        submit = st.form_submit_button("Start Training")

        if submit:
            validate_and_start_training(
                lora_name, placeholder, uploaded_images,
                epochs, lr, lora_rank, lora_alpha,
                batch_size, resolution, target_lora
            )

    # Отображение прогресса обучения
    render_training_progress()


def render_cloning_tab():
    """Вкладка для клонирования моделей"""
    st.subheader("Clone Existing LoRA Model")
    st.info("Создайте копию существующей модели для экспериментов")

    available_loras = get_available_loras()

    if not available_loras:
        st.warning("Нет доступных моделей для клонирования")
        return

    with st.form("clone_form"):
        source_lora = st.selectbox(
            "Source Model",
            options=list(available_loras.keys()),
            help="Модель которую будем клонировать"
        )

        new_name = st.text_input(
            "New Model Name",
            value=f"{source_lora}_copy",
            help="Уникальное имя для новой модели"
        )

        new_placeholder = st.text_input(
            "New Trigger Phrase",
            value=f"<{source_lora}-copy>",
            help="Новый токен активации"
        )

        if st.form_submit_button("Clone Model"):
            if new_name in available_loras:
                st.error(f"Модель '{new_name}' уже существует!")
            else:
                with st.spinner("Cloning model..."):
                    success = st.session_state.image_generator.clone_lora(
                        source_name=source_lora,
                        new_name=new_name,
                        new_placeholder=new_placeholder
                    )

                    if success:
                        st.success(f"Модель {source_lora} успешно клонирована как {new_name}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()


def render_management_tab():
    """Вкладка для управления моделями"""
    st.subheader("LoRA Model Management")
    available_loras = get_available_loras()

    if not available_loras:
        st.info("Нет созданных моделей")
        return

    # Таблица с моделями
    st.write("### Available Models")
    for model_name, config in available_loras.items():
        with st.expander(f"📁 {model_name}"):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.json({
                    "Placeholder": config.get("placeholder", ""),
                    "Description": config.get("description", ""),
                    "Created": config.get("created", ""),
                    "Filename": config.get("filename", ""),
                    "Base Model": config.get("base_model", SD_MODEL_NAME)
                }, expanded=False)

            with col2:
                # Проверяем, что модель не является предустановленной
                if model_name not in LORA_MODELS:
                    if st.button("🗑️", key=f"del_{model_name}", help="Удалить модель"):
                        delete_lora_model(model_name)


def render_training_progress():
    """Отображение прогресса обучения"""
    if hasattr(st.session_state, "training_started") and st.session_state.training_started:
        trainer = st.session_state.model_trainer
        progress = trainer.get_training_progress()

        st.subheader("Training Progress")

        # Прогресс бар и метрики
        progress_col, metrics_col = st.columns([3, 1])

        with progress_col:
            st.progress(progress["progress"])

            if progress["loss_history"]:
                st.line_chart({"Loss": progress["loss_history"]})

        with metrics_col:
            st.metric("Epoch", f"{progress['epoch']}")
            st.metric("Current LoRA", progress.get("current_lora", ""))

        # Кнопка остановки
        if st.button("⛔ Stop Training"):
            trainer.stop_training()
            st.session_state.training_started = False
            st.rerun()


def get_available_loras() -> Dict:
    """Получение списка доступных LoRA моделей"""
    if "image_generator" in st.session_state:
        return st.session_state.image_generator.all_loras
    return {}


def delete_lora_model(model_name: str):
    """Удаление модели с подтверждением"""
    # Защита от удаления предустановленных моделей
    if model_name in LORA_MODELS:
        st.error("Нельзя удалять предустановленные модели!")
        return

    if st.session_state.image_generator.delete_dynamic_lora(model_name):
        st.success(f"Модель {model_name} удалена!")
        time.sleep(1)
        st.rerun()
    else:
        st.error("Ошибка при удалении модели")


def validate_and_start_training(
        lora_name: str,
        placeholder: str,
        uploaded_images: List,
        epochs: int,
        lr: float,
        lora_rank: int,
        lora_alpha: int,
        batch_size: int,
        resolution: int,
        target_lora: Optional[str] = None
):
    """Валидация и запуск процесса обучения"""
    # Проверка входных данных
    errors = []

    if not lora_name:
        errors.append("Укажите имя модели")

    # Проверка placeholder
    if not placeholder.strip() or not placeholder.startswith("<") or not placeholder.endswith(">"):
        st.warning("⚠️ Trigger phrase should be in format <your-style>")
        placeholder = f"<{lora_name}-style>"  # Автоматическое исправление
        st.info(f"Using auto-generated trigger: {placeholder}")

    if len(uploaded_images) < 3:
        errors.append("Загрузите минимум 3 изображения")
    elif len(uploaded_images) > 30:
        errors.append("Слишком много изображений (макс. 30)")

    # Проверка существования имени
    if lora_name in get_available_loras():
        errors.append(f"Модель '{lora_name}' уже существует")

    # Вывод ошибок
    if errors:
        for error in errors:
            st.error(error)
        return

    # Создание временной директории для обучения
    session_id = str(int(time.time()))
    temp_folder = Path(TRAINING_DATA_PATH) / session_id
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Сохранение изображений
    for i, img in enumerate(uploaded_images):
        img_path = temp_folder / f"img_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(img.getbuffer())

    # Запуск обучения
    with st.spinner("Starting training process..."):
        st.session_state.training_started = True

        success = st.session_state.model_trainer.train_model(
            dataset_path=str(temp_folder),
            placeholder_token=placeholder,
            lora_name=lora_name,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            resolution=resolution,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_lora=target_lora if target_lora else None
        )

        if success:
            st.success("Обучение успешно завершено!")
            st.balloons()
        else:
            st.error("Ошибка в процессе обучения")