"""Streamlit interface for training and managing LoRA models."""

import streamlit as st
import time
import torch
from typing import Dict, List, Optional
from pathlib import Path
from src.config import TRAINING_DATA_PATH, SD_MODEL_NAME, LORA_MODELS
from src.model_trainer import ModelTrainer


def render_training_interface():
    """Render the main interface for creating and managing LoRA models.

    This function handles the main training dashboard, including GPU availability checks,
    model initialization, and tab navigation.
    """
    st.header("🔄 Model Training Dashboard")

    # Check GPU availability
    if not torch.cuda.is_available():
        st.error("⚠️ CUDA-compatible GPU required")
        st.info("Training requires an NVIDIA GPU")
        return

    # Initialize trainer
    if "model_trainer" not in st.session_state:
        if "image_generator" in st.session_state and st.session_state.image_generator.text2img_pipe:
            st.session_state.model_trainer = ModelTrainer(
                st.session_state.image_generator.text2img_pipe
            )
        else:
            st.warning("Please load the image generation model first")
            return

    # Check pipeline readiness
    if not st.session_state.model_trainer.is_ready():
        st.warning("Image model not loaded!")
        if st.button("Load Default Model"):
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
    """Render the tab for training new LoRA models.

    This includes the training form, parameter configuration, and training initiation.
    """
    st.subheader("Create New LoRA Model")

    with st.form("training_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            lora_name = st.text_input(
                "Model Name",
                value="my_style",
                help="Unique name for your model"
            )
            placeholder = st.text_input(
                "Trigger Phrase",
                value="<my-style>",
                help="Token to activate the style in prompts"
            )

            # Select existing model for fine-tuning
            available_loras = get_available_loras()
            target_lora = st.selectbox(
                "Continue Training (optional)",
                options=[""] + list(available_loras.keys()),
                format_func=lambda x: x if x else "-- Create New --",
                help="Select existing LoRA for fine-tuning"
            )

        with col2:
            # Advanced training parameters
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
                help="Images in consistent style for training"
            )

        # Training start button
        submit = st.form_submit_button("Start Training")

        if submit:
            validate_and_start_training(
                lora_name, placeholder, uploaded_images,
                epochs, lr, lora_rank, lora_alpha,
                batch_size, resolution, target_lora
            )

    # Display training progress
    render_training_progress()


def render_cloning_tab():
    """Render the tab for cloning existing LoRA models.

    Allows users to create copies of existing models for experimentation.
    """
    st.subheader("Clone Existing LoRA Model")
    st.info("Create a copy of an existing model for experimentation")

    available_loras = get_available_loras()

    if not available_loras:
        st.warning("No models available for cloning")
        return

    with st.form("clone_form"):
        source_lora = st.selectbox(
            "Source Model",
            options=list(available_loras.keys()),
            help="Model to clone"
        )

        new_name = st.text_input(
            "New Model Name",
            value=f"{source_lora}_copy",
            help="Unique name for the new model"
        )

        new_placeholder = st.text_input(
            "New Trigger Phrase",
            value=f"<{source_lora}-copy>",
            help="New activation token"
        )

        if st.form_submit_button("Clone Model"):
            if new_name in available_loras:
                st.error(f"Model '{new_name}' already exists!")
            else:
                with st.spinner("Cloning model..."):
                    success = st.session_state.image_generator.clone_lora(
                        source_name=source_lora,
                        new_name=new_name,
                        new_placeholder=new_placeholder
                    )

                    if success:
                        st.success(f"Model {source_lora} cloned as {new_name}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()


def render_management_tab():
    """Render the model management tab.

    Displays available models and provides deletion functionality.
    """
    st.subheader("LoRA Model Management")
    available_loras = get_available_loras()

    if not available_loras:
        st.info("No models created yet")
        return

    # Model table
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
                # Prevent deletion of preset models
                if model_name not in LORA_MODELS:
                    if st.button("🗑️", key=f"del_{model_name}", help="Delete model"):
                        delete_lora_model(model_name)


def render_training_progress():
    """Display the current training progress.

    Shows progress bar, loss chart, and provides stop training functionality.
    """
    if hasattr(st.session_state, "training_started") and st.session_state.training_started:
        trainer = st.session_state.model_trainer
        progress = trainer.get_training_progress()

        st.subheader("Training Progress")

        # Progress bar and metrics
        progress_col, metrics_col = st.columns([3, 1])

        with progress_col:
            st.progress(progress["progress"])

            if progress["loss_history"]:
                st.line_chart({"Loss": progress["loss_history"]})

        with metrics_col:
            st.metric("Epoch", f"{progress['epoch']}")
            st.metric("Current LoRA", progress.get("current_lora", ""))

        # Stop button
        if st.button("⛔ Stop Training"):
            trainer.stop_training()
            st.session_state.training_started = False
            st.rerun()


def get_available_loras() -> Dict:
    """Get available LoRA models.

    Returns:
        Dict: Dictionary of available LoRA models with their configurations.
    """
    if "image_generator" in st.session_state:
        return st.session_state.image_generator.all_loras
    return {}


def delete_lora_model(model_name: str):
    """Delete a LoRA model with confirmation.

    Args:
        model_name (str): Name of the model to delete.
    """
    # Prevent deletion of preset models
    if model_name in LORA_MODELS:
        st.error("Cannot delete preset models!")
        return

    if st.session_state.image_generator.delete_dynamic_lora(model_name):
        st.success(f"Model {model_name} deleted!")
        time.sleep(1)
        st.rerun()
    else:
        st.error("Error deleting model")


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
    """Validate inputs and start the training process.

    Args:
        lora_name (str): Name for the new model
        placeholder (str): Trigger phrase for the model
        uploaded_images (List): List of uploaded training images
        epochs (int): Number of training epochs
        lr (float): Learning rate
        lora_rank (int): LoRA rank parameter
        lora_alpha (int): LoRA alpha parameter
        batch_size (int): Training batch size
        resolution (int): Image resolution for training
        target_lora (Optional[str]): Existing model to fine-tune
    """
    # Input validation
    errors = []

    if not lora_name:
        errors.append("Please specify a model name")

    # Validate placeholder format
    if not placeholder.strip() or not placeholder.startswith("<") or not placeholder.endswith(">"):
        st.warning("⚠️ Trigger phrase should be in format <your-style>")
        placeholder = f"<{lora_name}-style>"  # Auto-correct
        st.info(f"Using auto-generated trigger: {placeholder}")

    if len(uploaded_images) < 3:
        errors.append("Upload at least 3 images")
    elif len(uploaded_images) > 30:
        errors.append("Too many images (max 30)")

    # Check for existing model name
    if lora_name in get_available_loras():
        errors.append(f"Model '{lora_name}' already exists")

    # Display errors
    if errors:
        for error in errors:
            st.error(error)
        return

    # Create temporary training directory
    session_id = str(int(time.time()))
    temp_folder = Path(TRAINING_DATA_PATH) / session_id
    temp_folder.mkdir(parents=True, exist_ok=True)

    # Save uploaded images
    for i, img in enumerate(uploaded_images):
        img_path = temp_folder / f"img_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(img.getbuffer())

    # Start training
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
            st.success("Training completed successfully!")
            st.balloons()
        else:
            st.error("Error during training")