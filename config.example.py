import os

# API configuration for LM Studio
PORT = 0000  # LM Studio API server port (default: 1234)

# Path configurations
MODEL_DIR = r"models"  # Root directory for all model files
SD_MODEL_NAME = "Model name"
SD_MODEL_DIR = os.path.join(MODEL_DIR,"model.safetensors")  # Main Stable Diffusion model
LORA_DIR = os.path.join(MODEL_DIR, "LoRA")  # Directory for LoRA adapters (.safetensors files)
SD_SAVE_PATH = "generated_images"  # Output directory for generated images
TRAINING_DATA_PATH = "training_data"  # Dataset directory for model training
LM_STUDIO_PATH = r"path/to/LM Studio.exe"  # LM Studio executable location

# LoRA configuration (Stable Diffusion adapters)
LORA_MODELS = {
    "Image-Style": {
        "filename": "image_style.safetensors",  # LoRA model filename
        "description": "Art style transfer adapter"  # Short model description
    },
    "character": {
        "filename": "character.safetensors",  # LoRA model filename
        "description": "Character-specific generation adapter"  # Short model description
    }
}

# Language Model configuration (GGUF format models)
MODEL_MAPPING = {
    "Model": {
        "path": [  # Relative path components from MODEL_DIR
            "Model",        # Brand/organization
            "model-chat",   # Model family
            "model-chat.gguf"  # Specific model file
        ],
        "internal_name": "chat model"  # Model identifier for API
    }
}