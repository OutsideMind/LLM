import streamlit as st
import random
import torch
from src.config import *
import traceback


def render_lora_settings():
    """Render LoRA adapter configuration UI in sidebar"""
    with st.sidebar:
        # Generation settings header
        st.header("Image Generation Settings")
        st.info(f"Active Model: {SD_MODEL_NAME}")  # Display current SD model
        st.caption("Recommended: 35 steps, CFG 6.0, Euler a")
        st.caption("Required tags: masterpiece, best quality")

        # Multi-select for LoRA styles
        selected_loras = st.multiselect(
            "Style Adapters (LoRA)",
            options=list(LORA_MODELS.keys()),
            help="Combine up to 3 artistic styles",
            max_selections=3
        )

        # Sync selected LoRAs with active adapters
        current_loras = [l["name"] for l in st.session_state.image_generator.loaded_loras]

        # Remove deselected adapters
        for lora_name in current_loras.copy():
            if lora_name not in selected_loras:
                st.session_state.image_generator.remove_lora(lora_name)

        # Add new adapters
        for lora_name in selected_loras:
            if lora_name not in current_loras:
                try:
                    st.session_state.image_generator.add_lora(lora_name)
                except Exception as e:
                    st.error(f"Failed to load {lora_name}: {str(e)}")

        # Weight adjustment panel
        if selected_loras:
            with st.container(border=True):
                st.markdown("### ⚙️ Adapter Settings")

                # Create sliders for each active LoRA
                for lora_name in selected_loras:
                    lora_item = next(
                        (item for item in st.session_state.image_generator.loaded_loras
                         if item["name"] == lora_name), None
                    )

                    if lora_item:
                        lora_info = LORA_MODELS[lora_name]
                        with st.container(border=True):
                            # Style header with description
                            st.markdown(
                                f"""<div style='margin-bottom: 8px;'>
                                    <strong style='font-size: 1.1rem;'>{lora_name}</strong><br>
                                    <span style='color: #666; font-size: 0.85rem; line-height: 1.3;'>
                                    {lora_info['description']}</span>
                                </div>""",
                                unsafe_allow_html=True
                            )

                            # Weight slider with clamping
                            new_weight = st.slider(
                                f"{lora_name} Weight",
                                0.0, 2.0,
                                value=lora_item["weight"],
                                step=0.1,
                                key=f"weight_{lora_name}",
                                help="Adapter influence strength (1.0 = default)"
                            )
                            lora_item["weight"] = round(new_weight, 1)

                # Weight synchronization controls
                if st.button("🔄 Apply Changes", help="Update model with new weights"):
                    try:
                        st.session_state.image_generator.update_lora_weights()
                        st.toast("Settings applied successfully!", icon="✅")
                    except Exception as e:
                        st.error(f"Update failed: {str(e)}")

                # Current configuration summary
                st.caption("Active weights: " + ", ".join(
                    [f'{l["name"]}: {l["weight"]:.1f}'
                     for l in st.session_state.image_generator.loaded_loras]
                ))


def render_image_size_settings():
    """Render image size selection UI with presets and custom options

    Returns:
        tuple: (width, height) in pixels
    """
    size_preset = st.selectbox(
        "Image Dimensions",
        options=[
            ("1024x1024 (Square)", 1024, 1024),  # Standard SDXL size
            ("1152x896 (Landscape)", 1152, 896),  # Wide aspect ratio
            ("896x1152 (Portrait)", 896, 1152),  # Tall aspect ratio
            ("Custom Size", "custom")  # User-defined dimensions
        ],
        format_func=lambda x: x[0],  # Display first element of tuple
        index=0  # Default to first option
    )

    # Initialize with default values
    width, height = 1024, 1024

    if size_preset[0] == "Custom Size":
        # Create two columns for side-by-side inputs
        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input(
                "Width",
                min_value=512,  # Minimum viable SDXL size
                max_value=2048,  # Maximum supported resolution
                value=1024,  # Default width
                step=64  # SDXL latent space alignment
            )
        with col2:
            height = st.number_input(
                "Height",
                min_value=512,
                max_value=2048,
                value=1024,
                step=64
            )
    else:
        # Extract dimensions from preset
        width, height = size_preset[1], size_preset[2]

    return width, height


def render_advanced_image_settings():
    """Render advanced image generation controls

    Returns:
        tuple: (steps, guidance_scale, seed, strength, uploaded_image)
    """
    with st.sidebar.expander("⚙️ Advanced Settings"):
        # Generation process controls
        steps = st.slider(
            "Sampling Steps",
            min_value=25,  # Minimum for decent quality
            max_value=50,  # Diminishing returns beyond
            value=35,  # Optimal for Euler a
            help="Quality vs speed tradeoff (35-45 recommended)"
        )

        # Prompt adherence control
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=5.0,  # More creative
            max_value=9.0,  # More strict
            value=6.0,  # Balanced default
            step=0.5,
            help="Prompt adherence strength (5.5-7.0 optimal)"
        )

        # Randomization control
        seed = st.number_input(
            "Random Seed",
            value=0,  # 0 = random
            help="0 for random, fixed value for reproducibility"
        )

        # Image-to-image strength
        strength = st.slider(
            "Image Influence",
            min_value=0.0,  # Full regeneration
            max_value=1.0,  # Original preservation
            value=0.7,  # Balanced modification
            step=0.1,
            help="Original image preservation strength"
        )

        # Reference image upload
        uploaded_image = st.file_uploader(
            "Source Image (optional)",
            type=["png", "jpg", "jpeg"],
            help="Upload base image for img2img processing"
        )

    return steps, guidance_scale, seed, strength, uploaded_image


def handle_image_generation(active_chat, width, height):
    """Handle complete image generation workflow

    Args:
        active_chat: Current chat session data
        width: Image width from settings
        height: Image height from settings

    Features:
        - Prompt parsing with negative prompt detection
        - Generation parameter management
        - Seed randomization logic
        - Automatic cache cleanup
        - Error handling with user feedback
    """

    if user_input := st.chat_input("Describe the image (use '--neg' for negative prompts)..."):
        # Parse prompt and negative prompt
        if "--neg" in user_input:
            parts = user_input.split("--neg", 1)
            prompt = parts[0].strip()
            negative_prompt = parts[1].strip() if len(parts) > 1 else ""
        else:
            prompt = user_input.strip()
            negative_prompt = ""

        # Display user input with formatting
        with st.chat_message("user"):
            st.markdown(f"**Prompt:** {prompt}")
            if negative_prompt:
                st.caption(f"**Exclusions:** {negative_prompt}")

        # Save to chat history
        active_chat["messages"].append({
            "role": "user",
            "content": user_input,
            "type": "image_prompt"
        })

        # Generation block
        with st.chat_message("assistant"):
            with st.spinner("Generating image..."):
                try:
                    # Get parameters from session state
                    steps = st.session_state.get('steps', 35)
                    guidance_scale = st.session_state.get('guidance_scale', 6.0)
                    seed = st.session_state.get('seed', 0)
                    strength = st.session_state.get('strength', 0.7)
                    uploaded_image = st.session_state.get('uploaded_image', None)

                    # Seed randomization logic
                    final_seed = seed if seed != 0 else random.randint(0, 2 ** 32 - 1)

                    # Choose generation method
                    if uploaded_image:
                        image_path, used_seed = st.session_state.image_generator.generate_image_from_image(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            init_image=uploaded_image,
                            strength=strength,  # Image preservation strength
                            steps=steps,
                            guidance_scale=guidance_scale,
                            seed=final_seed,
                            width=width,
                            height=height
                        )
                    else:
                        image_path, used_seed = st.session_state.image_generator.generate_image(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            steps=steps,
                            guidance_scale=guidance_scale,
                            seed=final_seed,
                            width=width,
                            height=height
                        )

                    # Handle generation result
                    if image_path:
                        st.image(image_path)
                        st.caption(f"**Seed:** `{used_seed}`")

                        # Save to chat history
                        active_chat["messages"].append({
                            "role": "assistant",
                            "content": image_path,
                            "type": "image",
                            "seed": used_seed
                        })

                        # Memory management
                        st.session_state.request_count += 1
                        if st.session_state.request_count % 5 == 0:
                            torch.cuda.empty_cache()
                            st.session_state.image_generator.load_models()
                            st.toast("🔄 Automatic cache clearance after 5 requests", icon="🧹")
                    else:
                        st.error("Generation failed - check parameters")

                except Exception as e:
                    st.error(f"🚨 Generation error: {str(e)}")
                    # Log full error details for debugging
                    print(f"Image generation error: {traceback.format_exc()}")