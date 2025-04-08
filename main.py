import streamlit as st
from ui_main import (
    setup_page,
    initialize_session_state,
    render_sidebar_controls,
    render_chat_container,
    handle_no_active_chat
)
from ui_text import handle_text_mode
from ui_image import (
    render_lora_settings,
    render_image_size_settings,
    render_advanced_image_settings,
    handle_image_generation
)


def main():
    """Main application entry point handling UI composition and mode routing"""

    # Initialize core UI components and session state
    setup_page()  # Configure page settings
    initialize_session_state()  # Setup default app state

    # Render sidebar and get selected operation mode
    mode = render_sidebar_controls()  # Returns: "Text", "Image Generation" or "Training"

    # Handle empty chat state
    if not st.session_state.active_chat:
        handle_no_active_chat()  # Show chat creation prompt
        return  # Stop further execution

    # Get current chat context
    active_chat = st.session_state.chats[st.session_state.active_chat]

    # Render main chat interface
    render_chat_container(active_chat)  # Display message history

    # Route execution based on selected mode
    if mode == "Text":
        handle_text_mode(active_chat, st.session_state.model_manager)

    elif mode == "Image Generation":
        # Render image-specific controls in sidebar
        with st.sidebar:
            render_lora_settings()  # LoRA adapter settings
            width, height = render_image_size_settings()  # Dimension controls
            steps, guidance_scale, seed, strength, uploaded_image = render_advanced_image_settings()

        # Persist generation parameters
        st.session_state.update({
            'steps': steps,  # Number of diffusion steps
            'guidance_scale': guidance_scale,  # Prompt adherence level
            'seed': seed,  # Randomization seed
            'strength': strength,  # img2img influence strength
            'uploaded_image': uploaded_image,  # Reference image
            'width': width,  # Selected image width
            'height': height  # Selected image height
        })

        # Handle image generation workflow
        handle_image_generation(active_chat, width, height)

    elif mode == "Training":
        # Placeholder for future implementation
        st.warning("Training mode under development")


if __name__ == "__main__":
    main()  # Launch Streamlit application