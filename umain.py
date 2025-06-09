import streamlit as st
import uuid
from utils import format_math, process_thoughts
from src.model_manager import ModelManager
from src.image_generator import ImageGenerator
from src.config import MODEL_MAPPING
from ui.utraining import render_training_interface


def setup_page():
    """Configure Streamlit page settings and inject custom CSS."""
    st.set_page_config(
        page_title="AI Assistant Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Inject custom CSS styles
    st.markdown("""
    <style>
        .chat-container {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
        }
        .stButton>button {
            width: 100%;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager()

    if "image_generator" not in st.session_state:
        st.session_state.image_generator = ImageGenerator()

    if "chats" not in st.session_state:
        st.session_state.chats: dict = {}

    if "active_chat" not in st.session_state:
        st.session_state.active_chat: str = None

    # Initialize request counter
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0


def render_sidebar_controls() -> str:
    """Render sidebar controls and return selected operation mode."""
    with st.sidebar:
        st.header("Control Panel")

        # Model selection
        selected_model = st.selectbox(
            "Model:",
            list(MODEL_MAPPING.keys()),
            key="model_selector"
        )

        # Model application button
        if st.button("🔄 Apply Model", key="apply_model"):
            with st.spinner(f"Loading {selected_model}..."):
                try:
                    st.session_state.model_manager.start_server(selected_model)
                    st.success(f"{selected_model} activated!")
                except Exception as e:
                    st.error(f"Model error: {str(e)}")

        st.markdown("---")
        st.header("Operation Mode")
        # Mode selection
        mode = st.radio(
            "Select mode:",
            ["Text", "Image Generation", "Training"],
            key="mode_selector"
        )

        st.markdown("---")
        # Chat management controls
        _render_chat_controls()

        # Active chat selection
        if st.session_state.chats:
            _render_active_chat_selector()

        st.markdown("---")

    return mode


def _render_chat_controls():
    """Render chat management UI elements."""
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("✨ New Chat", key="new_chat"):
            chat_id = str(uuid.uuid4())
            st.session_state.chats[chat_id] = {"messages": []}
            st.session_state.active_chat = chat_id
            st.rerun()
    with col2:
        if st.session_state.chats and st.button(
            "🗑",
            key="delete_chat",
            help="Delete current chat"
        ):
            if st.session_state.active_chat:
                del st.session_state.chats[st.session_state.active_chat]
                st.session_state.active_chat = next(
                    iter(st.session_state.chats),
                    None
                )
                st.rerun()


def _render_active_chat_selector():
    """Render UI for selecting active chat."""
    options = list(st.session_state.chats.keys())
    selected = st.selectbox(
        "Active Chats:",
        options=options,
        index=options.index(st.session_state.active_chat)
            if st.session_state.active_chat else 0,
        format_func=lambda x: f"Chat {options.index(x) + 1}",
        key="chat_selector"
    )
    st.session_state.active_chat = selected


def render_chat_container(active_chat: dict):
    """Render chat message container with scrollable history."""
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for msg in active_chat["messages"]:
            _render_message(msg)

        st.markdown('</div>', unsafe_allow_html=True)


def _render_message(msg: dict):
    """Render individual chat message based on content type."""
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"])
            if msg.get("seed"):
                st.caption(f"Seed: {msg['seed']}")
        elif msg.get("type") == "image_prompt":
            st.markdown(f"Prompt: {msg['content']}")
            if msg.get("negative_prompt"):
                st.caption(f"Exclusions: {msg['negative_prompt']}")
        else:
            content = format_math(process_thoughts(msg["content"]))
            st.markdown(content, unsafe_allow_html=True)


def handle_no_active_chat():
    """Handle UI state when no chats exist."""
    st.info("Create a new chat using the sidebar controls")
    if st.button("Create First Chat"):
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = {"messages": []}
        st.session_state.active_chat = chat_id
        st.rerun()


def handle_text_mode(active_chat: dict):
    """Process text-based interactions."""
    if user_input := st.chat_input("Type your message..."):
        # Add user message to chat history
        active_chat["messages"].append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            # Generate and display response
            try:
                response = st.session_state.model_manager.generate_response(
                    user_input
                )
                st.markdown(response)
                active_chat["messages"].append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")


def handle_image_mode(active_chat: dict):
    """Handle image generation workflow."""
    # Local import to avoid circular dependencies
    from ui.uimage import (
        render_lora_settings,
        render_image_size_settings,
        render_advanced_image_settings,
        handle_image_generation
    )

    # Render settings in sidebar
    with st.sidebar:
        render_lora_settings()
        width, height = render_image_size_settings()
        steps, guidance_scale, seed, strength, uploaded_image = (
            render_advanced_image_settings()
        )

    # Store parameters in session state
    st.session_state.update({
        'steps': steps,
        'guidance_scale': guidance_scale,
        'seed': seed,
        'strength': strength,
        'uploaded_image': uploaded_image,
        'width': width,
        'height': height
    })

    # Handle image generation process
    handle_image_generation(active_chat, width, height)


def main():
    """Main application entry point."""
    setup_page()
    initialize_session_state()

    # Determine operation mode
    mode = render_sidebar_controls()

    # Handle no active chat scenario
    if not st.session_state.active_chat:
        handle_no_active_chat()
        return

    # Get active chat
    active_chat = st.session_state.chats[st.session_state.active_chat]

    # Display chat history
    render_chat_container(active_chat)

    # Process selected mode
    if mode == "Text":
        handle_text_mode(active_chat)
    elif mode == "Image Generation":
        handle_image_mode(active_chat)
    elif mode == "Training":
        render_training_interface()


if __name__ == "__main__":
    main()