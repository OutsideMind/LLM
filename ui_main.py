import streamlit as st
import uuid
from utils import format_math, process_thoughts
from model_manager import ModelManager
from image_generator import ImageGenerator
from model_trainer import ModelTrainer
from config import MODEL_MAPPING


def setup_page():
    """Initialize Streamlit page configuration

    Sets up:
    - Page title and favicon
    - Wide layout mode
    - Expanded sidebar state by default
    """
    st.set_page_config(
        page_title="AI Assistant Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def initialize_session_state():
    """Initialize core application state

    Creates default values for:
    - Request counter
    - Model management instances
    - Chat history storage
    - Active chat tracking
    """
    session_defaults = {
        "request_count": 0,  # Track total API requests
        "model_manager": ModelManager(),  # LLM server controller
        "image_generator": ImageGenerator(),  # Image generation pipeline
        "model_trainer": ModelTrainer(),  # Training module
        "chats": {},  # Chat history storage
        "active_chat": None  # Currently active chat
    }

    # Set defaults only if not initialized
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar_controls():
    """Build interactive sidebar components

    Returns:
        str: Selected application mode
    """
    with st.sidebar:
        st.header("Control Panel")

        # Model selection dropdown
        selected_model = st.selectbox("Model:", list(MODEL_MAPPING.keys()))

        # Model activation button
        if st.button("🔄 Apply Model"):
            try:
                st.session_state.model_manager.start_server(selected_model)
                st.success(f"{selected_model} activated!")
            except Exception as e:
                st.error(str(e))

        st.markdown("---")
        st.header("Operation Mode")
        # Application mode selector
        mode = st.radio("Select mode:", ["Text", "Image Generation", "Training"])

        st.markdown("---")
        # Chat management controls
        render_chat_controls()

        # Active chat selector
        if st.session_state.chats:
            selected_chat = st.selectbox(
                "Active Chats:",
                options=list(st.session_state.chats.keys()),
                format_func=lambda x: f"Chat {list(st.session_state.chats.keys()).index(x) + 1}",
                key="chat_selector"
            )
            st.session_state.active_chat = selected_chat

        st.markdown("---")

    return mode


def render_chat_controls():
    """Render chat creation/deletion controls

    Features:
    - New chat button with UUID generation
    - Chat deletion button with safety checks
    - Automatic UI refresh after changes
    """
    chat_col1, chat_col2 = st.columns([3, 1])
    with chat_col1:
        if st.button("✨ New Chat"):
            chat_id = str(uuid.uuid4())
            st.session_state.chats[chat_id] = {"messages": []}
            st.session_state.active_chat = chat_id
            st.rerun()  # Refresh to update chat list
    with chat_col2:
        if st.session_state.chats and st.button("🗑️", help="Delete current chat"):
            if st.session_state.active_chat is not None:
                del st.session_state.chats[st.session_state.active_chat]
                st.session_state.active_chat = next(
                    iter(st.session_state.chats.keys())) if st.session_state.chats else None
                st.rerun()  # Refresh UI


def render_chat_container(active_chat):
    """Display chat message history

    Handles:
    - Text messages with math formatting
    - Image generations with metadata
    - Image prompts with negative prompts
    - Custom CSS styling container
    """
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)

        for msg in active_chat["messages"]:
            with st.chat_message(msg["role"]):
                if msg.get("type") == "image":
                    st.image(msg["content"])
                    if msg.get("seed"):
                        st.caption(f"**Seed:** `{msg['seed']}`")
                elif msg.get("type") == "image_prompt":
                    st.markdown(f"**Prompt:** {msg['content']}")
                    if msg.get("negative_prompt"):
                        st.caption(f"**Exclusions:** {msg['negative_prompt']}")
                else:
                    content = format_math(process_thoughts(msg["content"]))
                    st.markdown(content, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


def handle_no_active_chat():
    """Display empty state notification

    Shows instructional message when:
    - No chats exist
    - All chats have been deleted
    - New session initialized
    """
    st.info("Create a new chat using the sidebar controls")