import streamlit as st
from utils import stream_response

def handle_text_mode(active_chat, model_manager):
    """Handle text-based chat interactions with LLM

    Args:
        active_chat: Current chat session data
        model_manager: Model controller instance

    Features:
        - Message history management
        - Context window (last 5 messages)
        - Error handling with rollback
    """

    # Get user input from chat widget
    if user_input := st.chat_input("Type your message..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        # Add to message history
        active_chat["messages"].append({"role": "user", "content": user_input})

        # Prepare assistant response area
        with st.chat_message("assistant"):
            response_container = st.empty()

        try:
            with st.spinner("Generating response..."):
                # Generate streaming response from LLM
                response = model_manager.client.chat.completions.create(
                    model=model_manager.current_model_internal_name,
                    messages=active_chat["messages"][-5:],  # Last 5 messages as context
                    temperature=0.7,  # Creativity control (0-2)
                    stream=True  # Enable real-time streaming
                )

                full_response = stream_response(response, response_container)
                # Save final response to history
                active_chat["messages"].append({
                    "role": "assistant",
                    "content": full_response
                })

        except Exception as e:
            # Error handling and state rollback
            st.error(f"🚨 Ошибка генерации: {str(e)}")
            active_chat["messages"].pop() # Remove failed response