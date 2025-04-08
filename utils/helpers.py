import re
import time


def format_math(content: str) -> str:
    """Format mathematical expressions in content using Markdown code blocks.

    Converts inline math expressions wrapped in $ symbols to
    ```math blocks for proper rendering in Streamlit.

    Example:
        $E=mc^2$ -> ```math\nE=mc^2\n```
    """
    return re.sub(r'\$(.*?)\$', r'\n\n```math\n\1\n```\n\n', content, flags=re.DOTALL)


def process_thoughts(content: str) -> str:
    """Convert <think> tags to styled HTML div elements.

    Transforms:
    <think>content</think> ->
    <div style="padding:10px;...">content</div>

    Used for displaying model's internal reasoning in a visually distinct format.
    """
    return re.sub(
        r'<think>(.*?)</think>',
        r'<div style="padding:10px;margin:10px 0;border-left:3px solid #ddd;color:#666;font-style:italic;">\1</div>',
        content,
        flags=re.DOTALL
    )


def stream_response(response, container) -> str:
    """Process and display streaming response with real-time formatting.

    Handles two special cases:
    1. <think> blocks - buffers content until closing tag
    2. Math expressions - formats with format_math()

    Args:
        response: Generator from chat completion API
        container: Streamlit container for displaying output

    Returns:
        Full processed response as string
    """
    full_response = ""
    thought_buffer = ""
    in_thought = False

    for chunk in response:
        if chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content

            # Handle think tag opening
            if '<think>' in delta:
                in_thought = True
                delta = delta.replace('<think>', '')

            # Handle think tag closing
            elif '</think>' in delta:
                in_thought = False
                delta = delta.replace('</think>', '')
                full_response += f'<think>{thought_buffer}{delta}</think>'
                thought_buffer = ""
                # Update container with processed thoughts
                container.markdown(process_thoughts(full_response), unsafe_allow_html=True)
                continue

            if in_thought:
                # Buffer content between <think> tags
                thought_buffer += delta
                # Display temporary thought with typing indicator
                current_thought = process_thoughts(full_response + f'<think>{thought_buffer}</think>')
                container.markdown(current_thought, unsafe_allow_html=True)
            else:
                # Build main response and display with math formatting
                full_response += delta
                container.markdown(
                    format_math(process_thoughts(full_response + "▌")),  # ▌ shows typing indicator
                    unsafe_allow_html=True
                )

            # Simulate typing speed
            time.sleep(0.02)

    # Final render without typing indicator
    final_content = format_math(process_thoughts(full_response))
    container.markdown(final_content, unsafe_allow_html=True)
    return full_response