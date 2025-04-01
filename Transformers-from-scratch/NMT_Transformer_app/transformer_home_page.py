import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import os

# Import the feature page functionality
from transformer_feature_app import display_feature_page


st.set_page_config(
    page_title="Transformer Architecture Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Feature page"])

    if page == "Home":
        show_home_page()
    else:
        display_feature_page()


def show_home_page():
    st.title("Understanding Transformer Architecture")

    st.markdown("""
    ## The Architecture That Changed NLP

    Transformers have revolutionized natural language processing since being introduced 
    in the paper "Attention Is All You Need" (Vaswani et al., 2017). Unlike previous 
    sequence models that process data sequentially, transformers process all input tokens 
    in parallel, using attention mechanisms to capture relationships between tokens 
    regardless of their position in the sequence.
    """)

    # Display transformer architecture image
    st.subheader("Transformer Architecture")

    # Option 1: If you have a downloaded image
    try:
        image_path = "transformers-diagram.png"  # Update this path to your image
        image = Image.open(image_path)
        st.image(image, caption="Transformer Architecture", use_column_width=True)
    except FileNotFoundError:
        # Option 2: Generate a simplified visual if image is not found
        st.warning("Transformer image not found. Displaying a simplified diagram instead.")
        fig = visualize_transformer_architecture()
        st.pyplot(fig)

    st.markdown("""
    ## Key Components

    ### Encoder
    - **Multi-Head Attention**: Allows the model to focus on different parts of the input sequence simultaneously
    - **Feed-Forward Networks**: Apply transformations to each position independently
    - **Add & Norm Layers**: Residual connections and layer normalization to stabilize training

    ### Decoder
    - **Masked Multi-Head Attention**: Prevents positions from attending to subsequent positions
    - **Encoder-Decoder Attention**: Allows the decoder to focus on relevant parts of the input sequence
    - **Feed-Forward Networks**: Similar to those in the encoder

    ### Other Components
    - **Positional Encoding**: Injects information about token positions since transformers don't process sequentially
    - **Input/Output Embeddings**: Convert tokens to vectors and back
    """)

    st.markdown("""
    ## Applications

    Transformer architectures power many modern AI systems, including:

    - Large language models like GPT, Claude, and LLaMA
    - Translation systems like Google Translate
    - Text summarization tools
    - Code generation assistants
    - Image generation models (with modifications)
    """)


def show_other_features():
    st.title("Other Features")
    st.write("This is where your other application features would go.")

    # Add your other functionality here
    st.write("You can add your custom visualizations or other tools in this section.")


def visualize_transformer_architecture():
    """Create a simplified visualization of the transformer architecture."""
    # Create a diagram using matplotlib
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')

    # Define components and their positions
    components = [
        {"name": "Input Embedding", "pos": (0.5, 0.9), "width": 0.3, "height": 0.05, "color": "lightblue"},
        {"name": "Positional Encoding", "pos": (0.5, 0.83), "width": 0.3, "height": 0.05, "color": "lightblue"},

        # Encoder
        {"name": "Encoder", "pos": (0.3, 0.65), "width": 0.35, "height": 0.15, "color": "lightgreen"},
        {"name": "Multi-Head Attention", "pos": (0.3, 0.7), "width": 0.25, "height": 0.03, "color": "white"},
        {"name": "Feed Forward", "pos": (0.3, 0.65), "width": 0.25, "height": 0.03, "color": "white"},
        {"name": "Add & Norm", "pos": (0.3, 0.6), "width": 0.25, "height": 0.03, "color": "white"},

        # Decoder
        {"name": "Decoder", "pos": (0.7, 0.4), "width": 0.35, "height": 0.25, "color": "lightyellow"},
        {"name": "Masked Multi-Head Attention", "pos": (0.7, 0.5), "width": 0.25, "height": 0.03, "color": "white"},
        {"name": "Multi-Head Attention", "pos": (0.7, 0.45), "width": 0.25, "height": 0.03, "color": "white"},
        {"name": "Feed Forward", "pos": (0.7, 0.4), "width": 0.25, "height": 0.03, "color": "white"},
        {"name": "Add & Norm", "pos": (0.7, 0.35), "width": 0.25, "height": 0.03, "color": "white"},

        {"name": "Linear", "pos": (0.5, 0.2), "width": 0.3, "height": 0.05, "color": "lightpink"},
        {"name": "Softmax", "pos": (0.5, 0.13), "width": 0.3, "height": 0.05, "color": "lightpink"},
        {"name": "Output", "pos": (0.5, 0.05), "width": 0.3, "height": 0.05, "color": "lightsalmon"}
    ]

    # Draw each component
    for comp in components:
        x, y = comp["pos"]
        w, h = comp["width"], comp["height"]
        ax.add_patch(plt.Rectangle(
            (x - w / 2, y - h / 2), w, h,
            facecolor=comp["color"],
            edgecolor='black',
            alpha=0.8
        ))
        ax.text(x, y, comp["name"], ha='center', va='center', fontsize=9)

    # Add arrows
    ax.arrow(0.5, 0.875, 0, -0.02, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.5, 0.78, -0.15, -0.05, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.3, 0.55, 0, -0.1, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.3, 0.45, 0.3, -0.1, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.5, 0.78, 0.15, -0.2, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.7, 0.3, -0.15, -0.05, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.5, 0.25, 0, -0.02, head_width=0.02, head_length=0.01, fc='black', ec='black')
    ax.arrow(0.5, 0.18, 0, -0.02, head_width=0.02, head_length=0.01, fc='black', ec='black')

    return fig


if __name__ == "__main__":
    main()