import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import numpy as np
import io

# -----------------
# 1. Load the pre-trained model
# Use st.cache_resource to load the model only once for performance
# -----------------
@st.cache_resource
def load_and_compile_model():
    """Loads the pre-trained Keras model."""
    try:
        model = load_model('my_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
model = load_and_compile_model()

# -----------------
# 2. Set up the Streamlit app interface
# -----------------
st.set_page_config(
    page_title="Visual CNN App",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Visualize CNN Layers 🖼️")
st.markdown("Upload an image to see how a Convolutional Neural Network (CNN) processes it layer by layer.")

# -----------------
# 3. Handle file upload
# -----------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file:
    # -----------------
    # 4. Preprocess the uploaded image
    # -----------------
    try:
        # Open the image file
        image = Image.open(uploaded_file).convert('L') # Convert to grayscale
        # Resize to the model's input size (e.g., 28x28 for MNIST)
        image = image.resize((28, 28))
        
        # Display the original image
        st.subheader("Original Image")
        st.image(image, caption='Original Image', use_column_width=False, width=150)

        # Convert the image to a NumPy array and preprocess for the model
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension
        img_array = img_array / 255.0 # Normalize pixel values
        
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        st.stop()
        
    # -----------------
    # 5. Perform classification and display prediction
    # -----------------
    with st.spinner('Classifying...'):
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        
    st.subheader("Classification Result")
    st.success(f"Predicted Class: {predicted_class_index}")
    st.write(f"Confidence: {prediction[0][predicted_class_index]:.2f}")

    # -----------------
    # 6. Visualize the feature maps of the convolutional layers
    # -----------------
    st.subheader("Layer-by-Layer Visualization")
    st.info("The images below are 'feature maps' that show what the network has learned to detect at each layer. As the network gets deeper, the features become more abstract.")
    
    # Create a new model that outputs the activation maps of the convolutional layers
    layer_names = [layer.name for layer in model.layers if 'conv2d' in layer.name]
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # Get the activations (feature maps) for the uploaded image
    with st.spinner("Generating visualizations..."):
        activations = activation_model.predict(img_array)

    for layer_name, activation in zip(layer_names, activations):
        st.markdown(f"**Layer: `{layer_name}`**")
        
        # We only display a few feature maps to avoid cluttering the page
        n_features = activation.shape[-1]
        display_features = min(8, n_features)
        
        # Create columns to display the images side-by-side
        cols = st.columns(display_features)
        
        for i in range(display_features):
            # Process and display each feature map
            feature_map = activation[0, :, :, i]
            # Normalize to 0-255 range for image display
            feature_map_normalized = (255 * (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))).astype(np.uint8)
            
            # Display the feature map in its own column
            cols[i].image(feature_map_normalized, use_column_width=True, caption=f'Filter {i+1}')
            
else:
    st.markdown("Upload a file to get started! 🚀")