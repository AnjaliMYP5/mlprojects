import os
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)
# The model file path should be adjusted if your model is in a different location
MODEL_PATH = 'model/crop_disease_model.h5'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the uploads folder exists, if not create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Define class names based on your model's output
    # This is an example, you should replace with your actual classes
    CLASS_NAMES = ['Healthy', 'Blight', 'Rust', 'Powdery Mildew']
    st.info("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.warning("Please make sure the 'model' directory exists and contains a valid 'crop_disease_model.h5' file.")
    model = None
    CLASS_NAMES = []


def preprocess_image(image_path):
    """
    Loads an image, resizes it, and converts it to a numpy array for model prediction.
    Assumes the model expects a 224x224 input size.
    """
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    return img_array


@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and model prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Preprocess the uploaded image
            img_array = preprocess_image(file_path)
            
            # Make the prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = float(np.max(predictions[0]))
            
            # Clean up the uploaded file
            os.remove(file_path)

            return jsonify({
                'prediction': predicted_class_name,
                'confidence': f'{confidence:.2f}',
                'status': 'success'
            })
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return jsonify({'error': f'Prediction failed: {e}'})
    
    return jsonify({'error': 'Invalid file type'})


if __name__ == '__main__':
    # Use threaded=True for better concurrency on a development server
    app.run(debug=True, threaded=True)
