import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import logging
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask application setup
app = Flask(__name__)

# Define constants for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Corrected file extension list
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit to 16MB

# Load the pre-trained Keras model
try:
    model = load_model('plant_disease_model.h5')
    # Fix: Print a success message after the model is loaded.
    logger.info("Model loaded successfully.")
except Exception as e:
    # Handle the error appropriately, and log the error
    logger.error(f"Error loading the model: {e}")
    # Consider raising an exception or exiting if the model fails to load.  For now, we'll keep going, which might be OK for some deployment scenarios.
    # exit(1)  #  <--  You might uncomment this in a production environment if model loading is critical.

# Class names for plant disease predictions.  Important:  Make sure this matches your model's output order!
class_names = [ 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path):
    """Loads and preprocesses an image for the model."""
    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
        return img_array
    except Exception as e:
        logger.error(f"Error preparing image: {e}")
        return None

def predict_disease(img_path):
    """Predicts the plant disease from the given image path."""
    img = prepare_image(img_path)
    if img is None:
        return "Error: Unable to process image."  # Return an error message

    try:
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index] * 100
        predicted_class_name = class_names[predicted_class_index]
        return predicted_class_name, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return "Error: Prediction failed." # Return error

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    """Handles file upload, prediction, and result display."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.warning("No file part")
            return render_template('index.html', error='No file part')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            logger.warning("No selected file")
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)  # Save the file
                logger.info(f"File saved successfully at {filepath}")

                # Perform prediction
                prediction_result = predict_disease(filepath)
                if isinstance(prediction_result, str) and prediction_result.startswith("Error:"):
                    os.remove(filepath) #delete the file
                    return render_template('index.html', error=prediction_result)  # Display the error message
                else:
                    predicted_class_name, confidence = prediction_result
                    # Render the result, passing the filename for display
                    return render_template('result.html',
                                           filename=filename,  # Pass filename for display
                                           predicted_class=predicted_class_name,
                                           confidence=confidence)
            except Exception as e:
                logger.error(f"Error during file processing: {e}")
                return render_template('index.html', error=f'Error: {e}')
        else:
            logger.warning("Invalid file type")
            return render_template('index.html', error='Invalid file type. Allowed types are png, jpg, jpeg.')
    return render_template('index.html', error=None)  # Ensure error is initialized to None for the GET request.

@app.route('/uploads/<filename>')
def display_image(filename):
    """Route to display the uploaded image."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Start the Flask development server
    app.run(debug=True, host='0.0.0.0',port=10000)
    