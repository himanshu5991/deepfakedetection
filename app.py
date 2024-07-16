import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your model
model_path = '/Users/himanshusharma/Desktop/Deepfake_detection/final_model.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_frame(frame):
    # Resize the frame to the model's expected input size
    resized_frame = cv2.resize(frame, (224, 224))
    # Normalize the frame
    normalized_frame = resized_frame / 255.0
    return normalized_frame

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the video file
        cap = cv2.VideoCapture(filepath)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            preprocessed_frame = preprocess_frame(frame)
            frames.append(preprocessed_frame)
        
        cap.release()
        
        # Convert frames to numpy array
        frames_array = np.array(frames)
        
        # Make predictions on the video frames
        predictions = model.predict(frames_array)
        average_prediction = np.mean(predictions, axis=0)
        
        # Define a threshold to interpret the prediction
        threshold = 0.5
        result = "Deepfake" if average_prediction > threshold else "Real"
        
        # Return the prediction result
        return jsonify({'prediction': average_prediction.tolist(), 'result': result}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
