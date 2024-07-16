import tensorflow as tf

# Load your original model
model = tf.keras.models.load_model('/Users/himanshusharma/Desktop/Deepfake_detection/final_model.h5')

# Save the model in the TensorFlow SavedModel format
model.save('/Users/himanshusharma/Desktop/Deepfake_detection', save_format='tf')
