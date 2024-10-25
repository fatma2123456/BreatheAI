import numpy as np
from PIL import Image
import tensorflow as tf
import os
import cv2  # For overlaying heatmap
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model('X-ray-VGG19-Best-model.h5')  # Ensure the path is correct

# Function to make predictions and generate Grad-CAM
def predict_image(file):
    if file.filename == '':
        print("No file selected.")
        return None, None, None

    try:
        # Load and preprocess the image
        image = Image.open(file.stream).convert('RGB')  # Ensure it's in RGB format
        image = image.resize((224, 224))  # Resize to match model input size
        image_array = np.array(image)  # Convert to NumPy array
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

        # Predict
        predictions = model.predict(image_array)
        print(f"Predictions: {predictions}")

        # Assuming the model has 3 classes: Normal, Viral Pneumonia, Covid
        class_names = ['COVID-19', 'Lung_Opacity', 'Normal', 'Pneumonia-Bacterial', 'Pneumonia_Viral', 'Tuberculosis']
        predicted_class = class_names[np.argmax(predictions)]
        print(f"Predicted class: {predicted_class}")

        # Generate Grad-CAM
        gradcam_image_path = generate_gradcam(image_array, np.argmax(predictions))
        print(f"Grad-CAM image saved at: {gradcam_image_path}")

        return predicted_class, image, gradcam_image_path
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

# Function to generate Grad-CAM
def generate_gradcam(image_array, predicted_class_index):
    try:
        # Get the last convolutional layer in the model
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer("block5_conv4").output, model.output]  # VGG16 specific layer name
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_array)
            loss = predictions[:, predicted_class_index]

        # Compute the gradient of the loss with respect to the output feature map
        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]

        # Compute the channel-wise mean of the gradients
        weights = tf.reduce_mean(grads, axis=(0, 1))

        # Multiply the feature maps by the corresponding weights
        cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_outputs[:, :, i]

        # Apply ReLU to the heatmap (so that negative values become 0)
        cam = np.maximum(cam, 0)

        # Normalize the heatmap to a range between 0 and 1
        cam = cam / cam.max()

        # Resize the heatmap to match the input image size
        cam = cv2.resize(cam, (224, 224))

        # Convert the heatmap to a 3-channel RGB image
        heatmap = np.uint8(255 * cam)

        # Create a color map (you can choose any colormap you like)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay the heatmap on the original image (assuming image_array is normalized)
        original_image = np.uint8(255 * image_array[0])  # Convert normalized image back to uint8

        # Overlay heatmap onto original image
        overlay_image = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

        # Save the overlay image
        gradcam_image_path = os.path.join('static', 'gradcam_result.jpg')
        plt.imsave(gradcam_image_path, overlay_image)

        return gradcam_image_path
    except Exception as e:
        print(f"Error during Grad-CAM generation: {e}")
        return None
