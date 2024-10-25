from flask import Flask, request, render_template
import os
from model import predict_image  # Import the predict_image function

app = Flask(__name__)

# Ensure the directory to save images exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/action.html')
def action():
    return render_template('action.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('action.html', prediction=None, image_url=None, gradcam_url=None)

    file = request.files['image']

    # Get prediction and Grad-CAM image path from the model
    predicted_class, original_image, gradcam_image_path = predict_image(file) 

    if predicted_class is None or gradcam_image_path is None:
        return render_template('action.html', prediction="Error during prediction", image_url=None, gradcam_url=None)

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    original_image.save(image_path)

    # Render the result.html with the prediction, uploaded image URL, and Grad-CAM image URL
    return render_template('result.html', prediction=predicted_class, image_url=image_path, gradcam_url=gradcam_image_path)


if __name__ == '__main__':
    app.run(debug=True)
