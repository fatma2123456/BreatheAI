from flask import Flask, request, render_template, redirect, url_for, session
import os
from model import predict_image  # Import the predict_image function

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/action.html')
def action():
    return render_template('action.html')





@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('action.html', prediction=None, image_url=None)

    file = request.files['image']
    

    predicted_class, image = predict_image(file)  # Get prediction from the model

    # Save the uploaded image temporarily
    image_path = os.path.join('static/uploads', file.filename)
    image.save(image_path)

    # Render the result.html with the prediction and image URL
    return render_template('result.html', prediction=predicted_class, image_url=image_path)

if __name__ == '__main__':
    app.run(debug=True)
