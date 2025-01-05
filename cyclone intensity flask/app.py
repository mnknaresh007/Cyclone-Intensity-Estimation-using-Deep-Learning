from flask import Flask, request, jsonify, render_template,redirect, url_for
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("C:/Users/nares/mj/model_1.h5")

# Define route for prediction
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        image_file = request.files['file']

        if image_file.filename == '':
            return redirect(request.url)

        if image_file:
    # Get the image file from the request
            image_file = request.files['file']
    # Save the image in the uploads directory
            image_filename = secure_filename(image_file.filename)
            image_path = os.path.join("static", "uploads", image_filename)
            image_file.save(image_path)
    # Preprocess the image
            img = load_and_prep_image(image_path)
    # Make prediction
            prediction = model.predict(img)
            return render_template('predict.html', prediction=prediction.tolist(), image_filename=image_filename)
    return render_template('predict.html')
# Define function to preprocess image
def load_and_prep_image(image_path, img_shape=256):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = img / 255.0
    return tf.expand_dims(img, axis=0)

# Define route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
