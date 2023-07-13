import numpy as np
import os
from PIL import Image
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def load_model():
    return keras.models.load_model('model/mnist.h5')


def transform(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    img = img / 255.  # 스케일링
    img = img.reshape(-1, 28, 28, 1)
    return img


@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    f.save('imgs/' + secure_filename(f.filename))
    img = transform('imgs/' + f.filename)
    pred = np.argmax(model.predict(img), axis=1)[0]
    return jsonify({'result': str(pred)})


if __name__ == "__main__":
    model = load_model()
    app.run(host='0.0.0.0', debug=True)
