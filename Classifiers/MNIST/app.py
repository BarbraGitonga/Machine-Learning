from flask import Flask, render_template, request
import numpy as np
import joblib
from PIL import Image
import os

app = Flask(__name__)

model = joblib.load(open('/home/barbra/Desktop/Machine-Learning/Classifiers/MNIST/Number_App/model/forest.pkl', 'rb') preprocessing(image)):
    image = image.convert('L')
    image = image.resize((28,28), Image.LANCZOS)
    image_array = np.array(image)
    flattened_array = image_array.flatten()
    return flattened_array


@app.route('/', methods = ['POST', 'GET'])
def predict_image():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            #saving the image in a temporary folder:
            image_path = 'temp_image.jpg'
            image.save(image_path)
            img = Image.open(image_path)
            #preprocessing
            processed_img = preprocessing(img)
            predict_probs = model.predict([processed_img])
            print(predict_probs)
            prediction = np.argmin(predict_probs[0])
            print(f"debug printed class:{prediction}")
            os.remove(image_path)
            
            return f"Predicted Class: {prediction}"
            
        else:
            return 'No image uploaded'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)