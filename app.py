# Adrian Sypos - G00309646
# Below are the sources from where I have adapted some of the code for this project
# http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/
# https://stackoverflow.com/questions/41957490/send-canvas-image-data-uint8clampedarray-to-flask-server-via-ajax
# http://flask.pocoo.org/docs/0.12/patterns/fileuploads/
# https://nextjournal.com/a/17592186058848
# https://www.tensorflow.org/get_started/mnist/beginners

from flask import Flask, render_template, request, redirect, url_for
import os, re, base64
import keras.models, sys
import numpy as np
from scipy.misc import imsave, imread, imresize
# importing load.py from model folder for initializing the model
sys.path.append(os.path.abspath("./model"))
from load import *
from werkzeug.utils import secure_filename

app = Flask(__name__)

# global variables for model and graph
global model, graph
model, graph = init()

UPLOAD_FOLDER = os.path.basename('./upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['png'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
	
    # read parsed image back in 8-bit, black and white mode (L)
    x = imread(file, mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))
	
    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)

    # graph.as_default() overrides the current default graph for the lifetime of the context
    with graph.as_default():
        # generates output predictions for the input x
        out = model.predict(x)
        print(out)
        # returns the indices of the maximum values along an axis
        print(np.argmax(out, axis=1))
        # return a string representation of the data in an array
        response = np.array_str(np.argmax(out, axis=1))
        # return the response
        return render_template('index.html', error=response)
		
@app.route("/")
def index():
    return render_template("index.html")
	
@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from canvas and save as an image
    parseImage(request.get_data())
	
    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))
	
    # reshape image data for use in neural network
    x = x.reshape(1,28,28,1)

    # graph.as_default() overrides the current default graph for the lifetime of the context
    with graph.as_default():
        # generates output predictions for the input x
        out = model.predict(x)
        print(out)
        # returns the indices of the maximum values along an axis
        print(np.argmax(out, axis=1))
        # return a string representation of the data in an array
        response = np.array_str(np.argmax(out, axis=1))
        # return the response
        return response 

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)