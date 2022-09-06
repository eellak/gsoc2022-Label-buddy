# Imports
import os
from joblib import load
from flask import Flask, request
from utils import mk_preds_fa, define_YOHO
import requests

# Set environnment variables
# root_dir = "Models"
# model_name = 'YOHO-1'

# MODEL_DIR = os.environ['MODELS']
# MODEL_FILE_LDA = os.environ["YOHO"]
# MODEL_PATH_YOHO = os.path.join(MODEL_DIR, MODEL_FILE_LDA)

# Creation of the Flask app
app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def result():

    if request.method == 'POST':
        print(request.files)
        audio_file = request.files['audio_data'].read()
        print(audio_file)

        model = define_YOHO()
        model.load_weights('./Models/YOHO-1/model-best.h5')
        prediction_yoho = mk_preds_fa(model, audio_file)

        return {'prediction YOHO': prediction_yoho}

@app.route('/train', methods=['GET', 'POST'])
def train():
    print("Starting trainig...")
    os.system('python3 training_inference.py') 


@app.route('/get_training_data', methods=['GET', 'POST'])
def get_trainin_data():
    # if key doesn't exist, returns None
    # dataset = request.args.get('dataset')

    data = {"data": "training"}

    # if dataset is not None:
    lb_dtst_url = "http://127.0.0.1:8000/api/v1/projects/get_dataset"
    r = requests.post(lb_dtst_url, data=data)

    #/home/baku/Desktop/DockerTesting/train-zipped/d1.zip
    with open("./train-zipped2/D1.zip", "wb") as fd:
        for chunk in r.iter_content(chunk_size=512):
            fd.write(chunk)

    return True


@app.route('/get_validation_data', methods=['GET', 'POST'])
def get_validation_data():
    # if key doesn't exist, returns None
    # dataset = request.args.get('dataset')

    data = {"data": "validation"}

    # if dataset is not None:
    lb_dtst_url = "http://127.0.0.1:8000/api/v1/projects/get_dataset"
    r = requests.post(lb_dtst_url, data=data)

    with open("val-zipped/BBC-Val.zip", "wb") as fd:
        for chunk in r.iter_content(chunk_size=512):
            fd.write(chunk)
    

@app.route('/')
def index():
    return 'Web App with Python Flask!'


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')