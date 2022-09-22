# Imports
import os
from flask import Flask, request, jsonify
from utils import mk_preds_vector, define_YOHO
import requests
import glob
import json
import numpy as np
from utils import get_log_melspectrogram
import pickle
import zipfile


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

        print('Processing audio...')

        # print(request.files)
        audio_file_bytes = request.files['audio_data'].read()

        path_to_audio_file = './audios/audio_file.wav'

        with open(path_to_audio_file, mode='bx') as f:
            f.write(audio_file_bytes)

        f.close()

        print("Defining YOHO...")
        model = define_YOHO()
        print('Loading weights...')

        path_of_trained_weight = './Models/YOHO-1/model-best.h5'
        path_of_pretrained_weight = './Models/YOHO-1/YOHO-music-speech.h5'

        if os.path.isfile(path_of_trained_weight):
            model.load_weights(path_of_trained_weight)
        else:
            model.load_weights(path_of_pretrained_weight)

        print('Making predictions...')
        prediction_yoho = mk_preds_vector(path_to_audio_file, model)
        print('Predictions made.')
        print(prediction_yoho)

        files = glob.glob('./audios/*')
        for f in files:
            os.remove(f)

        print(prediction_yoho)

        return {'prediction YOHO': prediction_yoho}

@app.route('/train', methods=['GET', 'POST'])
def train():
    print("Starting trainig...")
    os.system('python3 training_inference.py') 
    print('Training done.')

    resp = jsonify(success=True)
    return resp


@app.route('/get_training_data', methods=['GET', 'POST'])
def get_trainin_data():
    # if key doesn't exist, returns None
    # dataset = request.args.get('dataset')
    
    print("Getting trainig data...")
    data = {"data": "training"}

    # if dataset is not None:
    lb_dtst_url = "http://127.0.0.1:8000/api/v1/projects/get_dataset"
    r = requests.post(lb_dtst_url, data=data)

    print(f"Request: {r}")

    #/home/baku/Desktop/DockerTesting/train-zipped/d1.zip
    with open("train-zipped/d1.zip", "wb") as fd:
        for chunk in r.iter_content(chunk_size=512):
            fd.write(chunk)

    print('Training data saved.')

    resp = jsonify(success=True)
    return resp



@app.route('/get_validation_data', methods=['GET', 'POST'])
def get_validation_data():

    print("Getting validation data...")

    data = {"data": "validation"}

    # if dataset is not None:
    lb_dtst_url = "http://127.0.0.1:8000/api/v1/projects/get_dataset"
    r = requests.post(lb_dtst_url, data=data)

    print(f"Request: {r}")

    with open("val-zipped/BBC-Val.zip", "wb") as fd:
        for chunk in r.iter_content(chunk_size=512):
            fd.write(chunk)
    
    print('Validation data saved.')

    resp = jsonify(success=True)
    return resp


@app.route('/get_approved_data_annotations', methods=['GET', 'POST'])
def get_approved_data_annotations():

    project_id = request.args.get('project_id')

    url = "http://127.0.0.1:8000/api/v1/projects/" + project_id + "/tasks/export_to_container"
    response = requests.post(url)

    with open(f"data/data_project{project_id}.zip", "wb") as fd:
        for chunk in response.iter_content(chunk_size=512):
            fd.write(chunk)

    flask_resp = jsonify(success=True)
    return flask_resp


@app.route('/enrich_dataset', methods=['GET', 'POST'])
def enrich_dataset():

    zip_data_paths = glob.glob("./data/*")
    for zip_file_path in zip_data_paths:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(('.mp3', '.wav')):
                    zip_ref.extract(file, "./audios/")
                if file.endswith(('.npy')):
                    zip_ref.extract(file, "./labels/")

    audio_files = glob.glob("./audios/*")
    for audio_file_path in audio_files:
        audio_name = audio_file_path.split("audio/")[1].split(".")[0]
        log_melspectrogram = get_log_melspectrogram(audio_file_path)
        np.save(f"./log_mel_spectogram/{audio_name}.npy", log_melspectrogram)

    label_files = glob.glob("./labels/*")
    for label_file_path in label_files:
        annotations = np.load(label_file_path).tolist()
        audio_names = list(annotations.keys())
        for audio_name in audio_names:
            with open(f'./labels/{audio_name}.pkl', 'wb') as f:
                pickle.dump(annotations[audio_name], f)

    files_in_directory = os.listdir('./labels')
    filtered_files = [file for file in files_in_directory if file.endswith(".npy")]
    for file in filtered_files:
        path_to_file = os.path.join('./labels', file)
        os.remove(path_to_file)


@app.route('/')
def index():
    ip_addr = request.remote_addr
    return '<h1> Your IP address is:' + ip_addr


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)