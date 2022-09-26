# Imports
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from utils import mk_preds_vector, define_YOHO
import requests
import glob
from utils import training_inference, enrich_dataset

# Creation of the Flask app
app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['GET', 'POST'])
def result():

    '''
    Route to perform the predictions through a post request.
    '''

    if request.method == 'POST':

        print('Processing audio...')
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

    '''
    Route to perform training process through a post request.
    '''

    if request.method == 'POST':

        # get data (training parameters) from the request
        epochs = int(request.args.get('epochs'))
        patience = int(request.args.get('patience'))
        initial_lr = float(request.args.get('lr'))

        if os.path.isfile('train-zipped/d1.zip') and os.path.isfile('train-zipped/d1.zip'):
            print("Starting trainig...")
            loss, binary_acc = training_inference(epochs, patience, initial_lr)
            print('Training done.')

            # respond with the current loss and binary accuracy
            response_data = {
                "loss": loss,
                "binary_accuracy": binary_acc
            }

            return response_data

        else: 
            print('There are no trainig/validation data to start the trianing process.')
            return 'Missing Data1!', 400


@app.route('/get_training_data', methods=['GET', 'POST'])
def get_trainin_data():
    
    '''
    Route to get basic training data. 
    '''

    if os.path.isfile('train-zipped/d1.zip'):
        print("Trainig data already exist!")
    else:
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

    '''
    Route to get basic validation data. 
    '''

    if os.path.isfile('val-zipped/BBC-Val.zip'):
        print("Validation data already exist!")
    else:
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

    '''
    Route to get data from all the approved annotation of the current project. 
    '''

    project_id = request.args.get('project_id')

    url = "http://127.0.0.1:8000/api/v1/projects/" + project_id + "/tasks/export_to_container"
    response = requests.post(url)

    with open(f"data/data_project{project_id}.zip", "wb") as fd:
        for chunk in response.iter_content(chunk_size=512):
            fd.write(chunk)

    flask_resp = jsonify(success=True)
    return flask_resp


@app.route('/send_model_weights', methods=['GET', 'POST'])
def send_model_weights():

    '''
    Route to send current model weights for download. 
    '''

    path_of_trained_weight = './Models/YOHO-1/model-best.h5'
    path_of_pretrained_weight = './Models/YOHO-1/YOHO-music-speech.h5'

    if os.path.isfile(path_of_trained_weight):
        path = path_of_trained_weight
    else:
        path = path_of_pretrained_weight

    return send_file(path, as_attachment=True)


@app.route('/enrich_data', methods=['GET', 'POST'])
def enrich_data():

    '''
    Helper route to perform data enrichment on demand. 
    '''
    done = enrich_dataset()

    print(done)

    resp = jsonify(success=True)
    return resp


@app.route('/')
def index():
    ip_addr = request.remote_addr
    return '<h1> Your IP address is:' + ip_addr


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)