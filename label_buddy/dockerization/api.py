# Imports
import os
from joblib import load
from flask import Flask, request
from utils import mk_preds_fa, define_YOHO

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
        model.load_weights(MODEL_PATH_YOHO)
        prediction_yoho = mk_preds_fa(model, audio_file)

        return {'prediction YOHO': prediction_yoho}

@app.route('/train', methods=['GET', 'POST'])
def train():
    print("Starting trainig...")
    os.system('python3 training_inference.py') 


@app.route('/get_data', methods=['GET', 'POST'])
def get_data():
    print("Starting trainig...")
    os.system('python3 training_inference.py') 
    print("Training done!")


@app.route('/')
def index():
    return 'Web App with Python Flask!'


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')