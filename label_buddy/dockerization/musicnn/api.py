# Imports
import os
from flask import Flask, request
from flask_cors import CORS
from musicnn.extractor import extractor
import numpy as np
import glob

# Creation of the Flask app
app = Flask(__name__)
CORS(app)

def musicnn_prediction_formating(tags_with_max_likelihoods):

    '''
    Function that returns the prediction by the musicnn model for every 3 deconds,
    taking as input the tags_with_max_likelihoods.
    '''

    final_preds = []

    starting_time = 0.0
    for tag_with_max_likelihoods in tags_with_max_likelihoods:
        ending_time = starting_time + 3.0
        final_pred = [starting_time, ending_time, tag_with_max_likelihoods]
        starting_time = ending_time
        final_preds.append(final_pred)

    return final_preds


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

        print("Extracting taggram and tags...")
        taggram, tags, features = extractor(path_to_audio_file, input_length=3, model='MTT_musicnn', extract_features=True)
        
        print("Getting max likelihoods...")
        max_likelihoods_pes_timestep = np.argmax(taggram, axis=1)
        tags_with_max_likelihoods = [tags[i] for i in max_likelihoods_pes_timestep]

        print('Making predictions...')
        preds = musicnn_prediction_formating(tags_with_max_likelihoods)
        print('Predictions made.')

        files = glob.glob('./audios/*')
        for f in files:
            os.remove(f)

        print(preds)

        return {'prediction musicnn': preds}


@app.route('/')
def index():
    ip_addr = request.remote_addr
    return '<h1> Your IP address is:' + ip_addr


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)