# Imports
import os
from flask import Flask, request
from flask_cors import CORS
from panns_inference import SoundEventDetection, labels as panns_labels
import librosa
import numpy as np
import glob

# Creation of the Flask app
app = Flask(__name__)
CORS(app)

def panns_preds(framewise_output, frame_step=300):

  '''
  Function that returns the prediction by the PANNs model for frame_step/100 seconds,
  taking as input the framewise_output.
  '''

  steps = framewise_output.shape[1] // frame_step
  preds = []

  curr_step = 0
  for step in range(steps):
    end_step = curr_step + frame_step
    classwise_output = np.max(framewise_output[0, curr_step:end_step, :], axis=0) # (classes_num,)
    idxes = np.argsort(classwise_output)[::-1]
    label = panns_labels[idxes[0]]

    # remove the commas from labels
    if "," in label:
      label = label.replace(",", "")
    
    # give prediction in [start, finish, label] format
    pred = [curr_step/100, end_step/100, label]
    curr_step = end_step
    preds.append(pred)

  return preds


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

        device = 'cpu' # 'cuda' | 'cpu'
        (audio, _) = librosa.core.load(path_to_audio_file, sr=32000, mono=True)
        audio = audio[None, :]  # (batch_size, segment_samples)

        print("Performing Sound Event Detection...")
        sed = SoundEventDetection(checkpoint_path=None, device=device)

        print('Getting framewise output...')
        framewise_output = sed.inference(audio)
        

        print('Making predictions...')
        preds = panns_preds(framewise_output)
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