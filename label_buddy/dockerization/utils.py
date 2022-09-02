import numpy as np
import IPython
import math
import glob
import sed_eval
import dcase_util
import pickle
import os
import soundfile as sf
import librosa
from zipfile import ZipFile
import keras
import re
import glob
import random
import datetime
import tensorflow as tf


def extract_train_zip():

  """
  Extract the .zip files into the 'train data' folder.
  """

  for i in range(0, 4):
    zip_name = "./train-zipped/d" + str(i + 1) + ".zip"
    with ZipFile(zip_name, 'r') as zip:
      zip.extractall('yoho_train_data')
      print("Extracted all sound files into the folder {}".format(i + 1))

  zip_name = "./train-zipped/BBC-Train.zip"
  with ZipFile(zip_name, 'r') as zip:
    zip.extractall('yoho_train_data')
    print("Extracted all sound files into the folder")


def extract_val_zip():

  """
  Extract the .zip files into the 'val data' folder.
  """

  zip_name = "./val-zipped/BBC-Val.zip"
  with ZipFile(zip_name, 'r') as zip:
    zip.extractall('yoho_validation_data')
    print("Extracted all sound files into the folder")


def smoothe_events(events):

  """
  Event smoothing.
  """

  music_events = []
  speech_events = []
  for e in events:
    if e[2] == "speech":
      speech_events.append(e)
    elif e[2] == "music":
      music_events.append(e)

  speech_events.sort(key=lambda x: x[0])
  music_events.sort(key=lambda x: x[0])

  max_speech_silence = 0.4
  max_music_silence = 0.6
  min_dur_speech = 1.3
  min_dur_music = 3.4

  count = 0

  while count < len(speech_events) - 1:
    if (speech_events[count][1] >= speech_events[count + 1][0]) or (speech_events[count + 1][0] - speech_events[count][1] <= max_speech_silence):
      speech_events[count][1] = max(speech_events[count + 1][1], speech_events[count][1])
      del speech_events[count + 1]
    else:
      count += 1

  count = 0

  while count < len(music_events) - 1:
    if (music_events[count][1] >= music_events[count + 1][0]) or (music_events[count + 1][0] - music_events[count][1] <= max_music_silence):
      music_events[count][1] = max(music_events[count + 1][1], music_events[count][1])
      del music_events[count + 1]
    else:
      count += 1


  smooth_events = music_events + speech_events

  for i in range(len(smooth_events)):
    smooth_events[i][0] = round(smooth_events[i][0], 3)
    smooth_events[i][1] = round(smooth_events[i][1], 3)

  smooth_events.sort(key=lambda x: x[0])

  return smooth_events


def get_universal_labels(events, no_of_div = 26):

  """
  Generate the label matrix of an audio sample based on its annotations.
  """
  
  events = smoothe_events(events)
  win_length = 8.0/no_of_div
  labels = np.zeros((no_of_div, 6))

  for e in events:

    start_time = float(e[0])
    stop_time = float(e[1])

    start_bin = int(start_time // win_length)
    stop_bin = int(stop_time // win_length)

    start_time_2 = start_time - start_bin * win_length
    stop_time_2 = stop_time - stop_bin * win_length

    n_bins = stop_bin - start_bin

    if n_bins == 0:
      if e[2] == "speech":
        labels[start_bin, 0:3] = [1, start_time_2, stop_time_2]

      elif e[2] == "music":
        labels[start_bin, 3:6] = [1, start_time_2, stop_time_2]

    elif n_bins == 1:
      if e[2] == "speech":
        labels[start_bin, 0:3] = [1, start_time_2, win_length]

      elif e[2] == "music":
        labels[start_bin, 3:6] = [1, start_time_2, win_length]

      if stop_time_2 > 0.0:
        if e[2] == "speech":
          labels[stop_bin, 0:3] = [1, 0.0, stop_time_2]

        elif e[2] == "music":
          labels[stop_bin, 3:6] = [1, 0.0, stop_time_2]

    elif n_bins > 1:
      if e[2] == "speech":
        labels[start_bin, 0:3] = [1, start_time_2, win_length]

      if e[2] == "music":
        labels[start_bin, 3:6] = [1, start_time_2, win_length]

      for i in range(1, n_bins):
        if e[2] == "speech":
          labels[start_bin + i, 0:3] = [1, 0.0, win_length]

        if e[2] == "music":
          labels[start_bin + i, 3:6] = [1, 0.0, win_length]

      if stop_time_2 > 0.0:
        if e[2] == "speech":
          labels[stop_bin, 0:3] = [1, 0.0, stop_time_2]

        elif e[2] == "music":
          labels[stop_bin, 3:6] = [1, 0.0, stop_time_2]

  labels[:, [1, 2, 4, 5]] /= win_length

  return labels


def get_only_binary_labels(events, win_length = 8.0/13):

  """
  Generate the label matrix of an audio sample based on its annotations.
  """

  events = smoothe_events(events)
  labels = np.zeros((2,))

  class_list = [e[2] for e in events]

  if "speech" in class_list:
    labels[0] = 1

  if "music" in class_list:
    labels[1] = 1

  return labels


def construct_labels(labels, no_of_steps = 4):
  # labels = smoothe_events(labels)

  new_labels = np.zeros((no_of_steps, 6))
  win_width = 8.0 / no_of_steps


  for i in range(len(labels)):
    s = labels[i][0] / win_width
    s = min(np.floor(s), no_of_steps - 1)

    r = s * win_width

    if labels[i][2] == "speech":
      new_labels[int(s)][0] = 1

      t1 = (labels[i][0] - r) / win_width
      t2 = (labels[i][1] - labels[i][0]) / win_width

      new_labels[int(s)][1] = t1
      new_labels[int(s)][2] = t2

    elif labels[i][2] == "music":
      new_labels[int(s)][3] = 1

      t1 = (labels[i][0] - r) / win_width
      t2 = (labels[i][1] - labels[i][0]) / win_width

      new_labels[int(s)][4] = t1
      new_labels[int(s)][5] = t2

  return new_labels


def to_seg_by_class(events, n_frames = 801):
  
  events = smoothe_events(events)
  labels = np.zeros((n_frames, 2), dtype=np.float32)

  for e in events:
    t1 = float(e[0])
    t1 = int(t1 / 160 * 16000)
    t2 = float(e[1])
    t2 = int(t2 / 160 * 16000)

    if e[2] == 'speech':
      labels[t1:t2, 0] = 1
    elif e[2] == 'music':
      labels[t1:t2, 1] = 1
  
  return labels 


def get_log_melspectrogram(audio, sr = 16000, hop_length = 160, win_length = 400, n_fft = 512, n_mels = 64, fmin = 125, fmax = 7500):
    
    """
    Return the log-scaled Mel bands of an audio signal.
    """

    bands = librosa.feature.melspectrogram(
        y=audio, sr=sr, hop_length=hop_length, win_length = win_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=np.float32)
    
    return librosa.core.power_to_db(bands, amin=1e-7)


def get_labels():

  labels = glob.glob("./yoho_train_data/content/**/mel-id-label-[0-9]*.pickle", recursive=True)

  counter = 0
  for ll in labels:
    print(f'Training label {counter} from {len(labels)}.')
    with open(ll, 'rb') as f:
      n = pickle.load(f)
    n2 = get_universal_labels(n)
    np.save(ll.replace(".pickle", ".npy"), n2)
    counter += 1

  labels = glob.glob("./yoho_validation_data/**/mel-id-label-[0-9]*.pickle", recursive=True)

  counter = 0
  for ll in labels:
    print(f'Validation label {counter} from {len(labels)}.')
    with open(ll, 'rb') as f:
      n = pickle.load(f)
    n2 = get_universal_labels(n)
    np.save(ll.replace(".pickle", ".npy"), n2)
    counter += 1


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_examples, batch_size=128, epoch_size = 16384, dim=(1, ),
                 n_classes=2, shuffle=True):
        'Initialization'
        print("Constructor called!!!")
        self.dim = dim
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.list_examples = list_examples
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #print("The self.list_examples is {}".format(self.list_examples))
        return int(np.floor(len(self.list_examples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_examples[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
      self.indexes = np.arange(len(self.list_examples))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # # Initialization
        X = np.empty([self.batch_size, 801, 64, 1], dtype=np.float64)
        y = np.empty([self.batch_size, 26, 6], dtype=np.float64)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
          # Store sample

          xx = np.load(ID[0])
          X[i, :, :, 0] = xx

          # Store class
          yy = np.load(ID[1])
          # yy2 = yy[:, [1, 2, 4, 5]]
          y[i, :, :] = yy
          
        return X, y
    

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
    
def alphanum_key(s):
    """
     Turn a string into a list of string and number chunks.
     "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ 
    Sort the given list in the way that humans expect.
    """
    
    return l.sort(key=alphanum_key)


def get_np_arrays():

  """
  Load the individual numpy arrays into partition
  """

  data = glob.glob("./yoho_train_data/content/**/mel-id-[0-9]*.npy", recursive=True) 
#   data = sort_nicely(data)

  labels = glob.glob("./yoho_train_data/content/**/mel-id-label-[0-9]*.npy", recursive=True)
#   labels = sort_nicely(labels)

  train_examples = [(data[i], labels[i]) for i in range(len(data))]

  random.seed(4)
  random.shuffle(train_examples)

  return train_examples


def create_train_partition(train_examples):

  """
  Creating the train partition.
  """

  m_train = 25184
  random.seed()
  random.shuffle(train_examples)

  data_MS = glob.glob("./yoho_train_data/MuSpeak/content/Mel Files/**/mel-id-[0-9]*.npy", recursive=True) 
#   data_MS = sort_nicely(data_MS)

  labels_MS = glob.glob("./yoho_train_data/MuSpeak/content/Mel Files/**/mel-id-label-[0-9]*.npy", recursive=True)
#   labels_MS = sort_nicely(labels_MS)

  train_examples_MS = [(data_MS[i], labels_MS[i]) for i in range(len(data_MS))]

  partition = {}
  partition['train'] = train_examples[0:m_train] + train_examples_MS

  random.shuffle(partition['train'])

  return partition['train']


def validation_data():

  """
  This loads data for the validation set.
  """

  data = glob.glob("./yoho_validation_data/**/mel-id-[0-9]*.npy", recursive=True)
#   data = sort_nicely(data)

  labels = glob.glob("./yoho_validation_data/**/mel-id-label-[0-9]*.npy", recursive=True)
#   labels = sort_nicely(labels)

  validation_examples = [(data[i], labels[i]) for i in range(len(data))]

  random.seed(4)
  random.shuffle(validation_examples)

  partition = {}
  partition['validation'] = validation_examples

  return partition['validation']


def smooth_output(output, min_speech=1.3, min_music=3.4, max_silence_speech=0.4, max_silence_music=0.6):

  """
  This function was adapted from Lemaire et al. 2019. 
  """

  duration_frame = 220 / 22050
  n_frame = output.shape[1]

  start_music = -1000
  start_speech = -1000

  for i in range(n_frame):
      if output[0, i] == 1:
          if i - start_speech > 1:
              if (i - start_speech) * duration_frame <= max_silence_speech:
                  output[0, start_speech:i] = 1

          start_speech = i

      if output[1, i] == 1:
          if i - start_music > 1:
              if (i - start_music) * duration_frame <= max_silence_music:
                  output[1, start_music:i] = 1

          start_music = i

  start_music = -1000
  start_speech = -1000

  for i in range(n_frame):
      if i != n_frame - 1:
          if output[0, i] == 0:
              if i - start_speech > 1:
                  if (i - start_speech) * duration_frame <= min_speech:
                      output[0, start_speech:i] = 0

              start_speech = i

          if output[1, i] == 0:
              if i - start_music > 1:
                  if (i - start_music) * duration_frame <= min_music:
                      output[1, start_music:i] = 0

              start_music = i
      else:
          if i - start_speech > 1:
              if (i - start_speech) * duration_frame <= min_speech:
                  output[0, start_speech:i + 1] = 0

          if i - start_music > 1:
              if (i - start_music) * duration_frame <= min_music:
                  output[1, start_music:i + 1] = 0

  return output


def preds_to_se(p, audio_clip_length = 8.0):

  """
  This function converts the predictions made by the neural network into a format that is understood by sed_eval.
  """

  start_speech = -100
  start_music = -100
  stop_speech = -100
  stop_music = -100

  audio_events = []

  n_frames = p.shape[0]

  if p[0, 0] == 1:
    start_speech = 0
  
  if p[0, 1] == 1:
    start_music = 0

  for i in range(n_frames - 1):
    if p[i, 0] == 0 and p[i + 1, 0] == 1:
      start_speech = i + 1

    elif p[i, 0] == 1 and p[i + 1, 0] == 0:
      stop_speech = i
      start_time = frames_to_time(start_speech)
      stop_time = frames_to_time(stop_speech)
      audio_events.append((start_time, stop_time, "speech"))
      start_speech = -100
      stop_speech = -100

    if p[i, 1] == 0 and p[i + 1, 1] == 1:
      start_music = i + 1
    elif p[i, 1] == 1 and p[i + 1, 1] == 0:
      stop_music = i
      start_time = frames_to_time(start_music)
      stop_time = frames_to_time(stop_music)      
      audio_events.append((start_time, stop_time, "music"))
      start_music = -100
      stop_music = -100

  if start_speech != -100:
    start_time = frames_to_time(start_speech)
    stop_time = audio_clip_length
    audio_events.append((start_time, stop_time, "speech"))
    start_speech = -100
    stop_speech = -100

  if start_music != -100:
    start_time = frames_to_time(start_music)
    stop_time = audio_clip_length
    audio_events.append((start_time, stop_time, "music"))
    start_music = -100
    stop_music = -100

  audio_events.sort(key = lambda x: x[0]) 
  return audio_events

def frames_to_time(f, sr = 22050.0, hop_size = 220):
  return f * hop_size / sr

def get_log_melspectrogram(audio, sr = 16000, hop_length = 160, win_length = 400, n_fft = 512, n_mels = 64, fmin = 125, fmax = 7500):
    
    """
    Return the log-scaled Mel bands of an audio signal.
    """

    bands = librosa.feature.melspectrogram(
        y=audio, sr=sr, hop_length=hop_length, win_length = win_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=np.float32)
    
    return librosa.core.power_to_db(bands, amin=1e-7)


def mk_preds_fa(model, audio_path, hop_size = 6.0, discard = 1.0, win_length = 8.0, sampling_rate = 22050):

  """
  Make predictions for full audio.
  """

  # load the audio file with one channel
  in_signal, in_sr = librosa.load(audio_path, mono=True)

  # Resample the audio file.
  in_signal_22k = librosa.resample(in_signal, orig_sr=in_sr, target_sr=sampling_rate)
  in_signal = np.copy(in_signal_22k)

  audio_clip_length_samples = in_signal.shape[0]
  print('audio_clip_length_samples is {}'.format(audio_clip_length_samples))

  #hop_size_samples = int(hop_size * sampling_rate)
  hop_size_samples = 220 * 602 - 1

  #win_length_samples = int(win_length * sampling_rate)
  win_length_samples = 220 * 802 - 1

  n_preds = int(math.ceil((audio_clip_length_samples - win_length_samples) / hop_size_samples)) + 1

  in_signal_pad = np.zeros((n_preds * hop_size_samples + 200 * 220))

  in_signal_pad[0:audio_clip_length_samples] = in_signal

  preds = np.zeros((n_preds, 802, 2))
  mss_in = np.zeros((n_preds, 802, 80))

  for i in range(n_preds):
    seg = in_signal_pad[i * hop_size_samples:(i * hop_size_samples) + win_length_samples]
    #print('seg.shape is {}'.format(seg.shape))
    seg = librosa.util.normalize(seg)

    mss = get_log_melspectrogram(seg)
    M = mss.T
    mss_in[i, :, :] = M

  preds = (model.predict(mss_in) >= (0.5, 0.5)).astype(np.float)

  preds_mid = np.copy(preds[1:-1, 100:702, :])

  preds_mid_2 = preds_mid.reshape(-1, 2)

  oa_preds = preds[0, 0:702, :] # oa stands for overall predictions

  oa_preds = np.concatenate((oa_preds, preds_mid_2), axis = 0)

  oa_preds = np.concatenate((oa_preds, preds[-1, 100:, :]), axis = 0)

  return oa_preds


def my_loss_fn(y_true, y_pred):

  weight = tf.constant([1.0])
  squared_difference = tf.square(y_true - y_pred)

  ss0 = squared_difference[:, :, 0] * 0 + 1
  ss1 = y_true[:, :, 0] 
  ss2 = y_true[:, :, 0]

  ss3 = squared_difference[:, :, 3] * 0 + 1
  ss4 = y_true[:, :, 3] 
  ss5 = y_true[:, :, 3]

  sss = tf.stack((ss0, ss1, ss2, ss3, ss4, ss5), axis = 2)
  
  squared_difference =  tf.multiply(squared_difference, sss)


  return tf.reduce_sum(squared_difference, axis=[-1, -2])  # Note the `axis=-1`


def binary_acc(y_true, y_pred):

  threshold = tf.constant([0.5])

  binary_true = tf.stack((y_true[:, :, 0], y_true[:, :, 3]), axis=1)
  binary_pred = tf.stack((y_pred[:, :, 0], y_pred[:, :, 3]), axis=1)

  binary_true = tf.greater_equal(binary_true, threshold)
  binary_pred = tf.greater_equal(binary_pred, threshold)

  # acc = tf.square(y_true - y_pred)
  acc = tf.cast((binary_true == binary_pred), tf.float32)

  # Note the `axis=-1`
  return tf.reduce_mean(acc, axis=[-1, -2])  


class TimeAcc(tf.keras.metrics.Metric):

  def __init__(self, name='time_accuracy', **kwargs):
    super(TimeAcc, self).__init__(name=name, **kwargs)
    self.time_correct_1 = self.add_weight(name='time_correct_1', initializer='zeros')
    self.time_count_1 = self.add_weight(name='time_count_1', initializer='zeros')
    self.time_correct_3 = self.add_weight(name='time_correct_3', initializer='zeros')
    self.time_count_3 = self.add_weight(name='time_count_3', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    tolerance = tf.constant([0.25])
    zero_lim = tf.constant([0.0])

    y_true_0 = y_true[:, :, 0]
    y_true_1 = y_true[:, :, 1]

    y_pred_0 = y_pred[:, :, 0]
    y_pred_1 = y_pred[:, :, 1]

    time_diff_1 = tf.abs(y_pred_1 - y_true_1)

    time_correct_1 = tf.cast(tf.less_equal(time_diff_1, tolerance), dtype = np.float32)

    time_correct_1 = tf.multiply(time_correct_1, y_true_0)

    self.time_count_1.assign_add(tf.reduce_sum(y_true_0, axis=None))

    self.time_correct_1.assign_add(tf.reduce_sum(time_correct_1, axis = None))

  def result(self):
    time_acc_1 = self.time_correct_1 / self.time_count_1
    return time_acc_1

  def reset_states(self):
    self.time_correct_1.assign(0)
    self.time_count_1.assign(0)


class MyCustomCallback_3(tf.keras.callbacks.Callback):

  # This optimises val loss for Wave-U-Net YOHO
  # Back to Val Binary acc.

  def __init__(self, model_dir, patience=0):
    super(MyCustomCallback_3, self).__init__()
    self.patience = patience
    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None
    self.model_best_path = os.path.join(model_dir, 'model-best.h5')
    self.model_last_path = os.path.join(model_dir, 'model-last-epoch.h5')
    self.custom_params = {"best_loss":np.inf, "last_epoch":0}
    
    self.custom_params_path = os.path.join(model_dir, 'custom_params.pickle')
    if os.path.isfile(self.custom_params_path):
      with open(self.custom_params_path, 'rb') as f:
        self.custom_params = pickle.load(f)

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best F1 as 0.0.
    self.is_impatient = False

  def on_train_end(self, logs=None):
    if not self.is_impatient:
      print("Restoring model weights from the end of the best epoch.")
      self.model.set_weights(self.best_weights)

  def on_epoch_end(self, epoch, logs=None):
    current_val_loss = logs.get("val_loss")
    self.model.save_weights(self.model_last_path)
    self.custom_params["last_epoch"] = self.custom_params["last_epoch"] + 1

    if current_val_loss < self.custom_params['best_loss']:
      self.custom_params['best_loss'] = current_val_loss
      self.wait = 0
      self.best_weights = self.model.get_weights()
      self.model.save_weights(self.model_best_path)

    else:
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.is_impatient = True
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)
    with open(self.custom_params_path, 'wb') as f:
      pickle.dump(self.custom_params, f, pickle.HIGHEST_PROTOCOL)
    

def define_YOHO():

  """
  Manually define YOHO
  """

  LAYER_DEFS = [
    # (layer_function, kernel, stride, num_filters)
    ([3, 3], 1,   64),
    ([3, 3], 2,  128),
    ([3, 3], 1,  128),
    ([3, 3], 2,  256),
    ([3, 3], 1,  256),
    ([3, 3], 2,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 2, 1024),
    ([3, 3], 1, 1024),
    ([3, 3], 1, 512),
    ([3, 3], 1, 256),
    ([3, 3], 1, 128),
]

  # params = yamnet_params.Params()
  m_features = tf.keras.Input(shape=(801, 64, 1), name="mel_input")
  X = m_features
  # X = tf.keras.layers.Reshape((801, 64, 1))(X)
  X = tf.keras.layers.Conv2D(filters = 32, kernel_size=[3, 3], strides=2, padding='same', use_bias=False, activation=None, name = "layer1/conv")(X)
  X = tf.keras.layers.BatchNormalization(center=True, scale=False, epsilon=1e-4, name = "layer1/bn")(X)
  X = tf.keras.layers.ReLU(name="layer1/relu")(X)

  for i in range(len(LAYER_DEFS)):
    X = tf.keras.layers.DepthwiseConv2D(kernel_size=LAYER_DEFS[i][0], strides = LAYER_DEFS[i][1], depth_multiplier=1, padding='same', use_bias=False,
                                        activation=None, name="layer"+ str(i + 2)+"/depthwise_conv")(X)
    X = tf.keras.layers.BatchNormalization(center=True, scale=False, epsilon=1e-4, name = "layer"+ str(i + 2)+"/depthwise_conv/bn")(X)
    X = tf.keras.layers.ReLU(name="layer"+ str(i + 2)+"/depthwise_conv/relu")(X)
    X = tf.keras.layers.Conv2D(filters = LAYER_DEFS[i][2], kernel_size=[1, 1], strides=1, padding='same', use_bias=False, activation=None,
                              name = "layer"+ str(i + 2)+"/pointwise_conv")(X)
    X = tf.keras.layers.BatchNormalization(center=True, scale=False, epsilon=1e-4, name = "layer"+ str(i + 2)+"/pointwise_conv/bn")(X)
    X = tf.keras.layers.ReLU(name="layer"+ str(i + 2)+"/pointwise_conv/relu")(X)


  _, _, sx, sy = X.shape
  X = tf.keras.layers.Reshape((-1, int(sx * sy)))(X)

  pred = tf.keras.layers.Conv1D(6,kernel_size=1, activation="sigmoid")(X)
  model = tf.keras.Model(
        name='yamnet_frames', inputs=m_features,
        outputs=[pred])
  
  return model


