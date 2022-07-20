import math
import librosa
import numpy as np
import tensorflow as tf


def define_YOHO(): 

    '''
    Manual definition of YOHO model.
    '''

    # input: log-scaled Mel bands of an audio signal.
    m_features = tf.keras.Input(shape=(801, 64, 1), name="mel_input")
    X = m_features
    # X = tf.keras.layers.Reshape((801, 64, 1))(X)
    X = tf.keras.layers.Conv2D(filters = 32, kernel_size=[3, 3], strides=2, 
                                padding='same', use_bias=False, activation=None, 
                                name = "layer1/conv")(X)
    X = tf.keras.layers.BatchNormalization(center=True, scale=False, epsilon=1e-4,
                                            name = "layer1/bn")(X)
    X = tf.keras.layers.ReLU(name="layer1/relu")(X)

    # layer settings for the upcoming ones
    # (layer_function, kernel, stride, num_filters)
    LAYER_DEFS = [
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

    for i in range(len(LAYER_DEFS)):
        X = tf.keras.layers.DepthwiseConv2D(kernel_size=LAYER_DEFS[i][0], 
                                            strides = LAYER_DEFS[i][1], 
                                            depth_multiplier=1, padding='same', 
                                            use_bias=False, activation=None, 
                                            name="layer"+ str(i + 2)+"/depthwise_conv")(X)
        X = tf.keras.layers.BatchNormalization(center=True, scale=False, 
                                                epsilon=1e-4, 
                                                name = "layer"+ str(i + 2)+"/depthwise_conv/bn")(X)
        X = tf.keras.layers.ReLU(name="layer"+ str(i + 2)+"/depthwise_conv/relu")(X)
        X = tf.keras.layers.Conv2D(filters = LAYER_DEFS[i][2], 
                                    kernel_size=[1, 1], strides=1, 
                                    padding='same', use_bias=False, activation=None,
                                    name = "layer"+ str(i + 2)+"/pointwise_conv")(X)
        X = tf.keras.layers.BatchNormalization(center=True, 
                                                scale=False, epsilon=1e-4, 
                                                name = "layer"+ str(i + 2)+"/pointwise_conv/bn")(X)
        X = tf.keras.layers.ReLU(name="layer"+ str(i + 2)+"/pointwise_conv/relu")(X)


    _, _, sx, sy = X.shape
    X = tf.keras.layers.Reshape((-1, int(sx * sy)))(X)

    pred = tf.keras.layers.Conv1D(6,kernel_size=1, activation="sigmoid")(X)
    model = tf.keras.Model(
        name='yamnet_frames', inputs=m_features,
        outputs=[pred])

    return model


def smoothe_events(events, max_speech_silence = 0.8, max_music_silence = 0.8, 
                   min_dur_speech = 0.8, min_dur_music = 3.4):

    '''
    Smoothing is performed over the output events to eliminate the occurrence of spurious audio events.
    Filtering - if the duration of the audio event is too short or if the silence between consecutive events
    of the same acoustic class is too short, we remove the occurrence.
    '''

    # Initialize the output events
    music_events = []
    speech_events = []

    # append to each array
    for e in events:
        if e[2] == "speech":
            speech_events.append(e)
        elif e[2] == "music":
            music_events.append(e)

    # sort the arrays by start time
    speech_events.sort(key=lambda x: x[0])
    music_events.sort(key=lambda x: x[0])

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

    # filtering
    count = 0
    while count < len(smooth_events):
        if smooth_events[count][1] - smooth_events[count][0] < min_dur_speech and smooth_events[count][2] == "speech":
            del smooth_events[count]

        elif smooth_events[count][1] - smooth_events[count][0] < min_dur_music and smooth_events[count][2] == "music":
            del smooth_events[count]

    else:
        count += 1

    for i in range(len(smooth_events)):
        smooth_events[i][0] = round(smooth_events[i][0], 3)
        smooth_events[i][1] = round(smooth_events[i][1], 3)

    smooth_events.sort(key=lambda x: x[0])

    return smooth_events


def get_log_melspectrogram(audio, sr = 16000, hop_length = 160, win_length = 400, n_fft = 512, n_mels = 64, fmin = 125, fmax = 7500):
    
    """
    Return the log-scaled Mel bands of an audio signal.
    """

    bands = librosa.feature.melspectrogram(
        y=audio, sr=sr, hop_length=hop_length, win_length = win_length, 
        n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=np.float32)
    
    return librosa.core.power_to_db(bands, amin=1e-7)


def mk_preds_vector(audio_path, model, hop_size = 6.0, discard = 1.0, win_length = 8.0, sampling_rate = 22050):

    '''
    This function takes an audio file and returns the predictions vector.
    '''

    # load the audio file with one channel
    in_signal, in_sr = librosa.load(audio_path, mono=True)

    # Resample the audio file.
    in_signal_22k = librosa.resample(in_signal, orig_sr=in_sr, target_sr=sampling_rate)
    in_signal = np.copy(in_signal_22k)

    # calculate constants
    audio_clip_length_samples = in_signal.shape[0]  # length of the audio clip in samples
    hop_size_samples = int(hop_size * sampling_rate)  # hop size in samples
    win_length_samples = int(win_length * sampling_rate)  # window length in samples
    n_preds = int(math.ceil((audio_clip_length_samples - win_length_samples) / hop_size_samples)) + 1  # number of predictions

    # padding
    in_signal_pad = np.zeros(((n_preds - 1) * hop_size_samples) + win_length_samples)  # padding the input signal
    in_signal_pad[0:audio_clip_length_samples] = in_signal  

    # initialization
    preds = np.zeros((n_preds, 26, 2))
    mss_in = np.zeros((n_preds, 801, 64))
    events = []

    for i in range(n_preds):
        seg = in_signal_pad[i * hop_size_samples:(i * hop_size_samples) + win_length_samples]
        seg_normalized = librosa.util.normalize(seg)
        seg_resampled = librosa.resample(seg_normalized, orig_sr=22050, target_sr=16000)

        mss = get_log_melspectrogram(seg_resampled)  # get the log-scaled Mel bands
        M = mss.T  # transpose the matrix
        mss_in[i, :, :] = M

    # get the predictions
    preds = model.predict(mss_in)  

    # initialize events
    events = []

    for j in range(n_preds):

        p = preds[j, :, :]
        events_curr = []
        win_width = win_length / 26

        for i in range(len(p)):
            if p[i][0] >= 0.5:
                start = win_width * i + win_width * p[i][1]
                end = p[i][2] * win_width + start
                events_curr.append([start, end, "speech"])

            if p[i][3] >= 0.5:
                start = win_width * i + win_width * p[i][4]
                end = p[i][5] * win_width + start
                events_curr.append([start, end, "music"])

        se = events_curr
        if j == 0:
            start = 0.0
            end = start + win_length
            if preds.shape[0] > 1:
                end -= discard

        elif j == n_preds - 1:
            start = j * hop_size + discard
            end = start - discard + win_length

        else:
            start = j * hop_size + discard
            end = start + win_length - discard

        for k in range(len(se)):
            se[k][0] = max(start, se[k][0] + j * hop_size)
            se[k][1] = min(end, se[k][1] + j * hop_size)

        for see in se:
            events.append(see) 

    # smoothing with filtering
    smooth_events = smoothe_events(events)

    return smooth_events
