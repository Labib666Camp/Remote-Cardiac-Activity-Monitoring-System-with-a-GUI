import numpy as np
import keras
from scipy.interpolate import interp1d

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

def reshape_data(signal):
    signal = np.reshape(signal,(signal.shape[0],))
    signal = downsample(signal,188)
    return np.reshape(signal,(1,signal.shape[0],1))

def predict_normality(signal):
    signal = reshape_data(signal)
    savedModel = keras.models.load_model("my-ECG-model")
    pred = np.round(savedModel.predict(signal),3)[0]
    return pred