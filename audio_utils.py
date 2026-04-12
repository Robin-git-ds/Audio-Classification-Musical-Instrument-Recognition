"""
Audio Classification Utilities
Common functions for audio processing, feature extraction, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


def envelope(y, rate, threshold=0.0005):
    """
    Extract signal envelope using rolling mean threshold.

    Parameters:
    y (array): Audio signal
    rate (int): Sample rate
    threshold (float): Threshold for envelope detection

    Returns:
    list: Boolean mask indicating signal above threshold
    """
    mask = []
    y_abs = pd.Series(y).apply(np.abs)
    y_mean = y_abs.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    mask = y_mean > threshold  # Vectorized instead of loop
    return mask.tolist()


def plot_signals(signals, figsize=(20, 5)):
    """
    Visualize time-domain signals.

    Parameters:
    signals (list): List of signal arrays
    figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(nrows=len(signals), ncols=1, figsize=figsize)
    if len(signals) == 1:
        axes = [axes]

    for i, signal in enumerate(signals):
        axes[i].plot(signal)
        axes[i].set_title(f'Signal {i+1}')
    plt.tight_layout()
    plt.show()


def plot_fft(fft, figsize=(20, 5)):
    """
    Visualize Fourier transform.

    Parameters:
    fft (array): FFT data
    figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(nrows=len(fft), ncols=1, figsize=figsize)
    if len(fft) == 1:
        axes = [axes]

    for i, f in enumerate(fft):
        axes[i].plot(f)
        axes[i].set_title(f'FFT {i+1}')
    plt.tight_layout()
    plt.show()


def plot_fbank(fbank, figsize=(20, 5)):
    """
    Visualize filter bank coefficients.

    Parameters:
    fbank (array): Filter bank data
    figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(nrows=len(fbank), ncols=1, figsize=figsize)
    if len(fbank) == 1:
        axes = [axes]

    for i, fb in enumerate(fbank):
        axes[i].imshow(fb, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Filter Bank {i+1}')
    plt.tight_layout()
    plt.show()


def plot_mfccs(mfccs, figsize=(20, 5)):
    """
    Visualize MFCC coefficients.

    Parameters:
    mfccs (array): MFCC data
    figsize (tuple): Figure size
    """
    fig, axes = plt.subplots(nrows=len(mfccs), ncols=1, figsize=figsize)
    if len(mfccs) == 1:
        axes = [axes]

    for i, mfcc in enumerate(mfccs):
        axes[i].imshow(mfcc, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'MFCC {i+1}')
    plt.tight_layout()
    plt.show()


def calc_fft(y, rate):
    """
    Calculate FFT of signal.

    Parameters:
    y (array): Audio signal
    rate (int): Sample rate

    Returns:
    tuple: FFT data and frequencies
    """
    n = len(y)
    freq = np.fft.rfftfreq(n, d=1/rate)
    Y = np.fft.rfft(y) / n
    return Y, freq


def load_audio_metadata(csv_path):
    """
    Load CSV metadata and calculate file lengths.

    Parameters:
    csv_path (str): Path to CSV file

    Returns:
    DataFrame: Metadata with lengths
    """
    df = pd.read_csv(csv_path)
    df['length'] = df['fname'].apply(lambda x: len(librosa.load(x, sr=None)[0]) / librosa.load(x, sr=None)[1])
    return df


def clean_audio_files(input_dir, output_dir, df, sr=16000, threshold=0.0005):
    """
    Batch process audio cleaning using envelope method.

    Parameters:
    input_dir (str): Input directory with raw audio
    output_dir (str): Output directory for cleaned audio
    df (DataFrame): Metadata dataframe
    sr (int): Target sample rate
    threshold (float): Envelope threshold
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for f in df.fname:
        signal, rate = librosa.load(os.path.join(input_dir, f), sr=sr)
        mask = envelope(signal, rate, threshold)
        wavfile.write(filename=os.path.join(output_dir, f), rate=rate, data=signal[mask])