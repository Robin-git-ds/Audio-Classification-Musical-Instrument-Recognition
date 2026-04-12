"""
Audio Classification Configuration
Centralized configuration for audio processing parameters.
"""

class AudioConfig:
    """
    Configuration class for audio classification parameters.
    """
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, sr=16000, threshold=0.0005):
        self.mode = mode  # 'conv' or 'lstm'
        self.nfilt = nfilt  # Number of filters for filter bank
        self.nfeat = nfeat  # Number of MFCC features
        self.nfft = nfft  # FFT window size
        self.sr = sr  # Sample rate
        self.threshold = threshold  # Envelope threshold
        self.step = int(sr / 10)  # Step size (160ms at 16kHz)
        self.min = None  # Min value for normalization (set after training)
        self.max = None  # Max value for normalization (set after training)

    def save(self, filepath):
        """Save config to pickle file."""
        import pickle
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath):
        """Load config from pickle file."""
        import pickle
        with open(filepath, 'rb') as file:
            return pickle.load(file)