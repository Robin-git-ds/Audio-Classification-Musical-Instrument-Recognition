# Audio Classification: Musical Instrument Recognition

A comprehensive machine learning project for classifying audio clips into 10 musical instrument categories using a CNN-LSTM hybrid model. The project features a systematic pipeline from data exploration to model evaluation, with modular utilities for audio processing and visualization.

## 🎯 Project Overview

This project implements an end-to-end audio classification system that:
- Processes raw audio files using advanced signal processing techniques
- Extracts meaningful features (MFCC, FFT, Filter Banks)
- Trains deep learning models (CNN and LSTM architectures)
- Achieves 83% accuracy and 0.863 F1-score on instrument classification
- Provides a scalable, modular pipeline for audio and signal processing tasks

## 📁 Project Structure

```
Audio-Classification-Musical-Instrument-Recognition/
├── 01_Data_Exploration_Visualization.ipynb    # Initial data analysis and visualization
├── 02_Data_Preparation_Cleaning.ipynb         # Audio preprocessing and cleaning
├── 03_Feature_Extraction_Model_Training.ipynb # Feature extraction and model training
├── 04_Model_Evaluation_Predictions.ipynb      # Model evaluation and inference
├── archive_UrbanSound8K_classification.ipynb  # Archived separate project
├── audio_utils.py                              # Audio processing utilities
├── config.py                                   # Configuration management
├── pyproject.toml                              # Poetry dependency management
├── requirements.txt                            # Pip requirements
├── README.md                                   # This file
└── saved_models/                               # Trained model checkpoints
    └── audio_classification.hdf5
```

## 🚀 Features

### Core Audio Processing Functions (`audio_utils.py`)

#### Signal Processing
- **`envelope(y, rate, threshold=0.0005)`**: Extract signal envelope using rolling mean threshold detection
- **`calc_fft(y, rate)`**: Calculate Fast Fourier Transform with frequency bins

#### Visualization
- **`plot_signals(signals, figsize=(20, 5))`**: Plot time-domain audio signals
- **`plot_fft(fft, figsize=(20, 5))`**: Visualize frequency domain representations
- **`plot_fbank(fbank, figsize=(20, 5))`**: Display filter bank coefficients
- **`plot_mfccs(mfccs, figsize=(20, 5))`**: Show Mel-frequency cepstral coefficients

#### Data Pipeline
- **`load_audio_metadata(csv_path)`**: Load CSV metadata and calculate audio file lengths
- **`clean_audio_files(input_dir, output_dir, df, sr=16000, threshold=0.0005)`**: Batch process audio cleaning using envelope method

### Configuration Management (`config.py`)

- **`AudioConfig`**: Centralized configuration class for model parameters
  - Audio processing settings (sample rate, FFT parameters)
  - Feature extraction configuration (MFCC, filter banks)
  - Model architecture parameters
  - Normalization bounds (min/max values)
  - Save/load functionality for configuration persistence

## 📋 Dependencies

### Core Dependencies
- **pandas** (>=1.3.0): Data manipulation and CSV handling
- **numpy** (>=1.21.0): Numerical computing
- **matplotlib** (>=3.5.0): Data visualization
- **scipy** (>=1.7.0): Scientific computing (signal processing)
- **python-speech-features** (>=0.6): Audio feature extraction
- **librosa** (>=0.9.0): Audio and music analysis
- **tensorflow** (>=2.8.0): Deep learning framework
- **scikit-learn** (>=1.0.0): Machine learning utilities
- **tqdm** (>=4.62.0): Progress bars
- **split-folders** (>=0.5.1): Dataset splitting utility

### Development Dependencies
- **jupyter** (>=1.0.0): Interactive notebook environment

## 🛠️ Installation

### Option 1: Using Poetry (Recommended)
```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and navigate to project
cd Audio-Classification-Musical-Instrument-Recognition

# Install dependencies
poetry install
```

### Option 2: Using pip
```bash
# Clone and navigate to project
cd Audio-Classification-Musical-Instrument-Recognition

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### 1. Data Preparation
Run the notebooks in sequence:

1. **01_Data_Exploration_Visualization.ipynb**
   - Load and explore audio dataset
   - Visualize signal waveforms, FFT, and MFCC features
   - Understand data distribution

2. **02_Data_Preparation_Cleaning.ipynb**
   - Split dataset into train/test folders
   - Apply envelope-based audio cleaning
   - Generate metadata CSV files

3. **03_Feature_Extraction_Model_Training.ipynb**
   - Extract MFCC features from cleaned audio
   - Train CNN and LSTM models
   - Save trained models and configuration

4. **04_Model_Evaluation_Predictions.ipynb**
   - Load trained models
   - Evaluate performance metrics
   - Make predictions on new audio files

### 2. Using Audio Utilities

```python
from audio_utils import envelope, plot_signals, clean_audio_files
import librosa

# Load audio
signal, sr = librosa.load('audio.wav', sr=16000)

# Extract envelope
mask = envelope(signal, sr, threshold=0.0005)
clean_signal = signal[mask]

# Visualize
plot_signals([signal, clean_signal])
```

### 3. Configuration Management

```python
from config import AudioConfig

# Create configuration
config = AudioConfig(mode='conv', nfilt=26, nfeat=13, sr=16000)

# Save configuration
config.save('model_config.pkl')

# Load configuration
loaded_config = AudioConfig.load('model_config.pkl')
```

## 🎵 Supported Instruments

The model classifies audio into 10 instrument categories:
- Piano
- Guitar
- Violin
- Drums
- Flute
- Trumpet
- Saxophone
- Cello
- Clarinet
- Trombone

## 📊 Performance Metrics

- **Accuracy**: 83%
- **F1-Score**: 0.863
- **Architecture**: CNN-LSTM Hybrid
- **Features**: 13 MFCC coefficients, 26 filter banks
- **Sample Rate**: 16kHz
- **Window Size**: 512 samples (32ms)

## 🔧 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfilt` | 26 | Number of filter banks |
| `nfeat` | 13 | Number of MFCC features |
| `nfft` | 512 | FFT window size |
| `sr` | 16000 | Sample rate (Hz) |
| `threshold` | 0.0005 | Envelope detection threshold |
| `step` | 1600 | Step size (samples) |

## 🚀 Future Enhancements

- [ ] Add support for real-time audio classification
- [ ] Implement data augmentation techniques
- [ ] Add more instrument categories
- [ ] Optimize model for edge deployment
- [ ] Add web interface for audio upload and classification
- [ ] Implement ensemble methods for better accuracy

## 📝 Notes

- The project uses Google Colab paths in some notebooks - update paths for local execution
- Models are trained on preprocessed audio segments
- Configuration files ensure reproducible results
- The pipeline is designed to be modular and extensible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.
