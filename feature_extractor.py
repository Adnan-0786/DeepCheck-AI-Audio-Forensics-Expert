# feature_extractor.py

import numpy as np
import librosa
from scipy.fftpack import dct

# ==========================================
# 1. CONSTANTS & HYPERPARAMETERS
# ==========================================
SR = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 320
N_LFCC_FILTERS = 40  # Number of linear filters
N_LFCC_COEFS = 20    # Number of static cepstral coefficients to retain
N_MGDC_COEFS = 20  # Number of MGD Cepstral Coefficients
MGD_GAMMA = 0.9    # Denominator smoothing factor (reduces resonance spikes)
MGD_ALPHA = 0.4    # Power compression factor


# ==========================================
# Other functions (filterbanks)
# ==========================================

def bark_filterbank(sr, n_fft, n_bark=22):
    freqs = np.linspace(0, sr/2, n_fft//2 + 1)
    bark = 6 * np.arcsinh(freqs / 600)
    bark_bins = np.linspace(bark.min(), bark.max(), n_bark + 1)  # n_bark+1 not +2
    
    fb = np.zeros((n_bark, len(freqs)))
    for i in range(n_bark):
        lower = bark_bins[i]
        center = bark_bins[i + 1]
        if i + 2 < len(bark_bins):
            upper = bark_bins[i + 2]
        else:
            upper = bark.max()
        
        for j, b in enumerate(bark):
            if lower <= b <= center:
                fb[i, j] = (b - lower) / (center - lower + 1e-8)
            elif center < b <= upper:
                fb[i, j] = (upper - b) / (upper - center + 1e-8)
    
    # Normalize so weights sum to 1
    fb = fb / (fb.sum(axis=1, keepdims=True) + 1e-8)
    return fb

bark_fb = bark_filterbank(SR, N_FFT)

def linear_filterbank(sr, n_fft, n_filters=40):
    """Generates a linearly spaced triangular filterbank."""
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    filter_edges = np.linspace(0, sr / 2, n_filters + 2)
    
    fb = np.zeros((n_filters, len(freqs)))
    for i in range(n_filters):
        lower = filter_edges[i]
        center = filter_edges[i + 1]
        upper = filter_edges[i + 2]
        
        for j, f in enumerate(freqs):
            if lower <= f <= center:
                fb[i, j] = (f - lower) / (center - lower + 1e-8)
            elif center < f <= upper:
                fb[i, j] = (upper - f) / (upper - center + 1e-8)
                
    # Normalize filters
    fb = fb / (fb.sum(axis=1, keepdims=True) + 1e-8)
    return fb

lfcc_fb = linear_filterbank(SR, N_FFT, N_LFCC_FILTERS)

# ==========================================
# 2. INDIVIDUAL EXTRACTORS
# ==========================================
def extract_lfcc(audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Computes 60-dim LFCC (20 Static + 20 Delta + 20 Delta-Delta).
    Returns shape: (T, 60)
    """
    # 1. Magnitude Spectrogram
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                        win_length=win_length, window='hann')
    magnitude = np.abs(stft)
    
    # 2. Linear Subband Energy
    linear_energy = np.dot(lfcc_fb, magnitude**2)
    log_energy = np.log(linear_energy + 1e-8)
    
    # 3. Discrete Cosine Transform (Decorrelation)
    lfcc_static = dct(log_energy, type=2, axis=0, norm='ortho')[:N_LFCC_COEFS, :]
    
    # 4. Dynamic Derivatives (Velocity and Acceleration)
    delta = librosa.feature.delta(lfcc_static, order=1)
    delta_delta = librosa.feature.delta(lfcc_static, order=2)
    
    # 5. Fusion
    # Shapes: static (20, T), delta (20, T), delta_delta (20, T)
    features = np.concatenate([
        lfcc_static,
        delta,
        delta_delta
    ], axis=0) # Shape: (60, T)
    
    return features.T # Shape: (T, 60)

def extract_mgd(audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Computes 40-dim Modified Group Delay Cepstral Coefficients (20 Static + 20 Delta).
    Returns shape: (T, 40)
    """
    # 1. Create standard window and time-ramped window
    w = librosa.filters.get_window('hann', win_length)
    
    # Time vector centered at 0 to prevent extreme linear phase shifts
    n_vector = np.arange(win_length) - (win_length // 2)
    n_w = w * n_vector
    
    # 2. Compute the two STFTs
    # X: Standard STFT
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                     win_length=win_length, window=w)
    
    # Y: Time-ramped STFT
    Y = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                     win_length=win_length, window=n_w)
    
    # Extract real and imaginary parts
    X_R, X_I = np.real(X), np.imag(X)
    Y_R, Y_I = np.real(Y), np.imag(Y)
    magnitude = np.abs(X)
    
    # 3. Compute Modified Group Delay
    numerator = (X_R * Y_R) + (X_I * Y_I)
    denominator = (magnitude ** (2 * MGD_GAMMA)) + 1e-8
    
    tau_m = numerator / denominator
    
    # Apply compression to reduce dynamic range
    mgd_spec = np.sign(tau_m) * (np.abs(tau_m) ** MGD_ALPHA)
    
    # 4. DCT for Cepstral Coefficients (Decorrelation)
    mgdc_static = dct(mgd_spec, type=2, axis=0, norm='ortho')[:N_MGDC_COEFS, :]
    
    # 5. Dynamic Derivatives (Velocity)
    mgdc_delta = librosa.feature.delta(mgdc_static, order=1)
    
    # 6. Fusion
    # Shapes: static (20, T), delta (20, T)
    features = np.concatenate([
        mgdc_static,
        mgdc_delta
    ], axis=0) # Shape: (40, T)
    
    return features.T # Shape: (T, 40)

def extract_bark_prosody(audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """Computes your existing 42-dim Bark, BFCC, and Pitch features."""
    # ... adapted from your notebook ...
    # STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, 
                         win_length=win_length, window='hann')
    magnitude = np.abs(stft)
    
    # === FIX: Use squared magnitude for energy ===
    bark_energy = np.dot(bark_fb, magnitude**2)
    log_bark = np.log1p(bark_energy)
    
    # BFCC (DCT of log Bark energies)
    bfcc = dct(log_bark, type=2, axis=0, norm='ortho')
    
    # === FIX: Don't transpose yet ===
    # Shape: (22, n_frames)
    
    # Delta features of FIRST 6 BFCCs only
    delta = librosa.feature.delta(bfcc[:6], order=1)
    delta_delta = librosa.feature.delta(bfcc[:6], order=2)
    
    '''# === PITCH FEATURES (simplified approximation) ===
    # Compute pitch using autocorrelation
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=100, fmax=400, sr=sr,
        frame_length=win_length, hop_length=hop_length
    )
    pitch_period = np.nan_to_num(f0, nan=0.0)
    pitch_period = 1.0 / (pitch_period + 1e-8)  # Convert F0 to period
    pitch_period = pitch_period.reshape(1, -1)  # (1, n_frames)
    
    # Simplified pitch correlation approximation using voiced probability
    pitch_corr = voiced_probs.reshape(1, -1)  # Use as proxy
    pitch_corr = np.tile(pitch_corr, (6, 1))  # Repeat for 6 bands'''

    # === PITCH FEATURES (Speed Optimized for Deadline) ===
    f0 = librosa.yin(
        audio, fmin=100, fmax=400, sr=sr,
        frame_length=win_length, hop_length=hop_length
    )
    pitch_period = np.nan_to_num(f0, nan=0.0)
    pitch_period = 1.0 / (pitch_period + 1e-8)  
    pitch_period = pitch_period.reshape(1, -1)  
    
    # Mock voiced probabilities to preserve 42-feature tensor shape
    voiced_probs = np.where(f0 > 0, 1.0, 0.0) 
    pitch_corr = voiced_probs.reshape(1, -1)  
    pitch_corr = np.tile(pitch_corr, (6, 1))
    
    # === NON-STATIONARITY (spectral flux approximation) ===
    spectral_flux = np.sqrt(np.sum(np.diff(magnitude**2, axis=1)**2, axis=0))
    
    # Pad with one zero at the start to restore original n_frames length
    spectral_flux = np.pad(spectral_flux, (1, 0), mode='constant')
    spectral_flux = spectral_flux.reshape(1, -1)
    
    # Concatenate all features
    # bfcc: (22, n_frames), delta: (6, n_frames), etc.
    features = np.concatenate([
        bfcc,              # 22
        delta,             # 6
        delta_delta,       # 6
        pitch_corr[:6],    # 6 (placeholder)
        pitch_period,      # 1
        spectral_flux      # 1
    ], axis=0)
    
    return features.T

# ==========================================
# 3. NORMALIZATION & PROCESSING
# ==========================================
def apply_utterance_cmvn(features):
    """Applies Utterance-Level CMVN."""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-8
    return (features - mean) / std

def stack_frames(X, context=2):
    """Add temporal context by stacking neighboring frames."""
    num_frames, num_features = X.shape
    padded = np.pad(X, ((context, context), (0, 0)), mode='edge')
    stacked = np.zeros((num_frames, num_features * (2 * context + 1)))
    for i in range(num_frames):
        stacked[i] = padded[i : i + 2*context + 1].reshape(-1)
    return stacked

# ==========================================
# 4. THE MAIN PIPELINE API
# ==========================================
def process_audio_file(audio_path):
    """
    The main public function called by your dataloader.
    Returns the fully processed T x (142 * (2*context + 1)) tensor.
    """
    audio, _ = librosa.load(audio_path, sr=SR)
    
    # 1. Extract
    lfcc = extract_lfcc(audio)
    mgd = extract_mgd(audio)
    bark = extract_bark_prosody(audio)
    
    # Ensure time alignment (truncate to shortest temporal length)
    min_len = min(lfcc.shape[0], mgd.shape[0], bark.shape[0])
    
    # 2. Fuse
    fused_features = np.concatenate([
        lfcc[:min_len, :], 
        mgd[:min_len, :], 
        bark[:min_len, :]
    ], axis=1) # Shape: (min_len, 142)
    
    # 3. Normalize
    normalized_features = apply_utterance_cmvn(fused_features)
    
    # 4. Contextualize (if your model requires 2D inputs rather than sequence inputs)
    # Note: For CNNs/Conformers, you might skip stacking and pass the 2D matrix directly.
    final_tensor = stack_frames(normalized_features, context=2)
    
    return final_tensor