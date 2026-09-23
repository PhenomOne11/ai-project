"""
Signal Processing Utilities
Spectral analysis, windowing, and frequency-domain operations
for real-time waveform diagnostics.

Implements:
- FFT-based power spectral density
- Windowing functions (Hann, Hamming, Blackman)
- Spectral entropy
- Signal-to-noise ratio estimation
- Peak detection
- Band energy extraction
"""

import numpy as np
from typing import Tuple, Optional, List


def compute_fft(signal: np.ndarray, sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute single-sided FFT magnitude spectrum.
    
    Args:
        signal: 1D time-domain signal
        sample_rate: Sampling frequency (Hz)
    
    Returns:
        (frequencies, magnitudes) - both 1D arrays
    """
    signal = signal.flatten().astype(np.float64)
    n = len(signal)
    
    if n == 0:
        return np.array([]), np.array([])
    
    # Apply Hann window
    windowed = signal * np.hanning(n)
    
    fft_vals = np.fft.rfft(windowed)
    magnitudes = np.abs(fft_vals) * 2.0 / n
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    
    return frequencies, magnitudes


def power_spectral_density(
    signal: np.ndarray,
    sample_rate: float = 1.0,
    window: str = "hann",
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate power spectral density using Welch's method (simplified).
    
    Args:
        signal: 1D time-domain signal
        sample_rate: Sampling frequency
        window: Window function name
        nperseg: Segment length (default: len(signal)//4)
    
    Returns:
        (frequencies, psd_values)
    """
    signal = signal.flatten().astype(np.float64)
    n = len(signal)
    
    if n == 0:
        return np.array([]), np.array([])
    
    if nperseg is None:
        nperseg = min(n, max(n // 4, 16))
    
    nperseg = min(nperseg, n)
    
    # Window function
    if window == "hann":
        win = np.hanning(nperseg)
    elif window == "hamming":
        win = np.hamming(nperseg)
    elif window == "blackman":
        win = np.blackman(nperseg)
    else:
        win = np.ones(nperseg)
    
    # Segment with 50% overlap
    step = nperseg // 2
    segments = []
    for start in range(0, n - nperseg + 1, step):
        seg = signal[start:start + nperseg] * win
        segments.append(seg)
    
    if len(segments) == 0:
        segments = [signal[:nperseg] * win[:len(signal[:nperseg])]]
    
    # Average periodograms
    psd_sum = np.zeros(nperseg // 2 + 1)
    for seg in segments:
        fft_seg = np.fft.rfft(seg)
        psd_sum += np.abs(fft_seg) ** 2
    
    psd = psd_sum / len(segments)
    win_norm = np.sum(win ** 2)
    if win_norm > 0:
        psd /= (sample_rate * win_norm)
    
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)
    
    return frequencies, psd


def spectral_entropy(signal: np.ndarray, normalize: bool = True) -> float:
    """
    Compute spectral entropy of a signal.
    
    High entropy → broadband noise
    Low entropy → concentrated spectral peaks
    
    Returns value in [0, 1] if normalized.
    """
    signal = signal.flatten().astype(np.float64)
    
    if len(signal) < 2:
        return 0.0
    
    _, magnitudes = compute_fft(signal)
    
    power = magnitudes ** 2
    total_power = power.sum()
    
    if total_power < 1e-12:
        return 0.0
    
    pdf = power / total_power
    pdf = pdf[pdf > 0]
    
    se = -np.sum(pdf * np.log2(pdf + 1e-12))
    
    if normalize and len(pdf) > 1:
        se /= np.log2(len(pdf))
    
    return float(np.clip(se, 0.0, 1.0))


def signal_to_noise_ratio(signal: np.ndarray, noise_estimate: Optional[np.ndarray] = None) -> float:
    """
    Estimate signal-to-noise ratio in dB.
    
    If noise_estimate is None, uses the signal's own high-frequency content
    as a noise proxy.
    """
    signal = signal.flatten().astype(np.float64)
    
    if len(signal) < 4:
        return 0.0
    
    if noise_estimate is not None:
        noise = noise_estimate.flatten().astype(np.float64)
    else:
        # Estimate noise from high-frequency content
        _, mags = compute_fft(signal)
        n_mags = len(mags)
        if n_mags < 4:
            return 0.0
        
        # Top 25% frequencies as noise
        cutoff = int(0.75 * n_mags)
        signal_power = np.sum(mags[:cutoff] ** 2)
        noise_power = np.sum(mags[cutoff:] ** 2)
        
        if noise_power < 1e-12:
            return 100.0
        
        return float(10.0 * np.log10(signal_power / noise_power))
    
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-12:
        return 100.0
    
    return float(10.0 * np.log10(signal_power / noise_power))


def find_peaks(
    signal: np.ndarray,
    min_height: Optional[float] = None,
    min_distance: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple peak detection in 1D signal.
    
    Args:
        signal: 1D array
        min_height: Minimum peak height threshold
        min_distance: Minimum samples between peaks
    
    Returns:
        (peak_indices, peak_values)
    """
    signal = signal.flatten().astype(np.float64)
    n = len(signal)
    
    if n < 3:
        return np.array([], dtype=int), np.array([])
    
    # Find local maxima
    peaks = []
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if min_height is None or signal[i] >= min_height:
                peaks.append(i)
    
    if len(peaks) == 0:
        return np.array([], dtype=int), np.array([])
    
    # Enforce minimum distance
    if min_distance > 1 and len(peaks) > 1:
        filtered = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered[-1] >= min_distance:
                filtered.append(p)
        peaks = filtered
    
    peak_indices = np.array(peaks, dtype=int)
    peak_values = signal[peak_indices]
    
    return peak_indices, peak_values


def band_energy(
    signal: np.ndarray,
    sample_rate: float,
    low_freq: float,
    high_freq: float
) -> float:
    """
    Compute energy in a specific frequency band.
    
    Returns the fraction of total energy in [low_freq, high_freq].
    """
    frequencies, magnitudes = compute_fft(signal, sample_rate)
    
    if len(frequencies) == 0:
        return 0.0
    
    power = magnitudes ** 2
    total_power = power.sum()
    
    if total_power < 1e-12:
        return 0.0
    
    mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    band_power = power[mask].sum()
    
    return float(band_power / total_power)


def zero_crossing_rate(signal: np.ndarray) -> float:
    """
    Compute zero-crossing rate of a signal.
    
    ZCR = (1/N) * Σ |sign(x[n]) - sign(x[n-1])| / 2
    """
    signal = signal.flatten().astype(np.float64)
    
    if len(signal) < 2:
        return 0.0
    
    signs = np.sign(signal)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    
    return float(crossings / (len(signal) - 1))


def apply_window(signal: np.ndarray, window_type: str = "hann") -> np.ndarray:
    """Apply a window function to a signal."""
    n = len(signal)
    
    windows = {
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
        "bartlett": np.bartlett,
        "rectangular": lambda n: np.ones(n)
    }
    
    win_fn = windows.get(window_type, np.hanning)
    return signal * win_fn(n)