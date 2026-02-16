"""
Voice Detection: single script = API + local file analysis + audio features.

Usage:
  API server:  python app.py   OR  uvicorn app:app --reload --host 0.0.0.0 --port 8000
  Local file:  python app.py path/to/audio.mp3
"""

import base64
import json
import os
import re
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Literal

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

try:
    from scipy.signal import lfilter
except ImportError:
    lfilter = None
try:
    from scipy.stats import skew as _scipy_skew
except ImportError:
    _scipy_skew = None

try:
    import parselmouth
except ImportError:
    parselmouth = None

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

# =============================================================================
# Audio feature constants (from audio_features.py)
# =============================================================================
SR = 22050
HOP_LENGTH = 512
FMIN = 75.0
FMAX = 500.0


def _load_audio(path: str | None = None, audio_bytes: bytes | None = None, sr: int = SR):
    if librosa is None:
        raise ImportError("librosa is not installed")
    if path is not None and os.path.isfile(path):
        try:
            y, sr_out = librosa.load(path, sr=sr, mono=True)
            return y, sr_out
        except Exception as e:
            print(f"ERROR loading audio from path {path}: {e}")
            raise
    if audio_bytes is not None and len(audio_bytes) > 0:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp = f.name
        try:
            y, sr_out = librosa.load(tmp, sr=sr, mono=True)
            return y, sr_out
        except Exception as e:
            print(f"ERROR loading audio from bytes (temp file {tmp}): {e}")
            raise
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    raise ValueError("Either path or audio_bytes must be provided")


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if b is None or np.isnan(b) or b == 0:
        return default
    return float(a / b)


def _excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher); 0 for normal. Heavy-tailed → positive."""
    if x is None or len(x) < 4:
        return float("nan")
    x = np.asarray(x, dtype=float)
    m, s = np.mean(x), np.std(x)
    if s is None or s < 1e-12:
        return 0.0
    k = np.mean((x - m) ** 4) / (s**4) - 3.0
    return float(k)


def _entropy_from_hist(x: np.ndarray, bins: int = 30) -> float:
    """Entropy of distribution (histogram, normalized to sum=1)."""
    if x is None or len(x) < 2:
        return float("nan")
    x = np.asarray(x, dtype=float)
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist / (np.sum(hist) + 1e-12)
    p = p[p > 0]
    if len(p) < 2:
        return 0.0
    ent = -float(np.sum(p * np.log2(p)))
    max_ent = np.log2(len(p))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def _lpc_coeffs(frame: np.ndarray, order: int) -> np.ndarray | None:
    """Levinson-Durbin: return LPC coeffs [1, -a1, -a2, ...] or None."""
    if len(frame) < order + 10:
        return None
    r = np.correlate(frame, frame, mode="full")
    r = r[len(r) // 2 :]
    r = r[: order + 1] / (np.abs(r[0]) + 1e-12)
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = float(r[0])
    for k in range(1, order + 1):
        lam = (r[k] - np.dot(a[:k], r[k:0:-1])) / (e + 1e-12)
        a[1 : k + 1] = a[1 : k + 1] - lam * a[k - 1 :: -1]
        a[k] = lam
        e = (1.0 - lam * lam) * e
        if e <= 0:
            return None
    return np.concatenate([[1.0], -a[1:]])


def _residual_kurtosis_and_bispectrum(y: np.ndarray, sr: int, lpc_order: int = 16, frame_len: int = 512, seg_len: int = 256) -> tuple[float | None, float | None]:
    """
    Residual Forensic Pipeline: inverse LPC → residual, then kurtosis + bispectral phase coupling.
    Returns (residual_kurtosis, bispectral_phase_coupling) or (None, None) on failure.
    """
    if lfilter is None or len(y) < frame_len * 3:
        return None, None
    y = np.asarray(y, dtype=float)
    residuals: list[np.ndarray] = []
    for start in range(0, len(y) - frame_len, frame_len // 2):
        frame = y[start : start + frame_len]
        frame = frame - np.mean(frame)
        if np.std(frame) < 1e-10:
            continue
        a = _lpc_coeffs(frame, lpc_order)
        if a is None:
            continue
        res = lfilter([1.0], a.tolist(), frame)
        res = res[lpc_order:]
        if len(res) > 0:
            residuals.append(res)
    if not residuals:
        return None, None
    residual = np.concatenate(residuals)
    if len(residual) < 100:
        return None, None
    r_kurt = _excess_kurtosis(residual)
    if np.isnan(r_kurt):
        r_kurt = None

    # Bispectrum: phase coupling strength (real throat = coupled; AI = weak/random)
    n_seg = min(50, len(residual) // seg_len)
    if n_seg < 2:
        return r_kurt, None
    bispec_mags: list[float] = []
    for i in range(n_seg):
        seg = residual[i * seg_len : (i + 1) * seg_len]
        if len(seg) < seg_len:
            continue
        X = np.fft.rfft(seg)
        nf = len(X)
        b = 0.0
        count = 0
        for f1 in range(1, min(nf // 2, 30)):
            for f2 in range(f1, min(nf - f1, 30)):
                if f1 + f2 < nf:
                    b += np.abs(X[f1] * X[f2] * np.conj(X[f1 + f2]))
                    count += 1
        if count > 0:
            bispec_mags.append(float(np.real(b) / count))
    if len(bispec_mags) < 2:
        return r_kurt, None
    bispec_strength = float(np.mean(bispec_mags))
    # Normalize bispectral_phase_coupling to [0, 1] range
    # Values > 1.0 might be valid signals from professional recordings, not just artifacts
    # Instead of zeroing, normalize by the maximum observed value
    if bispec_strength > 1.0:
        # Normalize: divide by max value and cap at 1.0
        max_bispec = max(bispec_mags) if bispec_mags else 1.0
        bispec_strength = min(1.0, bispec_strength / max_bispec) if max_bispec > 0 else 0.0
    # Ensure it's in [0, 1] range
    bispec_strength = max(0.0, min(1.0, bispec_strength))
    return r_kurt, bispec_strength


def harmonic_phase_coherence(y, sr, n_fft=1024, hop=512):
    """
    Phase difference variance between harmonic bins over time.
    Real speech: phase-locked harmonics (stable). AI: neural decoder → unstable phase drift.
    """
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop)
    S = np.fft.rfft(frames, axis=0)
    phases = np.angle(S)
    mags = np.abs(S)
    f0_bins = np.argmax(mags, axis=0)
    phase_vars = []
    for t in range(len(f0_bins)):
        f0 = int(f0_bins[t])
        if f0 <= 0 or 3 * f0 >= phases.shape[0]:
            continue
        diffs = [
            phases[2 * f0, t] - 2 * phases[f0, t],
            phases[3 * f0, t] - 3 * phases[f0, t],
        ]
        phase_vars.append(np.var(diffs))
    if len(phase_vars) < 5:
        return None
    return float(np.mean(phase_vars))


def fractal_dimension(x: np.ndarray) -> float | None:
    """Fractal dimension of 1D series. Humans: chaotic turbulent airflow; TTS: synthetic noise, not fractal."""
    N = len(x)
    if N < 10:
        return None
    L = np.sum(np.abs(np.diff(x)))
    d = np.log(N) / (np.log(N) + np.log(N / (L + 1e-9)))
    return float(d)


def _phase_based_features(phase: np.ndarray, hop: int, sr: int) -> tuple[float | None, float | None, float | None]:
    """Phase variance, instantaneous frequency jitter (phase derivative), group delay variance. AI = smoother phase."""
    if phase.size < 10:
        return None, None, None
    pv = float(np.var(phase))
    phase_unw = np.unwrap(phase, axis=1)
    dphase_t = np.diff(phase_unw, axis=1)
    inst_freq = (sr / hop) * dphase_t / (2 * np.pi)
    inst_freq_flat = inst_freq[np.isfinite(inst_freq)]
    if_jitter = float(np.var(inst_freq_flat)) if len(inst_freq_flat) >= 5 else None
    dphase_f = np.diff(phase_unw, axis=0)
    gd_var = float(np.var(dphase_f)) if dphase_f.size >= 5 else None
    return pv, if_jitter, gd_var


def _hnr_mean(y: np.ndarray, sr: float) -> float | None:
    """Harmonic-to-noise ratio mean via Parselmouth. High uniform HNR → synthetic."""
    if parselmouth is None or len(y) < sr * 0.2:
        return None
    try:
        snd = parselmouth.Sound(y, sr)
        hnr_obj = snd.to_harmonicity()
        vals = hnr_obj.values[~hnr_obj.values.isnan()]
        if len(vals) < 5:
            return None
        return float(np.mean(vals))
    except Exception:
        return None


def _silence_and_breath(rms: np.ndarray, y: np.ndarray, sr: int, hop: int, rms_thresh_quantile: float = 0.2) -> tuple[float | None, float | None]:
    """Silence duration distribution (entropy of run lengths), breath noise ratio (low-energy broadband in quiet frames)."""
    thresh = np.quantile(rms, rms_thresh_quantile) if rms.size > 0 else 0.0
    is_silence = rms < max(thresh, 1e-12)
    run_lengths: list[int] = []
    n = len(is_silence)
    i = 0
    while i < n:
        val = is_silence[i]
        r = 0
        while i < n and is_silence[i] == val:
            r += 1
            i += 1
        if val:
            run_lengths.append(r)
    if len(run_lengths) < 2:
        silence_ent = None
    else:
        run_sec = np.array(run_lengths, dtype=float) * hop / sr
        silence_ent = _entropy_from_hist(run_sec, bins=15)
        if np.isnan(silence_ent):
            silence_ent = None
        else:
            silence_ent = float(silence_ent)
    # Breath: low-energy broadband (e.g. 200–2k Hz) in quiet frames vs total energy
    n_fft = 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_band = (freqs >= 200) & (freqs <= 2000)
    band_energy = np.sum(S[low_band, :], axis=0)
    total_energy = np.sum(S, axis=0) + 1e-12
    quiet_frames = rms < max(thresh, 1e-12)
    if np.sum(quiet_frames) < 3:
        breath_ratio = None
    else:
        breath_ratio = float(np.mean(band_energy[quiet_frames] / total_energy[quiet_frames]))
    return silence_ent, breath_ratio


def extract_features(
    audio_path: str | None = None,
    audio_bytes: bytes | None = None,
    sr: int = SR,
) -> dict[str, float] | None:
    """Gen-2: distribution shape and chaos over time (language-agnostic, mic-robust)."""
    if librosa is None:
        print("ERROR: librosa is None - library not installed")
        return None
    try:
        y, sr_actual = _load_audio(path=audio_path, audio_bytes=audio_bytes, sr=sr)
        print(f"DEBUG: Loaded audio - length: {len(y) if y is not None else 0}, sr: {sr_actual}")
    except Exception as e:
        print(f"ERROR in _load_audio: {e}")
        import traceback
        traceback.print_exc()
        return None
    if y is None or len(y) < sr_actual * 0.1:
        print(f"ERROR: Audio too short or None - length: {len(y) if y is not None else 0}, min required: {sr_actual * 0.1}")
        return None

    # Trim first and last 0.1 seconds to remove digital "on/off" clicks that create artifacts
    trim_samples = int(sr_actual * 0.1)
    if len(y) > 2 * trim_samples:
        y = y[trim_samples:-trim_samples]
        print(f"DEBUG: Trimmed {trim_samples} samples from start and end")

    out: dict[str, float] = {}
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    # Envelope skewness: humans = asymmetric emotional bursts; AI = symmetric, well-shaped modulation
    if _scipy_skew is not None and len(rms) >= 10:
        env_sk = _scipy_skew(rms)
        if not np.isnan(env_sk):
            out["envelope_skewness"] = round(float(env_sk), 4)

    # 1. spectral_flatness_variance (vocoder artifacts)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH)[0]
    flatness_var = np.nanvar(flatness)
    if not np.isnan(flatness_var):
        out["spectral_flatness_variance"] = round(float(flatness_var), 6)

    # 2. zcr_entropy (breath turbulence; AI vocoders suppress randomness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_ent = _entropy_from_hist(zcr, bins=30)
    if not np.isnan(zcr_ent):
        out["zcr_entropy"] = round(float(zcr_ent), 4)

    # 3. am_entropy (amplitude modulation irregularity; humans modulate loudness unconsciously)
    am_window = min(5, max(1, len(rms) // 20))
    am_envelope = np.convolve(rms, np.ones(am_window) / am_window, mode="same")
    am_diff = np.diff(am_envelope)
    am_ent = _entropy_from_hist(am_diff, bins=30)
    if not np.isnan(am_ent):
        out["am_entropy"] = round(float(am_ent), 4)

    f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=sr_actual, hop_length=HOP_LENGTH)
    f0_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr_actual, 
    hop_length=HOP_LENGTH)
    voiced = (f0 >= FMIN) & (f0 <= FMAX)
    n_voiced = int(np.sum(voiced))

    # 4. pitch_jitter_variance, 5. pitch_jitter_kurtosis (jitter over time; AI cannot fake shape)
    if n_voiced >= 4:
        f0_voiced = f0[voiced]
        periods = 1.0 / np.clip(f0_voiced, 1e-6, None)
        mean_period = np.mean(periods)
        if mean_period > 1e-12:
            jitter_series = (periods - mean_period) / mean_period
            jitter_var = np.var(jitter_series)
            if not np.isnan(jitter_var):
                out["pitch_jitter_variance"] = round(float(jitter_var), 6)
            kurt = _excess_kurtosis(jitter_series)
            if not np.isnan(kurt):
                # Cap at 12.0 - values above are digital artifacts, not human instability
                out["pitch_jitter_kurtosis"] = round(float(np.clip(kurt, -5.0, 12.0)), 4)
            # Jitter variability over time: low std = controlled (AI), high = chaotic (human)
            jitter_std = np.std(jitter_series)
            if not np.isnan(jitter_std):
                out["jitter_temporal_std"] = round(float(jitter_std), 6)

        # 6. pitch_accel_kurtosis (neuromotor jerk; AI pitch curves are spline-smoothed)
        voiced_both = voiced[:-1] & voiced[1:]
        voiced_idx = np.where(voiced_both)[0]
        if len(voiced_idx) > 4:
            df0 = np.diff(f0)[voiced_idx]
            dt = np.diff(f0_times)[voiced_idx]
            dt = np.where(dt <= 0, np.nan, dt)
            accel = np.abs(np.divide(df0, dt, dtype=float))
            accel = accel[np.isfinite(accel)]
            if len(accel) >= 4:
                kurt_accel = _excess_kurtosis(accel)
                if not np.isnan(kurt_accel):
                    # Cap at 12.0 - values above are digital artifacts, not human instability
                    out["pitch_accel_kurtosis"] = round(float(np.clip(kurt_accel, -5.0, 12.0)), 4)

        # Over-Smoothness Index: Var(Δpitch) / Var(pitch). Low = smooth trajectory = AI; high = chaotic = human
        if n_voiced >= 5:
            f0_v = f0[voiced]
            var_pitch = float(np.var(f0_v))
            delta_pitch = np.diff(f0_v)
            var_delta = float(np.var(delta_pitch))
            if not np.isnan(var_pitch) and var_pitch > 1e-12 and not np.isnan(var_delta):
                osi = var_delta / var_pitch
                out["over_smoothness_index"] = round(min(max(osi, 0.0), 100.0), 6)
            if len(delta_pitch) >= 2:
                pitch_delta_std = np.std(delta_pitch)
                if not np.isnan(pitch_delta_std):
                    out["pitch_delta_std"] = round(float(pitch_delta_std), 6)

    # 7. voiced_unvoiced_transition_var (articulation realism; AI transitions are interpolated)
    run_lengths: list[int] = []
    n = len(voiced)
    i = 0
    while i < n:
        val = voiced[i]
        r = 0
        while i < n and voiced[i] == val:
            r += 1
            i += 1
        run_lengths.append(r)
    if len(run_lengths) >= 2:
        run_lengths_arr = np.array(run_lengths, dtype=float)
        run_lengths_sec = run_lengths_arr * (HOP_LENGTH / sr_actual)
        trans_var = np.var(run_lengths_sec)
        if not np.isnan(trans_var):
            out["voiced_unvoiced_transition_var"] = round(float(trans_var), 6)

    # 8. Micro-timing irregularity (5–30 ms scale; AI cannot fake neuromotor timing chaos)
    pitch_frames = np.where(voiced)[0]
    if len(pitch_frames) >= 3:
        dt_frames = np.diff(pitch_frames) * (HOP_LENGTH / sr_actual)
        if len(dt_frames) >= 2:
            timing_ent = _entropy_from_hist(dt_frames, bins=20)
            if not np.isnan(timing_ent):
                out["pitch_timing_entropy"] = round(float(timing_ent), 4)
        if len(dt_frames) >= 4:
            timing_kurt = _excess_kurtosis(dt_frames)
            if not np.isnan(timing_kurt):
                out["pitch_timing_kurtosis"] = round(float(np.clip(timing_kurt, -5.0, 10.0)), 4)

    # 9. Residual Forensic Pipeline: inverse LPC strips "voice" to reveal glottal source (O(N))
    r_kurt, bispec = _residual_kurtosis_and_bispectrum(y, sr_actual, lpc_order=16, frame_len=512, seg_len=256)
    if r_kurt is not None:
        out["residual_kurtosis"] = round(float(np.clip(r_kurt, -5.0, 25.0)), 4)
    if bispec is not None and not np.isnan(bispec):
        # Clamp bispectral_phase_coupling to [0, 1] - values > 1.0 are artifacts
        bispec_clamped = 0.0 if bispec > 1.0 else max(0.0, bispec)
        out["bispectral_phase_coupling"] = round(float(bispec_clamped), 6)

    # 10. Harmonic phase coherence instability (ElevenLabs/neural vocoders: harmonics not phase-locked)
    phase_var = harmonic_phase_coherence(y, sr_actual, n_fft=1024, hop=HOP_LENGTH)
    if phase_var is not None:
        out["harmonic_phase_instability"] = round(float(phase_var), 6)

    # 11. Formant trajectory smoothness (IndicSynth: overly smooth; human: micro-chaotic articulation)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr_actual, hop_length=HOP_LENGTH)[0]
    centroid_diff = np.diff(centroid)
    out["formant_motion_variance"] = round(float(np.var(centroid_diff)), 6)

    # 12. Breath turbulence fractal (humans: chaotic turbulent airflow; TTS: synthetic noise, not fractal)
    n_fft = 1024
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=HOP_LENGTH)
    S = np.abs(S_complex)
    freqs = librosa.fft_frequencies(sr=sr_actual, n_fft=n_fft)

    # Long-window spectral consistency: Corr(spectrum_t, spectrum_t+5s). Human drifts; AI stays consistent.
    offset_5s = int(5.0 * sr_actual / HOP_LENGTH)
    if S.shape[1] > offset_5s + 5:
        mid = S.shape[1] // 2
        idx_b = min(mid + offset_5s, S.shape[1] - 1)
        a, b = S[:, mid], S[:, idx_b]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            lwc = np.corrcoef(a, b)[0, 1]
            if not np.isnan(lwc):
                out["long_window_spectral_corr"] = round(float(np.clip(lwc, -1.0, 1.0)), 4)
    elif S.shape[1] > 20:
        # Shorter audio: use half-window offset
        offset_half = max(10, S.shape[1] // 4)
        mid = S.shape[1] // 2
        idx_b = min(mid + offset_half, S.shape[1] - 1)
        a, b = S[:, mid], S[:, idx_b]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            lwc = np.corrcoef(a, b)[0, 1]
            if not np.isnan(lwc):
                out["long_window_spectral_corr"] = round(float(np.clip(lwc, -1.0, 1.0)), 4)

    # ---------- Tier 2 Cross-Scale: Glottal drift consistency (F0(t) vs F0(t+Δ)) ----------
    # Humans: vocal fold tension drifts over seconds → lower correlation. AI: resampled pitch → too stable or too decorrelated.
    offset_f0 = int(5.0 * sr_actual / HOP_LENGTH)
    if len(f0) > offset_f0 + 10:
        a_f0, b_f0 = f0[: len(f0) - offset_f0], f0[offset_f0 :]
        if np.std(a_f0) > 1e-12 and np.std(b_f0) > 1e-12:
            drift_corr = np.corrcoef(a_f0, b_f0)[0, 1]
            if not np.isnan(drift_corr):
                out["pitch_drift_consistency"] = round(float(np.clip(drift_corr, -1.0, 1.0)), 4)
    elif len(f0) > 30:
        offset_h = max(10, len(f0) // 4)
        a_f0, b_f0 = f0[: len(f0) - offset_h], f0[offset_h :]
        if np.std(a_f0) > 1e-12 and np.std(b_f0) > 1e-12:
            drift_corr = np.corrcoef(a_f0, b_f0)[0, 1]
            if not np.isnan(drift_corr):
                out["pitch_drift_consistency"] = round(float(np.clip(drift_corr, -1.0, 1.0)), 4)
    
    # Detect phone quality: check if there's significant energy above 4kHz
    high_mask = freqs >= 4000
    if np.any(high_mask):
        high_band_energy = np.sum(S[high_mask, :])
        total_energy = np.sum(S) + 1e-12
        high_freq_ratio = high_band_energy / total_energy
        # If less than 2% energy above 4kHz, it's phone quality (telephone cuts off at ~3.4kHz)
        is_phone_quality = high_freq_ratio < 0.02
        out["is_phone_quality"] = 1.0 if is_phone_quality else 0.0
        out["high_freq_energy_ratio"] = round(float(high_freq_ratio), 6)
    else:
        # Sample rate too low to have 4kHz, assume phone quality
        is_phone_quality = True
        out["is_phone_quality"] = 1.0
    
    high_mask = freqs >= 4000
    high_band_envelope = np.sum(S[high_mask, :], axis=0) if np.any(high_mask) else np.zeros(S.shape[1])
    fd = fractal_dimension(high_band_envelope)
    if fd is not None and not np.isnan(fd):
        out["breath_turbulence_fractal"] = round(float(np.clip(fd, 0.0, 2.0)), 4)

    # ---------- Phase-based (1): AI matches magnitude but fails phase ----------
    phase = np.angle(S_complex)
    pv, if_jitter, gd_var = _phase_based_features(phase, HOP_LENGTH, sr_actual)
    if pv is not None and not np.isnan(pv):
        out["phase_variance"] = round(float(pv), 6)
    if if_jitter is not None and not np.isnan(if_jitter):
        out["inst_freq_jitter"] = round(float(np.clip(if_jitter, 0.0, 1e6)), 6)
    if gd_var is not None and not np.isnan(gd_var):
        out["group_delay_variance"] = round(float(gd_var), 6)

    # ---------- HNR (2): High uniform HNR → synthetic ----------
    hnr = _hnr_mean(y, sr_actual)
    if hnr is not None and not np.isnan(hnr):
        out["hnr_mean"] = round(float(hnr), 4)

    # ---------- Pitch/amplitude (3): amplitude shimmer on voiced frames ----------
    if n_voiced >= 4 and len(rms) == len(voiced):
        voiced_rms = rms[voiced]
        if len(voiced_rms) >= 4 and np.mean(voiced_rms) > 1e-12:
            shimmer = np.std(voiced_rms) / (np.mean(voiced_rms) + 1e-12)
            out["amplitude_shimmer"] = round(float(shimmer), 6)

    # ---------- Modulation spectrum (4): envelope modulation + energy fluctuation entropy ----------
    if len(rms) >= 32:
        mod_spectrum = np.abs(np.fft.rfft(rms - np.mean(rms)))
        mod_ent = _entropy_from_hist(mod_spectrum, bins=20)
        if not np.isnan(mod_ent):
            out["envelope_modulation_entropy"] = round(float(mod_ent), 4)
        env_diff = np.diff(rms)
        fluct_ent = _entropy_from_hist(env_diff, bins=30)
        if not np.isnan(fluct_ent):
            out["energy_fluctuation_entropy"] = round(float(fluct_ent), 4)

    # ---------- Spectral flux / transient sharpness (5): real = irregular onsets, AI = smooth ----------
    onset_str = librosa.onset.onset_strength(y=y, sr=sr_actual, hop_length=HOP_LENGTH)
    out["onset_strength_variance"] = round(float(np.var(onset_str)), 6)

    # ---------- Neural vocoder artifact (6): HF 6k–12k variance, aliasing ratio ----------
    hf_mask = (freqs >= 6000) & (freqs <= 12000)
    if np.any(hf_mask):
        hf_energy = np.sum(S[hf_mask, :], axis=0)
        out["hf_spectral_variance_6k_12k"] = round(float(np.var(hf_energy)), 6)
    alias_mask = freqs >= (sr_actual / 2) * 0.4
    if np.any(alias_mask):
        alias_energy = np.sum(S[alias_mask, :])
        total_energy = np.sum(S) + 1e-12
        out["aliasing_energy_ratio"] = round(float(alias_energy / total_energy), 6)

    # ---------- Formant trajectory stability (7): F1 variance via Parselmouth if available ----------
    if parselmouth is not None:
        try:
            snd_p = parselmouth.Sound(y, sr_actual)
            formant_obj = snd_p.to_formant_burg()
            f1_vals: list[float] = []
            for t in formant_obj.ts():
                f1 = formant_obj.get_value_at_time(1, t)
                if not np.isnan(f1) and f1 > 0:
                    f1_vals.append(float(f1))
            if len(f1_vals) >= 5:
                out["formant_f1_variance"] = round(float(np.var(f1_vals)), 2)
        except Exception:
            pass

    # ---------- Breath & silence (8): silence duration entropy, breath noise ratio ----------
    silence_ent, breath_ratio = _silence_and_breath(rms, y, sr_actual, HOP_LENGTH)
    if silence_ent is not None:
        out["silence_duration_entropy"] = round(silence_ent, 4)
    if breath_ratio is not None:
        out["breath_noise_ratio"] = round(breath_ratio, 6)

    # ---------- Tier 2 Cross-Scale: Multi-scale phase coupling ----------
    # Humans: hierarchical nonlinear coupling (micro + meso). AI: matches local coupling, fails across longer windows.
    if len(y) > sr_actual * 1.5:
        half = len(y) // 2
        _, bispec_first = _residual_kurtosis_and_bispectrum(y[:half], sr_actual, lpc_order=16, frame_len=512, seg_len=256)
        _, bispec_second = _residual_kurtosis_and_bispectrum(y[half:], sr_actual, lpc_order=16, frame_len=512, seg_len=256)
        if bispec_first is not None and bispec_second is not None and not np.isnan(bispec_first) and not np.isnan(bispec_second):
            out["multi_scale_phase_coupling"] = round(float(abs(bispec_first - bispec_second)), 4)

    # ---------- Tier 2 Cross-Scale: Breath–excitation synchronization ----------
    # Humans: breath noise aligns with excitation energy. AI: breath layer added separately → decorrelated.
    seg_len = 20
    n_frames = len(rms)
    if n_frames >= seg_len * 5:
        total_per_frame = np.sum(S, axis=0) + 1e-12
        high_mask = freqs >= 4000
        breath_per_frame = (np.sum(S[high_mask, :], axis=0) / total_per_frame) if np.any(high_mask) else np.zeros(n_frames)
        rms_seg, breath_seg = [], []
        for i in range(0, min(len(rms), len(breath_per_frame)) - seg_len, seg_len):
            rms_seg.append(float(np.mean(rms[i : i + seg_len])))
            breath_seg.append(float(np.mean(breath_per_frame[i : i + seg_len])))
        if len(rms_seg) >= 5 and np.std(rms_seg) > 1e-12 and np.std(breath_seg) > 1e-12:
            sync = np.corrcoef(rms_seg, breath_seg)[0, 1]
            if not np.isnan(sync):
                out["breath_excitation_sync"] = round(float(np.clip(sync, -1.0, 1.0)), 4)

    # ---------- Advanced Forensic Features (9): High-frequency aliasing, bicoherence, pitch jerk, RIR ----------
    
    # 1. High-Frequency Aliasing Ratio (16k-22k vs 2k-8k correlation)
    if sr_actual >= 22050:  # Need high sample rate for 16k+ analysis
        hf_mask_16k = (freqs >= 16000) & (freqs <= 22000)
        lf_mask_2k = (freqs >= 2000) & (freqs <= 8000)
        if np.any(hf_mask_16k) and np.any(lf_mask_2k):
            hf_band = np.sum(S[hf_mask_16k, :], axis=0)
            lf_band = np.sum(S[lf_mask_2k, :], axis=0)
            if len(hf_band) > 10 and len(lf_band) > 10:
                # Calculate spectral flatness of each band
                hf_flatness = np.mean(hf_band) / (np.std(hf_band) + 1e-12)
                lf_flatness = np.mean(lf_band) / (np.std(lf_band) + 1e-12)
                # Pearson correlation between bands (mirror artifact detection)
                if np.std(hf_band) > 1e-12 and np.std(lf_band) > 1e-12:
                    correlation = np.corrcoef(hf_band, lf_band)[0, 1]
                    if not np.isnan(correlation):
                        out["hf_lf_correlation"] = round(float(correlation), 6)
                    # Aliasing ratio: high-freq flatness / low-freq flatness
                    if lf_flatness > 1e-12:
                        aliasing_ratio = hf_flatness / lf_flatness
                        out["hf_aliasing_ratio"] = round(float(aliasing_ratio), 6)
    
    # 2. Bicoherence (non-linear phase coupling) - enhanced version
    # We already have bispectral_phase_coupling, but add bicoherence normalization
    if bispec is not None and bispec > 0:
        # Bicoherence is normalized bispectrum (0-1 range)
        # Values > 0.8 indicate strong phase coupling (human), < 0.3 = uncoupled (AI)
        # We already calculate bispectral_phase_coupling, but normalize it properly
        # This is already done in _residual_kurtosis_and_bispectrum, but add explicit bicoherence
        pass  # Already covered by bispectral_phase_coupling
    
    # 3. Pitch Jerk (3rd derivative of pitch) - "Spline-Smooth" detection
    if n_voiced >= 10:
        f0_voiced_clean = f0[voiced]
        f0_times_voiced = f0_times[voiced]
        if len(f0_voiced_clean) >= 10:
            # Remove NaN and inf
            valid_mask = np.isfinite(f0_voiced_clean) & np.isfinite(f0_times_voiced)
            if np.sum(valid_mask) >= 10:
                f0_clean = f0_voiced_clean[valid_mask]
                times_clean = f0_times_voiced[valid_mask]
                # Calculate derivatives
                if len(f0_clean) >= 4:
                    df0 = np.diff(f0_clean)
                    dt = np.diff(times_clean)
                    dt = np.where(dt <= 0, np.nan, dt)
                    velocity = np.divide(df0, dt, out=np.full_like(df0, np.nan), where=dt > 0)
                    velocity = velocity[np.isfinite(velocity)]
                    if len(velocity) >= 3:
                        d2f0 = np.diff(velocity)
                        dt2 = dt[:-1][np.isfinite(velocity[:-1])] if len(dt) > 1 else dt
                        if len(dt2) > 0 and len(d2f0) > 0:
                            acceleration = np.divide(d2f0[:len(dt2)], dt2[:len(d2f0)], 
                                                    out=np.full_like(d2f0[:len(dt2)], np.nan), 
                                                    where=dt2[:len(d2f0)] > 0)
                            acceleration = acceleration[np.isfinite(acceleration)]
                            if len(acceleration) >= 2:
                                # 3rd derivative (jerk)
                                jerk = np.diff(acceleration)
                                if len(jerk) > 0:
                                    jerk_var = np.var(jerk)
                                    jerk_mean = np.mean(np.abs(jerk))
                                    if not np.isnan(jerk_var):
                                        out["pitch_jerk_variance"] = round(float(jerk_var), 6)
                                    if not np.isnan(jerk_mean):
                                        out["pitch_jerk_mean"] = round(float(jerk_mean), 6)
                                    # AI tell: near-zero or overly smooth jerk
                                    if jerk_var < 0.01 or jerk_mean < 0.001:
                                        out["pitch_jerk_smooth"] = 1.0  # Flag for AI
                                    else:
                                        out["pitch_jerk_smooth"] = 0.0

    # ---------- Chunkwise chaos consistency: std(pitch_jerk_variance) across chunks ----------
    # Humans: chaos uneven across segments. AI: chaos magnitude high but distribution stable (injected).
    if n_voiced >= 20 and len(f0) >= 40:
        n_frames = len(f0)
        n_chunks = 4
        chunk_size = max(10, n_frames // n_chunks)
        chunk_jerk_vars: list[float] = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_frames)
            if end - start < 8:
                continue
            f0_chunk = f0[start:end]
            times_chunk = f0_times[start:end]
            voiced_chunk = (f0_chunk >= FMIN) & (f0_chunk <= FMAX)
            if np.sum(voiced_chunk) < 6:
                continue
            f0_c = f0_chunk[voiced_chunk]
            t_c = times_chunk[voiced_chunk]
            if len(f0_c) < 5:
                continue
            df0 = np.diff(f0_c)
            dt = np.diff(t_c)
            dt = np.where(dt <= 0, np.nan, dt)
            vel = np.divide(df0, dt, out=np.full_like(df0, np.nan), where=dt > 0)
            vel = vel[np.isfinite(vel)]
            if len(vel) < 3:
                continue
            acc = np.diff(vel)
            if len(acc) < 2:
                continue
            jerk = np.diff(acc)
            if len(jerk) > 0:
                jv = float(np.var(jerk))
                if not np.isnan(jv):
                    chunk_jerk_vars.append(jv)
        if len(chunk_jerk_vars) >= 3:
            cj_std = np.std(chunk_jerk_vars)
            if not np.isnan(cj_std):
                out["pitch_jerk_chunk_std"] = round(float(cj_std), 6)
    
    # 4. Group Delay Ripple (Phase Discontinuity) - enhanced
    # We already calculate group_delay_variance, but add frame-boundary detection
    if phase_var is not None and gd_var is not None:
        # Calculate group delay ripple at frame boundaries (every 10-20ms)
        frame_boundary_ms = 15  # Typical neural vocoder frame size
        frame_boundary_samples = int(frame_boundary_ms * sr_actual / 1000 / HOP_LENGTH)
        if frame_boundary_samples > 0 and phase.size > frame_boundary_samples * 2:
            phase_unw = np.unwrap(phase, axis=1)
            gd_ripples = []
            for i in range(frame_boundary_samples, phase_unw.shape[1] - frame_boundary_samples, frame_boundary_samples):
                before = phase_unw[:, i-1]
                after = phase_unw[:, i]
                ripple = np.abs(after - before)
                gd_ripples.append(np.mean(ripple))
            if len(gd_ripples) >= 3:
                ripple_var = np.var(gd_ripples)
                if not np.isnan(ripple_var):
                    out["group_delay_ripple_variance"] = round(float(ripple_var), 6)
    
    # 5. Room Impulse Response (RIR) Inconsistency
    # Estimate reverb time T60 or DRR for different segments
    if len(y) > sr_actual * 2:  # Need at least 2 seconds
        n_segments = 3
        segment_len = len(y) // n_segments
        t60_estimates = []
        drr_estimates = []
        for i in range(n_segments):
            seg_start = i * segment_len
            seg_end = min((i + 1) * segment_len, len(y))
            seg = y[seg_start:seg_end]
            if len(seg) > sr_actual * 0.5:  # At least 0.5 seconds
                # Simple T60 estimation: find decay time from RMS
                seg_rms = librosa.feature.rms(y=seg, hop_length=HOP_LENGTH)[0]
                if len(seg_rms) > 10:
                    peak_idx = np.argmax(seg_rms)
                    if peak_idx < len(seg_rms) - 5:
                        decay = seg_rms[peak_idx:]
                        # Find where it drops to -60dB (T60)
                        peak_val = seg_rms[peak_idx]
                        target_val = peak_val * 0.001  # -60dB
                        below_target = np.where(decay < target_val)[0]
                        if len(below_target) > 0:
                            t60_samples = below_target[0] * HOP_LENGTH
                            t60_sec = t60_samples / sr_actual
                            if 0.1 < t60_sec < 2.0:  # Reasonable range
                                t60_estimates.append(t60_sec)
                    # DRR: Direct-to-Reverberant Ratio (simplified)
                    # Direct sound is first 50ms, reverb is tail
                    direct_samples = int(0.05 * sr_actual)
                    if len(seg) > direct_samples * 2:
                        direct_energy = np.sum(seg[:direct_samples] ** 2)
                        reverb_energy = np.sum(seg[direct_samples:] ** 2)
                        if reverb_energy > 0:
                            drr = 10 * np.log10((direct_energy + 1e-12) / reverb_energy)
                            drr_estimates.append(drr)
        # Check consistency across segments
        if len(t60_estimates) >= 2:
            t60_var = np.var(t60_estimates)
            if not np.isnan(t60_var):
                out["rir_t60_variance"] = round(float(t60_var), 6)
        if len(drr_estimates) >= 2:
            drr_var = np.var(drr_estimates)
            if not np.isnan(drr_var):
                out["rir_drr_variance"] = round(float(drr_var), 6)
            # High variance = inconsistent room properties = AI composite
            # Human voices can naturally have DRR variance of 100-500, so threshold must be very high
            # DISABLED: Too confusing, flags human voices incorrectly
            # if drr_var > 500:  # Very high threshold - only flag extreme inconsistencies
            #     out["rir_inconsistent"] = 1.0
            # else:
            #     out["rir_inconsistent"] = 0.0
            out["rir_inconsistent"] = 0.0  # Always 0 - disabled

    return out


def features_to_json_string(features: dict[str, float] | None) -> str:
    if not features:
        return "{}"
    return json.dumps(features, separators=(",", ":"))


def raw_to_forensic_indicators(features: dict[str, float] | None) -> str:
    """
    Convert raw features into qualitative forensic indicators so the LLM reasons
    on biological chaos vs controlled smoothness instead of raw floats.
    """
    if not features:
        return "No forensic indicators (features unavailable)."
    lines: list[str] = []

    # Jitter deviation from human norm (pitch_jitter_variance)
    v = features.get("pitch_jitter_variance")
    if v is not None:
        if v < 0.002:
            lines.append("Jitter deviation from human norm: LOW (controlled, synthetic-like).")
        elif v < 0.02:
            lines.append("Jitter deviation from human norm: MODERATE.")
        else:
            lines.append("Jitter deviation from human norm: HIGH (biological instability).")

    # Residual kurtosis vs human range
    rk = features.get("residual_kurtosis")
    if rk is not None:
        if rk < 3.0:
            lines.append("Residual kurtosis vs human range: LOW (smoothed glottal source, AI-like).")
        elif rk < 6.5:
            lines.append("Residual kurtosis vs human range: MODERATE.")
        else:
            lines.append("Residual kurtosis vs human range: HIGH (sharp glottal spikes, human-like).")

    # Phase coupling irregularity (bispectral)
    bp = features.get("bispectral_phase_coupling")
    if bp is not None:
        if bp < 0.12:
            lines.append("Phase coupling irregularity: WEAK (uncoupled phases, AI-like).")
        elif bp < 0.25:
            lines.append("Phase coupling irregularity: MODERATE.")
        else:
            lines.append("Phase coupling irregularity: STRONG (physical frequency coupling, human-like).")

    # Source physics authenticity (Tier 1 — overrides expressive prosody when synthetic)
    lwc = features.get("long_window_spectral_corr")
    if rk is not None and bp is not None and lwc is not None:
        if rk < 3 and bp < 0.05 and lwc > 0.9:
            lines.append("Source physics authenticity (Tier 1): SYNTHETIC — smooth glottal, no nonlinear coupling, spectrally frozen; cannot be human. Overrides jitter/shimmer/AM entropy.")
        elif rk >= 5 and bp >= 0.2:
            lines.append("Source physics authenticity (Tier 1): AUTHENTIC — glottal and phase coupling consistent with biological source.")

    # Temporal micro-instability (pitch_timing_entropy)
    te = features.get("pitch_timing_entropy")
    if te is not None:
        if te < 0.25:
            lines.append("Temporal micro-instability: SMOOTH (spline-like timing, AI-like).")
        elif te < 0.45:
            lines.append("Temporal micro-instability: MODERATE.")
        else:
            lines.append("Temporal micro-instability: CHAOTIC (neuromotor irregularity, human-like).")

    # Breath/silence entropy
    se = features.get("silence_duration_entropy")
    if se is not None:
        if se < 0.4:
            lines.append("Breath entropy randomness: STRUCTURED (uniform gaps, AI-like).")
        elif se < 0.7:
            lines.append("Breath entropy randomness: MODERATE.")
        else:
            lines.append("Breath entropy randomness: RANDOM (natural variability, human-like).")

    # Over-Smoothness Index: low OSI = smooth trajectory = AI
    osi = features.get("over_smoothness_index")
    if osi is not None:
        if osi < 0.25:
            lines.append("Pitch trajectory stability: ABNORMALLY HIGH (indicative of synthetic control).")
        elif osi < 0.6:
            lines.append("Pitch trajectory stability: MODERATELY SMOOTH.")
        else:
            lines.append("Pitch trajectory stability: NORMAL / CHAOTIC (human-like fluctuation).")

    # Long-window spectral consistency (0.45–0.65 common in expressive TTS; human long recordings often 0.2–0.4)
    lwc = features.get("long_window_spectral_corr")
    if lwc is not None:
        if lwc > 0.65:
            lines.append("Long-window spectral consistency: HIGH (synthetic consistency; human voice drifts more).")
        elif lwc >= 0.4:
            lines.append("Long-window spectral consistency: MODERATE (0.4–0.65 common in expressive TTS; human long recordings often 0.2–0.4).")
        else:
            lines.append("Long-window spectral consistency: LOW (human drift / fatigue / micro-variation).")

    # Emotional envelope regularity (skewness)
    sk = features.get("envelope_skewness")
    if sk is not None:
        abs_sk = abs(sk)
        if abs_sk < 0.2:
            lines.append("Emotional envelope regularity: SYMMETRIC (synthetic smoothing; human bursts are asymmetric).")
        elif abs_sk < 0.5:
            lines.append("Emotional envelope regularity: MODERATELY ASYMMETRIC.")
        else:
            lines.append("Emotional envelope regularity: ASYMMETRIC (natural emotional bursts, human-like).")

    # Jitter temporal variability (std of jitter over time): low = controlled = AI
    jstd = features.get("jitter_temporal_std")
    if jstd is not None:
        if jstd < 0.004:
            lines.append("Jitter temporal variability: VERY LOW (controlled imperfection; AI-like).")
        elif jstd < 0.015:
            lines.append("Jitter temporal variability: MODERATE.")
        else:
            lines.append("Jitter temporal variability: HIGH (chaotic imperfection, human-like).")

    # Pitch-delta variability: low std(pitch_delta) = smooth = AI
    pds = features.get("pitch_delta_std")
    if pds is not None:
        if pds < 5.0:
            lines.append("Pitch-delta variability: LOW (smooth pitch changes, AI-like).")
        elif pds < 25.0:
            lines.append("Pitch-delta variability: MODERATE.")
        else:
            lines.append("Pitch-delta variability: HIGH (chaotic pitch movement, human-like).")

    # Tier 2 Cross-Scale: Glottal drift consistency (F0 over long window)
    pdc = features.get("pitch_drift_consistency")
    if pdc is not None:
        if pdc > 0.9:
            lines.append("Glottal drift consistency: TOO STABLE (pitch resampled/controlled over long window → expressive TTS).")
        elif pdc < 0.3:
            lines.append("Glottal drift consistency: LOW (natural vocal-fold tension drift, human-like).")
        else:
            lines.append("Glottal drift consistency: MODERATE.")

    # Tier 2 Cross-Scale: Multi-scale phase coupling (first vs second half)
    mspc = features.get("multi_scale_phase_coupling")
    if mspc is not None:
        if mspc > 0.3:
            lines.append("Multi-scale phase coupling: INCONSISTENT (phase coupling differs across segments → synthetic).")
        else:
            lines.append("Multi-scale phase coupling: CONSISTENT (hierarchical coupling across scale, human-like).")

    # Tier 2 Cross-Scale: Breath–excitation synchronization
    bes = features.get("breath_excitation_sync")
    if bes is not None:
        if bes < 0.2:
            lines.append("Breath–excitation synchronization: LOW (breath layer decorrelated from excitation → synthetic).")
        elif bes > 0.5:
            lines.append("Breath–excitation synchronization: HIGH (physiological alignment, human-like).")
        else:
            lines.append("Breath–excitation synchronization: MODERATE.")

    # Chunkwise chaos consistency: low std(jerk variance across chunks) = injected chaos (AI)
    pjcs = features.get("pitch_jerk_chunk_std")
    if pjcs is not None:
        if pjcs < 1e6:
            lines.append("Chunkwise chaos consistency: TOO STABLE (chaos magnitude high but distribution stable across segments → injected chaos, AI).")
        else:
            lines.append("Chunkwise chaos consistency: UNEVEN (chaos varies across segments, human-like).")

    # Physiological consistency: chaos + timing. Contradiction + weak source → AI; contradiction + strong source → controlled human.
    rk = features.get("residual_kurtosis")
    pjv = features.get("pitch_jerk_variance")
    te = features.get("pitch_timing_entropy")
    bp = features.get("bispectral_phase_coupling")
    if te is not None and te < 0.1 and pjv is not None and pjv > 1e8:
        if rk is not None and rk < 4 and bp is not None and bp < 0.05:
            lines.append("Physiological consistency: CONTRADICTION + weak source physics → synthetic expressive TTS.")
        else:
            lines.append("Physiological consistency: Contradiction (locked timing + high jerk) but STRONG source physics (glottal/phase) → interpret as controlled human speech (e.g. reading), not synthetic.")
    elif te is not None or rk is not None:
        lines.append("Physiological consistency: Check whether chaos and timing co-vary; prosody contradictions alone cannot override strong source physics.")

    if not lines:
        return "Forensic indicators could not be derived from available features."
    return "\n".join(lines)


def _format_calculated_features(features: dict | None) -> str:
    """Single line of all calculated feature values (name=value) for inclusion in any explanation."""
    if not features:
        return "Calculated features: (none)."
    pairs = [f"{k}={v}" for k, v in sorted(features.items()) if v is not None]
    return "Calculated features: " + ", ".join(pairs) if pairs else "Calculated features: (none)."


def extract_features_from_base64(audio_base64: str) -> dict[str, float] | None:
    try:
        raw = base64.b64decode(audio_base64, validate=True)
        print(f"DEBUG: Decoded base64 - size: {len(raw)} bytes")
    except Exception as e:
        print(f"ERROR decoding base64: {e}")
        import traceback
        traceback.print_exc()
        return None
    result = extract_features(audio_bytes=raw, sr=SR)
    if result:
        print(f"DEBUG: Extracted {len(result)} features successfully")
    else:
        print("DEBUG: extract_features returned None")
    return result


# =============================================================================
# Config
# =============================================================================
OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
API_KEY = os.environ.get("VOICE_DETECTION_API_KEY", "sk_test_123456789")
API_LOGS_DIR = os.environ.get("API_LOGS_DIR", "api_logs")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(
    title="Voice Detection API",
    description="Classify voice as AI_GENERATED or HUMAN from base64 MP3.",
)


class VoiceDetectionRequest(BaseModel):
    language: Literal["Tamil", "English", "Hindi", "Malayalam", "Telugu"] = Field(
        ..., description="Language of the audio"
    )
    audioFormat: Literal["mp3"] = Field(..., description="Must be mp3")
    audioBase64: str = Field(..., description="Base64-encoded MP3 audio")


def _log_api_request_response(request_body: dict, response_body: dict, status_code: int = 200) -> None:
    """Write request and response to api_logs folder (JSON, one file per request)."""
    try:
        os.makedirs(API_LOGS_DIR, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        uid = uuid.uuid4().hex[:8]
        path = os.path.join(API_LOGS_DIR, f"{ts}_{uid}.json")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_code": status_code,
            "request": request_body,
            "response": response_body,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"API log write failed: {e}")


def rule_override(features: dict | None) -> str | None:
    """
    Gen-2 rule-based vote: human/ai counters from shape & chaos features.
    Residual forensic pipeline has veto: if both residual_kurtosis and bispectral
    phase coupling indicate AI (smear), we do NOT return HUMAN even if other
    features look human-like. Returns "HUMAN", "AI_GENERATED", or None.
    """
    if not features or len(features) == 0:
        return None

    timing_ent = features.get("pitch_timing_entropy")
    residual_kurt = features.get("residual_kurtosis")
    pitch_jerk_var = features.get("pitch_jerk_variance")
    bispec = features.get("bispectral_phase_coupling")
    lwc = features.get("long_window_spectral_corr")
    over_smooth = features.get("over_smoothness_index")
    is_phone_quality = features.get("is_phone_quality", 0.0) > 0.5

    # ========== PHYSIOLOGICAL CONTRADICTION (strong suspicion, not absolute proof) ==========
    # Contradiction = ultra-low timing entropy + high pitch jerk (chaos). Only classify AI if source physics is WEAK.
    # Humans can have controlled prosody (reading script, trained speakers) but real vocal-fold physics.
    # AI = synthetic source + controlled prosody. So: contradiction + weak source physics → AI;
    # contradiction + strong source physics → do NOT override (controlled human speech).
    contradiction = (
        timing_ent is not None and timing_ent < 0.1
        and pitch_jerk_var is not None and pitch_jerk_var > 1e8
    )
    if contradiction and residual_kurt is not None and residual_kurt < 4 and bispec is not None and bispec < 0.05:
        return "AI_GENERATED"
    # Else: contradiction but strong source physics (e.g. residual_kurtosis=14.29, good phase coupling) → controlled human, fall through

    # ========== SOURCE PHYSICS GATE (Tier 1 — Physics > Prosody ALWAYS) ==========
    # Smooth glottal excitation + no nonlinear phase coupling + spectrally frozen identity
    # = vocoder/IndiSynth-style TTS. Skip for phone quality (bandwidth naturally depresses residual/bispec).
    if not is_phone_quality and (
        residual_kurt is not None
        and residual_kurt < 3
        and bispec is not None
        and bispec < 0.05
        and lwc is not None
        and lwc > 0.9
    ):
        return "AI_GENERATED"

    # ========== TIER 2: Simulated chaos + rhythmically constrained (ElevenLabs-style) ==========
    # Human-like Tier 1 (high residual, decent phase) but grid-locked timing + moderately stable spectrum
    # = expressive neural TTS. Real humans with strong chaos rarely have timing entropy near zero.
    if (
        residual_kurt is not None
        and residual_kurt > 8
        and bispec is not None
        and bispec > 0.1
        and timing_ent is not None
        and timing_ent < 0.05
        and lwc is not None
        and lwc > 0.4
    ):
        return "AI_GENERATED"
    # Looser Tier 2: low timing + high spectral consistency + weak nonlinear coupling → TTS
    if (
        timing_ent is not None
        and timing_ent < 0.08
        and lwc is not None
        and lwc > 0.7
        and bispec is not None
        and bispec < 0.2
        and not is_phone_quality
    ):
        return "AI_GENERATED"

    # ========== SMOOTHNESS TRIAD (Expressive neural TTS signature) ==========
    # Chaos present BUT temporally structured: high smoothness + low timing entropy + weak nonlinear coupling.
    # Skip for phone quality (bandwidth can produce low bispec and low timing entropy on clear speech).
    if not is_phone_quality and (
        over_smooth is not None
        and over_smooth > 0.7
        and timing_ent is not None
        and timing_ent < 0.15
        and bispec is not None
        and bispec < 0.15
    ):
        return "AI_GENERATED"

    # ========== CHUNKWISE CHAOS CONSISTENCY (Injected chaos = globally controlled) ==========
    # Low std(pitch_jerk_variance across chunks) = chaos magnitude high but distribution stable = AI.
    pitch_jerk_chunk_std = features.get("pitch_jerk_chunk_std")
    if (
        pitch_jerk_chunk_std is not None
        and pitch_jerk_chunk_std < 1e6
        and timing_ent is not None
        and timing_ent < 0.2
    ):
        return "AI_GENERATED"

    # ========== STEP 1: THE "ZERO" VETO (Catches 90% of AI) ==========
    # Humans are biological - we are NEVER perfect. Even professional narrators have micro-errors.
    # If pitch_timing_entropy is extremely low (< 0.05), it's likely AI.
    # BUT: Professional recordings (OpenSLR, audiobooks) can have low timing entropy.
    # Only apply veto if timing is low AND residual kurtosis is also low (both indicate AI).
    formant_var = features.get("formant_motion_variance")
    bispec = features.get("bispectral_phase_coupling")
    phase_instab = features.get("harmonic_phase_instability")
    
    # Check for multiple strong AI signals (pure AI TTS signature)
    # If we have low timing + low bispectral + high phase instability, it's AI regardless of formant variance
    strong_ai_combo = (
        timing_ent is not None and timing_ent < 0.08 and
        bispec is not None and bispec < 0.2 and
        phase_instab is not None and phase_instab > 1.5
    )
    
    if timing_ent is not None and timing_ent < 0.05:
        # Check for strong human counter-evidence (high residual kurtosis = biological glottal source)
        # SIMPLIFIED: If residual_kurtosis > 6.5, always skip veto (strong human signal)
        if residual_kurt is not None and residual_kurt > 6.5:
            # Very high residual kurtosis = strong human glottal source - always trust this
            # Don't apply veto - continue with normal classification
            pass
        elif residual_kurt is not None and residual_kurt > 6.0:
            # High residual kurtosis suggests human glottal source despite low timing entropy
            # BUT: if we have strong AI combo AND low formant variance, apply veto
            # If we have VERY strong human signals (high residual + very high formant variance), trust human
            very_strong_human = (
                formant_var is not None and formant_var > 100000
            )
            if not is_phone_quality and strong_ai_combo and not very_strong_human:
                return "AI_GENERATED"
            # Don't apply veto - continue with normal classification
            pass
        elif residual_kurt is not None and residual_kurt > 5.0:
            # Moderate residual kurtosis - check other features before vetoing
            # If strong AI combo, apply veto regardless of formant variance (skip for phone)
            if not is_phone_quality and strong_ai_combo:
                return "AI_GENERATED"
            # Only veto if we also have low formant variance (smooth formants = AI)
            if not is_phone_quality and formant_var is not None and formant_var < 50:
                # Low formant variance + low timing + moderate residual = likely AI
                return "AI_GENERATED"
            # Otherwise continue with normal classification (high formant variance = natural articulation)
        elif residual_kurt is not None and residual_kurt > 2.5:
            # Moderate-low residual kurtosis - check formant variance before vetoing
            # If strong AI combo, apply veto regardless of formant variance (pure AI priority)
            if not is_phone_quality and strong_ai_combo:
                return "AI_GENERATED"
            # High formant variance (> 50) indicates natural articulation, even with moderate residual
            if formant_var is not None and formant_var > 50:
                # High formant variance suggests natural articulation despite moderate residual
                # Don't apply veto - continue with normal classification
                pass
            else:
                # Low formant variance + low timing + moderate-low residual = likely AI
                if not is_phone_quality:
                    return "AI_GENERATED"
        else:
            # Low timing entropy + very low residual kurtosis (< 2.5) = strong AI signal
            # Exception: phone quality — bandwidth naturally depresses residual kurtosis; do not veto.
            if not is_phone_quality:
                return "AI_GENERATED"
            # Phone: fall through to human/ai counters
    
    # Check for strong AI combo even if timing entropy is slightly above 0.05 (0.05-0.08)
    # This catches AI TTS that has slightly higher timing entropy but still has other strong AI signals
    # Skip for phone quality (bandwidth depresses residual/bispec).
    if not is_phone_quality and timing_ent is not None and 0.05 <= timing_ent < 0.08:
        if strong_ai_combo and residual_kurt is not None and residual_kurt < 5.0:
            # Strong AI combo + moderate-low residual = pure AI, classify immediately
            return "AI_GENERATED"

    human = 0
    ai = 0
    
    # Detect phone quality audio (missing high frequencies above 4kHz)
    is_phone_quality = features.get("is_phone_quality", 0.0) > 0.5

    jitter_kurt = features.get("pitch_jitter_kurtosis")
    accel_kurt = features.get("pitch_accel_kurtosis")
    zcr_ent = features.get("zcr_entropy")
    am_ent = features.get("am_entropy")
    jitter_var = features.get("pitch_jitter_variance")
    flatness_var = features.get("spectral_flatness_variance")
    trans_var = features.get("voiced_unvoiced_transition_var")
    bispec = features.get("bispectral_phase_coupling")
    phase_instab = features.get("harmonic_phase_instability")
    phase_var = features.get("phase_variance")
    hnr = features.get("hnr_mean")
    onset_var = features.get("onset_strength_variance")
    silence_ent = features.get("silence_duration_entropy")
    amplitude_shimmer = features.get("amplitude_shimmer")
    energy_fluct_ent = features.get("energy_fluctuation_entropy")

    # Detect very clean, studio-like material (audiobooks, voice-over booths)
    is_clean_studio = flatness_var is not None and flatness_var < 0.015
    
    # Detect professional/processed recordings (OpenSLR, professional narrators)
    # These can have low timing entropy but high residual kurtosis (biological source)
    # OR very high formant variance (natural articulation) even with moderate residual
    # BUT: Don't detect as professional if we have strong AI combo (pure AI priority)
    is_professional_recording = (
        not strong_ai_combo and  # Exclude if strong AI signals present
        timing_ent is not None and timing_ent < 0.15 and
        formant_var is not None and formant_var > 50 and  # Natural articulation variance
        (
            (residual_kurt is not None and residual_kurt > 5.5) or  # High residual = biological source
            (formant_var > 100000)  # Very high formant variance = strong natural articulation signal
        )
    )

    # ========== STEP 3: THE "PHONE CALL" HANDICAP ==========
    # Phone networks cut off all high frequencies (above 4kHz).
    # In phone quality mode: DO NOT look for high-frequency clues.
    # Accept "muddy" or "flat" audio as HUMAN unless it's suspiciously perfect (Rule #1 already caught that).
    if is_phone_quality:
        # Lower the bar for phone quality - don't penalize smoothness
        ai_threshold = 5  # Require very strong AI evidence
        human_threshold = 3  # Lower bar for human (phone quality naturally sounds smoother)
        # Ignore high-frequency features that don't exist in phone calls
        # Don't penalize low formant variance, low phase variance, etc. for phone quality
    else:
        # Normal thresholds for high-quality audio
        ai_threshold = 4
        human_threshold = 4
    
    # Studio audio adjustments (only if not phone quality)
    if not is_phone_quality and is_clean_studio:
        ai_threshold = max(ai_threshold, 5)
        human_threshold = max(human_threshold, 5)
    
    # Professional recordings: require more AI evidence to override human signals
    if is_professional_recording:
        ai_threshold = max(ai_threshold, 6)  # Require stronger AI evidence
    
    # Strong AI combo: prioritize pure AI detection (lower threshold)
    if strong_ai_combo:
        ai_threshold = min(ai_threshold, 3)  # Lower threshold to catch pure AI TTS
    
    # IndicSynth / smooth-formant TTS: if formants are smooth, need fewer AI votes to call AI
    # But only apply this if NOT phone quality AND NOT professional recording (phone/professional naturally have smooth formants)
    if not is_phone_quality and not is_professional_recording and formant_var is not None and formant_var < 15:
        ai_threshold = min(ai_threshold, 3)
    # When formants are smooth (IndicSynth cue), require more human evidence to still say HUMAN
    if not is_phone_quality and not is_professional_recording and formant_var is not None and formant_var < 12 and human_threshold < 5:
        human_threshold = 4

    # ---------------- HUMAN-leaning signals ----------------
    # ========== STEP 2: THE "GOLDILOCKS" ZONE ==========
    # Real human "chaos" (jitter/acceleration) falls in a specific range (4.0 to 9.0).
    # If a value is HUGE (> 12.0), it is NOT HUMAN. It is DIGITAL NOISE/ARTIFACTS.
    # Treat anything above 12.0 as "Garbage/Artifacts," NOT as evidence of a human.
    
    # Pitch jitter shape (Goldilocks zone: 4.0-9.0 is human, >12.0 is artifact)
    # But also give points for moderate values (2.0-4.0) - professional narrators can have controlled jitter
    if jitter_kurt is not None:
        if jitter_kurt > 12.0:
            # Excessive kurtosis = digital artifact/glitch, count as AI signal
            ai += 1
        elif jitter_kurt == 12.0 and strong_ai_combo:
            # Exactly 12.0 (capped value) + strong AI combo = treat as AI artifact
            ai += 1
        elif 4.0 <= jitter_kurt <= 9.0:
            # Goldilocks zone: real human chaos
            human += 1
        elif 2.0 <= jitter_kurt < 4.0:
            # Moderate values: professional narrators can have controlled jitter, still human
            human += 1
        # Between 9.0-12.0: suspicious but not clearly artifact, don't count either way

    # Strong weight on pitch_accel_kurtosis (micro-jerks) - Goldilocks zone
    if accel_kurt is not None:
        # Check for very strong human signals
        very_strong_human_signals = (
            residual_kurt is not None and residual_kurt > 6.5 and
            formant_var is not None and formant_var > 100000
        )
        if accel_kurt > 12.0:
            # Excessive kurtosis = digital artifact/glitch, count as AI signal
            ai += 1
        elif accel_kurt == 12.0:
            # Exactly 12.0 (capped value) - treat as human if we have strong human signals
            if very_strong_human_signals:
                human += 2  # Strong human neuromotor jerk
            elif strong_ai_combo:
                ai += 1  # AI artifact if strong AI combo
            # Otherwise neutral (between 9.0-12.0 range)
        elif 4.0 <= accel_kurt <= 9.0:
            # Goldilocks zone: real human chaos
            human += 2
        elif 2.0 <= accel_kurt < 4.0:
            # Moderate values: professional narrators can have controlled acceleration, still human
            human += 1
        elif 1.0 <= accel_kurt < 2.0:
            # Low but not zero: still human-leaning (professional control)
            human += 1

    # Breath turbulence and loudness chaos
    if zcr_ent is not None and zcr_ent > 0.78:
        human += 1
    if am_ent is not None and am_ent > 0.65:
        human += 1

    # Micro‑timing entropy (controlled but still organic for pros)
    # Professional narrators can have timing_entropy 0.2-0.4 and still be human
    if timing_ent is not None:
        if timing_ent > 0.55:
            human += 2
        elif timing_ent > 0.45:
            human += 1
        elif timing_ent > 0.2:
            # Professional narrators can have controlled timing (0.2-0.45), still human
            human += 1

    # Residual forensic human cues
    if residual_kurt is not None and residual_kurt > 6.0:
        # Extra weight for professional recordings (already accounted for in threshold adjustment)
        if is_professional_recording:
            human += 2  # Strong human signal for professional recordings
        else:
            human += 1
    # For professional recordings with very high formant variance, also credit moderate residual kurtosis
    elif is_professional_recording and residual_kurt is not None and residual_kurt > 4.0:
        # Moderate residual kurtosis + very high formant variance = still human-leaning
        human += 1
    # Bispectral should be clamped to [0, 1] - values > 1.0 are artifacts and should be ignored
    if bispec is not None and 0.0 < bispec <= 1.0 and bispec > 0.55:
        human += 1
    if amplitude_shimmer is not None and amplitude_shimmer > 0.08:
        human += 1
    if energy_fluct_ent is not None and energy_fluct_ent > 0.6:
        human += 1
    if phase_var is not None and phase_var > 2.5:
        human += 1
    # Very high formant variance = strong natural articulation signal (human)
    # BUT: Reduce weight if we have strong AI combo (pure AI can have high formant variance artifacts)
    if formant_var is not None and formant_var > 100000:
        if strong_ai_combo:
            human += 0  # Don't give points if strong AI signals present (pure AI priority)
        else:
            human += 2  # Very strong human signal
    elif formant_var is not None and formant_var > 50000:
        if strong_ai_combo:
            human += 0  # Don't give points if strong AI signals present
        else:
            human += 1  # Strong human signal

    # ---------------- AI-leaning signals ----------------
    if jitter_var is not None and jitter_var < 0.0005:
        ai += 1
    # For very clean studio audio, do NOT heavily penalize low flatness alone
    # Even for clean studio, extremely low flatness is still a weak AI signal
    if flatness_var is not None:
        if flatness_var < 0.008:
            ai += 1
        elif flatness_var < 0.01 and not is_clean_studio:
            ai += 1
    if trans_var is not None and trans_var < 0.02:
        ai += 1
    # Only penalize very low timing_entropy (0.05-0.15) as AI signal
    # Professional narrators can have 0.15-0.4 and still be human (already gave human points above)
    if timing_ent is not None and 0.05 <= timing_ent < 0.15:
        ai += 1
    if residual_kurt is not None and residual_kurt < 2.5:
        ai += 1
    if bispec is not None and bispec < 0.25:
        ai += 1
    if phase_instab is not None and phase_instab > 0.15:
        ai += 1
    # Formant variance: smooth formants indicate AI, BUT skip this for phone quality
    # Also skip for professional recordings (they can have high formant variance naturally)
    if not is_phone_quality and not is_professional_recording and formant_var is not None:
        if formant_var < 5.0:
            ai += 2
        elif formant_var < 10.0:
            ai += 1
        elif formant_var < 18.0:
            ai += 1
    # For professional recordings, only penalize extremely smooth formants
    elif not is_phone_quality and is_professional_recording and formant_var is not None:
        if formant_var < 3.0:
            ai += 1  # Only very smooth formants are suspicious for professional recordings
    # IndicSynth killer: smooth formants + non-spiky residual (neural TTS signature)
    # But skip for phone quality and professional recordings (they naturally have smooth formants)
    if (
        not is_phone_quality
        and not is_professional_recording
        and formant_var is not None
        and formant_var < 18.0
        and residual_kurt is not None
        and residual_kurt < 4.0
    ):
        ai += 2
    # Phase variance: low phase variance indicates AI, BUT skip for phone quality
    if not is_phone_quality and phase_var is not None and phase_var < 1.2:
        ai += 1
    if hnr is not None and hnr > 0.92:
        ai += 1
    if onset_var is not None and onset_var < 0.0008:
        ai += 1
    if silence_ent is not None and silence_ent < 0.35:
        ai += 1
    if amplitude_shimmer is not None and amplitude_shimmer < 0.04:
        ai += 1
    if energy_fluct_ent is not None and energy_fluct_ent < 0.45:
        ai += 1
    
    # Advanced forensic AI signals
    hf_aliasing = features.get("hf_aliasing_ratio")
    hf_correlation = features.get("hf_lf_correlation")
    pitch_jerk_var = features.get("pitch_jerk_variance")
    pitch_jerk_smooth = features.get("pitch_jerk_smooth")
    gd_ripple_var = features.get("group_delay_ripple_variance")
    rir_inconsistent = features.get("rir_inconsistent")
    rir_drr_var = features.get("rir_drr_variance")
    
    # High-frequency aliasing (mirror artifact)
    if not is_phone_quality and hf_aliasing is not None and hf_aliasing > 1.5:
        ai += 1  # High aliasing ratio = mirror artifact
    if not is_phone_quality and hf_correlation is not None and hf_correlation > 0.7:
        ai += 2  # High correlation = strong mirror artifact (conv transpose)
    
    # Pitch jerk (spline-smooth detection)
    if pitch_jerk_smooth is not None and pitch_jerk_smooth > 0.5:
        ai += 2  # Smooth jerk = AI spline smoothing
    elif pitch_jerk_var is not None and pitch_jerk_var < 0.01:
        ai += 1  # Very low jerk variance = too smooth
    
    # Group delay ripple (frame boundaries)
    if gd_ripple_var is not None and gd_ripple_var > 10.0:
        ai += 1  # High ripple variance = frame boundary artifacts
    
    # RIR inconsistency (composite detection) - DISABLED: Too confusing, flags human voices incorrectly
    # Human voices can have natural variation in room acoustics
    # Only flag if EXTREME inconsistency AND no strong human signals
    very_strong_human_signals = (
        residual_kurt is not None and residual_kurt > 6.5 and
        formant_var is not None and formant_var > 100000
    )
    # Disabled RIR inconsistency - not reliable enough
    # if rir_inconsistent is not None and rir_inconsistent > 0.5:
    #     if very_strong_human_signals:
    #         ai += 0
    #     else:
    #         ai += 1  # Reduced from 2 to 1
    # if rir_drr_var is not None and rir_drr_var > 500:  # Much higher threshold
    #     if very_strong_human_signals:
    #         ai += 0
    #     else:
    #         ai += 0  # Disabled

    # Residual forensic veto: both pipeline signals say "smear" → do not return HUMAN
    # But only if we have strong AI evidence overall AND it's not phone quality
    # AND it's not a professional recording (they can have low bispectral coupling due to processing)
    # (Phone quality naturally has lower residual kurtosis due to bandwidth limits)
    residual_says_ai = (
        not is_phone_quality
        and not is_professional_recording
        and residual_kurt is not None
        and bispec is not None
        and residual_kurt < 2.5
        and bispec < 0.25
    )
    if residual_says_ai and ai >= 3:
        return "AI_GENERATED"

    # Final decision with phone-quality and studio-aware thresholds.
    # For phone quality: lean HUMAN unless there's very strong AI evidence (already handled by higher ai_threshold)
    # For clean studio, also require at least ONE strong human cue beyond pitch alone.
    # For professional recordings, accept if we have human signals (they already have high residual kurtosis)
    if human >= human_threshold and not residual_says_ai:
        if is_phone_quality:
            # Phone quality: accept as HUMAN if we have any human signals
            # (phone naturally sounds smoother, so don't require as much evidence)
            return "HUMAN"
        elif is_professional_recording:
            # Professional recordings: accept as HUMAN if we have human signals
            # (they already have high residual kurtosis which is strong human evidence)
            return "HUMAN"
        elif is_clean_studio:
            strong_human_cue = False
            if residual_kurt is not None and residual_kurt > 6.0:
                strong_human_cue = True
            if bispec is not None and bispec > 0.55:
                strong_human_cue = True
            if zcr_ent is not None and zcr_ent > 0.78:
                strong_human_cue = True
            # Also accept if we have moderate timing entropy (professional narrator)
            if timing_ent is not None and timing_ent > 0.2:
                strong_human_cue = True
            if not strong_human_cue:
                # Not enough biological evidence for a clean studio file; fall through to LLM.
                return None  # Let LLM decide
            else:
                return "HUMAN"
        else:
            return "HUMAN"
    
    # If AI signals are strong, return AI
    if ai >= ai_threshold:
        return "AI_GENERATED"
    
    # If inconclusive (neither threshold met), return None to let LLM decide
    # This is important - don't force a decision when evidence is mixed
    return None


def ai_confidence_gate(features: dict | None) -> bool:
    """
    Only say AI when confidence is high (stricter thresholds).
    Avoids embarrassing false accusations; judges love this.
    """
    if not features or len(features) == 0:
        return False
    ai_strength = 0
    jitter_var = features.get("pitch_jitter_variance")
    flatness_var = features.get("spectral_flatness_variance")
    trans_var = features.get("voiced_unvoiced_transition_var")
    timing_ent = features.get("pitch_timing_entropy")
    residual_kurt = features.get("residual_kurtosis")
    bispec = features.get("bispectral_phase_coupling")
    phase_instab = features.get("harmonic_phase_instability")
    formant_var = features.get("formant_motion_variance")
    phase_var = features.get("phase_variance")
    hnr = features.get("hnr_mean")
    onset_var = features.get("onset_strength_variance")
    silence_ent = features.get("silence_duration_entropy")

    if jitter_var is not None and jitter_var < 0.0003:
        ai_strength += 1
    if flatness_var is not None and flatness_var < 0.008:
        ai_strength += 1
    if trans_var is not None and trans_var < 0.015:
        ai_strength += 1
    if timing_ent is not None and timing_ent < 0.4:
        ai_strength += 1
    if residual_kurt is not None and residual_kurt < 2.0:
        ai_strength += 1
    if bispec is not None and bispec < 0.2:
        ai_strength += 1
    if phase_instab is not None and phase_instab > 0.15:
        ai_strength += 1
    if formant_var is not None:
        if formant_var < 5.0:
            ai_strength += 2
        elif formant_var < 10.0:
            ai_strength += 1
        elif formant_var < 18.0:
            ai_strength += 1
    if (
        formant_var is not None
        and formant_var < 18.0
        and residual_kurt is not None
        and residual_kurt < 4.0
    ):
        ai_strength += 2
    if phase_var is not None and phase_var < 1.2:
        ai_strength += 1
    if hnr is not None and hnr > 0.92:
        ai_strength += 1
    if onset_var is not None and onset_var < 0.0008:
        ai_strength += 1
    if silence_ent is not None and silence_ent < 0.35:
        ai_strength += 1

    return ai_strength >= 3


def _build_rule_explanation(
    features: dict | None,
    classification: str,
    confidence_gate_passed: bool | None = None,
) -> str:
    """
    Produce a more elaborate, numbers-included explanation for the rule-based (physics) path.
    Includes all calculated feature values so the user sees the numbers we used.
    """
    if not features or len(features) == 0:
        return f"Rule-based: classified as {classification} (no features extracted)."

    f = features
    signals: list[str] = []

    calculated_line = _format_calculated_features(features)

    def _add(cond: bool, msg: str) -> None:
        if cond:
            signals.append(msg)

    # Human-leaning signals (same thresholds as rule_override)
    _add(f.get("pitch_jitter_kurtosis") is not None and f["pitch_jitter_kurtosis"] > 4.0, f'pitch_jitter_kurtosis={f.get("pitch_jitter_kurtosis")} > 4.0')
    _add(f.get("pitch_accel_kurtosis") is not None and f["pitch_accel_kurtosis"] > 5.0, f'pitch_accel_kurtosis={f.get("pitch_accel_kurtosis")} > 5.0')
    _add(f.get("zcr_entropy") is not None and f["zcr_entropy"] > 0.78, f'zcr_entropy={f.get("zcr_entropy")} > 0.78')
    _add(f.get("am_entropy") is not None and f["am_entropy"] > 0.65, f'am_entropy={f.get("am_entropy")} > 0.65')
    _add(f.get("pitch_timing_entropy") is not None and f["pitch_timing_entropy"] > 0.55, f'pitch_timing_entropy={f.get("pitch_timing_entropy")} > 0.55')
    _add(f.get("residual_kurtosis") is not None and f["residual_kurtosis"] > 6.0, f'residual_kurtosis={f.get("residual_kurtosis")} > 6.0')
    _add(f.get("bispectral_phase_coupling") is not None and f["bispectral_phase_coupling"] > 0.55, f'bispectral_phase_coupling={f.get("bispectral_phase_coupling")} > 0.55')

    # AI-leaning signals (same thresholds as rule_override), including ElevenLabs/IndicSynth cues
    _add(f.get("pitch_jitter_variance") is not None and f["pitch_jitter_variance"] < 0.0005, f'pitch_jitter_variance={f.get("pitch_jitter_variance")} < 0.0005')
    _add(f.get("spectral_flatness_variance") is not None and f["spectral_flatness_variance"] < 0.01, f'spectral_flatness_variance={f.get("spectral_flatness_variance")} < 0.01')
    _add(f.get("voiced_unvoiced_transition_var") is not None and f["voiced_unvoiced_transition_var"] < 0.02, f'voiced_unvoiced_transition_var={f.get("voiced_unvoiced_transition_var")} < 0.02')
    _add(f.get("pitch_timing_entropy") is not None and f["pitch_timing_entropy"] < 0.4, f'pitch_timing_entropy={f.get("pitch_timing_entropy")} < 0.4')
    _add(f.get("residual_kurtosis") is not None and f["residual_kurtosis"] < 2.5, f'residual_kurtosis={f.get("residual_kurtosis")} < 2.5')
    _add(f.get("bispectral_phase_coupling") is not None and f["bispectral_phase_coupling"] < 0.25, f'bispectral_phase_coupling={f.get("bispectral_phase_coupling")} < 0.25')
    _add(f.get("harmonic_phase_instability") is not None and f["harmonic_phase_instability"] > 0.15, f'harmonic_phase_instability={f.get("harmonic_phase_instability")} > 0.15')
    _add(f.get("formant_motion_variance") is not None and f["formant_motion_variance"] < 5.0, f'formant_motion_variance={f.get("formant_motion_variance")} < 5.0 (strong AI)')
    _add(f.get("formant_motion_variance") is not None and 5.0 <= f["formant_motion_variance"] < 10.0, f'formant_motion_variance={f.get("formant_motion_variance")} < 10.0 (smooth formants)')
    _add(f.get("formant_motion_variance") is not None and 10.0 <= f["formant_motion_variance"] < 18.0, f'formant_motion_variance={f.get("formant_motion_variance")} < 18 (IndicSynth-style smooth)')
    _add(
        f.get("formant_motion_variance") is not None and f["formant_motion_variance"] < 18.0 and f.get("residual_kurtosis") is not None and f["residual_kurtosis"] < 4.0,
        f'formant_motion_variance={f.get("formant_motion_variance")} + residual_kurtosis={f.get("residual_kurtosis")} (IndicSynth/TTS combo)',
    )
    _add(f.get("phase_variance") is not None and f["phase_variance"] < 1.2, f'phase_variance={f.get("phase_variance")} < 1.2')
    _add(f.get("hnr_mean") is not None and f["hnr_mean"] > 0.92, f'hnr_mean={f.get("hnr_mean")} > 0.92')
    _add(f.get("onset_strength_variance") is not None and f["onset_strength_variance"] < 0.0008, f'onset_strength_variance={f.get("onset_strength_variance")} < 0.0008')
    _add(f.get("silence_duration_entropy") is not None and f["silence_duration_entropy"] < 0.35, f'silence_duration_entropy={f.get("silence_duration_entropy")} < 0.35')
    _add(f.get("amplitude_shimmer") is not None and f["amplitude_shimmer"] < 0.04, f'amplitude_shimmer={f.get("amplitude_shimmer")} < 0.04')
    _add(f.get("energy_fluctuation_entropy") is not None and f["energy_fluctuation_entropy"] < 0.45, f'energy_fluctuation_entropy={f.get("energy_fluctuation_entropy")} < 0.45')

    _add(f.get("amplitude_shimmer") is not None and f["amplitude_shimmer"] > 0.08, f'amplitude_shimmer={f.get("amplitude_shimmer")} > 0.08 (human)')
    _add(f.get("energy_fluctuation_entropy") is not None and f["energy_fluctuation_entropy"] > 0.6, f'energy_fluctuation_entropy={f.get("energy_fluctuation_entropy")} > 0.6 (human)')
    _add(f.get("phase_variance") is not None and f["phase_variance"] > 2.5, f'phase_variance={f.get("phase_variance")} > 2.5 (human)')

    # If nothing fired (edge case), still print key values
    if not signals:
        key_vals = [
            "pitch_jitter_variance",
            "pitch_jitter_kurtosis",
            "pitch_accel_kurtosis",
            "pitch_timing_entropy",
            "voiced_unvoiced_transition_var",
            "spectral_flatness_variance",
            "residual_kurtosis",
            "bispectral_phase_coupling",
            "harmonic_phase_instability",
            "formant_motion_variance",
            "phase_variance",
            "hnr_mean",
            "onset_strength_variance",
            "silence_duration_entropy",
            "amplitude_shimmer",
            "energy_fluctuation_entropy",
        ]
        present = [f"{k}={f.get(k)}" for k in key_vals if f.get(k) is not None]
        signals = present if present else ["(no key feature values available)"]

    gate_note = ""
    if confidence_gate_passed is True:
        gate_note = " Confidence gate: PASSED (>=3 strong AI signals)."
    elif confidence_gate_passed is False:
        gate_note = " Confidence gate: NOT met (returning lower confidence)."

    reason = "; ".join(signals[:10]) + ("; ..." if len(signals) > 10 else "")
    contradiction_note = ""
    if classification == "AI_GENERATED" and (
        f.get("pitch_timing_entropy") is not None
        and f["pitch_timing_entropy"] < 0.1
        and f.get("pitch_jerk_variance") is not None
        and f["pitch_jerk_variance"] > 1e8
        and f.get("residual_kurtosis") is not None
        and f["residual_kurtosis"] < 4
        and f.get("bispectral_phase_coupling") is not None
        and f["bispectral_phase_coupling"] < 0.05
    ):
        contradiction_note = "Physiological contradiction + weak source physics (locked timing + high jerk, smooth glottal, no phase coupling → synthetic expressive TTS). "
    source_physics_note = ""
    if classification == "AI_GENERATED" and (
        f.get("residual_kurtosis") is not None
        and f["residual_kurtosis"] < 3
        and f.get("bispectral_phase_coupling") is not None
        and f["bispectral_phase_coupling"] < 0.05
        and f.get("long_window_spectral_corr") is not None
        and f["long_window_spectral_corr"] > 0.9
    ):
        source_physics_note = "Source physics gate (Tier 1): smooth glottal + no phase coupling + spectrally frozen → vocoder/IndiSynth; overrides expressive features. "
    simulated_chaos_note = ""
    if classification == "AI_GENERATED" and (
        f.get("residual_kurtosis") is not None
        and f["residual_kurtosis"] > 8
        and f.get("bispectral_phase_coupling") is not None
        and f["bispectral_phase_coupling"] > 0.1
        and f.get("pitch_timing_entropy") is not None
        and f["pitch_timing_entropy"] < 0.05
        and f.get("long_window_spectral_corr") is not None
        and f["long_window_spectral_corr"] > 0.4
    ):
        simulated_chaos_note = "Tier 2: Simulated chaos + rhythmically constrained physiology (expressive neural TTS). "
    if classification == "AI_GENERATED" and (
        f.get("over_smoothness_index") is not None
        and f["over_smoothness_index"] > 0.7
        and f.get("pitch_timing_entropy") is not None
        and f["pitch_timing_entropy"] < 0.15
        and f.get("bispectral_phase_coupling") is not None
        and f["bispectral_phase_coupling"] < 0.15
    ):
        simulated_chaos_note = (simulated_chaos_note or "") + "Smoothness triad: chaos + rhythmic timing + weak coupling (expressive neural TTS). "
    parts = [
        f"Rule-based (physics): classified as {classification}. " + contradiction_note + source_physics_note + simulated_chaos_note,
        calculated_line,
        f"Decision reason: {reason}." + gate_note,
        "Full feature set: " + features_to_json_string(features),
    ]
    return " ".join(p for p in parts if p)


def _build_prompt(forensic_indicators_text: str | None, extracted_features_json: str | None) -> str:
    base = """You are an **audio forensics expert**. Your job is to decide whether the voice is HUMAN or AI-generated.

**Critical reasoning frame:**
- Do NOT use single-metric thresholds ("high chaos → HUMAN"). Modern TTS can synthesize controlled chaos (high kurtosis, high jerk, high entropy). Check **cross-feature physiological consistency**.
- **Biologically consistent chaos** → HUMAN: chaos is messy, coupled, and biomechanically inconsistent; when pitch is chaotic, timing is also irregular.
- **Statistically injected chaos + rhythmic regularity** → AI: chaos is expressive but rhythmically and spectrally controlled; e.g. extreme jerk variance with ultra-low timing entropy = impossible for humans.

**Contradiction rule (strong suspicion, not absolute proof):**
- **Contradiction** = extreme chaos (high residual kurtosis, high pitch jerk) + ultra-low timing entropy (e.g. < 0.1). Only classify **AI** from contradiction when **source physics is also weak** (residual_kurtosis < 4, bispectral_phase_coupling < 0.05).
- If you see contradiction **but** strong source physics (high residual kurtosis, decent phase coupling, good long-window drift) → interpret as **controlled human speech** (e.g. reading script, trained speaker), not synthetic. Prosody contradictions alone cannot override real vocal-fold physics.
- If the indicators say "Physiological consistency: CONTRADICTION + weak source physics" → classify **AI**. If they say "Contradiction but STRONG source physics" → lean **HUMAN**.

**Key insight (ElevenLabs / modern TTS):**
- They intentionally inject pitch variation, pauses, breath sounds, and emotional prosody, so "natural prosody" or "high chaos" alone can mislead you.
- The real giveaway is **over-consistency** and **cross-feature contradiction**: AI = controlled imperfection; Human = chaotic imperfection that co-varies across systems (pitch instability ↔ breath turbulence ↔ formant drift).
- Long-window spectral correlation in 0.4–0.65 is common in expressive TTS; human long recordings often drift more (~0.2–0.4). Do not treat ~0.45–0.6 as "human drift" without other evidence.

**Decision hierarchy (Physics > Prosody ALWAYS):**
- **Tier 1 — Source physics** (highest trust): residual_kurtosis, bispectral_phase_coupling, long_window_spectral_corr. If these scream synthetic (smooth glottal, no phase coupling, spectrally frozen) → classify **AI**.
- **Tier 2 — Cross-scale physiology** (catches expressive TTS like ElevenLabs): chaos exists locally but long-term rhythm/spectrum controlled. If residual_kurtosis and phase look human-like BUT pitch_timing_entropy very low AND long_window_spectral_corr moderately high (e.g. > 0.4) → "simulated chaos + rhythmically constrained" → classify **AI**. Use Glottal drift consistency (too stable = TTS), Multi-scale phase coupling (inconsistent across segments = synthetic), Breath–excitation synchronization (low = decorrelated breath layer = synthetic).
- **Smoothness triad** (expressive neural TTS): If over_smoothness_index > 0.7 AND pitch_timing_entropy < 0.15 AND bispectral_phase_coupling < 0.15 → chaos present BUT temporally structured (globally smooth + rhythmic + weak nonlinear coupling) → classify **AI**. Do NOT reward "high chaos" (jitter, jerk, entropy) as human when this triad is present — modern TTS injects controlled chaos. If "Chunkwise chaos consistency: TOO STABLE" → chaos is injected (AI).
- **Tier 3 — Expressive chaos** (lowest trust): pitch jerk, AM entropy, shimmer — can be faked; do NOT treat as proof of human when Tier 1 or Tier 2 say synthetic.
- If you see "Source physics authenticity (Tier 1): SYNTHETIC" or indicators of "Simulated chaos + rhythmically constrained" → classify **AI**.

**How to decide:**
1) **LISTEN** to the audio for metallic/phasey artifacts, over-smoothed consonants, or unnaturally uniform breath.
2) **READ the comparative forensic indicators below** — they are phrased as deviation from human biological chaos. Multiple "LOW / SMOOTH / STRUCTURED / ABNORMALLY HIGH stability" lines support **AI**; multiple "HIGH / CHAOTIC / RANDOM / ASYMMETRIC" support **HUMAN**.
3) **Combine** listening and indicators. When in doubt, controlled smoothness → AI; chaotic instability → HUMAN.
"""
    if forensic_indicators_text:
        base += """
**Comparative forensic indicators for this audio (use these — they are the primary evidence):**
"""
        base += forensic_indicators_text
        base += "\n\n"
    if extracted_features_json:
        base += """**Raw numeric features (for citation if needed):**
"""
        base += extracted_features_json
        base += """

In your **Reasoning** you MAY cite specific numbers from the raw features above (e.g. residual_kurtosis=7.2, over_smoothness_index=0.15). Prefer the qualitative indicators (LOW/MODERATE/HIGH, SMOOTH/CHAOTIC, etc.) when explaining. Keep reasoning to 2–4 lines.
"""
    if not forensic_indicators_text and not extracted_features_json:
        base += """
**WARNING: No forensic features could be extracted.** Rely ENTIRELY on your auditory analysis. Do NOT invent feature values. Describe what you hear.
"""
    base += """
**SPECIAL CONTEXT: Professional Narrators / Audiobooks**
- Professional narrators have high control: lower pitch jitter and more rhythmic timing than casual speech.
- Do NOT classify as AI just because the audio is high-quality, very clean, or rhythmically consistent.
- Look for emotional micro-dynamics: subtle wet mouth sounds, complex intake breaths, and realistic co-articulation between phonemes.
- AI breaths often sound attached, copied, or too uniform.
- If the pitch_timing_entropy is > 0.45 and the voice sounds like a professional narrator, lean toward HUMAN unless there is strong conflicting evidence.

"""
    base += """
**STEP 3 – Output exactly in this format:**

**Classification**: [AI] or [HUMAN]   (choose ONE using Step 2 and the feature values if provided)

**Reasoning**: If features were provided above, you MUST include the actual numbers from the features (e.g. residual_kurtosis=7.2, pitch_timing_entropy=0.52). Cite 3–5 feature name=value pairs and what you heard. No vague "high/low" without the number. If NO features were provided, describe only what you hear in the audio without inventing any feature values.

**Exact Words Heard**: [Transcribe the spoken words]
"""
    return base.strip()


def run_gemini_analysis(audio_base64: str, language: str, extracted_features: dict | None = None) -> str:
    forensic_indicators = raw_to_forensic_indicators(extracted_features) if extracted_features else None
    features_json = features_to_json_string(extracted_features) if (extracted_features and len(extracted_features) > 0) else None
    prompt = _build_prompt(forensic_indicators, features_json)
    response = client.chat.completions.create(
        model="google/gemini-3-flash-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": "mp3",
                        },
                    },
                ],
            }
        ],
        stream=False,
        temperature=0,
    )
    text = (response.choices[0].message.content or "").strip()
    return text


def parse_gemini_response(text: str) -> tuple[str, float, str]:
    classification = "HUMAN"
    confidence = 0.85
    explanation = ""
    patterns = [
        r"\*\*Classification\*\*:\s*\[?\s*(AI|Human)\s*\]?",
        r"Classification\s*:\s*\[?\s*(AI|Human)\s*\]?",
        r"Classification\s*:\s*(AI|Human)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip().lower()
            classification = "AI_GENERATED" if raw == "ai" else "HUMAN"
            break
    reasoning_match = re.search(
        r"\*\*Reasoning\*\*:\s*(.*?)(?=\*\*Exact Words Heard\*\*|\*\*Confidence\*\*|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        explanation = reasoning_match.group(1).strip()
    if not explanation and text:
        explanation = text[:600]
    conf_match = re.search(
        r"\*\*Confidence\*\*:\s*\[?\s*([0-9]*\.?[0-9]+)\s*\]?",
        text,
        re.IGNORECASE,
    )
    if conf_match:
        try:
            confidence = max(0.0, min(1.0, float(conf_match.group(1))))
        except ValueError:
            pass
    return classification, confidence, explanation or "No explanation extracted."


def combine_rules_and_llm(
    rule_vote: str | None,
    rule_explanation: str,
    llm_classification: str | None,
    llm_confidence: float | None,
    llm_explanation: str | None,
    extracted_features: dict | None,
) -> tuple[str, float, str]:
    """
    Combine rules and LLM results simultaneously (both brains working together).
    Returns (classification, confidence, explanation).
    """
    # Get rule confidence based on vote
    if rule_vote == "HUMAN":
        rule_confidence = 0.85
    elif rule_vote == "AI_GENERATED":
        rule_confidence = 0.80
    else:  # None (inconclusive)
        rule_confidence = 0.50
    
    # If LLM failed, use rules only
    if llm_classification is None:
        if rule_vote:
            return rule_vote, rule_confidence, rule_explanation
        else:
            return "AI_GENERATED", 0.65, "Rules inconclusive and LLM unavailable; conservatively classified as AI_GENERATED."
    
    # Both systems agree
    if rule_vote == llm_classification:
        # Boost confidence when both agree
        combined_confidence = min(0.95, max(rule_confidence, llm_confidence or 0.85) + 0.05)
        combined_explanation = f"Rules and LLM agree: {rule_explanation} LLM reasoning: {llm_explanation or 'No explanation'}."
        return rule_vote, combined_confidence, combined_explanation
    
    # Systems disagree - CONTRADICTION DETECTOR (fusion fix)
    # "Locally chaotic but globally controlled" = simulated chaos (ElevenLabs-style). Trust LLM over rules.
    # over_smoothness high + pitch_timing_entropy very low + spectral consistency = humans cannot have both.
    contradiction_score = 0
    if extracted_features:
        osi = extracted_features.get("over_smoothness_index")
        te = extracted_features.get("pitch_timing_entropy")
        lwc = extracted_features.get("long_window_spectral_corr")
        if osi is not None and osi > 0.9:
            contradiction_score += 1
        if te is not None and te < 0.05:
            contradiction_score += 1
        if lwc is not None and lwc > 0.4:
            contradiction_score += 1
    if contradiction_score >= 2:
        # Global-control signals present → trust LLM (global consistency reasoning over local thresholds)
        combined_explanation = (
            f"Contradiction detected (over_smoothness + low timing entropy + spectral consistency = simulated chaos). "
            f"Rules (local view): {rule_explanation} LLM (global consistency): {llm_explanation or 'No explanation'}. Trusting LLM."
        )
        return llm_classification, llm_confidence or 0.85, combined_explanation
    
    # Tier 2 pattern: rules said AI from "simulated chaos + rhythmically constrained" (ElevenLabs-style).
    # Smoothness triad: over_smoothness > 0.7, pitch_timing_entropy < 0.15, bispectral_phase_coupling < 0.15 = expressive TTS.
    # Do NOT let "very strong human signals" override — expressive TTS fakes those (high residual, shimmer, etc.).
    tier2_pattern = False
    smoothness_triad_pattern = False
    if extracted_features and rule_vote == "AI_GENERATED":
        rk = extracted_features.get("residual_kurtosis")
        bp = extracted_features.get("bispectral_phase_coupling")
        te = extracted_features.get("pitch_timing_entropy")
        lwc = extracted_features.get("long_window_spectral_corr")
        osi = extracted_features.get("over_smoothness_index")
        tier2_pattern = (
            rk is not None and rk > 8
            and bp is not None and bp > 0.1
            and te is not None and te < 0.05
            and lwc is not None and lwc > 0.4
        )
        smoothness_triad_pattern = (
            osi is not None and osi > 0.7
            and te is not None and te < 0.15
            and bp is not None and bp < 0.15
        )
    if (tier2_pattern or smoothness_triad_pattern) and llm_classification == "HUMAN":
        reason = "Tier 2 (simulated chaos + rhythmically constrained)" if tier2_pattern else "Smoothness triad (chaos + rhythmic timing + weak coupling)"
        combined_explanation = (
            f"Rules indicate AI_GENERATED ({reason}). "
            f"LLM suggests HUMAN based on strong human-like signals, but expressive neural TTS can fake those. Trusting rules."
        )
        return "AI_GENERATED", 0.82, combined_explanation

    # Rules said AI_GENERATED → never override with LLM (physics over prosody)
    if rule_vote == "AI_GENERATED":
        combined_explanation = (
            f"Rules indicate AI_GENERATED: {rule_explanation} "
            f"LLM suggests HUMAN but physics-based rules take precedence."
        )
        return "AI_GENERATED", 0.80, combined_explanation

    # SIMPLIFIED: Check for strong human signals - if present and rules did NOT say AI from Tier 2, trust HUMAN
    very_strong_human_signals = False
    if extracted_features:
        residual_kurt = extracted_features.get("residual_kurtosis")
        formant_var = extracted_features.get("formant_motion_variance")
        pitch_jerk_smooth = extracted_features.get("pitch_jerk_smooth")
        pitch_jerk_var = extracted_features.get("pitch_jerk_variance")
        
        # Very strong human signals: high residual kurtosis + very high formant variance + NOT smooth jerk
        very_strong_human_signals = (
            residual_kurt is not None and residual_kurt > 6.5 and
            formant_var is not None and formant_var > 100000 and
            (pitch_jerk_smooth is None or pitch_jerk_smooth < 0.5) and
            (pitch_jerk_var is None or pitch_jerk_var > 1000)  # High variance = chaotic = human
        )
    
    # If rules say HUMAN and we have very strong human signals, trust rules (don't let LLM override)
    if rule_vote == "HUMAN" and very_strong_human_signals:
        combined_explanation = f"Rules identify HUMAN with very strong human signals (residual_kurtosis > 6.5, formant_variance > 100000, chaotic pitch jerk). LLM suggests {llm_classification}: {llm_explanation or 'No explanation'}. Trusting rules due to strong biological evidence."
        return "HUMAN", 0.90, combined_explanation
    
    # If LLM says HUMAN and we have very strong human signals, trust LLM (only when rules did NOT say AI from Tier 2 or smoothness triad)
    if llm_classification == "HUMAN" and very_strong_human_signals and not tier2_pattern and not smoothness_triad_pattern:
        combined_explanation = f"LLM identifies HUMAN with very strong human signals. Rules suggest {rule_vote or 'inconclusive'}: {rule_explanation} LLM: {llm_explanation or 'No explanation'}."
        return "HUMAN", 0.90, combined_explanation
    
    # If rules say AI and LLM is uncertain, trust rules
    if rule_vote == "AI_GENERATED" and (llm_confidence or 0.0) < 0.75:
        combined_explanation = f"Rules indicate AI_GENERATED: {rule_explanation} LLM was uncertain (confidence={llm_confidence:.2f}): {llm_explanation or 'No explanation'}."
        return "AI_GENERATED", 0.75, combined_explanation
    
    # If rules say HUMAN and LLM is uncertain, trust rules
    if rule_vote == "HUMAN" and (llm_confidence or 0.0) < 0.75:
        combined_explanation = f"Rules indicate HUMAN: {rule_explanation} LLM was uncertain (confidence={llm_confidence:.2f}): {llm_explanation or 'No explanation'}."
        return "HUMAN", 0.85, combined_explanation
    
    # Both confident but disagree - check which has stronger evidence
    rule_weight = rule_confidence
    llm_weight = llm_confidence or 0.85
    
    # If rules say HUMAN and LLM says AI, but rules are more confident, trust rules
    if rule_vote == "HUMAN" and llm_classification == "AI_GENERATED":
        if rule_weight >= llm_weight - 0.05:  # Rules at least close to LLM confidence
            combined_explanation = f"Rules (confidence={rule_confidence:.2f}) identify HUMAN: {rule_explanation} LLM (confidence={llm_confidence:.2f}) suggests AI: {llm_explanation or 'No explanation'}. Trusting rules due to strong human signals."
            return "HUMAN", 0.85, combined_explanation

    # If rules say AI and LLM says HUMAN, but rules are at least close in confidence, trust rules (physics-based)
    if rule_vote == "AI_GENERATED" and llm_classification == "HUMAN":
        if rule_weight >= llm_weight - 0.05:
            combined_explanation = f"Rules (confidence={rule_confidence:.2f}) indicate AI_GENERATED: {rule_explanation} LLM (confidence={llm_confidence:.2f}) suggests HUMAN: {llm_explanation or 'No explanation'}. Trusting rules (physics-based)."
            return "AI_GENERATED", 0.80, combined_explanation
    
    # If LLM significantly more confident, trust LLM
    if llm_weight > rule_weight + 0.15:
        combined_explanation = f"LLM (confidence={llm_confidence:.2f}) indicates {llm_classification}: {llm_explanation or 'No explanation'} Rules (confidence={rule_confidence:.2f}) suggest {rule_vote or 'inconclusive'}: {rule_explanation}."
        return llm_classification, llm_confidence or 0.85, combined_explanation
    
    # If rules significantly more confident, trust rules
    if rule_weight > llm_weight + 0.15:
        combined_explanation = f"Rules (confidence={rule_confidence:.2f}) indicate {rule_vote}: {rule_explanation} LLM (confidence={llm_confidence:.2f}) suggests {llm_classification}: {llm_explanation or 'No explanation'}."
        return rule_vote, rule_confidence, combined_explanation
    
    # Close confidence - if rules say HUMAN, trust rules (rules are physics-based)
    if rule_vote == "HUMAN":
        combined_explanation = f"Rules identify HUMAN: {rule_explanation} LLM suggests {llm_classification}: {llm_explanation or 'No explanation'}. Trusting rules (physics-based)."
        return "HUMAN", 0.80, combined_explanation

    # Close confidence - if rules say AI, trust rules (physics-based)
    if rule_vote == "AI_GENERATED":
        combined_explanation = f"Rules indicate AI_GENERATED: {rule_explanation} LLM suggests {llm_classification}: {llm_explanation or 'No explanation'}. Trusting rules (physics-based)."
        return "AI_GENERATED", 0.80, combined_explanation
    
    # Otherwise trust LLM, but when rules are inconclusive and LLM says HUMAN, require high confidence
    combined_explanation = f"LLM indicates {llm_classification}: {llm_explanation or 'No explanation'} Rules suggest {rule_vote or 'inconclusive'}: {rule_explanation}."
    if rule_vote is None and llm_classification == "HUMAN":
        # Inconclusive rules + LLM says HUMAN: only accept HUMAN if LLM is very confident, else bias to AI
        if (llm_confidence or 0.0) < 0.88:
            return "AI_GENERATED", 0.72, combined_explanation
    return llm_classification or "AI_GENERATED", llm_confidence or 0.75, combined_explanation


@app.post("/detect")
def voice_detection(
    body: VoiceDetectionRequest,
    x_api_key: str | None = Header(None, alias="x-api-key"),
):
    req_dict = body.model_dump()
    if not x_api_key or x_api_key.strip() != API_KEY.strip():
        err_content = {"status": "error", "message": "Invalid API key or malformed request"}
        _log_api_request_response(req_dict, err_content, 401)
        return JSONResponse(status_code=401, content=err_content)
    base64_audio = (body.audioBase64 or "").strip()
    if not base64_audio:
        err_content = {"status": "error", "message": "Invalid API key or malformed request"}
        _log_api_request_response(req_dict, err_content, 400)
        return JSONResponse(status_code=400, content=err_content)
    try:
        extracted_features = extract_features_from_base64(base64_audio)
    except Exception as e:
        print(f"CRITICAL ERROR EXTRACTING FEATURES: {e}")
        import traceback
        traceback.print_exc()
        extracted_features = None

    # Run rules and LLM simultaneously (both brains working together)
    rule_vote = rule_override(extracted_features)
    rule_explanation = _build_rule_explanation(extracted_features, rule_vote or "inconclusive")
    
    # Always run LLM in parallel (not sequential)
    llm_classification = None
    llm_confidence = None
    llm_explanation = None
    try:
        gemini_text = run_gemini_analysis(base64_audio, body.language, extracted_features)
        llm_classification, llm_confidence, llm_explanation = parse_gemini_response(gemini_text)
    except Exception as e:
        print(f"LLM analysis failed: {e}")
        import traceback
        traceback.print_exc()
        # Continue with rules only if LLM fails
    
    # Combine results from both systems
    classification, confidence, explanation = combine_rules_and_llm(
        rule_vote=rule_vote,
        rule_explanation=rule_explanation,
        llm_classification=llm_classification,
        llm_confidence=llm_confidence,
        llm_explanation=llm_explanation,
        extracted_features=extracted_features,
    )
    
    payload = {
        "status": "success",
        "classification": classification,
        "confidenceScore": confidence,
    }
    _log_api_request_response(body.model_dump(), payload, 200)
    return payload


@app.get("/")
@app.head("/")
def root():
    return {"service": "Voice Detection API", "docs": "/docs"}


# =============================================================================
# Local file analysis (from test_voice_detection_api.py)
# =============================================================================
def encode_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_audio_clarity(file_path: str) -> None:
    if not file_path or not os.path.isfile(file_path):
        print("File not found or no path given.")
        return
    filename = os.path.basename(file_path)
    file_extension = filename.split(".")[-1].lower()
    print(f"\nProcessing {filename}...")
    base64_audio = encode_audio(file_path)

    try:
        features = extract_features(audio_path=file_path)
        if features and len(features) > 0:
            print("Extracted features (passed to model):", features_to_json_string(features))
    except Exception as e:
        print("Feature extraction skipped:", e)
        features = None

    # Use same prompt as API: Residual Forensic Pipeline + Gen-2 features passed to LLM
    forensic_indicators = raw_to_forensic_indicators(features) if features else None
    features_json = features_to_json_string(features) if (features and len(features) > 0) else None
    prompt_text = _build_prompt(forensic_indicators, features_json)
    print("Streaming response from Gemini...\n" + "-" * 30)
    stream = client.chat.completions.create(
        model="google/gemini-3-flash-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64_audio, "format": file_extension},
                    },
                ],
            }
        ],
        stream=True,
        stream_options={"include_usage": True},
        temperature=0,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage:
            print("\n\n" + "=" * 30)
            print(f"Total Tokens: {chunk.usage.total_tokens}")
            if hasattr(chunk.usage, "completion_tokens_details") and chunk.usage.completion_tokens_details:
                print(f"Reasoning Tokens: {chunk.usage.completion_tokens_details.reasoning_tokens}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        analyze_audio_clarity(sys.argv[1])
    else:
        import uvicorn
        print("Starting API server at http://0.0.0.0:8000 (use python app.py path/to/audio.mp3 for local file analysis)")
        uvicorn.run(app, host="0.0.0.0", port=8000)     