"""
Core signal-processing functions for the diode / phantom-radio simulator.

This module exposes two related but deliberately separate outputs:

* ``am`` is a conventional full-carrier AM waveform used internally by the
  accidental-diode simulation. Because the RF carrier is represented at an
  audible frequency, listening to this diagnostic waveform produces a strong
  steady whistle. That is expected and is not the finished audio effect.
* ``am_effect`` is a carrier-rejected (DSB-SC) render intended for listening
  and saving. It preserves the metallic AM sidebands that made the diagnostic
  AM stage appealing, but it omits the constant carrier tone.

The physical recovery path remains:
  source -> bandwidth limiting -> full-carrier AM -> pickup/channel ->
  asymmetric junction (half-wave rectification) -> low-pass recovery.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

_EPS = 1e-12


def _as_mono_float(audio: np.ndarray) -> np.ndarray:
    """Return a finite, one-dimensional float64 signal."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("audio must be a one-dimensional mono signal")
    if x.size == 0:
        raise ValueError("audio must contain at least one sample")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _peak_normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= _EPS:
        return np.zeros_like(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) / peak


def _safe_sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase filter when possible, with a short-buffer fallback."""
    try:
        return sps.sosfiltfilt(sos, x)
    except ValueError:
        # sosfiltfilt needs enough samples for padding. The causal fallback
        # keeps very short clips usable instead of crashing the GUI.
        return sps.sosfilt(sos, x)


def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    try:
        return sps.filtfilt(b, a, x)
    except ValueError:
        return sps.lfilter(b, a, x)


def effective_message_bandwidth(
    fs: float,
    carrier_freq: float,
    requested_bandwidth: float = 6000.0,
    guard_hz: float = 250.0,
) -> float:
    """Return a message bandwidth that is safe for sampled AM.

    Three limits matter:
      1. the user's requested radio/audio bandwidth;
      2. the message should remain below the carrier for usable envelope
         detection; and
      3. the upper sideband must remain below Nyquist.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    nyquist = fs / 2.0
    if not 0.0 < carrier_freq < nyquist:
        raise ValueError(
            f"carrier_freq must be between 0 and Nyquist ({nyquist:.1f} Hz)"
        )
    if requested_bandwidth <= 0:
        raise ValueError("requested_bandwidth must be positive")

    # 80% of carrier leaves transition room between recovered audio and the
    # carrier. Nyquist guard prevents sideband aliasing at high carrier values.
    safe = min(
        float(requested_bandwidth),
        carrier_freq * 0.80,
        nyquist - carrier_freq - max(0.0, float(guard_hz)),
    )
    if safe < 40.0:
        raise ValueError(
            "carrier frequency leaves too little room for an AM sideband; "
            "move the carrier farther from Nyquist"
        )
    return safe


def bandlimit_message(
    audio: np.ndarray,
    fs: float,
    carrier_freq: float,
    requested_bandwidth: float = 6000.0,
    order: int = 6,
) -> tuple[np.ndarray, float]:
    """Normalize and anti-alias the source before AM modulation."""
    x = _as_mono_float(audio).copy()
    # A DC offset becomes a residual tone exactly at the carrier even in a
    # nominally suppressed-carrier render. Remove it both before and after
    # filtering so arbitrary user WAVs cannot reintroduce the whistle.
    x -= np.mean(x)
    x = _peak_normalize(x)
    bandwidth = effective_message_bandwidth(
        fs, carrier_freq, requested_bandwidth=requested_bandwidth
    )
    sos = sps.butter(order, bandwidth, btype="lowpass", fs=fs, output="sos")
    filtered = _safe_sosfiltfilt(sos, x)
    filtered -= np.mean(filtered)
    return _peak_normalize(filtered), bandwidth


# ---------------------------------------------------------------------------
# 1. Synthetic voice generator
# ---------------------------------------------------------------------------
def generate_synthetic_voice(duration=4.0, fs=44100, seed=None):
    """Build a crude speech-like signal for demonstration."""
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    t = np.arange(n) / fs

    n_words = int(duration * 2.2)
    word_env = np.zeros(n)
    pos = 0
    for _ in range(n_words):
        word_len = int(rng.uniform(0.12, 0.35) * fs)
        gap_len = int(rng.uniform(0.03, 0.12) * fs)
        end = min(pos + word_len, n)
        if pos < n:
            ramp = min(400, (end - pos) // 3) or 1
            seg = np.ones(end - pos)
            seg[:ramp] = np.linspace(0, 1, ramp)
            seg[-ramp:] = np.linspace(1, 0, ramp)
            word_env[pos:end] = seg
        pos = end + gap_len
        if pos >= n:
            break

    formants = [500, 1200, 2400]
    voice = np.zeros(n)
    for f0 in formants:
        wobble = sps.lfilter([1], [1, -0.995], rng.normal(0, 1, n)) * 0.35
        wobble = np.clip(wobble, -1, 1)
        freq_jitter = f0 * (1 + 0.02 * wobble)
        phase = 2 * np.pi * np.cumsum(freq_jitter) / fs
        voice += np.sin(phase) * (0.5 + 0.5 * wobble)

    voice *= word_env
    return t, _peak_normalize(voice)


# ---------------------------------------------------------------------------
# 2. AM modulation and the listenable carrier-rejected render
# ---------------------------------------------------------------------------
def am_modulate(
    audio,
    fs,
    carrier_freq,
    mod_index=0.8,
    source_bandwidth=6000.0,
):
    """Create conventional full-carrier AM for the physical simulation.

    ``(1 + m*x) * cos(2*pi*fc*t)`` necessarily contains a pure tone at
    ``carrier_freq``. In this audio-rate simulation that tone is audible;
    use :func:`carrier_rejected_am_effect` for a listenable/exportable effect.
    """
    if not 0.0 <= mod_index <= 1.0:
        raise ValueError("mod_index must be between 0 and 1")
    message, _ = bandlimit_message(
        audio, fs, carrier_freq, requested_bandwidth=source_bandwidth
    )
    t = np.arange(message.size) / fs
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    return _peak_normalize((1.0 + mod_index * message) * carrier)


def carrier_rejected_am_effect(
    audio,
    fs,
    carrier_freq,
    mod_index=0.8,
    residual_carrier=0.0,
    source_bandwidth=6000.0,
    carrier_guard_hz=80.0,
):
    """Render the metallic AM sidebands without the continuous whistle.

    This is double-sideband suppressed-carrier modulation when
    ``residual_carrier == 0``. A very small residual value can be introduced
    deliberately, but the default is completely carrier-free.
    """
    if not 0.0 <= mod_index <= 1.0:
        raise ValueError("mod_index must be between 0 and 1")
    if not 0.0 <= residual_carrier <= 1.0:
        raise ValueError("residual_carrier must be between 0 and 1")

    message, bandwidth = bandlimit_message(
        audio, fs, carrier_freq, requested_bandwidth=source_bandwidth
    )
    # Remove very-low-frequency source energy as well as DC. In suppressed-
    # carrier modulation, rumble at (for example) 20 Hz becomes a pair of
    # nearly steady components at fc +/- 20 Hz and can still be perceived as
    # a whistle. The guard creates a genuinely quiet zone around the carrier.
    if 0.0 < carrier_guard_hz < bandwidth * 0.5:
        sos_guard = sps.butter(2, carrier_guard_hz, btype="highpass", fs=fs, output="sos")
        message = _safe_sosfiltfilt(sos_guard, message)
        message = _peak_normalize(message - np.mean(message))

    t = np.arange(message.size) / fs
    carrier = np.cos(2 * np.pi * carrier_freq * t)

    # Omitting the constant "1" term is exact carrier rejection. This is
    # cleaner than applying a narrow notch after synthesis: no notch ringing,
    # and no accidental removal of nearby modulation sidebands.
    effect = (residual_carrier + mod_index * message) * carrier

    if residual_carrier == 0.0:
        # Null any finite-buffer carrier residue exactly. Even a zero-mean
        # source can leave a tiny coherent component because real clips do not
        # always contain an integer number of carrier cycles and zero-phase
        # filters have edge transients. Solve the two-component (cos/sin)
        # projection without allocating a large N x 2 design matrix.
        quadrature = np.sin(2 * np.pi * carrier_freq * t)
        gram = np.array(
            [
                [np.dot(carrier, carrier), np.dot(carrier, quadrature)],
                [np.dot(carrier, quadrature), np.dot(quadrature, quadrature)],
            ],
            dtype=np.float64,
        )
        rhs = np.array(
            [np.dot(effect, carrier), np.dot(effect, quadrature)],
            dtype=np.float64,
        )
        try:
            coeff = np.linalg.solve(gram, rhs)
            effect = effect - coeff[0] * carrier - coeff[1] * quadrature
        except np.linalg.LinAlgError:
            pass

    return _peak_normalize(effect)


# ---------------------------------------------------------------------------
# 3. Channel effects
# ---------------------------------------------------------------------------
def apply_channel_effects(
    am_signal,
    fs,
    snr_db=20.0,
    detune_hz=0.0,
    carrier_freq=5000.0,
    rng=None,
):
    """Apply weak pickup noise and slow fading.

    Older builds multiplied the entire RF-like waveform by a bipolar cosine.
    That forced periodic polarity inversions and produced a large artificial
    sub-audio beat after rectification. Here the same control produces bounded,
    positive fading (0.35..1.0), which keeps the unstable-pickup character
    without manufacturing an unrelated tone.
    """
    del carrier_freq  # retained for API compatibility
    sig = _as_mono_float(am_signal).copy()
    n = sig.size

    if detune_hz != 0.0:
        t = np.arange(n) / fs
        fade = 0.675 + 0.325 * np.cos(2 * np.pi * abs(detune_hz) * t)
        sig *= fade

    sig_power = float(np.mean(sig**2))
    if sig_power <= _EPS:
        return sig
    noise_power = sig_power / (10 ** (snr_db / 10))
    generator = rng if rng is not None else np.random.default_rng()
    noise = generator.normal(0, np.sqrt(noise_power), n)
    return sig + noise


# ---------------------------------------------------------------------------
# 4. Accidental diode / corroded-junction demodulation
# ---------------------------------------------------------------------------
def diode_demodulate(
    received,
    fs,
    lowpass_cutoff=3000.0,
    diode_threshold=0.0,
    order=8,
    carrier_freq=None,
    notch_carrier=True,
    oversample=4,
    highpass_cutoff=25.0,
):
    """Half-wave rectify and low-pass filter an AM-like pickup.

    Nonlinear rectification is performed after oversampling to keep generated
    harmonics from aliasing into the audible band. The requested recovery
    cutoff is automatically kept below the simulated carrier, preventing the
    invalid GUI combinations that previously let the carrier pass through.
    """
    x = _as_mono_float(received)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if oversample < 1 or int(oversample) != oversample:
        raise ValueError("oversample must be a positive integer")
    if diode_threshold < 0:
        raise ValueError("diode_threshold must be non-negative")

    oversample = int(oversample)
    fs_os = fs * oversample
    received_os = sps.resample_poly(x, oversample, 1)

    # Crude passive junction: asymmetric threshold followed by half-wave
    # conduction. This is intentionally less clean than an ideal Hilbert
    # envelope detector.
    rectified = np.maximum(received_os - diode_threshold, 0.0)
    nyq_os = fs_os / 2.0

    if notch_carrier and carrier_freq is not None:
        # Remove narrow residuals at the true, unfolded carrier harmonics.
        # The later low-pass does most of the work; these notches protect edge
        # cases where the carrier is close to the recovery passband.
        for harmonic in range(1, 8):
            f0 = carrier_freq * harmonic
            if 10.0 < f0 < nyq_os * 0.98:
                b_notch, a_notch = sps.iirnotch(f0, Q=35.0, fs=fs_os)
                rectified = _safe_filtfilt(b_notch, a_notch, rectified)

    # Explicit anti-alias filtering before returning to the original rate.
    aa_cutoff = (fs / 2.0) * 0.90
    sos_aa = sps.butter(order, aa_cutoff, btype="lowpass", fs=fs_os, output="sos")
    filtered_os = _safe_sosfiltfilt(sos_aa, rectified)
    envelope = sps.resample_poly(filtered_os, 1, oversample)

    if envelope.size != x.size:
        n = x.size
        envelope = (
            envelope[:n]
            if envelope.size > n
            else np.pad(envelope, (0, n - envelope.size))
        )

    nyquist = fs / 2.0
    cutoff = min(float(lowpass_cutoff), nyquist * 0.95)
    if carrier_freq is not None:
        # Keep transition room below the carrier. This fixes settings such as
        # a 1 kHz carrier with a 3 kHz recovery low-pass, which cannot remove
        # the carrier by definition.
        cutoff = min(cutoff, float(carrier_freq) * 0.80)
    if cutoff <= 20.0:
        raise ValueError("effective demodulation cutoff is too low")

    sos_lp = sps.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    envelope = _safe_sosfiltfilt(sos_lp, envelope)

    # Remove DC plus the exaggerated 1-10 Hz wobble generated by the old
    # bipolar 'detune' implementation. This is below speech/music fundamentals
    # and prevents saved WAVs from wasting headroom on inaudible excursions.
    if 0.0 < highpass_cutoff < cutoff * 0.5:
        sos_hp = sps.butter(2, highpass_cutoff, btype="highpass", fs=fs, output="sos")
        envelope = _safe_sosfiltfilt(sos_hp, envelope)
    else:
        envelope -= np.mean(envelope)

    return _peak_normalize(envelope)


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------
def run_pipeline_with_effect(
    audio,
    fs,
    carrier_freq=5000.0,
    mod_index=0.8,
    snr_db=20.0,
    detune_hz=0.0,
    diode_threshold=0.0,
    demod_cutoff=3000.0,
    notch_carrier=True,
    source_bandwidth=6000.0,
    residual_carrier=0.0,
    rng=None,
):
    """Return ``raw_am, carrier_free_effect, received, demodulated``."""
    raw_am = am_modulate(
        audio,
        fs,
        carrier_freq,
        mod_index,
        source_bandwidth=source_bandwidth,
    )
    am_effect = carrier_rejected_am_effect(
        audio,
        fs,
        carrier_freq,
        mod_index,
        residual_carrier=residual_carrier,
        source_bandwidth=source_bandwidth,
    )
    received = apply_channel_effects(
        raw_am,
        fs,
        snr_db,
        detune_hz,
        carrier_freq,
        rng=rng,
    )
    demod = diode_demodulate(
        received,
        fs,
        demod_cutoff,
        diode_threshold,
        carrier_freq=carrier_freq,
        notch_carrier=notch_carrier,
    )
    return raw_am, am_effect, received, demod


def run_pipeline(
    audio,
    fs,
    carrier_freq=5000.0,
    mod_index=0.8,
    snr_db=20.0,
    detune_hz=0.0,
    diode_threshold=0.0,
    demod_cutoff=3000.0,
    notch_carrier=True,
    source_bandwidth=6000.0,
    rng=None,
):
    """Backward-compatible three-output physical simulation pipeline.

    Returns ``raw_am, received, demodulated`` as earlier builds did. Use
    :func:`run_pipeline_with_effect` when the carrier-free listening render is
    also required.
    """
    raw_am, _effect, received, demod = run_pipeline_with_effect(
        audio,
        fs,
        carrier_freq=carrier_freq,
        mod_index=mod_index,
        snr_db=snr_db,
        detune_hz=detune_hz,
        diode_threshold=diode_threshold,
        demod_cutoff=demod_cutoff,
        notch_carrier=notch_carrier,
        source_bandwidth=source_bandwidth,
        residual_carrier=0.0,
        rng=rng,
    )
    return raw_am, received, demod


if __name__ == "__main__":
    fs = 44100
    _, voice = generate_synthetic_voice(duration=3.0, fs=fs, seed=1)
    raw_am, am_effect, received, demod = run_pipeline_with_effect(
        voice,
        fs,
        carrier_freq=5000.0,
        mod_index=0.9,
        snr_db=12.0,
        detune_hz=1.5,
        diode_threshold=0.05,
        demod_cutoff=2500.0,
        rng=np.random.default_rng(1),
    )

    print("voice    :", voice.shape, voice.min(), voice.max())
    print("raw AM   :", raw_am.shape, raw_am.min(), raw_am.max())
    print("AM effect:", am_effect.shape, am_effect.min(), am_effect.max())
    print("received :", received.shape, received.min(), received.max())
    print("demod    :", demod.shape, demod.min(), demod.max())

    from scipy.io import wavfile

    def save(name, x):
        x16 = np.int16(np.clip(_peak_normalize(x), -1, 1) * 32767)
        wavfile.write(name, fs, x16)

    save("test_original.wav", voice)
    save("test_raw_am_diagnostic.wav", raw_am)
    save("test_carrier_free_am_effect.wav", am_effect)
    save("test_received_diagnostic.wav", received)
    save("test_diode_recovered.wav", demod)
    print("wrote test_*.wav files")
