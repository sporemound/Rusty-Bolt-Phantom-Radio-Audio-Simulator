"""Small regression tests; run with: python test_dsp_core.py"""

import numpy as np
from scipy import signal as sps

from dsp_core import (
    am_modulate,
    carrier_rejected_am_effect,
    diode_demodulate,
    run_pipeline,
    run_pipeline_with_effect,
)

FS = 44100
DURATION = 1.0
N = int(FS * DURATION)
T = np.arange(N) / FS
FC = 5000.0
FM = 700.0
SOURCE = np.sin(2 * np.pi * FM * T)


def bin_magnitude(x, frequency):
    spectrum = np.fft.rfft(np.asarray(x) * np.hanning(len(x)))
    freqs = np.fft.rfftfreq(len(x), 1 / FS)
    return abs(spectrum[np.argmin(abs(freqs - frequency))])



def fitted_tone_amplitude(x, frequency):
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x)) / FS
    cosine = np.cos(2 * np.pi * frequency * t)
    sine = np.sin(2 * np.pi * frequency * t)
    gram = np.array(
        [
            [np.dot(cosine, cosine), np.dot(cosine, sine)],
            [np.dot(cosine, sine), np.dot(sine, sine)],
        ]
    )
    rhs = np.array([np.dot(x, cosine), np.dot(x, sine)])
    coeff = np.linalg.solve(gram, rhs)
    return float(np.hypot(coeff[0], coeff[1]))

def test_carrier_rejection():
    raw = am_modulate(SOURCE, FS, FC, mod_index=0.8)
    effect = carrier_rejected_am_effect(SOURCE, FS, FC, mod_index=0.8)
    raw_carrier = fitted_tone_amplitude(raw, FC)
    effect_carrier = fitted_tone_amplitude(effect, FC)
    sideband = max(
        bin_magnitude(effect, FC - FM),
        bin_magnitude(effect, FC + FM),
    )
    rejection_db = 20 * np.log10((effect_carrier + 1e-15) / (raw_carrier + 1e-15))
    assert rejection_db < -180.0, rejection_db
    assert sideband > 1.0


def test_clean_diode_recovery():
    raw = am_modulate(SOURCE, FS, FC, mod_index=0.8)
    recovered = diode_demodulate(
        raw,
        FS,
        lowpass_cutoff=2500.0,
        diode_threshold=0.0,
        carrier_freq=FC,
        notch_carrier=True,
    )
    # Ignore filter edges and sign ambiguity.
    trim = int(0.1 * FS)
    corr = abs(np.corrcoef(SOURCE[trim:-trim], recovered[trim:-trim])[0, 1])
    assert corr > 0.97, corr
    carrier_residual = bin_magnitude(recovered, FC)
    wanted = bin_magnitude(recovered, FM)
    assert wanted > carrier_residual * 1000, (wanted, carrier_residual)


def test_pipeline_shapes_and_finiteness():
    outputs = run_pipeline_with_effect(
        SOURCE,
        FS,
        carrier_freq=FC,
        mod_index=0.8,
        snr_db=20,
        detune_hz=2,
        diode_threshold=0.05,
        demod_cutoff=3000,
        rng=np.random.default_rng(1234),
    )
    assert len(outputs) == 4
    for output in outputs:
        assert output.shape == SOURCE.shape
        assert np.all(np.isfinite(output))



def test_dc_offset_does_not_restore_carrier():
    offset_source = SOURCE + 0.35
    effect = carrier_rejected_am_effect(
        offset_source, FS, FC, mod_index=0.8, residual_carrier=0.0
    )
    carrier = fitted_tone_amplitude(effect, FC)
    sideband = max(
        bin_magnitude(effect, FC - FM),
        bin_magnitude(effect, FC + FM),
    )
    assert carrier < 1e-10, carrier
    assert sideband > 1.0


def test_silence_and_short_clip_are_safe():
    for source in (np.zeros(2048), np.linspace(-1, 1, 24)):
        outputs = run_pipeline_with_effect(
            source,
            FS,
            carrier_freq=FC,
            snr_db=20,
            rng=np.random.default_rng(2),
        )
        for output in outputs:
            assert output.shape == source.shape
            assert np.all(np.isfinite(output))


def test_legacy_pipeline_still_returns_three_outputs():
    outputs = run_pipeline(
        SOURCE, FS, carrier_freq=FC, rng=np.random.default_rng(3)
    )
    assert len(outputs) == 3
    for output in outputs:
        assert output.shape == SOURCE.shape

def test_low_carrier_clamps_recovery_band():
    raw = am_modulate(SOURCE, FS, 1000.0, mod_index=0.8)
    recovered = diode_demodulate(
        raw,
        FS,
        lowpass_cutoff=6000.0,
        carrier_freq=1000.0,
        notch_carrier=True,
    )
    # The 1 kHz carrier must not dominate even though the requested cutoff was
    # invalid and above the carrier.
    f, p = sps.welch(recovered, FS, nperseg=8192)
    at_carrier = p[np.argmin(abs(f - 1000.0))]
    below_800 = np.max(p[(f > 50) & (f < 800)])
    assert below_800 > at_carrier * 100, (below_800, at_carrier)


if __name__ == "__main__":
    tests = [
        test_carrier_rejection,
        test_clean_diode_recovery,
        test_pipeline_shapes_and_finiteness,
        test_dc_offset_does_not_restore_carrier,
        test_silence_and_short_clip_are_safe,
        test_legacy_pipeline_still_returns_three_outputs,
        test_low_carrier_clamps_recovery_band,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
