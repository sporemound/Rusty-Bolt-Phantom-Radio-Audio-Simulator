# Repair report

## Root cause

The output labelled **AM signal** was conventional full-carrier AM:

`(1 + modulation × source) × cos(carrier)`

The constant `1 × cos(carrier)` term is an uninterrupted sine wave. With the
simulated carrier set to the default 5,000 Hz, the WAV therefore contained a
loud 5 kHz tone. The previous **notch carrier** checkbox was located inside the
later diode-demodulation function and had no effect on the raw AM WAV.

## Repair

The application now generates two separate AM representations:

1. **Raw AM diagnostic** — retained for the internal physical-demonstration
   path and clearly labelled as containing a carrier.
2. **Carrier-free AM effect** — generated without the constant carrier term,
   with DC removal, an 80 Hz carrier guard, and a final finite-buffer
   cosine/sine projection that nulls residual carrier energy.

A **Residual carrier** slider allows the tone to be reintroduced deliberately;
the default value is `0.000`.

## Additional defects corrected

- Source audio is safely bandwidth-limited before modulation to prevent AM
  sideband aliasing.
- The diode recovery cutoff is constrained below the carrier, preventing
  impossible settings that allowed carrier leakage.
- The old detune operation performed full polarity reversals. It is replaced
  by positive, bounded slow fading.
- A 25 Hz high-pass keeps sub-audio recovery motion from consuming output
  headroom.
- Filters fall back safely for very short clips.
- DC-offset WAV inputs cannot recreate the carrier.
- Playback errors are marshalled back to Tk's GUI thread.
- The controls are scrollable at smaller window sizes.
- The legacy `run_pipeline()` three-output API remains available; the GUI uses
  `run_pipeline_with_effect()` for the added carrier-free render.

## Verification

- Seven regression tests pass.
- A 135-combination parameter sweep passes across carrier, threshold, cutoff,
  and noise extremes.
- In the included 16-bit comparison WAV, the coherent 5 kHz carrier amplitude
  is 119.39 dB lower in the repaired effect than in the raw AM diagnostic.
- The 5 kHz Welch-bin power is 92.21 dB below the strongest nearby sideband
  outside the ±80 Hz carrier guard.

Run `test_dsp_core.py` or `run_tests.bat` to repeat the regression checks.
