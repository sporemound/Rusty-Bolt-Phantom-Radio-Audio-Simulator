# Rusty Bolt / Phantom Radio Audio Simulator — repaired build

> **Scope note:** in RF engineering, the “rusty bolt” or passive
> intermodulation effect normally means two or more strong RF signals mixing at
> a passive nonlinear contact. This program’s diode-recovery path models the
> closely related accidental detection of an AM signal by a nonlinear/corroded
> junction. The carrier-free AM output is a deliberately creative audio render,
> not literal RF.

## The tone that was not a filter bug

The old **AM signal** output used ordinary amplitude modulation:

`(1 + message) × carrier`

The constant `1 × carrier` term is a pure sine wave. Because this simulator
places the would-be RF carrier at an audible frequency such as 5,000 Hz, the
saved WAV necessarily contains a loud 5 kHz whistle. The old **notch carrier**
checkbox operated only inside the later diode-demodulation stage, so it could
never remove the tone from the raw AM WAV.

This build keeps the raw full-carrier AM waveform only as a clearly labelled
diagnostic. The recommended output is now **Carrier-free AM effect**, generated
as:

`message × carrier`

This is double-sideband suppressed-carrier modulation. It retains the metallic,
frequency-shifted texture of the AM stage but contains no constant carrier when
**Residual carrier** is set to `0.000`. An 80 Hz baseband guard also prevents
DC, hum, and rumble from turning into near-carrier whistles.

## Other repairs

- Input audio is bandwidth-limited before modulation so upper sidebands do not
  alias at high carrier settings.
- Invalid combinations such as a 1 kHz carrier with a 3 kHz recovery filter are
  automatically constrained below the carrier.
- The old detune implementation caused complete polarity reversals and a large
  artificial low-frequency beat. It is now bounded slow fading.
- The recovered output receives a 25 Hz high-pass so sub-audio motion does not
  consume WAV headroom.
- The controls are scrollable, short clips no longer crash zero-phase filters,
  and playback errors are returned safely to Tk's GUI thread.

## Run on Windows

1. Open a terminal in this folder.
2. Install dependencies including in-app playback:

   `py -m pip install -r requirements-playback.txt`

   If `sounddevice` is unavailable, install `requirements.txt` instead; WAV
   loading and saving will still work.

3. Double-click `run_simulator.bat`, or run:

   `py diode_effect_simulator.py`

`sounddevice` is needed only for in-app playback. WAV loading and saving use
SciPy.

## Which output to use

- **Carrier-free AM effect (recommended):** closest to the old AM sound, minus
  the obnoxious steady tone.
- **Diode-recovered phantom radio:** physically closer to accidental AM
  demodulation by a corroded or nonlinear contact.
- **Raw AM diagnostic:** intentionally contains the carrier whistle.
- **Raw noisy pickup:** intentionally contains the carrier and is not intended
  as a final audio effect.
