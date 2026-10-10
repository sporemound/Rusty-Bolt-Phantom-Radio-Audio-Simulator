#!/usr/bin/env python3
"""
Rusty Bolt / Phantom Radio Audio Simulator
==========================================

The application now separates the *diagnostic* full-carrier AM waveform from
an actually listenable carrier-rejected AM effect. The former necessarily has
a steady carrier whistle because real RF has been scaled into the audible
range. The latter keeps the metallic AM sidebands while removing that tone at
the synthesis stage.

Run locally with:
    python -m pip install -r requirements.txt
    python diode_effect_simulator.py
"""

from __future__ import annotations

import threading
from math import gcd

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from dsp_core import (
    effective_message_bandwidth,
    generate_synthetic_voice,
    run_pipeline_with_effect,
)

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

try:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
except Exception:
    wavfile = None
    resample_poly = None


FS = 44100


class DiodeEffectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rusty Bolt / Phantom Radio Audio Simulator")
        self.root.geometry("1180x900")
        self.root.minsize(980, 760)

        self.fs = FS
        self.original = None
        self.raw_am_signal = None
        self.am_effect = None
        self.received = None
        self.demod = None

        self._build_layout()
        self._on_generate_voice()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        controls_outer = ttk.Frame(self.root)
        controls_outer.pack(side=tk.LEFT, fill=tk.Y)

        controls_canvas = tk.Canvas(
            controls_outer, width=315, highlightthickness=0, borderwidth=0
        )
        controls_scroll = ttk.Scrollbar(
            controls_outer, orient=tk.VERTICAL, command=controls_canvas.yview
        )
        controls = ttk.Frame(controls_canvas, padding=10)
        controls_window = controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )
        controls_canvas.configure(yscrollcommand=controls_scroll.set)
        controls_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        controls_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def update_scrollregion(_event=None):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def fit_width(event):
            controls_canvas.itemconfigure(controls_window, width=event.width)

        controls.bind("<Configure>", update_scrollregion)
        controls_canvas.bind("<Configure>", fit_width)

        plot_frame = ttk.Frame(self.root, padding=10)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        src_box = ttk.LabelFrame(controls, text="1. Source audio", padding=8)
        src_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(
            src_box, text="Generate synthetic voice", command=self._on_generate_voice
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            src_box, text="Load WAV file...", command=self._on_load_wav
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            src_box, text="▶ Play original", command=lambda: self._play(self.original)
        ).pack(fill=tk.X, pady=2)

        am_box = ttk.LabelFrame(controls, text="2. AM / metallic effect", padding=8)
        am_box.pack(fill=tk.X, pady=(0, 10))
        self.carrier_var = self._add_slider(
            am_box, "Carrier / shift frequency (Hz)", 1000, 15000, 5000
        )
        self.modidx_var = self._add_slider(
            am_box, "Modulation depth", 0.1, 1.0, 0.8, resolution=0.01
        )
        self.residual_var = self._add_slider(
            am_box,
            "Residual carrier in effect",
            0.0,
            0.20,
            0.0,
            resolution=0.005,
        )
        ttk.Label(
            am_box,
            text=(
                "0.000 removes the continuous carrier tone. Raise only when "
                "a little whistle is deliberately wanted."
            ),
            wraplength=270,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 2))

        ch_box = ttk.LabelFrame(controls, text="3. Pickup quality", padding=8)
        ch_box.pack(fill=tk.X, pady=(0, 10))
        self.snr_var = self._add_slider(
            ch_box, "Signal-to-noise (dB)", -5, 40, 15
        )
        self.detune_var = self._add_slider(
            ch_box, "Slow fading rate (Hz)", 0, 20, 2, resolution=0.1
        )

        dm_box = ttk.LabelFrame(controls, text="4. Diode recovery", padding=8)
        dm_box.pack(fill=tk.X, pady=(0, 10))
        self.threshold_var = self._add_slider(
            dm_box,
            "Junction threshold (weak contact)",
            0.0,
            0.5,
            0.05,
            resolution=0.01,
        )
        self.cutoff_var = self._add_slider(
            dm_box, "Recovered bandwidth (Hz)", 500, 6000, 3000
        )
        self.notch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            dm_box,
            text="Suppress residual carrier in recovered output",
            variable=self.notch_var,
        ).pack(anchor=tk.W, pady=(2, 0))

        ttk.Button(
            controls, text="Run simulation ▶", command=self._on_run
        ).pack(fill=tk.X, pady=(0, 10))

        play_box = ttk.LabelFrame(controls, text="Listen", padding=8)
        play_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(
            play_box,
            text="▶ Carrier-free AM effect (recommended)",
            command=lambda: self._play(self.am_effect),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            play_box,
            text="▶ Diode-recovered phantom radio",
            command=lambda: self._play(self.demod),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            play_box,
            text="▶ Raw AM diagnostic (contains whistle)",
            command=lambda: self._play(self.raw_am_signal),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            play_box,
            text="▶ Raw noisy pickup (contains carrier)",
            command=lambda: self._play(self.received),
        ).pack(fill=tk.X, pady=2)

        save_box = ttk.LabelFrame(controls, text="Save", padding=8)
        save_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(
            save_box,
            text="Save carrier-free AM effect...",
            command=lambda: self._on_save(
                self.am_effect, "rusty_bolt_carrier_free_effect.wav"
            ),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            save_box,
            text="Save diode-recovered effect...",
            command=lambda: self._on_save(
                self.demod, "phantom_radio_diode_recovered.wav"
            ),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            save_box,
            text="Save original...",
            command=lambda: self._on_save(self.original, "original_voice.wav"),
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            save_box,
            text="Save raw AM diagnostic...",
            command=lambda: self._on_save(
                self.raw_am_signal, "raw_am_diagnostic_contains_carrier.wav"
            ),
        ).pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar(value="")
        ttk.Label(
            controls,
            textvariable=self.status_var,
            wraplength=280,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(2, 8))

        if not AUDIO_AVAILABLE:
            ttk.Label(
                controls,
                text=(
                    "Playback disabled: sounddevice / PortAudio is not "
                    "available. Saving WAV files still works."
                ),
                foreground="#b00020",
                wraplength=280,
                justify=tk.LEFT,
            ).pack(fill=tk.X, pady=(4, 0))

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax_orig = self.fig.add_subplot(411)
        self.ax_am = self.fig.add_subplot(412)
        self.ax_recv = self.fig.add_subplot(413)
        self.ax_demod = self.fig.add_subplot(414)
        self.fig.tight_layout(pad=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_slider(self, parent, label, lo, hi, default, resolution=1):
        ttk.Label(parent, text=label).pack(anchor=tk.W)
        var = tk.DoubleVar(value=default)
        tk.Scale(
            parent,
            from_=lo,
            to=hi,
            orient=tk.HORIZONTAL,
            variable=var,
            resolution=resolution,
            length=255,
            showvalue=True,
        ).pack(fill=tk.X, pady=(0, 6))
        return var

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_generate_voice(self):
        _, voice = generate_synthetic_voice(duration=4.0, fs=self.fs)
        self.original = voice
        self._on_run()

    def _on_load_wav(self):
        if wavfile is None or resample_poly is None:
            messagebox.showerror("Error", "SciPy is required to load WAV files.")
            return
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        try:
            source_fs, data = wavfile.read(path)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not read WAV file:\n{exc}")
            return

        if data.ndim > 1:
            data = data.astype(np.float64).mean(axis=1)
        else:
            data = data.astype(np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        if peak > 0:
            data /= peak

        if source_fs != self.fs:
            factor = gcd(int(self.fs), int(source_fs))
            up, down = int(self.fs) // factor, int(source_fs) // factor
            data = resample_poly(data, up, down)

        self.original = data
        self._on_run()

    def _on_run(self):
        if self.original is None:
            return
        carrier = self.carrier_var.get()
        cutoff_requested = self.cutoff_var.get()
        try:
            raw_am, am_effect, received, demod = run_pipeline_with_effect(
                self.original,
                self.fs,
                carrier_freq=carrier,
                mod_index=self.modidx_var.get(),
                snr_db=self.snr_var.get(),
                detune_hz=self.detune_var.get(),
                diode_threshold=self.threshold_var.get(),
                demod_cutoff=cutoff_requested,
                notch_carrier=self.notch_var.get(),
                residual_carrier=self.residual_var.get(),
            )
            message_bw = effective_message_bandwidth(
                self.fs, carrier, requested_bandwidth=6000.0
            )
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            return

        self.raw_am_signal = raw_am
        self.am_effect = am_effect
        self.received = received
        self.demod = demod

        recovered_cutoff = min(cutoff_requested, carrier * 0.80)
        note = (
            f"Carrier-free effect ready. Source bandwidth: {message_bw:.0f} Hz. "
            f"Recovered cutoff: {recovered_cutoff:.0f} Hz. "
            "Raw AM and raw pickup remain diagnostic signals and intentionally "
            "contain the carrier."
        )
        self.status_var.set(note)
        self._update_plots()

    def _update_plots(self):
        if self.original is None:
            return
        t = np.arange(len(self.original)) / self.fs
        items = [
            (self.ax_orig, self.original, "1. Original source audio", "tab:blue"),
            (
                self.ax_am,
                self.am_effect,
                "2. Carrier-free AM effect (recommended export)",
                "tab:orange",
            ),
            (
                self.ax_recv,
                self.received,
                "3. Raw noisy pickup (diagnostic RF-like stage)",
                "tab:red",
            ),
            (
                self.ax_demod,
                self.demod,
                "4. Diode-recovered phantom radio",
                "tab:green",
            ),
        ]
        for ax, data, title, color in items:
            ax.clear()
            step = max(1, len(data) // 5000)
            ax.plot(t[::step], data[::step], linewidth=0.5, color=color)
            ax.set_title(title, fontsize=9)
            ax.set_xlim(0, t[-1] if len(t) else 1)
            ax.tick_params(labelsize=7)
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

    def _play(self, data):
        if data is None:
            return
        if not AUDIO_AVAILABLE:
            messagebox.showinfo(
                "Playback unavailable",
                "sounddevice/PortAudio is unavailable. Save a WAV and play it "
                "in your normal audio player.",
            )
            return

        def worker():
            try:
                peak = float(np.max(np.abs(data))) + 1e-12
                sd.play(np.asarray(data, dtype=np.float64) / peak, self.fs)
                sd.wait()
            except Exception as exc:
                # Tk calls must be made on the GUI thread.
                self.root.after(
                    0, lambda: messagebox.showerror("Playback error", str(exc))
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_save(self, data, default_name):
        if data is None:
            messagebox.showinfo("Nothing to save", "Run the simulation first.")
            return
        if wavfile is None:
            messagebox.showerror("Error", "SciPy is required to save WAV files.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=default_name,
        )
        if not path:
            return

        peak = float(np.max(np.abs(data))) + 1e-12
        normalized = np.asarray(data, dtype=np.float64) / peak
        x16 = np.int16(np.clip(normalized, -1, 1) * 32767)
        wavfile.write(path, self.fs, x16)
        messagebox.showinfo("Saved", f"Saved to:\n{path}")


def main():
    root = tk.Tk()
    DiodeEffectApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
