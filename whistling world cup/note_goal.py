'''
Listen to the mic for one calibrated note and publish an MQTT message when it's heard.

Audio acquisition/processing (device selection, sample-rate fallback, bandpass filtering,
pitch detection, loudness gating) is ported as-is from Hudson's frequency_drive.py:
https://github.com/hudsonbrown-tufts/AI_in_mobile_robotics/blob/main/HW3/frequency_drive.py
- same structure, minus the robot-driving/light-sensor parts, with a single MQTT publish
in place of a motor command.

Usage:
    python note_goal.py
    python note_goal.py --list-devices
'''
import sys
import time

import numpy as np
import pyaudio
from scipy.signal import butter, sosfilt, sosfilt_zi

from mqttlib import MQTTClient

SUB_TOPIC = "ME193/hudson"
MESSAGE = "GOAL!!!"  # published when the target note is heard

# Which microphone to use. Run `python note_goal.py --list-devices` to see indices/names.
# INPUT_DEVICE_INDEX (if set) takes priority; otherwise INPUT_DEVICE_NAME_HINT does a
# case-insensitive substring match against device names; leave both None to use the
# system's default input device.
INPUT_DEVICE_INDEX = None
INPUT_DEVICE_NAME_HINT = "FIIO UTWS5"

# Audio capture settings.
SAMPLE_RATE = 44100  # preferred rate; falls back to the device's own default if unsupported
CHUNK_SIZE = 1024

# Note calibration. At startup you play the note once and the program records its pitch. A
# heard frequency then counts as that note if it's within +/- tolerance_hz() of it, where
# the tolerance is NOTE_TOLERANCE_PERCENT of the note's frequency (6% ~= one semitone), but
# never narrower than NOTE_TOLERANCE_MIN_HZ (FFT bins are ~15-45 Hz wide, so low notes need
# a floor). Widen these if the note isn't being picked up reliably; narrow them if other
# sounds keep triggering it.
NOTE_TOLERANCE_PERCENT = 6
NOTE_TOLERANCE_MIN_HZ = 25

# How long (seconds) to listen during calibration, and the minimum number of loud-enough
# audio chunks needed to accept a reading.
CALIBRATION_SECONDS = 3.0
CALIBRATION_MIN_CHUNKS = 5

# Pause (seconds) between pressing Enter and the fixed CALIBRATION_SECONDS listening window
# actually starting, so you have time to get the note going instead of losing part of the
# window to silence while you react.
CALIBRATION_PREP_SECONDS = 1.5

# How many consecutive audio chunks (~23 ms each at 44.1 kHz) must hear the note before
# publishing, so a single noisy chunk can't trigger a false publish.
CONFIRM_CHUNKS = 3

# Minimum loudness (in dB, see amplitude_db()) to treat a chunk as a real tone rather than
# silence/background noise. Raise this if it reacts to quiet room noise; lower it if quiet
# notes aren't being picked up. Note this is a consistent relative loudness scale, not a
# calibrated dB SPL meter reading (that would require calibrating against your specific mic).
AMPLITUDE_THRESHOLD_DB = 60

# Bandpass filter applied to each audio chunk before frequency analysis, to suppress
# background noise (low rumble, HVAC hum, hiss, etc) outside the range of tones we care
# about. The note you calibrate must fall inside LOW/HIGH, or it'll be filtered out; ORDER
# is the Butterworth filter order (higher = sharper cutoff but more ringing).
BANDPASS_LOW_HZ = 400
BANDPASS_HIGH_HZ = 1000
BANDPASS_ORDER = 4


def design_bandpass(rate):
    """Design a Butterworth bandpass filter (as secqond-order sections) for the configured
    BANDPASS_LOW_HZ/BANDPASS_HIGH_HZ range."""
    nyquist = rate / 2
    low = BANDPASS_LOW_HZ / nyquist
    high = min(BANDPASS_HIGH_HZ, nyquist * 0.99) / nyquist
    return butter(BANDPASS_ORDER, [low, high], btype="bandpass", output="sos")


def dominant_frequency(samples, rate):
    """Return the frequency (Hz) of the strongest non-DC FFT bin."""
    windowed = samples * np.hanning(len(samples))
    spectrum = np.fft.rfft(windowed)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / rate)

    peak_index = np.argmax(magnitudes[1:]) + 1  # skip the DC bin
    return freqs[peak_index]


def amplitude_db(samples):
    """Return the RMS loudness of `samples` in dB. This is an uncalibrated, but consistent,
    relative loudness scale (0 dB ~= silence, ~90 dB ~= a full-scale 16-bit signal) - not a
    calibrated dB SPL meter reading."""
    rms = np.sqrt(np.mean(np.square(samples)))
    return 20 * np.log10(rms + 1e-6)


def tolerance_hz(note_freq):
    """Return the +/- window (Hz) around the calibrated note that still counts as a match."""
    return max(note_freq * NOTE_TOLERANCE_PERCENT / 100, NOTE_TOLERANCE_MIN_HZ)


def record_note(stream, rate, sos):
    """Wait CALIBRATION_PREP_SECONDS, then listen for exactly CALIBRATION_SECONDS and return
    the median dominant frequency (Hz) of the loud-enough chunks, or None if too few chunks
    were above AMPLITUDE_THRESHOLD_DB."""
    time.sleep(CALIBRATION_PREP_SECONDS)

    # Drop audio that piled up in the buffer while we were waiting on input()/the prep pause.
    stale_frames = stream.get_read_available()
    if stale_frames:
        stream.read(stale_frames, exception_on_overflow=False)

    print(f"Listening for {CALIBRATION_SECONDS:g}s...")
    filter_state = sosfilt_zi(sos) * 0
    freqs = []
    chunk_index = 0
    # Paced by an actual clock, not a chunk count: if any audio backlog is still sitting in
    # the device's buffer despite the flush above, stream.read() returns it instantly instead
    # of blocking for real time, which would otherwise drain a fixed number of "chunks" in a
    # fraction of CALIBRATION_SECONDS.
    start_time = time.monotonic()
    while time.monotonic() - start_time < CALIBRATION_SECONDS:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        filtered_samples, filter_state = sosfilt(sos, samples, zi=filter_state)
        chunk_index += 1
        if chunk_index <= 2:
            continue  # let the freshly-reset filter settle
        level_db = amplitude_db(filtered_samples)
        if level_db >= AMPLITUDE_THRESHOLD_DB:
            freq = dominant_frequency(filtered_samples, rate)
            freqs.append(freq)
            print(f"\r  Hearing {freq:7.1f} Hz at {level_db:5.1f} dB", end="", flush=True)
    print()

    if len(freqs) < CALIBRATION_MIN_CHUNKS:
        return None
    return float(np.median(freqs))


def calibrate_note(stream, rate, sos):
    """Ask the user to play the target note and record its pitch."""
    print("\n--- Note calibration ---")
    print(f"Press Enter, wait {CALIBRATION_PREP_SECONDS:g}s, then play and hold your note "
          f"for ~{CALIBRATION_SECONDS:g}s.")
    while True:
        input("\nPress Enter when ready...")
        freq = record_note(stream, rate, sos)
        if freq is None:
            print(
                f"  Didn't hear a loud enough note (need >= {AMPLITUDE_THRESHOLD_DB} dB). "
                "Play louder or closer to the mic and try again."
            )
            continue
        print(f"  Note = {freq:.1f} Hz (accepts {freq - tolerance_hz(freq):.1f}-{freq + tolerance_hz(freq):.1f} Hz)")
        return freq


def list_input_devices():
    """Print every available mic input device name."""
    audio = pyaudio.PyAudio()
    print("Available input devices:")
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            print(f"  [{i}] {info['name']!r} (default rate: {int(info['defaultSampleRate'])} Hz)")
    audio.terminate()


def choose_input_device(audio):
    """Pick an input device per INPUT_DEVICE_INDEX / INPUT_DEVICE_NAME_HINT, falling back to
    the system default input device."""
    if INPUT_DEVICE_INDEX is not None:
        return audio.get_device_info_by_index(INPUT_DEVICE_INDEX)

    if INPUT_DEVICE_NAME_HINT:
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0 and INPUT_DEVICE_NAME_HINT.lower() in info["name"].lower():
                return info
        raise RuntimeError(
            f"No input device matching {INPUT_DEVICE_NAME_HINT!r}. "
            "Run with --list-devices to see available devices."
        )

    return audio.get_device_info_by_index(audio.get_default_input_device_info()["index"])


def open_input_stream(audio, device_info):
    """Open the stream, preferring SAMPLE_RATE but falling back to the device's own default
    rate (Bluetooth headset mics often only support a lower rate like 16kHz). Returns
    (stream, actual_rate)."""
    device_index = device_info["index"]
    for rate in (SAMPLE_RATE, int(device_info["defaultSampleRate"])):
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE,
            )
            return stream, rate
        except OSError:
            continue
    raise RuntimeError(f"Could not open {device_info['name']!r} at any supported sample rate.")


def main():
    audio = pyaudio.PyAudio()
    device_info = choose_input_device(audio)
    stream, rate = open_input_stream(audio, device_info)
    print(f"Using input device: {device_info['name']!r} at {rate} Hz")

    sos = design_bandpass(rate)
    target_freq = calibrate_note(stream, rate, sos)
    print("Press Ctrl+C to quit.")

    filter_state = sosfilt_zi(sos) * 0
    match_streak = 0
    silence_streak = 0
    note_active = False  # True from the moment we publish until the note is confirmed stopped

    try:
        with MQTTClient() as client:
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)

                filtered_samples, filter_state = sosfilt(sos, samples, zi=filter_state)
                freq = dominant_frequency(filtered_samples, rate)
                level_db = amplitude_db(filtered_samples)
                heard_tone = level_db >= AMPLITUDE_THRESHOLD_DB
                is_match = heard_tone and abs(freq - target_freq) <= tolerance_hz(target_freq)

                print(f"\rFrequency: {freq:7.1f} Hz | Level: {level_db:5.1f} dB", end="", flush=True)

                # CONFIRM_CHUNKS consecutive matches trigger one publish; CONFIRM_CHUNKS
                # consecutive non-matches re-arm it. Using separate streaks (rather than one
                # counter that resets on any miss) means a single noisy chunk in the middle of
                # a held note can't cause a second publish for what's really one continuous note.
                match_streak = match_streak + 1 if is_match else 0
                silence_streak = 0 if is_match else silence_streak + 1

                if not note_active and match_streak >= CONFIRM_CHUNKS:
                    note_active = True
                    client.publish(SUB_TOPIC, MESSAGE)
                    print(f"\n  Published '{MESSAGE}' to '{SUB_TOPIC}'")
                elif note_active and silence_streak >= CONFIRM_CHUNKS:
                    note_active = False
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_input_devices()
    else:
        main()
