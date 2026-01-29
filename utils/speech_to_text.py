import subprocess
import sounddevice as sd
import wavio
import os

# Recording settings
DURATION = 5  # seconds
FS = 16000    # sample rate
MAX_SILENCE_SECONDS = 0.8

# Paths
WHISPER_CLI_PATH = "/home/zeek/Projects/linux-assistant_3/whisper-cli"
MODEL_PATH = "/home/zeek/Projects/linux-assistant_3/ggml-base.en.bin"
AUDIO_FILE = "audio.wav"

def record_audio():
    print(f"🎙️ Recording for {DURATION} seconds...")
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1)
    sd.wait()
    wavio.write(AUDIO_FILE, recording, FS, sampwidth=2)
    return AUDIO_FILE

def transcribe_audio(file_path):
    try:
        result = subprocess.run(
            [WHISPER_CLI_PATH, file_path, "--model", MODEL_PATH, "--language", "en"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("Error running whisper-cli:\n", result.stderr)
            return None
        return result.stdout
    except FileNotFoundError:
        print(f"whisper-cli not found at {WHISPER_CLI_PATH}")
        return None

def stt():
    audio_file = record_audio()
    text = transcribe_audio(audio_file)
    if text:
        print("Transcribed text:\n", text)
    else:
        print("No text was transcribed.")
    return text

if __name__ == "__main__":
    stt()