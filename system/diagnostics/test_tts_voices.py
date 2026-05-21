import wave
from pathlib import Path
import sys

SYSTEM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_DIR / "app"))
import bot_server


TEXT = (
    "Алексей, по договору 10240001 баланс 438 рублей. "
    "Последний платеж 900 рублей, 5 мая."
)
SPEAKERS = ["baya", "kseniya", "xenia", "aidar", "eugene"]


def save_wav(path, pcm_bytes):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(bot_server.SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def main():
    out_dir = SYSTEM_DIR / "output" / "phrases"
    out_dir.mkdir(exist_ok=True)
    model = bot_server._load_tts_model()

    for speaker in SPEAKERS:
        bot_server.TTS_SPEAKER = speaker
        pcm, elapsed_ms = bot_server.synthesize(model, TEXT)
        out_path = out_dir / f"tts_voice_{speaker}.wav"
        save_wav(out_path, pcm)
        print(f"{speaker}: {out_path} ({elapsed_ms} ms)")


if __name__ == "__main__":
    main()
