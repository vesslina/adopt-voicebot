import socket
import struct
import logging
import io
import time
import wave
import numpy as np
import torch
from silero_vad import load_silero_vad
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

HOST = "0.0.0.0"
PORT = 8090

KIND_HANGUP = 0x00
KIND_UUID   = 0x01
KIND_AUDIO  = 0x10
KIND_ERROR  = 0xFF

SAMPLE_RATE = 8000
CHUNK_SAMPLES = 256
CHUNK_BYTES = CHUNK_SAMPLES * 2

VAD_THRESHOLD = 0.5
SILENCE_TO_END_SEC = 0.6
MIN_SPEECH_SEC = 0.2
PRE_ROLL_SEC = 0.2   # сохраняем X сек "до" начала речи — фиксит обрез начала слов

WHISPER_MODEL = "small"
WHISPER_LANG = "ru"


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by Asterisk")
        buf += chunk
    return buf


def pcm_to_tensor(pcm_bytes):
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples)


def pcm_to_wav_bytes(pcm_bytes):
    """Конвертирует PCM 8kHz 16-bit mono в WAV-байты для Whisper."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf


def transcribe(whisper_model, pcm_bytes):
    wav_buf = pcm_to_wav_bytes(pcm_bytes)
    t0 = time.monotonic()
    segments, info = whisper_model.transcribe(
        wav_buf,
        language=WHISPER_LANG,
        beam_size=3,
        vad_filter=False,   # VAD уже сделан нашим Silero
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return text, elapsed_ms


def handle_call(conn, addr, vad_model, whisper_model):
    log.info("Call connected from %s:%d", *addr)

    pre_roll_bytes = int(PRE_ROLL_SEC * SAMPLE_RATE) * 2
    pre_roll_buf = b""     # скользящий буфер тишины перед речью

    audio_buf = b""        # буфер для нарезки на VAD-чанки
    speech_buf = b""       # накопленная речь текущей фразы
    in_speech = False
    silence_samples = 0
    speech_samples = 0
    phrase_idx = 0

    silence_threshold = int(SILENCE_TO_END_SEC * SAMPLE_RATE)
    min_speech_samples = int(MIN_SPEECH_SEC * SAMPLE_RATE)

    try:
        while True:
            header = recv_exact(conn, 3)
            kind = header[0]
            length = struct.unpack(">H", header[1:3])[0]
            payload = recv_exact(conn, length) if length > 0 else b""

            if kind == KIND_UUID:
                h = payload.hex()
                uuid = f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
                log.info("Call UUID: %s", uuid)

            elif kind == KIND_AUDIO:
                # Эхо пока отключено — будем слышать тишину (это нормально на этом этапе)
                # conn.sendall(header + payload)

                audio_buf += payload

                while len(audio_buf) >= CHUNK_BYTES:
                    chunk = audio_buf[:CHUNK_BYTES]
                    audio_buf = audio_buf[CHUNK_BYTES:]

                    tensor = pcm_to_tensor(chunk)
                    with torch.no_grad():
                        prob = vad_model(tensor, SAMPLE_RATE).item()

                    if prob >= VAD_THRESHOLD:
                        if not in_speech:
                            in_speech = True
                            silence_samples = 0
                            # Добавляем pre-roll — последние X мс тишины перед речью
                            speech_buf = pre_roll_buf
                            speech_samples = len(pre_roll_buf) // 2
                        speech_buf += chunk
                        speech_samples += CHUNK_SAMPLES
                        silence_samples = 0
                        pre_roll_buf = b""
                    else:
                        if in_speech:
                            speech_buf += chunk
                            speech_samples += CHUNK_SAMPLES
                            silence_samples += CHUNK_SAMPLES

                            if silence_samples >= silence_threshold:
                                duration_ms = speech_samples * 1000 // SAMPLE_RATE
                                if speech_samples >= min_speech_samples:
                                    phrase_idx += 1
                                    log.info("--- Фраза #%d (%d мс) → STT...", phrase_idx, duration_ms)
                                    text, stt_ms = transcribe(whisper_model, speech_buf)
                                    log.info("=== Фраза #%d: \"%s\"  [STT: %d мс]", phrase_idx, text, stt_ms)

                                in_speech = False
                                speech_buf = b""
                                speech_samples = 0
                                silence_samples = 0
                                pre_roll_buf = b""
                        else:
                            # Накапливаем скользящий pre-roll буфер
                            pre_roll_buf = (pre_roll_buf + chunk)[-pre_roll_bytes:]

            elif kind == KIND_HANGUP:
                log.info("Hangup received")
                if in_speech and speech_samples >= min_speech_samples:
                    phrase_idx += 1
                    duration_ms = speech_samples * 1000 // SAMPLE_RATE
                    log.info("--- Финальная фраза #%d (%d мс) → STT...", phrase_idx, duration_ms)
                    text, stt_ms = transcribe(whisper_model, speech_buf)
                    log.info("=== Фраза #%d: \"%s\"  [STT: %d мс]", phrase_idx, text, stt_ms)
                break

            elif kind == KIND_ERROR:
                log.error("Asterisk error: %s", payload.decode(errors="replace"))
                break

            else:
                log.warning("Unknown kind=0x%02X length=%d, skipping", kind, length)

    except ConnectionError as e:
        log.info("Connection ended: %s", e)
    except Exception as e:
        log.exception("Unexpected error: %s", e)
    finally:
        conn.close()
        log.info("Connection closed: %s:%d", *addr)


def main():
    log.info("Loading Silero VAD...")
    vad_model = load_silero_vad()
    log.info("VAD loaded OK")

    log.info("Loading Whisper '%s' model (первый раз — скачает ~244 МБ)...", WHISPER_MODEL)
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    log.info("Whisper loaded OK")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(5)
    log.info("STT AudioSocket server listening on %s:%d", HOST, PORT)
    log.info("Press Ctrl+C to stop")

    try:
        while True:
            conn, addr = srv.accept()
            handle_call(conn, addr, vad_model, whisper_model)
    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
