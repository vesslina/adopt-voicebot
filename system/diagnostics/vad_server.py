import socket
import struct
import logging
import os
import wave
import numpy as np
import torch
from silero_vad import load_silero_vad

logging.basicConfig(
    level=logging.DEBUG,
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
CHUNK_SAMPLES = 256        # Silero VAD требует ровно 256 сэмплов при 8kHz
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit mono = 2 байта на сэмпл

VAD_THRESHOLD = 0.5
SILENCE_TO_END_SEC = 0.6   # секунд тишины → конец фразы
MIN_SPEECH_SEC = 0.2       # фразы короче этого игнорируем

SYSTEM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(SYSTEM_DIR, "output", "phrases")


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


def save_wav(pcm_bytes, filepath):
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def handle_call(conn, addr, vad_model):
    log.info("Call connected from %s:%d", *addr)
    call_uuid = "unknown"

    audio_buf = b""      # буфер для нарезки на чанки VAD
    speech_buf = b""     # накопленная речь текущей фразы
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
                call_uuid = f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
                log.info("Call UUID: %s", call_uuid)

            elif kind == KIND_AUDIO:
                # Эхо — отправляем обратно, чтобы слышать входящий звук
                conn.sendall(header + payload)

                audio_buf += payload

                # Обрабатываем полные чанки по 256 сэмплов
                while len(audio_buf) >= CHUNK_BYTES:
                    chunk = audio_buf[:CHUNK_BYTES]
                    audio_buf = audio_buf[CHUNK_BYTES:]

                    tensor = pcm_to_tensor(chunk)
                    with torch.no_grad():
                        prob = vad_model(tensor, SAMPLE_RATE).item()

                    if prob >= VAD_THRESHOLD:
                        if not in_speech:
                            log.debug(">>> Речь началась (prob=%.2f)", prob)
                            in_speech = True
                            silence_samples = 0
                        speech_buf += chunk
                        speech_samples += CHUNK_SAMPLES
                        silence_samples = 0
                    else:
                        if in_speech:
                            # Накапливаем тишину внутри фразы
                            speech_buf += chunk
                            speech_samples += CHUNK_SAMPLES
                            silence_samples += CHUNK_SAMPLES

                            if silence_samples >= silence_threshold:
                                duration_ms = speech_samples * 1000 // SAMPLE_RATE
                                if speech_samples >= min_speech_samples:
                                    phrase_idx += 1
                                    filepath = f"{SAVE_DIR}\\phrase_{phrase_idx:03d}.wav"
                                    save_wav(speech_buf, filepath)
                                    log.info("=== Получена фраза #%d, длина %d мс → %s",
                                             phrase_idx, duration_ms, filepath)
                                else:
                                    log.debug("Короткий звук %d мс — игнорируем", duration_ms)

                                in_speech = False
                                speech_buf = b""
                                speech_samples = 0
                                silence_samples = 0

            elif kind == KIND_HANGUP:
                log.info("Hangup received")
                # Сохраняем остаток если абонент повесил трубку во время речи
                if in_speech and speech_samples >= min_speech_samples:
                    phrase_idx += 1
                    duration_ms = speech_samples * 1000 // SAMPLE_RATE
                    filepath = f"{SAVE_DIR}\\phrase_{phrase_idx:03d}.wav"
                    save_wav(speech_buf, filepath)
                    log.info("=== Финальная фраза #%d, длина %d мс → %s",
                             phrase_idx, duration_ms, filepath)
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
    log.info("Loading Silero VAD model...")
    vad_model = load_silero_vad()
    log.info("VAD model loaded OK")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(5)
    log.info("VAD AudioSocket server listening on %s:%d", HOST, PORT)
    log.info("Фразы сохраняются в: %s", SAVE_DIR)
    log.info("Press Ctrl+C to stop")

    try:
        while True:
            conn, addr = srv.accept()
            handle_call(conn, addr, vad_model)
    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
