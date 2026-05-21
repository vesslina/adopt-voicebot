import socket
import struct
import logging

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


def recv_exact(conn, n):
    """Read exactly n bytes from socket."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by Asterisk")
        buf += chunk
    return buf


def handle_call(conn, addr):
    log.info("Call connected from %s:%d", *addr)
    try:
        while True:
            header = recv_exact(conn, 3)
            kind = header[0]
            length = struct.unpack(">H", header[1:3])[0]

            payload = recv_exact(conn, length) if length > 0 else b""

            if kind == KIND_UUID:
                uuid_str = payload.hex()
                log.info("Call UUID: %s-%s-%s-%s-%s",
                         uuid_str[0:8], uuid_str[8:12],
                         uuid_str[12:16], uuid_str[16:20], uuid_str[20:])

            elif kind == KIND_AUDIO:
                # Echo: send audio back as-is
                conn.sendall(header + payload)

            elif kind == KIND_HANGUP:
                log.info("Hangup received, closing call")
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
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(5)
    log.info("AudioSocket echo server listening on %s:%d", HOST, PORT)
    log.info("Press Ctrl+C to stop")

    try:
        while True:
            conn, addr = srv.accept()
            handle_call(conn, addr)  # single-threaded, one call at a time
    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
