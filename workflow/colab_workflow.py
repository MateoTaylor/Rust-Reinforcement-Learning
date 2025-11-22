import socket
from pyngrok import ngrok
import time
from training_env.environment_control import EnvironmentControl
import json
import torch

# initialize ngrok tunnel
def init_ngrok(host="0.0.0.0", port=8501, proto="tcp"):
    # Open a local TCP socket to accept connections from colab
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)

    # Open an ngrok TCP tunnel to the given local port
    tunnel = ngrok.connect(addr=port, proto=proto)
    public_url = tunnel.public_url
    print(f" * ngrok tunnel \"{public_url}\" -> \"{host}:{port}\"")

    # Parse remote host and port from tcp://host:port
    remote_host, remote_port = None, None
    if public_url and public_url.startswith("tcp://"):
        try:
            host_port = public_url.split("tcp://", 1)[1]
            remote_host, remote_port = host_port.split(":")
            remote_port = int(remote_port)
            print(f" * remote endpoint is: \nNGROK_HOST= '{remote_host}' \nNGROK_PORT= {remote_port}\n")
        except Exception:
            pass

    return server, public_url, remote_host, remote_port

def colab_handshake(server, max_retries=5, wait_seconds=10):
    """Wait for a TCP signal from the colab """
    retries = 0
    while retries < max_retries:
        try:
            server.settimeout(30)  # allow accept to time out
            conn, addr = server.accept()  # wait for incoming connection from colab (via ngrok)
            with conn:
                conn.settimeout(30)
                raw = conn.recv(4096)
                if not raw:
                    return False
                msg = raw.decode("utf-8", errors="ignore").strip()
                print(f"Received handshake message: '{msg}' from {addr}")
                if msg == "Handshake1":
                    conn.sendall("Handshake2".encode("utf-8"))
                    print(f"Handshake from {addr} succeeded")
                    return True
                return False
        except socket.timeout:
            retries += 1
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            return False
        except Exception as exc:
            print("Handshake error:", exc)
            retries += 1
            time.sleep(wait_seconds)
    print(" * Max retries reached, colab handshake failed.")
    return False

if __name__ == "__main__":
    # call with port as a named argument so the host parameter remains a string
    server, public_url, remote_host, remote_port = init_ngrok(port=8501)

    # ensure received a remote TCP endpoint from ngrok before doing the handshake
    if remote_host is None or remote_port is None:
        print("ngrok did not provide a remote TCP endpoint.")
        exit(1)

    if not colab_handshake(server):
        print("Colab handshake failed.")
        server.close()
        exit(1)

    print("Colab handshake successful.")

    # poll indefinitely
    server.settimeout(200)
    conn, addr = server.accept()
    env = EnvironmentControl()
    while True:
        try:
            raw_bytes = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                raw_bytes += chunk
                if b"\n" in chunk:  # assuming newline-delimited messages
                    break
            if not raw_bytes:
                break
            # check for any re-handshakes
            msg = raw_bytes.decode("utf-8", errors="ignore").strip()
            if msg == "Handshake1":
                conn.sendall("Handshake2".encode("utf-8"))

            raw = json.loads(raw_bytes.decode("utf-8").strip())
            if raw["type"] == "reset":
                # reset env stuff
                try:
                    env.reset()
                    msg = json.dumps({"message": "reset complete"}) + "\n"
                    conn.sendall(msg.encode("utf-8"))
                except Exception as e:
                    msg = json.dumps({"message": f"reset error {e}"}) + "\n"
                    conn.sendall(msg.encode("utf-8"))
                    exit(1)

            elif raw["type"] == "step":
                # step env stuff
                action = torch.tensor(raw["action"], dtype=torch.long)
                state, done, info = env.step(action)
                msg = json.dumps({
                    "message": "step complete",
                    "state": state.numpy().tolist(),
                    "done": done,
                    "extra_info": info
                }) + "\n"
                conn.sendall(msg.encode("utf-8"))
        except socket.timeout:
            # allow timeout to continue polling
            continue