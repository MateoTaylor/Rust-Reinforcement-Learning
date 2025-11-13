'''
Controls messages to the Rust training environment.
Messages are communicated to C# plugin via TCP socket at port 5555.

Current messages:
- "GET_STATE": Request the current state of the environment. Returns JSON of nodes + players in JSON
- "RESET_ENV": Reset the environment. Empties inventories, kills player, clears + respawns all nodes.

Author: Mateo Taylor
'''

import socket
import json

def send_command(command: str, host="127.0.0.1", port=5555):
    """Send a simple text command to the Rust Oxide plugin and return JSON."""
    try:
        with socket.create_connection((host, port), timeout=2) as s:
            s.sendall(command.encode("utf-8"))
            s.shutdown(socket.SHUT_WR)  # tell server we're done sending

            # Read until the server closes the connection
            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk

        # Decode and parse JSON safely
        text = data.decode("utf-8").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print("Response: ", text)
            return None

    except (ConnectionRefusedError, TimeoutError) as e:
        print(f" Connection failed: {e}")
        return None

def get_state():
    """Request the current state of the environment."""
    return send_command("GET_STATE")

def reset_env():
    """Reset the environment."""
    return send_command("RESET_ENV")

def give_pickaxe():
    """Give the player a pickaxe."""
    return send_command("GIVE_ITEM")