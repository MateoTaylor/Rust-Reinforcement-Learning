from training_env.environment_control import EnvironmentControl
import json
import torch
import subprocess
from dotenv import load_dotenv
import os

import websockets
import asyncio

async def keepalive(websocket):
    """Send periodic pings to keep the WebSocket connection alive"""
    try:
        while True:
            await asyncio.sleep(30)  # Wait 30 seconds
            try:
                await websocket.ping()
                print("Sent keepalive ping")
            except Exception as e:
                print(f"Keepalive ping failed: {e}")
                break
    except asyncio.CancelledError:
        print("Keepalive task cancelled")

async def handler(websocket):
    env = EnvironmentControl()

    keepalive_task = asyncio.create_task(keepalive(websocket))

    try:
        while True:
            raw = await websocket.recv()
            data = json.loads(raw)
            print(f"Received data: {data}")
            
            if data["type"] == "ping":
                response = {"message": "pong"}
                await websocket.send(json.dumps(response))
            elif data["type"] == "reset":
                try:
                    env.reset()
                    response = {"message": "reset complete"}
                except Exception as e:
                    response = {"message": f"reset error {e}"}
                await websocket.send(json.dumps(response))
            elif data["type"] == "step":
                action = torch.tensor(data["action"], dtype=torch.long)
                state, done, info = env.step(action)
                response = {
                    "message": "step complete",
                    "state": state.numpy().tolist(),
                    "done": done,
                    "extra_info": info
                }
                await websocket.send(json.dumps(response))
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    finally:
        # Stop keepalive when connection closes
        keepalive_task.cancel()
        try:
            await keepalive_task
        except asyncio.CancelledError:
            pass

# Main function to start the WebSocket server
async def ws_server_main():
    async with websockets.serve(handler, "0.0.0.0", 8765, process_request=None):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever

async def main():
    # Start the WebSocket server
    ws_task = ws_server_main()

    # start the cloudflared tunnel

    subprocess.Popen([os.getenv("CLOUDFLARED_LOCATION"), "tunnel", "run", "rust-rl"])
    print("Cloudflared tunnel started")

    await ws_task

if __name__ == "__main__":
    load_dotenv()
    # Start the WebSocket server
    asyncio.run(main())

    