import time
from collections import deque

from conf.conf import Config
from model.model import Model
from training_env.environment_control_updated import EnvironmentControl
import torch


def workflow():
    print("Starting test workflow...")
    INTERVAL = Config.INTERVAL
    SEQUENCE_LENGTH = 1000 # 1000 frames @ 5FPS
    ACTION_SPACE = Config.ACTION_DIM
    CONTEXT_FRAMES = Config.CONTEXT_FRAMES
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)
    env = EnvironmentControl()

    agent = Model()
    agent.to(DEVICE)
    checkpoint = torch.load("model/checkpoint_30.pth", map_location=DEVICE)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    agent.load_state_dict(checkpoint, strict=False)
    agent.yolo.eval()
    agent.eval()

    print("Model loaded and set to eval mode. Beginning in 2s")
    time.sleep(2)
    env.reset()
    state, info = env.get_state()
    frame_buffer = deque([state.clone() for _ in range(CONTEXT_FRAMES)], maxlen=CONTEXT_FRAMES)

    for frame in range(SEQUENCE_LENGTH):
        start_time = time.time()

        state_window = torch.stack(list(frame_buffer), dim=0).unsqueeze(0).to(DEVICE)
        action, _, _, _ = agent.select_action(state_window, stochastic=False)
        state, next_info = env.step(action.cpu().numpy())
        frame_buffer.append(state)
        print("Frame:", frame, "Action:", action)

        elapsed = time.time() - start_time
        time_to_wait = INTERVAL - elapsed
        if time_to_wait > 0:
            time.sleep(time_to_wait)
    
    env.pause()

if __name__ == "__main__":
    workflow()