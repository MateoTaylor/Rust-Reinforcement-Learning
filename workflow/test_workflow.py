import time
from model.model import Model
from training_env.environment_control_updated import EnvironmentControl
import torch


def workflow():
    print("Starting test workflow...")
    INTERVAL = 0.2  # 200 ms per step (5 FPS)
    SEQUENCE_LENGTH = 1000 # 1000 frames @ 5FPS
    ACTION_SPACE = [2, 2, 3, 3]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)
    env = EnvironmentControl()

    agent = Model()
    agent.to(DEVICE)
    checkpoint = torch.load("model/checkpoint_3.pth", map_location=DEVICE)
    agent.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    agent.eval()

    print("Model loaded and set to eval mode. Beginning in 2s")
    time.sleep(2)
    env.reset()
    hidden = None
    state, info = env.step([0] * len(ACTION_SPACE))
    
    state = state.view(1, *state.shape) # broadcast to [1, C, H, W]
    for frame in range(SEQUENCE_LENGTH):
        start_time = time.time()

        state = state.to(DEVICE)
        action, _, hidden, _ = agent.select_action(state, hidden=hidden)
        state, next_info = env.step(action)
        state = state.view(1, *state.shape) # broadcast to [1, C, H, W]
        print("Frame:", frame, "Action:", action)

        elapsed = time.time() - start_time
        time_to_wait = INTERVAL - elapsed
        if time_to_wait > 0:
            time.sleep(time_to_wait)
    
    env.pause()

if __name__ == "__main__":
    workflow()