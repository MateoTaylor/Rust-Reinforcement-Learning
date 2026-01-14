import time
from model.model import Model
from training_env.environment_control_updated import EnvironmentControl
import torch


def workflow():
    print("Starting test workflow...")
    INTERVAL = 0.2  # 200 ms per step (5 FPS)
    SEQUENCE_LENGTH = 1000 # 1000 frames @ 5FPS
    ACTION_SPACE = [2, 2, 2, 2, 2, 2, 5, 5]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)
    env = EnvironmentControl()

    checkpoint = torch.load("checkpoint_0.pth", map_location=DEVICE)
    agent = Model()
    agent.load_state_dict(checkpoint['model_state_dict'])
    
    agent.to(DEVICE)
    agent.eval()

    print("Model loaded and set to eval mode. Beginning in 2s")
    time.sleep(2)
    env.reset()
    state, info = env.step([0] * len(ACTION_SPACE))
    hidden = agent.init_hidden()  # Initialize hidden state from model
    
    state = state.view(1, *state.shape) # broadcast to [1, C, H, W]
    for frame in range(SEQUENCE_LENGTH):
        start_time = time.time()

        action, log_prob, value, hidden = agent.select_action(state, hidden)
        next_state, next_info = env.step(action)

        next_state = next_state.view(1, *next_state.shape) # broadcast to [1, C, H, W]
        state = next_state

        elapsed = time.time() - start_time
        time_to_wait = INTERVAL - elapsed
        if time_to_wait > 0:
            time.sleep(time_to_wait)
    
    env.pause()

if __name__ == "__main__":
    workflow()