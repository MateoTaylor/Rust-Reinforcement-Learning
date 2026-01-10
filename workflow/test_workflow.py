import time
from model.model import Model
from training_env.environment_control_updated import EnvironmentControl
import torch


def workflow():
    print("Starting test workflow...")
    INTERVAL = 0.2  # 200 ms per step (5 FPS)
    SEQUENCE_LENGTH = 100 # 100 frames @ 5FPS
    ACTION_SPACE = [2, 2, 2, 2, 2, 2, 9, 9]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", DEVICE)
    env = EnvironmentControl()

    agent = Model("backups/1-7-Evening/model_episode_350.pth")
    agent.to(DEVICE)
    agent.eval()

    print("Model loaded and set to eval mode. Beginning in 2s")
    time.sleep(2)
    env.reset()
    state, info = env.step([0] * len(ACTION_SPACE))
    hidden = agent.init_hidden()  # Initialize hidden state from model
    

    for frame in range(SEQUENCE_LENGTH):
        start_time = time.time()

        action, log_prob, value, hidden = agent.select_action(state, hidden)
        next_state, next_info = env.step(action)
        state = next_state

        elapsed = time.time() - start_time
        time_to_wait = INTERVAL - elapsed
        if time_to_wait > 0:
            time.sleep(time_to_wait)
    
    state, info = env.step([0] * len(ACTION_SPACE))

if __name__ == "__main__":
    workflow()