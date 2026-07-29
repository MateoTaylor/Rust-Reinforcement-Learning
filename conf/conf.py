'''
Config file for PPO agent training
Author: Mateo Taylor
'''

import torch


class Config:
    FEATURE_DIM = [3, 320, 320]
    ACTION_DIM = [2, 2, 3, 3]
    CONTEXT_FRAMES = 5
    CHUNK_SIZE = 10
    
    INTERVAL = 0.25  # 200 ms per step (5 FPS)
    EPISODE_LENGTH = 640  # 2.5~ minutes
    CHUNK_LENGTH = 80  # 80 frames per chunk

    reward_info = {
        "resource_gathered": 0,
        "closest_node": 0,
        "swimming_penalty": 0,
        "looking_at_node": 0
    }
    # reward_info = {
    #     "distance_to_target_reward": 0,
    # }

    START_LEARNING_RATE = 1e-4
    TARGET_LEARNING_RATE = 5e-5
    EPISODES = 2000
    VALUE_HEAD_WARMUP_EPISODES = 75

    LSTM_HIDDEN_SIZE = 512

    GAMMA = 0.99
    EPS_CLIP = 0.2    
    LAMDA = 0.95
    ENTROPY = 0.05

    GRADIENT_CLIP = 0.5

    EPOCHS = 4

    TRANSFORMER_LAYERS = 2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
