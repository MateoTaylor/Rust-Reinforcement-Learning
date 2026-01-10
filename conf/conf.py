'''
Config file for PPO agent training
Author: Mateo Taylor
'''

import torch


class Config:
    FEATURE_DIM = [3, 320, 320] # FOR RESNET INPUT
    ACTION_DIM = [2, 2, 2, 2, 2, 2, 5, 5]
    
    INTERVAL = 0.20  # 200 ms per step (5 FPS)
    CHUNK_LENGTH = 256 # 256 frames @ 5FPS
    EPISODE_LENGTH = 768  # 2.5~ minutes

    # reward_info = {
    #     "resource_gathered": 0,
    #     "closest_node": 0,
    #     "swimming_penalty": 0,
    #     "looking_at_node": 0
    # }
    reward_info = {
        "distance_to_target_reward": 0,
    }

    LEARNING_RATE = 1e-5
    EPISODES = 1000

    LSTM_HIDDEN_SIZE = 256

    GAMMA = 0.99
    EPS_CLIP = 0.2    
    LAMDA = 0.95
    ENTROPY = 0.01

    GRADIENT_CLIP = 0.5

    EPOCHS = 4

    TRANSFORMER_LAYERS = 2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
