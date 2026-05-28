'''
Config file for PPO agent training
Author: Mateo Taylor
'''

import torch


class Config:
    FEATURE_DIM = [3, 640, 640] # FOR RESNET INPUT
    ACTION_DIM = [2, 2, 3, 3]
    
    INTERVAL = 0.20  # 200 ms per step (5 FPS)
    TRAIN_SEQUENCE_LENGTH = 8  # batch learning slices each chunk into 8-frame sequences
    EPISODE_LENGTH = 320  # 2.5~ minutes
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
    VALUE_HEAD_WARMUP_EPISODES = 5

    LSTM_HIDDEN_SIZE = 256

    GAMMA = 0.99
    EPS_CLIP = 0.2    
    LAMDA = 0.95
    ENTROPY = 0.01

    GRADIENT_CLIP = 0.5

    EPOCHS = 4

    TRANSFORMER_LAYERS = 2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
