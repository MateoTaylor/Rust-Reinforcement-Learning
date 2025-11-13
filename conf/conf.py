'''
Config file for PPO agent training
Author: Mateo Taylor
'''

class GameConfig:
    MAX_STEPS = 50
    MOVEMENT_SPEED = 0.25 # seconds per unit distance


class DIM_CONFIG:
    OBSERVATION_DIM = [None]  
    FEATURE_DIM = [3, 360, 640]  # RGB screenshot dimensions (C, H, W)   
    ACTION_DIM = [2, 2, 4, 2, 3, 4, 3]
    # 2 Swing pickaxe, [No, Yes]
    # 2 Move vs Look, [Move,  Look]
    # 4 Movement dir, [None, Forward, Left, Right]
    # 2 Jump, [No, Yes]
    # 3 Movement scale, [1x, 2x, 3x]
    # 4 Mouse Movement, [Up, Down, Left, Right]
    # 3 Mouse movement scale, [15deg, 30deg, 60deg]


class Config:
    reward_info = {
        "resource_gathered": 0,
        "closest_node": 0,
        "swimming_penalty": 0,
        "looking_at_node": 0
    }
    FEATURE_DIM = DIM_CONFIG.FEATURE_DIM
    CONV_DIM = 0

    TRAINING_FEATURE_CLASSIFIER = True

    LEARNING_RATE = 2.5e-4
    EPISODES = 2000
    MAX_STEPS = GameConfig.MAX_STEPS

    ACTION_SPACE = DIM_CONFIG.ACTION_DIM
    HIDDEN_SIZE = 256
    TIMESTEP_SKIPPED = 5  # number of initial timesteps to skip in loss calculation due to unreliable hidden state

    GAMMA = 0.99
    EPS_CLIP = 0.2    
    LAMDA = 0.95
    ENTROPY = 0.01
    FEATURE_CLASSIFIER_COEF = 0.2 

    BATCH_SIZE = None
    EPOCHS = 4
    GRADIENT_CLIP = 0.5
