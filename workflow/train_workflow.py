"""
Main training workflow for the PPO agent 
"""

import time
from conf.conf import Config
from feature.reward_calculation import calculate_reward, RewardNormalizer
from ppo.algorithm import Algorithm
from model.model import Model
from training_env.environment_control_updated import EnvironmentControl
import os
import torch
from monitoring.monitoring import Training_Logger

def workflow(model=None, output_dir="backups/new", start_episode=0):
    env = EnvironmentControl()
    agent = Algorithm(Model(model))
    
    print("Config.DEVICE:", Config.DEVICE)
    print(torch.cuda.is_available())
    
    # Create backups folder if it doesn't exist
    os.makedirs("backups", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger = Training_Logger(start_episode, output_dir)
    reward_norm = RewardNormalizer()

    for episode in range(start_episode, Config.EPISODES):
        env.reset()
        step_info = []
        state, info = env.step([0] * len(Config.ACTION_DIM))
        total_reward = 0
        total_reward_info = Config.reward_info.copy()
        total_normalized_reward = 0
        
        hidden = agent.model.init_hidden()  # Initialize hidden state from model
        chunk_start_hidden = hidden
        for step in range(Config.EPISODE_LENGTH):
            start_time = time.time()
            # selected action, log probability over all actions, and value estimate from the agent
            # pass in the current + hidden state of the LSTM
            action, log_prob, value, hidden = agent.select_action(state, hidden)

            # next_state is a PIL image converted, done is a bool,
            # next_info is extra info for reward calculation
            next_state, next_info = env.step(action)
            done = (step + 1) % Config.CHUNK_LENGTH == 0

            reward, reward_info = calculate_reward(info, next_info)
            normalized_reward = reward_norm.normalize(reward, done)
            #log total reward info
            total_reward += reward
            total_normalized_reward += normalized_reward
            for key, value in reward_info.items():
                total_reward_info[key] += value

            # collect step info for learning after the episode
            step_info.append((state.detach(), action.detach(), reward, log_prob.detach(), value))
            state = next_state  
            info = next_info     
            if (step + 1) % Config.CHUNK_LENGTH == 0:
                env.pause() # lift up all keys while learning
                new_hidden = agent.learn(step_info, next_state, chunk_start_hidden, hidden, logger)
                hidden = tuple(h.detach().clone() for h in new_hidden) 
                chunk_start_hidden = hidden
                step_info = []
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elapsed = time.time() - start_time
            time_to_wait = Config.INTERVAL - elapsed
            if time_to_wait > 0:
                time.sleep(time_to_wait)
                
        print(f"Episode {episode} learning complete.")
        # Log episode information
        logger.log_episode(total_reward, total_normalized_reward, total_reward_info)
        logger.save_logs()
        # Save model every 5 episodes
        if (episode + 1) % 5 == 0:
            agent.save(f"{output_dir}/model_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")

if __name__ == "__main__":
    print("Starting training in 2 seconds...")
    time.sleep(2)  # give user time to switch to the game window
    workflow(
        model=None,
        output_dir="backups/1-10-Morning",
        start_episode=0
        )