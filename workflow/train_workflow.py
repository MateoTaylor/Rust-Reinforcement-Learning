"""
Main training workflow for the PPO agent 
"""

import time
from conf.conf import Config
from feature.reward_calculation import calculate_reward
from ppo.algorithm import Algorithm
from model.model import Model
from training_env.environment_control_updated import EnvironmentControl
import os
import torch
from monitoring.monitoring import Training_Logger
import numpy as np

def workflow(model=None, output_dir="backups/new", start_episode=0):
    env = EnvironmentControl()
    agent = Algorithm(model)

    print("Config.DEVICE:", Config.DEVICE)
    print(torch.cuda.is_available())
    
    # Create backups folder if it doesn't exist
    os.makedirs("backups", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger = Training_Logger(start_episode, output_dir)

    for episode in range(start_episode, Config.EPISODES):
        train_policy = episode >= Config.VALUE_HEAD_WARMUP_EPISODES
        agent.model.set_trainable_components(train_policy)

        env.reset()
        step_info = []
        state, info = env.step([0] * len(Config.ACTION_DIM))
        
        total_reward = 0
        total_reward_info = Config.reward_info.copy()
        total_normalized_reward = 0
        
        hidden = agent.model.init_hidden()  # Initialize hidden state from model
        for step in range(Config.EPISODE_LENGTH):
            start_time = time.time()
            current_hidden = hidden
            # selected action, log probability over all actions, and value estimate from the agent
            # pass in the current + hidden state of the LSTM
            action, log_prob, value, hidden = agent.select_action(state, hidden)

            # next_state is a PIL image converted, done is a bool,
            # next_info is extra info for reward calculation
            
            next_state, next_info = env.step(action.cpu().numpy())

            reward, reward_info = calculate_reward(info, next_info)
            #log total reward info
            total_reward += reward
            for key, value in reward_info.items():
                total_reward_info[key] += value

            # Is this the last step of the chunk?
            is_chunk_terminal = (step + 1) % Config.CHUNK_LENGTH == 0
            mask = 0.0 if is_chunk_terminal else 1.0

            # collect step info for learning after the episode
            step_info.append((state.detach(), action.detach(), reward, log_prob.detach(), value, current_hidden.detach(), mask))
            
            state = next_state  
            info = next_info     
            elapsed = time.time() - start_time
            time_to_wait = Config.INTERVAL - elapsed
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            
            # If chunk is done but episode isn't completely finished, reset env for the next chunk
            if is_chunk_terminal and (step + 1) < Config.EPISODE_LENGTH:
                # reset the environemnt so it can learn from multiple episodes
                env.pause()
                env.reset()
                state, info = env.step([0] * len(Config.ACTION_DIM))
                hidden = agent.model.init_hidden()  # Initialize hidden state from model
                
        env.pause()
        if total_reward < 0.20:
            # skip learning 50% of the time when there's no reward
            if np.random.rand() < 0.50 and episode > Config.VALUE_HEAD_WARMUP_EPISODES:
                episode -= 1
                continue
        
        agent.learn(step_info, next_state, hidden, hidden, logger)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Episode {episode} learning complete.")
        
        # Step the scheduler if within the warmup period
        agent.step_scheduler(episode)
        
        # Log episode information
        logger.log_episode(total_reward, total_normalized_reward, total_reward_info)
        logger.save_logs()
        # Save model every 5 episodes
        if (episode + 1) % 5 == 0:
            agent.save(f"{output_dir}/model_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")

if __name__ == "__main__":
    print("Starting training in 2 seconds...")
    model = Model()
    checkpoint = torch.load("backups/5-26-Afternoon/model_episode_235.pth", map_location=Config.DEVICE)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.load_state_dict(checkpoint, strict=False)
    time.sleep(2)  # give user time to switch to the game window
    workflow(
        model=model,
        output_dir="backups/5-26-Afternoon",
        start_episode=176
        )