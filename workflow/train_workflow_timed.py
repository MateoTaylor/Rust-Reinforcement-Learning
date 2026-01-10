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
from monitoring.monitoring import monitor_episode
import torch


def workflow(model=None, output_dir="backups/new"):
    env = EnvironmentControl()
    agent = Algorithm(Model(model))
    
    print("Config.DEVICE:", Config.DEVICE)
    print(torch.cuda.is_available())
    
    # Create backups folder if it doesn't exist
    os.makedirs("backups", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    # delete old training logs
    if os.path.exists("monitoring/training_log.txt"):
        os.remove("monitoring/training_log.txt")
    if os.path.exists("monitoring/loss_log.txt"):
        os.remove("monitoring/loss_log.txt")

    for episode in range(Config.EPISODES):
        env.reset()
        step_info = []
        state, info = env.step([0] * len(Config.ACTION_DIM))
        total_reward = 0
        total_reward_info = Config.reward_info.copy()
        
        hidden = agent.model.init_hidden()  # Initialize hidden state from model
        for step in range(Config.EPISODE_LENGTH):
            hidden = tuple(h.detach() for h in hidden)
            start_time = time.time()
            
            # selected action, log probability over all actions, and value estimate from the agent
            # pass in the current + hidden state of the LSTM
            action, log_prob, value, hidden = agent.select_action(state, hidden)
            after_action_selection = time.time()
            print(f"  Action selection: {after_action_selection - start_time:.4f}s")

            # next_state is a PIL image converted, done is a bool,
            # next_info is extra info for reward calculation
            next_state, next_info = env.step(action)
            after_env_step = time.time()
            print(f"  Environment step: {after_env_step - after_action_selection:.4f}s")
            
            reward, reward_info = calculate_reward(info, next_info)
            after_reward_calc = time.time()
            print(f"  Reward calculation: {after_reward_calc - after_env_step:.4f}s")

            #log total reward info
            total_reward += reward
            for key, value in reward_info.items():
                total_reward_info[key] += value

            # collect step info for learning after the episode
            step_info.append((state, action, reward, log_prob, value))
            state = next_state  
            info = next_info     
            
            if (step + 1) % Config.CHUNK_LENGTH == 0:
                hidden = agent.learn(step_info, next_state, hidden)
                step_info = []
            elapsed = time.time() - start_time
            time_to_wait = Config.INTERVAL - elapsed
            print(f"Total step time: {elapsed:.4f}s")
            if time_to_wait > 0:
                time.sleep(time_to_wait)

                
        print(f"Episode {episode} learning complete.")
        # Log episode information
        monitor_episode(episode, total_reward, total_reward_info)

        # Save model every 5 episodes
        if (episode + 1) % 5 == 0:
            agent.save(f"{output_dir}/model_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")

if __name__ == "__main__":
    print("Starting training in 2 seconds...")
    time.sleep(2)  # give user time to switch to the game window
    workflow(
        model=None,
        output_dir="backups/1-2-Evening"
        )