"""
Main training workflow for the PPO agent 
"""

from time import sleep
from conf.conf import Config
from feature.reward_calculation import calculate_reward
from ppo.algorithm import Algorithm
from model.model import Model
from training_env.environment_control import EnvironmentControl
import os
from monitoring.monitoring import monitor_episode



def workflow(model=None, output_dir="backups/new"):
    env = EnvironmentControl()
    agent = Algorithm(Model(model), device="cpu")
    
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
        done = False
        step_info = []
        state, done, info = env.step([0] * len(Config.ACTION_SPACE))
        total_reward = 0
        total_reward_info = Config.reward_info.copy()
        
        hidden = agent.model.init_hidden()  # Initialize hidden state from model
        for step in range(Config.MAX_STEPS):
            # selected action, log probability over all actions, and value estimate from the agent
            # pass in the current + hidden state of the LSTM
            action, log_prob, value, hidden = agent.select_action(state, hidden)

            # next_state is a PIL image converted, done is a bool,
            # next_info is extra info for reward calculation
            next_state, done, next_info = env.step(action)
            reward, reward_info = calculate_reward(info, next_info)

            #log total reward info
            total_reward += reward
            for key, value in reward_info.items():
                total_reward_info[key] += value

            # will always be 0 if we're not training the feature classifier
            # NOTE: if doing multiple features later, this needs to be a list
            # TODO: ADD THING THAT MAKES THE ENV. SEATCH FOR nodeInView Conditional!
            extr_feat_ground_truth = next_info["players"][0]["nodeInView"]

            # collect step info for learning after the episode
            step_info.append((state, action, reward, next_state, done, log_prob, value, extr_feat_ground_truth))
            state = next_state  
            info = next_info     
            
            if done:
                break
        
        print(f"Episode {episode} finished in {step+1} steps.")
        # pass in big list of step info for learning
        agent.learn(step_info)
        print(f"Episode {episode} learning complete.")
        # Log episode information
        monitor_episode(episode, total_reward, total_reward_info)

        # Save model every 10 episodes
        if (episode + 1) % 10 == 0:
            agent.save(f"{output_dir}/model_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")

if __name__ == "__main__":
    print("Starting training in 2 seconds...")
    sleep(2)  # give user time to switch to the game window
    workflow(
        output_dir="backups/11-12-Evening"
        )  