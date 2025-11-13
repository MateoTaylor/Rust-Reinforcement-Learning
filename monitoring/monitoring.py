'''
Monitoring and logging utilities for the PPO agent training process.
Author: Mateo Taylor
'''

def monitor_loss(total_loss, actor_loss, critic_loss, entropy_loss, feature_loss):
    """ 
    Log loss values during training.
    """
    log_file = "monitoring/loss_log.txt"
    with open(log_file, "a") as f:
        f.write(f"Total Loss: {total_loss:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, Entropy Loss: {entropy_loss:.4f}, Feature Loss: {feature_loss:.4f}\n")

def monitor_episode(episode, reward, reward_info):
    """ 
    Log episode information such as episode number, state, reward, and reward details.
    """
    log_file = "monitoring/training_log.txt"
    with open(log_file, "a") as f:
        f.write(f"Episode {episode} - Total Reward: {reward} - Details: {reward_info}\n")
    