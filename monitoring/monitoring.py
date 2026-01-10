'''
Monitoring and logging utilities for the PPO agent training process.
Author: Mateo Taylor
'''


from conf.conf import Config
    
class Training_Logger:
    def __init__(self, start_episode, output_dir):
        self.output_dir = output_dir
        self.start_episode = start_episode
        self.total_rewards = []
        self.reward_details = []
        self.normalized_rewards = []
        
        self.total_loss = []
        self.actor_loss = []
        self.critic_loss = []
        self.entropy_loss = []

    def log_episode(self, reward, normalized_reward, reward_info):
        self.total_rewards.append(reward)
        self.normalized_rewards.append(normalized_reward)
        self.reward_details.append(reward_info)

    def log_loss(self, total_loss, actor_loss, critic_loss, entropy_loss):
        self.total_loss.append(total_loss)
        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)
        self.entropy_loss.append(entropy_loss)

    def save_logs(self):
        reward_log_file = f"{self.output_dir}/training_log.txt"
        loss_log_file = f"{self.output_dir}/loss_log.txt"
        
        with open(reward_log_file, "a") as f:
            for i, (reward, normalized_reward, details) in enumerate(zip(self.total_rewards, self.normalized_rewards, self.reward_details)):
                episode_num = self.start_episode + i
                f.write(f"Episode {episode_num} - Total Reward: {reward} - Normalized Reward: {normalized_reward} - Details: {details}\n")
        
        with open(loss_log_file, "a") as f:
            for i, (t_loss, a_loss, c_loss, e_loss) in enumerate(zip(self.total_loss, self.actor_loss, self.critic_loss, self.entropy_loss)):
                step_num = self.start_episode * Config.EPISODE_LENGTH + i // 4  * Config.CHUNK_LENGTH + i % 4 * Config.EPOCHS
                f.write(f"Step {step_num} - Total Loss: {t_loss:.4f}, Actor Loss: {a_loss:.4f}, Critic Loss: {c_loss:.4f}, Entropy Loss: {e_loss:.4f}\n")

        # reset after logging
        self.start_episode += 1
        self.total_rewards = []
        self.reward_details = []
        self.total_loss = []
        self.actor_loss = []
        self.critic_loss = []
        self.entropy_loss = []