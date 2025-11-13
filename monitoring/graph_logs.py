import matplotlib.pyplot as plt

def graph_loss():
    '''graph loss_log.txt and save as loss.pdf
    loss_log.txt format:
    Total Loss: 0.4372, Actor Loss: -0.0022, Critic Loss: 0.3248, Entropy Loss: -0.0704, Feature Loss: 0.1850
    '''
    log_file = "monitoring/loss_log.txt"
    total_losses = []
    actor_losses = []
    critic_losses = []
    entropy_losses = []
    feature_losses = []
    with open(log_file, "r") as file:
        for line in file:
            parts = line.split(", ")
            total_loss = float(parts[0].split(": ")[1])
            actor_loss = float(parts[1].split(": ")[1])
            critic_loss = float(parts[2].split(": ")[1])
            entropy_loss = float(parts[3].split(": ")[1])
            feature_loss = float(parts[4].split(": ")[1])
            
            total_losses.append(total_loss)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            entropy_losses.append(entropy_loss)
            feature_losses.append(feature_loss)
    epochs = list(range(1, len(total_losses) + 1))
    # Plotting loss details
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, actor_losses, label="Actor Loss")
    plt.plot(epochs, critic_losses, label="Critic Loss")
    plt.plot(epochs, entropy_losses, label="Entropy Loss")
    plt.plot(epochs, feature_losses, label="Feature Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.savefig("monitoring/loss_details.pdf")

    # graph total loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_losses, label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss Over Epochs")
    plt.legend()
    plt.savefig("monitoring/total_loss.pdf")


def graph_reward():
    '''graph training_log.txt and save as reward.pdf
    
    training_log.txt format:
    Episode 436 - Total Reward: -25.80000000000002 - Details: {'resource_gathered': 0, 'closest_node': 1.2, 'move_to_center': -9.999999999999996, 'swimming_penalty': -17.0}
    '''
    log_file = "monitoring/training_log.txt"
    episodes = []
    rewards = {}
    total_rewards = []
    with open(log_file, "r") as file:
        for line in file:
            parts = line.split(" - ")
            total_reward = parts[1]
            total_reward_value = float(total_reward.split(": ")[1])
            total_rewards.append(total_reward_value)

            episode_part = parts[0]
            episode_num = int(episode_part.split(" ")[1])

            reward_details = parts[2]
            reward_str = reward_details.split("Details: ")[1].strip()
            reward_dict = eval(reward_str)
            episodes.append(episode_num)
            # Accumulate rewards for each key
            for key, value in reward_dict.items():
                if key not in rewards:
                    rewards[key] = []
                rewards[key].append(value)
    
    # Plotting reward details
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for key, values in rewards.items():
        plt.plot(episodes, values, label=key, color=colors.pop(0))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.savefig("monitoring/reward_details.pdf")

    # graph total reward
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, total_rewards, label="Total Reward", color='b')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward Over Episodes")
    plt.legend()
    plt.savefig("monitoring/total_reward.pdf")

if __name__ == "__main__":
    graph_reward()
    graph_loss()