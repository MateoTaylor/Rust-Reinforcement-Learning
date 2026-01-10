'''
basic PPO algorithm implementation
Author: Mateo Taylor
'''

import torch
from conf.conf import Config
import torch.nn as nn

class Algorithm:
    def __init__(self, model):
        self.model = model.to(Config.DEVICE)  # Move model to device       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.LAMDA
        self.eps_clip = Config.EPS_CLIP
        self.entropy_coef = Config.ENTROPY
        self.device = Config.DEVICE
    
    def select_action(self, state, hidden_state  ):
        """Wrapper to call model's select_action method."""
        return self.model.select_action(state, hidden_state)

    def compute_gae(self, rewards, masks, values, next_value):
        """
        Compute Generalized Advantage Estimation.
        All inputs should be tensors.
        """
        advantages = []
        gae = 0
        values_list = values.tolist() + [next_value.item()]
        rewards_list = rewards.tolist()
        masks_list = masks.tolist()
        
        for step in reversed(range(len(rewards_list))):
            delta = rewards_list[step] + self.gamma * values_list[step + 1] * masks_list[step] - values_list[step]
            gae = delta + self.gamma * self.gae_lambda * masks_list[step] * gae
            advantages.insert(0, gae)
        return advantages

    def learn(self, sample_data, next_state, hidden_state_in, hidden_state_out, logger):
        # sample data contains tuples of (state, action, reward, next_state, done, log_prob, value)
        # unpack sample data
        states, actions, rewards, old_log_probs, values = zip(*sample_data)
        states = torch.stack(states)  # [seq_len, 3, 360, 640]
        actions = torch.stack(actions).long()  # [seq_len, num_action_dims]
        rewards = torch.tensor(rewards, dtype=torch.float32)  # [seq_len]
        old_log_probs = torch.stack(old_log_probs)  # [seq_len]
        values = torch.stack([torch.tensor(v).squeeze() for v in values])  # [seq_len]
        masks = torch.tensor([1.0] * len(rewards), dtype=torch.float32)  # [seq_len]

        with torch.no_grad():
            
            # Evaluate next_state with final hidden state'
            next_state_tensor = next_state.detach().clone().to(self.device)
            _, _, next_value, _ = self.model.evaluate(
                next_state_tensor,
                torch.zeros_like(actions[0]),
                hidden_state_out
            )
            next_value = next_value.cpu()

        # Compute advantages and returns
        advantages = self.compute_gae(rewards, masks, values.detach(), next_value)
        advantages = torch.tensor(advantages, dtype=torch.float32)  # [seq_len]
        returns = advantages + values.detach()  # [seq_len]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy for K epochs
        for _ in range(Config.EPOCHS):
            # Initialize fresh hidden state for the sequence
            hidden = tuple(h.to(Config.DEVICE).detach().clone() for h in hidden_state_in)
            
            all_log_probs = []
            all_state_values = []
            all_entropies = []
            
            # Process each timestep sequentially
            for t in range(states.size(0)):
                state_t = states[t].to(self.device)  # [3, 360, 640]
                action_t = actions[t].to(self.device)  # [num_action_dims]

                for h in hidden:
                    h.requires_grad_(True)
                    
                log_prob_t, state_value_t, entropy_t, hidden = torch.utils.checkpoint.checkpoint(
                    self.model.evaluate,
                    state_t,
                    action_t,
                    hidden,
                    use_reentrant=False
                )

                all_log_probs.append(log_prob_t.cpu())
                all_state_values.append(state_value_t.squeeze().cpu())
                all_entropies.append(entropy_t.cpu())

            # Stack all timesteps
            log_probs = torch.stack(all_log_probs)  # [seq_len - skipped]
            state_values = torch.stack(all_state_values)  # [seq_len - skipped]
            dist_entropy = torch.stack(all_entropies)  # [seq_len - skipped]
           
            # Move to GPU for loss computation
            log_probs = log_probs.to(self.device)
            state_values = state_values.to(self.device)
            dist_entropy = dist_entropy.to(self.device)
            old_log_probs_gpu = old_log_probs.to(self.device)
            advantages_gpu = advantages.to(self.device)
            returns_gpu = returns.to(self.device)
            # Finding the ratio (pi_theta / pi_theta__old) 
            ratios = torch.exp(log_probs - old_log_probs_gpu.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages_gpu
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_gpu

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns_gpu - state_values).pow(2).mean()
            entropy_loss = -self.entropy_coef * dist_entropy.mean()

            loss = actor_loss + critic_loss + entropy_loss

            logger.log_loss(loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item())
            # take gradient step
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
            self.optimizer.step()
        
        return hidden
            
    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)


def print_vram(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(Config.DEVICE) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(Config.DEVICE) / 1024**3  # GB
        print(f"{label}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
