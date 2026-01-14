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
        All inputs should be tensors on the same device.
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        values_extended = torch.cat([values, next_value.unsqueeze(0)])
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values_extended[step + 1] * masks[step] - values_extended[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            advantages[step] = gae
        return advantages

    def learn(self, sample_data, next_state, hidden_state_in, hidden_state_out, logger):
        # sample data contains tuples of (state, action, reward, next_state, done, log_prob, value)
        # unpack sample data
        states, actions, rewards, old_log_probs, values = zip(*sample_data)
        
        # Create all tensors directly on GPU
        states = torch.stack(states).to(self.device)  # [seq_len, 3, 360, 640]
        actions = torch.stack(actions).long().to(self.device)  # [seq_len, num_action_dims]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # [seq_len]
        old_log_probs = torch.stack(old_log_probs).to(self.device)  # [seq_len]
        values = torch.stack([torch.tensor(v).squeeze() for v in values]).to(self.device)  # [seq_len]
        masks = torch.ones(len(rewards), dtype=torch.float32, device=self.device)  # [seq_len]

        with torch.no_grad():
            # Evaluate next_state with final hidden state
            next_state_tensor = next_state.detach().clone().to(self.device)
            _, _, next_value, _ = self.model.evaluate(
                next_state_tensor,
                torch.zeros_like(actions[0]),
                hidden_state_out
            )
            next_value = next_value.squeeze()  # Keep on GPU

        # Compute advantages and returns on GPU
        advantages = self.compute_gae(rewards, masks, values.detach(), next_value)
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
                    
                # log_prob_t, state_value_t, entropy_t, hidden = torch.utils.checkpoint.checkpoint(
                #     self.model.evaluate,
                #     state_t,
                #     action_t,
                #     hidden,
                #     use_reentrant=False
                # )
                log_prob_t, entropy_t, state_value_t, hidden = self.model.evaluate(state_t, action_t, hidden)

                all_log_probs.append(log_prob_t)
                all_state_values.append(state_value_t.squeeze())
                all_entropies.append(entropy_t)

            # Stack all timesteps (already on GPU)
            log_probs = torch.stack(all_log_probs)  # [seq_len]
            state_values = torch.stack(all_state_values)  # [seq_len]
            dist_entropy = torch.stack(all_entropies)  # [seq_len]
           
            # All tensors already on GPU, no need to move
            # Finding the ratio (pi_theta / pi_theta__old) 
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - state_values).pow(2).mean()
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
