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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.START_LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.VALUE_HEAD_WARMUP_EPISODES,
            eta_min=Config.TARGET_LEARNING_RATE
        )
        self.scaler = torch.amp.GradScaler('cuda')
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.LAMDA
        self.eps_clip = Config.EPS_CLIP
        self.entropy_coef = Config.ENTROPY
        self.device = Config.DEVICE
    
    def select_action(self, state, hidden_state=None):
        """Wrapper to call model's select_action method."""
        return self.model.select_action(state, stochastic=True)

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

    def learn(self, sample_data, next_state, hidden_state_in=None, hidden_state_out=None, logger=None):
        # sample data contains tuples of (state, action, reward, log_prob, value, hidden_state, mask)
        # unpack sample data
        states, actions, rewards, old_log_probs, values, _, masks = zip(*sample_data)
        
        # Create all tensors directly on GPU
        states = torch.stack(states).to(self.device)  # [seq_len, 3, 360, 640]
        actions = torch.stack(actions).long().to(self.device)  # [seq_len, num_action_dims]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # [seq_len]
        old_log_probs = torch.stack(old_log_probs).to(self.device)  # [seq_len]
        values = torch.stack([v.squeeze() if torch.is_tensor(v) else torch.as_tensor(v) for v in values]).to(self.device)  # [seq_len]
        masks = torch.tensor(masks, dtype=torch.float32, device=self.device)  # [seq_len]
        with torch.no_grad():
            # Evaluate next_state without hidden state; the transformer is stateless.
            next_state_tensor = next_state.detach().clone().to(self.device)
            _, _, next_value, _ = self.model.evaluate(next_state_tensor, torch.zeros_like(actions[0]))
            next_value = next_value.squeeze()  # Keep on GPU

        # Compute advantages and returns on GPU
        advantages = self.compute_gae(rewards, masks, values.detach(), next_value)
        returns = advantages + values.detach()  # [seq_len]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        train_sequence_length = Config.CHUNK_LENGTH
        if states.size(0) % train_sequence_length != 0:
            raise ValueError(
                f"Episode length {states.size(0)} must be divisible by CHUNK_LENGTH={train_sequence_length}"
            )

        batch_sequence_count = states.size(0) // train_sequence_length
        states = states.view(batch_sequence_count, train_sequence_length, *states.shape[1:])
        actions = actions.view(batch_sequence_count, train_sequence_length, -1)
        old_log_probs = old_log_probs.view(batch_sequence_count, train_sequence_length)
        advantages = advantages.view(batch_sequence_count, train_sequence_length)
        returns = returns.view(batch_sequence_count, train_sequence_length)
        train_policy = getattr(self.model, "policy_training_enabled", True)

        # Optimize policy for K epochs
        for _ in range(Config.EPOCHS):
            autocast_context = torch.autocast(device_type='cuda') if torch.cuda.is_available() else torch.enable_grad()
            with autocast_context:
                if train_policy:
                    log_probs, dist_entropy, state_values, _ = self.model.evaluate(states, actions)

                    log_probs = log_probs.reshape(-1)
                    state_values = state_values.reshape(-1)
                    state_values = state_values.squeeze(-1)
                    dist_entropy = dist_entropy.reshape(-1)
                    old_log_probs_gpu = old_log_probs.reshape(-1)
                    advantages_gpu = advantages.reshape(-1)
                    returns_gpu = returns.reshape(-1)

                    # Finding the ratio (pi_theta / pi_theta__old)
                    ratios = torch.exp(log_probs - old_log_probs_gpu.detach())

                    # Finding Surrogate Loss
                    surr1 = ratios * advantages_gpu
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_gpu

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * (returns_gpu - state_values).pow(2).mean()
                    entropy_loss = -self.entropy_coef * dist_entropy.mean()

                    loss = actor_loss + critic_loss + entropy_loss
                else:
                    _, state_values, _ = self.model.forward(states)
                    state_values = state_values.reshape(-1)
                    returns_gpu = returns.reshape(-1)

                    actor_loss = torch.zeros((), device=self.device)
                    entropy_loss = torch.zeros((), device=self.device)
                    critic_loss = 0.5 * (returns_gpu - state_values).pow(2).mean()
                    loss = critic_loss

            if logger is not None:
                logger.log_loss(loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item())
            # take gradient step
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            if Config.GRADIENT_CLIP > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        return hidden_state_out
            
    def step_scheduler(self, episode):
        """Step the learning rate scheduler during the warmup period."""
        if episode < Config.VALUE_HEAD_WARMUP_EPISODES:
            self.scheduler.step()

    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)


def print_vram(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(Config.DEVICE) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(Config.DEVICE) / 1024**3  # GB
        print(f"{label}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
