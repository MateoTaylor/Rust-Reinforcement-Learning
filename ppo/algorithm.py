'''
basic PPO algorithm implementation
Author: Mateo Taylor
'''

import torch
from conf.conf import Config
import torch.nn as nn
from monitoring.monitoring import monitor_loss

class Algorithm:
    def __init__(self, model, device):
        self.model = model.to(device)  # Move model to device       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.gamma = Config.GAMMA
        self.gae_lambda = Config.LAMDA
        self.eps_clip = Config.EPS_CLIP
        self.entropy_coef = Config.ENTROPY
        self.feature_classifier_coef = Config.FEATURE_CLASSIFIER_COEF
        self.device = device
    
    def select_action(self, state, hidden_state):
        """Wrapper to call model's select_action method."""
        return self.model.select_action(state, hidden_state)

    def compute_gae(self, rewards, masks, values):
        """
        Compute Generalized Advantage Estimation.
        All inputs should be tensors.
        """
        advantages = []
        gae = 0
        # Append 0 for the value after the last state (terminal state has value 0)
        values_list = values.tolist() + [0]
        rewards_list = rewards.tolist()
        masks_list = masks.tolist()
        
        for step in reversed(range(len(rewards_list))):
            delta = rewards_list[step] + self.gamma * values_list[step + 1] * masks_list[step] - values_list[step]
            gae = delta + self.gamma * self.gae_lambda * masks_list[step] * gae
            advantages.insert(0, gae)
        return advantages
    
    def learn(self, sample_data):
        # sample data contains tuples of (state, action, reward, next_state, done, log_prob, value)
        assert len(sample_data) == Config.MAX_STEPS, "Sample data length does not match MAX_STEPS"

        # unpack sample data
        states, actions, rewards, _, dones, old_log_probs, values, extr_feat_ground_truth = zip(*sample_data)
        states = torch.stack(states).to(self.device)  # [seq_len, 3, 360, 640]
        actions = torch.stack(actions).long().to(self.device)  # [seq_len, num_action_dims]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # [seq_len]
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)  # [seq_len]
        old_log_probs = torch.stack(old_log_probs).to(self.device)  # [seq_len]
        values = torch.tensor(values, dtype=torch.float32).to(self.device)  # [seq_len]
        extr_feat_ground_truth = torch.tensor(extr_feat_ground_truth, dtype=torch.float32).to(self.device)  # [seq_len]
        masks = 1 - dones  # [seq_len]
        

        # Compute advantages and returns
        advantages = self.compute_gae(rewards, masks, values.detach())
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)  # [seq_len]
        returns = advantages + values.detach()  # [seq_len]

        # Slice the targets to match the skipped timesteps
        advantages_sliced = advantages[Config.TIMESTEP_SKIPPED:]
        returns_sliced = returns[Config.TIMESTEP_SKIPPED:]
        old_log_probs_sliced = old_log_probs[Config.TIMESTEP_SKIPPED:]
        extr_feat_ground_truth_sliced = extr_feat_ground_truth[Config.TIMESTEP_SKIPPED:]

        # Normalize advantages
        advantages_sliced = (advantages_sliced - advantages_sliced.mean()) / (advantages_sliced.std() + 1e-8)

        # Optimize policy for K epochs
        for _ in range(Config.EPOCHS):
            # Initialize fresh hidden state for the sequence
            hidden = self.model.init_hidden()
            
            all_log_probs = []
            all_state_values = []
            all_entropies = []
            all_extr_feat_logits = []
            
            # Process each timestep sequentially
            assert(states.size(0) == Config.MAX_STEPS)
            for t in range(Config.TIMESTEP_SKIPPED, states.size(0)):
                state_t = states[t]  # [3, 360, 640]
                action_t = actions[t]  # [num_action_dims]
                
                log_prob_t, state_value_t, entropy_t, hidden, extr_feat_logits = self.model.evaluate(
                    state_t, action_t, hidden
                )
                
                all_log_probs.append(log_prob_t)
                all_state_values.append(state_value_t)
                all_entropies.append(entropy_t)
                all_extr_feat_logits.append(extr_feat_logits)

            # Stack all timesteps
            log_probs = torch.stack(all_log_probs)  # [seq_len - skipped]
            state_values = torch.stack(all_state_values)  # [seq_len - skipped]
            dist_entropy = torch.stack(all_entropies)  # [seq_len - skipped]
            extr_feat_logits = torch.stack(all_extr_feat_logits)  # [seq_len - skipped]

            # Finding the ratio (pi_theta / pi_theta__old) 
            ratios = torch.exp(log_probs - old_log_probs_sliced.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages_sliced
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_sliced

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns_sliced - state_values).pow(2).mean()
            entropy_loss = -self.entropy_coef * dist_entropy.mean()

            feature_loss = self.feature_classifier_coef * nn.BCEWithLogitsLoss()(extr_feat_logits.squeeze(-1), extr_feat_ground_truth_sliced.float())

            loss = actor_loss + critic_loss + entropy_loss + feature_loss
            monitor_loss(loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item(), feature_loss.item())
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRADIENT_CLIP)
            self.optimizer.step()
            
    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
