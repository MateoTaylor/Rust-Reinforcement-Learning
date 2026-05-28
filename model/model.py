'''
basic model + Spatial Softmax + GRU layer
Refactored by Gemini
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from conf.conf import Config
from ultralytics import YOLO


class Model(nn.Module):

  def __init__(self, pretrained=None):
    super(Model, self).__init__()

    yolo_wrapper = YOLO("model/best.pt", verbose=False, task="detect")
    self.yolo = yolo_wrapper.model
    self.action_space = Config.ACTION_DIM

    for param in self.yolo.parameters():
      param.requires_grad = False

    self.yolo_features = None

    def hook_fn(module, input, output):
      self.yolo_features = output

    self.yolo.model[22].register_forward_hook(hook_fn)

    self.channel_squish = nn.Sequential(
      nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
      nn.BatchNorm2d(128),
      nn.SiLU(),
    )
    self.channel_squish.requires_grad_(False)
    self.flattened_dim = 128 * 20 * 20

    self.gru = nn.GRU(self.flattened_dim, Config.LSTM_HIDDEN_SIZE, batch_first=True)

    self.policy_heads = nn.ModuleList(
      [nn.Linear(Config.LSTM_HIDDEN_SIZE, action_dim) for action_dim in self.action_space]
    )
    self.value_head = nn.Linear(Config.LSTM_HIDDEN_SIZE, 1)
    self.policy_training_enabled = True

    if pretrained and isinstance(pretrained, str):
      pretrained_state = torch.load(pretrained, map_location=Config.DEVICE)
      self.load_state_dict(pretrained_state)

  def set_trainable_components(self, train_policy: bool):
    self.policy_training_enabled = train_policy

    self.channel_squish.requires_grad_(train_policy)
    self.gru.requires_grad_(train_policy)
    self.policy_heads.requires_grad_(train_policy)
    self.value_head.requires_grad_(True)

  def _prepare_input(self, x):
    if x.dim() == 3:
      return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 4:
      return x.unsqueeze(1)
    if x.dim() == 5:
      return x
    raise ValueError(f"Expected 3D, 4D, or 5D input, got shape {tuple(x.shape)}")

  def init_hidden(self, batch_size=1, device=None):
    device = device or Config.DEVICE
    return torch.zeros(1, batch_size, Config.LSTM_HIDDEN_SIZE, device=device)

  def forward(self, x, hidden=None):
    device = next(self.parameters()).device
    x = x.to(device)

    if hidden is not None:
      if isinstance(hidden, tuple):
        hidden = tuple(h.to(device) for h in hidden)
      else:
        hidden = hidden.to(device)

    x = self._prepare_input(x)
    batch_size, sequence_length, channels, height, width = x.shape

    x_reshaped_for_yolo = x.reshape(batch_size * sequence_length, channels, height, width)

    self.yolo_features = None
    yolo_generator = self.yolo(x_reshaped_for_yolo)
    for _ in yolo_generator:
      pass

    if self.yolo_features is None:
      raise RuntimeError("YOLO hook did not capture features during forward pass")

    features = self.yolo_features.clone()
    features = self.channel_squish(features)

    features = features.reshape(batch_size, sequence_length, -1)
    features, hidden = self.gru(features, hidden)

    value = self.value_head(features)
    policy_logits = [policy_head(features) for policy_head in self.policy_heads]
    return policy_logits, value, hidden

  def select_action(self, x, hidden=None, stochastic=False, temperature=1.0):
    with torch.no_grad():
      action_logits, value, hidden = self.forward(x, hidden=hidden)
      actions = []
      log_probs = []

      for logits in action_logits:
        scaled_logits = logits / max(temperature, 1e-6)
        final_step_logits = scaled_logits[:, -1, :]
        dist = torch.distributions.Categorical(logits=final_step_logits)

        if stochastic:
          action = dist.sample()
        else:
          action = torch.argmax(final_step_logits, dim=-1)

        actions.append(action.view(-1)[0])
        log_probs.append(dist.log_prob(action).view(-1)[0])

      action = torch.stack(actions)
      log_prob = torch.stack(log_probs).sum()
      value = value[:, -1, :].view(-1)[0]

    return action, log_prob, value, hidden

  def evaluate(self, state, action, hidden=None):
    policy_logits, value, hidden = self.forward(state, hidden)

    batch_size, sequence_length = value.shape[:2]
    action = action.to(value.device).reshape(batch_size, sequence_length, -1).long()

    log_probs = []
    entropies = []

    for head_index, logits in enumerate(policy_logits):
      dist = torch.distributions.Categorical(logits=logits)
      current_action = action[..., head_index]
      log_probs.append(dist.log_prob(current_action))
      entropies.append(dist.entropy())

    log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
    entropy = torch.stack(entropies, dim=-1).sum(dim=-1)
    value = value.squeeze(-1)
    return log_prob, entropy, value, hidden
