'''
basic model + Spatial Softmax + stateless transformer encoder
Refactored for transformer-based inference
'''

from contextlib import nullcontext
import math

import torch
import torch.nn as nn
import torchvision

from conf.conf import Config
from ultralytics import YOLO


class Model(nn.Module):

  def __init__(self, pretrained=None):
    super(Model, self).__init__()

    self.action_space = Config.ACTION_DIM
    self.d_model = 512
    self.max_objects = 8
    self.yolo_feature_dim = self.max_objects * 5
    self.policy_training_enabled = True
    self.yolo_internal_features = None

    self.channel_squish = nn.Sequential(
          nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
          nn.BatchNorm2d(256),
          nn.SiLU(),
      )
    self.flattened_dim = 256 * 2

    self.feature_proj = nn.Linear(self.flattened_dim + self.yolo_feature_dim, self.d_model)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=self.d_model,
        nhead=8,
        dim_feedforward=self.d_model * 2,
        batch_first=True,
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.TRANSFORMER_LAYERS)

    self.policy_heads = nn.ModuleList([nn.Linear(self.d_model, action_dim) for action_dim in self.action_space])
    self.value_head = nn.Linear(self.d_model, 1)

    self.yolo = YOLO("model/best.pt").model
    for param in self.yolo.parameters():
      param.requires_grad = False
    self.yolo.eval()

    def hook_fn(module, input, output):
      self.yolo_internal_features = output

    self.yolo.model[22].register_forward_hook(hook_fn)

    if pretrained and isinstance(pretrained, str):
      pretrained_state = torch.load(pretrained, map_location=Config.DEVICE)
      if isinstance(pretrained_state, dict) and "model_state_dict" in pretrained_state:
        pretrained_state = pretrained_state["model_state_dict"]
      self.load_state_dict(pretrained_state, strict=False)

  def set_trainable_components(self, train_policy: bool):
    self.policy_training_enabled = train_policy

    self.channel_squish.requires_grad_(train_policy)
    self.feature_proj.requires_grad_(train_policy)
    self.transformer.requires_grad_(train_policy)
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

  def _spatial_softmax(self, x):
    b, c, h, w = x.shape
    x = x.view(b, c, h * w)
    softmax = torch.softmax(x, dim=-1)

    xs = torch.linspace(-1.0, 1.0, w, device=x.device)
    ys = torch.linspace(-1.0, 1.0, h, device=x.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)

    exp_x = torch.sum(softmax * grid_x, dim=-1)
    exp_y = torch.sum(softmax * grid_y, dim=-1)
    return torch.cat([exp_x, exp_y], dim=-1)

  def _positional_encoding(self, sequence_length, device):
    position = torch.arange(sequence_length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / self.d_model)
    )
    encoding = torch.zeros(sequence_length, self.d_model, device=device)
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding

  def _non_maximum_suppression(self, yolo_raw, yolo_features, b, s):
    for i in range(b * s):
      preds = yolo_raw[i].transpose(0, 1)

      cx, cy, w_, h_ = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
      x1 = cx - w_ / 2
      y1 = cy - h_ / 2
      x2 = cx + w_ / 2
      y2 = cy + h_ / 2
      boxes = torch.stack((x1, y1, x2, y2), dim=1)

      class_scores = preds[:, 4:]
      max_scores, class_ids = torch.max(class_scores, dim=1)

      conf_mask = max_scores > 0.25
      f_boxes = boxes[conf_mask]
      f_scores = max_scores[conf_mask]
      f_classes = class_ids[conf_mask]

      if len(f_boxes) > 0:
        keep_idx = torchvision.ops.batched_nms(f_boxes, f_scores, f_classes, iou_threshold=0.45)
        keep_idx = keep_idx[:self.max_objects]
        num_objs = len(keep_idx)

        yolo_features[i, :num_objs, 0:4] = f_boxes[keep_idx]
        yolo_features[i, torch.arange(num_objs, device=yolo_features.device), 4] = f_scores[keep_idx]

    return yolo_features

  def forward(self, x, hidden=None):
    x = x.to(Config.DEVICE)
    x = self._prepare_input(x)
    b, s, c, h, w = x.shape
    x = x.view(-1, c, h, w)

    with torch.no_grad():
      yolo_output = self.yolo(x * 255.0)[0]

    yolo_features = torch.zeros((b * s, self.max_objects, 5), device=x.device)
    yolo_features = self._non_maximum_suppression(yolo_output, yolo_features, b, s)
    yolo_features = yolo_features.view(b, s, -1)

    if self.yolo_internal_features is None:
      raise RuntimeError("YOLO internal features were not captured by the forward hook.")

    spatial_features = self.channel_squish(self.yolo_internal_features)
    spatial_features = self._spatial_softmax(spatial_features)
    spatial_features = spatial_features.view(b, s, -1)

    combined_features = torch.cat([spatial_features, yolo_features], dim=-1)
    memory = self.feature_proj(combined_features)
    memory = memory + self._positional_encoding(s, memory.device).unsqueeze(0)
    transformer_out = self.transformer(memory)

    value = self.value_head(transformer_out)
    x_logits = [policy_head(transformer_out) for policy_head in self.policy_heads]

    return x_logits, value, hidden

  def select_action(self, x, hidden=None, stochastic=False, temperature=1.0):
    device_type = "cuda" if x.device.type == "cuda" and torch.cuda.is_available() else "cpu"
    autocast_context = torch.autocast(device_type=device_type, dtype=torch.float16) if device_type == "cuda" else nullcontext()

    with torch.no_grad():
      with autocast_context:
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
