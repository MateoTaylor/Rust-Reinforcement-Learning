'''
basic model + Spatial Softmax + LSTM layer
Refactored by Gemini
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # Needed for meshgrid
from conf.conf import Config

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, temperature=None):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.temperature = temperature or nn.Parameter(torch.ones(1))

        # Create normalized coordinate grid [-1, 1]
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height)
        )
        # Register as buffers (not trainable, but move to device with model)
        self.register_buffer('pos_x', torch.from_numpy(pos_x.reshape(height * width)).float())
        self.register_buffer('pos_y', torch.from_numpy(pos_y.reshape(height * width)).float())

    def forward(self, feature_map):
        # Input: [Batch, Channels, Height, Width]
        b, c, h, w = feature_map.shape

        # Flatten spatial dims: [B, C, H*W]
        flat = feature_map.view(b, c, -1)

        # Softmax over spatial dimensions (H*W) to find "center of mass"
        softmax_attention = F.softmax(flat / self.temperature, dim=2)

        # Calculate expected X and Y coordinates
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=2, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=2, keepdim=True)

        # Output: [Batch, Channels * 2] (Interleave X, Y)
        return torch.cat((expected_x, expected_y), dim=2).view(b, -1)

class Model(nn.Module):

    def __init__(self, pretrained=None):
        super(Model, self).__init__()
        self.action_space = Config.ACTION_DIM
        self.pickaxe_mask = 0

        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0), # [320, 320, 3] -> []
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Spatial Softmax
        self.spatial_softmax = SpatialSoftmax(height=38, width=38)

        # LSTM
        self.lstm = nn.LSTM(192, 256, batch_first=True)

        # Actor head
        self.policy_heads = nn.ModuleList([nn.Linear(256, action_dim) for action_dim in self.action_space])
        self.value_head = nn.Linear(256, 1)

        if pretrained and isinstance(pretrained, str):
            pretrained = torch.load(pretrained, map_location=Config.DEVICE)
            self.load_state_dict(pretrained)

        #  freeze everything except value head until critic loss stabilizes.
        # for param in self.parameters():
        #     param.requires_grad = False
        # self.value_head.weight.requires_grad = True
        # self.value_head.bias.requires_grad = True

    def forward(self, x, hidden=None):
        # x input: [1, 3, 320, 320]

        # compress x into just 250~ frames
        x = self.cnn(x)

        x_gap = torch.mean(x, dim=[2, 3]) # global avg pool [Batch*Seq_Len, 64]

        x = self.spatial_softmax(x) # -> [Batch*Seq_Len, 128]

        x = torch.cat((x_gap, x), dim=1) # [Batch*Seq_Len, 192]

        # pass through lstm
        x = x.unsqueeze(1)  # [add batch dim, 1, 192]
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(1)  # [add batch dim, 256]

        value = self.value_head(x)

        x_logits = [policy_head(x) for policy_head in self.policy_heads]
        return x_logits, value, hidden
  
    def init_hidden(self):
        # initialize LSTM hidden and cell states to zeros on the specified device
        h0 = torch.zeros(1, 1, 256)# [num_layers=1, batch=1, hidden_size=256]
        c0 = torch.zeros(1, 1, 256)
        return (h0, c0)
  
    def select_action(self, state, hidden):
        # state: [3, 360, 640]
        # Get the device from the model's parameters
        state = state.to(Config.DEVICE)
        hidden = (hidden[0].to(Config.DEVICE), hidden[1].to(Config.DEVICE))
        policy_logits, value, hidden = self.forward(state, hidden)  # list of [action_dim_i], [1], hidden_state

        policy_probs = [F.softmax(logits, dim=-1) for logits in policy_logits]  # list of [action_dim_i]
        dist = [torch.distributions.Categorical(prob) for prob in policy_probs]

        action = [d.sample() for d in dist]  # list of scalars
        action = torch.stack(action)  # [num_action_dims]

        log_probs = [d.log_prob(a) for d, a in zip(dist, action)]  # list of scalars
        log_prob = torch.stack(log_probs).sum()  # scalar

        return action, log_prob, value, hidden  # [num_action_dims], scalar, [1], hidden_state
    
    def evaluate(self, state, action, hidden):
        # state: [3, 360, 640]
        policy_logits, value, hidden = self.forward(state, hidden)  # list of [action_dim_i], [1], hidden_state

        policy_probs = [F.softmax(logits, dim=-1) for logits in policy_logits]  # list of [action_dim_i]
        dist = [torch.distributions.Categorical(prob) for prob in policy_probs]

        log_probs = [d.log_prob(a) for d, a in zip(dist, action)]  # list of scalars
        log_prob = torch.stack(log_probs).sum()  # scalar

        entropy = torch.stack([d.entropy() for d in dist]).sum()  # scalar

        return log_prob, entropy, value, hidden  # scalar, scalar, [1], hidden_state