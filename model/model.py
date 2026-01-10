'''
basic model + Spatial Softmax + LSTM layer
Refactored by Gemini
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # Needed for meshgrid
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
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
        
        # 1. Pretrained Backbone (ResNet18)
        resnet = models.resnet18(weights="DEFAULT") # [3, 320, 3] -> [512, 10, 10]
        
        # Freeze ResNet weights
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Force BatchNorm to eval mode (Crucial for Batch Size 1)
        for module in resnet.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False # Do not update stats
                module.requires_grad_(False)

        # Unfreeze Layer 4 (Optional: Keep frozen if convergence is still unstable)
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # 2. Compression Layer
        # 512 channels -> 128 channels. 
        # Spatial Softmax will turn 128 channels into 256 coordinates (128*2).
        self.compression = nn.Conv2d(512, 128, kernel_size=1)

        # 3. Spatial Softmax (Replaces Transformer)
        # Expects 10x10 feature map
        self.spatial_softmax = SpatialSoftmax(height=10, width=10)

        # 4. LSTM
        # Input 256 matches Spatial Softmax output (128 channels * 2 coords)
        self.lstm = nn.LSTM(256, 256, batch_first=True)

        # 5. Heads
        self.policy_heads = nn.ModuleList([nn.Linear(256, action_dim) for action_dim in self.action_space])
        self.value_head = nn.Linear(256, 1)

        if pretrained and isinstance(pretrained, str):
            pretrained = torch.load(pretrained, map_location=Config.DEVICE)
            self.load_state_dict(pretrained)
            
        self.pickaxe_mask = 0

    def forward(self, x, hidden=None):
        # x input: [3, 640, 640] or [Batch, 3, 320, 320]
        
        # Ensure correct shape if single frame
        if x.dim() == 3:
             x = x.reshape(1, 3, 320, 320)
        
        # Input Normalization Check (Safety)
        if x.max() > 1.0:
            x = x.float() / 255.0

        # Backbone
        # Use checkpointing for memory efficiency if needed
        # x = checkpoint(self.backbone, x, use_reentrant=False) 
        x = self.backbone(x)  # -> [Batch, 512, 10, 10]

        # Compression & Spatial Softmax
        x = self.compression(x)     # -> [Batch, 128, 10, 10]
        x = self.spatial_softmax(x) # -> [Batch, 256] (128 X-coords + 128 Y-coords)

        # LSTM
        # Reshape for LSTM: [Batch, Seq_Len=1, Features]
        # (Assuming we process 1 step at a time in this forward call)
        x = x.unsqueeze(1) 
        x, hidden = self.lstm(x, hidden) 
        x = x.squeeze(1) # -> [Batch, 256]

        # Heads
        value = self.value_head(x)
        x_logits = [policy_head(x) for policy_head in self.policy_heads]
        
        return x_logits, value, hidden

    def init_hidden(self):
        h0 = torch.zeros(1, 1, 256).to(Config.DEVICE)
        c0 = torch.zeros(1, 1, 256).to(Config.DEVICE)
        return (h0, c0)

    def select_action(self, state, hidden):
        # state: [3, 360, 640]
        state = state.to(Config.DEVICE)
        
        # Ensure hidden tuple is on device
        hidden = (hidden[0].to(Config.DEVICE), hidden[1].to(Config.DEVICE))
        
        policy_logits, value, hidden = self.forward(state, hidden)

        policy_probs = [F.softmax(logits, dim=-1) for logits in policy_logits]
        dist = [torch.distributions.Categorical(prob) for prob in policy_probs]

        action = [d.sample() for d in dist]
        action = torch.stack(action)

        # Mask pickaxe logic
        if self.pickaxe_mask > 0:
            action[5] = 0.0
            self.pickaxe_mask -= 1
        elif action[5] == 1:
            self.pickaxe_mask = 5

        log_probs = [d.log_prob(a) for d, a in zip(dist, action)]
        log_prob = torch.stack(log_probs).sum()

        return action, log_prob, value, hidden

    def evaluate(self, state, action, hidden):
        # Handles batch evaluation for PPO training
        policy_logits, value, hidden = self.forward(state, hidden)

        policy_probs = [F.softmax(logits, dim=-1) for logits in policy_logits]
        dist = [torch.distributions.Categorical(prob) for prob in policy_probs]

        log_probs = [d.log_prob(a) for d, a in zip(dist, action)]
        log_prob = torch.stack(log_probs).sum(dim=0) # Sum over action dims, keep batch dim

        entropy = torch.stack([d.entropy() for d in dist]).sum(dim=0)

        return log_prob, entropy, value, hidden