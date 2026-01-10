'''
basic model + LSTM layer
Author: Mateo Taylor
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import math
from conf.conf import Config

class Model(nn.Module):
    
    def __init__(self, pretrained=None):
        super(Model, self).__init__()
        self.action_space = Config.ACTION_DIM
        # Pretrained Backbone
        resnet = models.resnet18(weights="DEFAULT") # [3, 320, 3] -> [512, 10, 10]
        
        # freeze all layers of resnet
        for param in resnet.parameters():
            param.requires_grad = False
        
        # force all batch norm layers into eval mode (bc im batching 1 at a time for lstm)
        for module in resnet.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.requires_grad_(False)

        # unfreeze layer 4 and let that learn w/ transformer
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.compressed_dim = 256
        self.compression = nn.Conv2d(512, 256, kernel_size=1)

        # Transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 256))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.1
        )

        self.register_buffer('pos_encoding_2d', self.get_2d_positional_encoding())
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.spatial_pool_attention = nn.Linear(256, 1) # pool in addition to cls token
        # lstm
        self.lstm = nn.LSTM(256, 256, batch_first=True)

        # actor head
        self.policy_heads = nn.ModuleList([nn.Linear(256, action_dim) for action_dim in self.action_space])

        self.value_head = nn.Linear(256, 1)
        if pretrained and isinstance(pretrained, str):
            pretrained = torch.load(pretrained, map_location=Config.DEVICE)
            self.load_state_dict(pretrained)\
            
        self.pickaxe_mask = 0


    def forward(self, x, hidden=None):
        # x input: [3, 640, 640]
        
        # compress x into just 250~ frames
        x = x.reshape(1, 3, 320, 320) # Changed .view() to .reshape()

        # Backbone resnet
        # x = checkpoint(self.backbone, x, use_reentrant=False)  # -> [Batch * Seq Len, 512, 10, 10]
        x = self.backbone(x)  # -> [Batch * Seq Len, 512, 10, 10]

        x = self.compression(x)  # -> [Batch * Seq Len, 256, 10, 10]
        # Flatten spatial (10x10)) and transpose to sequence, also add CLS token
        # [Batch * Seq Len, 256, 10, 10] -> [Batch * Seq Len, 256, 100] -> [Batch * Seq Len, 100, 256]
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_encoding_2d
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # [Batch * Seq Len, 101, 256]

        # spatial transformer to cls token
        x = self.spatial_transformer(x) # [Batch * Seq Len, 26, 128]
        cls_token = x[:, 0]  # Extract CLS token -> [1, 256]
        
        # spatial attention pooling over other tokens
        spatial_tokens = x[:, 1:]  # [1, 100, 256]
        attn_weights = F.softmax(self.spatial_pool_attention(spatial_tokens), dim=1)  # [1, 100, 1]
        spatial_feat = (spatial_tokens * attn_weights).sum(dim=1)  # [1, 256]

        x = cls_token + spatial_feat  # combine cls token and pooled spatial features -> [1, 256]

        # pass through lstm
        x = x.view(1, 1, 256)  # Reshape to [batch=1, seq_len=1, features=256]

        x, hidden = self.lstm(x, hidden)  # -> [1, 1, 256], hidden

        x = x.squeeze(1)  # Remove seq dimension -> [1, 256]
        value = self.value_head(x)  # -> [1, 1]
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

        # now mask pickaxe swing until animation is done
        if self.pickaxe_mask > 0:
            action[5] = 0.0
            self.pickaxe_mask -= 1
        elif action[5] == 1:
            self.pickaxe_mask = 5

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


    def get_2d_positional_encoding(self, height=10, width=10, d_model=256):
        pe = torch.zeros(height, width, d_model)
        
        # Y-axis encoding
        y_pos = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        div_term_y = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        pe[:, :, 0:d_model//2:2] = torch.sin(y_pos * div_term_y).unsqueeze(1)
        pe[:, :, 1:d_model//2:2] = torch.cos(y_pos * div_term_y).unsqueeze(1)
        
        # X-axis encoding
        x_pos = torch.arange(0, width, dtype=torch.float).unsqueeze(0)
        div_term_x = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        pe[:, :, d_model//2::2] = torch.sin(x_pos.T * div_term_x).unsqueeze(0)
        pe[:, :, d_model//2+1::2] = torch.cos(x_pos.T * div_term_x).unsqueeze(0)
        
        return pe.flatten(0, 1).unsqueeze(0)  # [1, 100, 256]
