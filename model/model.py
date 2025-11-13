'''
basic model + LSTM layer
Author: Mateo Taylor
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from conf.conf import Config

class Model(nn.Module):

    def __init__(self, pretrained=None):
        super(Model, self).__init__()
        
        hidden_size = Config.HIDDEN_SIZE
        self.input_shape = Config.FEATURE_DIM # should be [3, 360, 640]
        self.action_space = Config.ACTION_SPACE # list of action dimension sizes
        self.conv_dim = 128 * 4 * 8  # calculated from CNN output size
        self.extr_feat = 1

        # pass screenshot through a CNN to extract features
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, ),   # [3,360,640] → [16,179,319]
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # [16,179,319] → [32,88,158]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, dilation=2), # [32,88,158] → [64,42,77]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1), # [64,42,77] → [64,40,75]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),# [64,40,75] → [128,38,73]
            nn.BatchNorm2d(128),   
            nn.ReLU(),

            nn.AdaptiveMaxPool2d((4, 8))                 # [128,38,73] → [128,4,8]
        )

        self.feature_classifier = nn.Sequential(
            nn.Linear(self.conv_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.extr_feat) # squash features down to binary
        )
        
        self.lstm = nn.LSTM(self.conv_dim + self.extr_feat, hidden_size, batch_first=True)
        
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # separate heads for policy and value function. note we need 1 policy head per action dimension
        self.policy_heads = nn.ModuleList([nn.Linear(hidden_size, action_dim) for action_dim in self.action_space])
        self.value_head = nn.Linear(hidden_size, 1) # critic

        # read pretrained filename if provided
        if pretrained and isinstance(pretrained, str):
            pretrained = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(pretrained)            

    def forward(self, x, hidden):
        # x: tensor of shape [3, 360, 640]
        # pass through CNN
        x = x.unsqueeze(0)  # add batch dimension for CNN: [1, 3, 360, 640]
        x = self.cnn_layers(x)  # [1, 128, 4, 8]
        x = x.view(-1)  # flatten to [conv_dim]

        extr_feat_logits = self.feature_classifier(x.unsqueeze(0)) # [conv_dim] -> [1, extr_feat]
        extr_feat_logits = extr_feat_logits.squeeze(0)
            
        # concatenate features to CNN output
        x = torch.cat([x, extr_feat_logits], dim=0)  # [conv_dim + extr_feat]

        # add batch and time dimensions for LSTM
        x = x.unsqueeze(0).unsqueeze(0) # [1, 1, conv_dim + extr_feat]
        x, hidden = self.lstm(x, hidden)  

        x = x.squeeze(0).squeeze(0)  # remove batch and time dimensions: [hidden_size]
        
        # shared layers
        x = self.shared(x)

        # predict policy and value
        policy_logits = [head(x) for head in self.policy_heads]  # list of [action_dim_i]
        value = self.value_head(x)  # [1]
        
        return policy_logits, value, hidden, extr_feat_logits
    
    def init_hidden(self):
        # initialize LSTM hidden and cell states to zeros
        h0 = torch.zeros(1, 1, Config.HIDDEN_SIZE)
        c0 = torch.zeros(1, 1, Config.HIDDEN_SIZE)
        return (h0, c0)
    
    def select_action(self, state, hidden):
        # state: [3, 360, 640]
        state = torch.FloatTensor(state)
        policy_logits, value, hidden, _ = self.forward(state, hidden)  # list of [action_dim_i], [1], hidden_state

        policy_probs = [F.softmax(logits, dim=-1) for logits in policy_logits]  # list of [action_dim_i]
        dist = [torch.distributions.Categorical(prob) for prob in policy_probs]

        action = [d.sample() for d in dist]  # list of scalars
        action = torch.stack(action)  # [num_action_dims]

        log_probs = [d.log_prob(a) for d, a in zip(dist, action)]  # list of scalars
        log_prob = torch.stack(log_probs).sum()  # scalar

        return action, log_prob, value, hidden  # [num_action_dims], scalar, [1], hidden_state

    def evaluate(self, states, actions, hidden):
        '''
        Evaluate actions and return log probabilities, state values, and entropy.
        states: tensor of shape [3, 360, 640] - single timestep
        actions: tensor of shape [num_action_dims] - single timestep
        hidden: LSTM hidden state tuple (h, c)
        '''
        # get policy logits and state values
        policy_logits, state_values, hidden, extr_feat_logits = self.forward(states, hidden)  # list of [action_dim_i], [1], hidden_state, [extr_feat]

        # create probability distributions for each action dimension
        dists = [torch.distributions.Categorical(logits=logits)
             for logits in policy_logits]
        
        # compute log probabilities of the taken actions
        log_probs = torch.stack(
        [dist.log_prob(actions[i]) for i, dist in enumerate(dists)]  # each: scalar
        ).sum()  # scalar
        
        # compute entropy for all action dimensions
        entropies = torch.stack([dist.entropy() for dist in dists])  # [num_action_dims]
        total_entropy = entropies.sum()  # scalar
        
        return log_probs, state_values, total_entropy, hidden, extr_feat_logits  # scalar, [1], scalar, hidden_state, [extr_feat]
