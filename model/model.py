'''
basic model + LSTM layer
Author: Mateo Taylor
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from conf.conf import Config

class Model(nn.Module):

    def __init__(self, pretrained=None):
        super(Model, self).__init__()

        hidden_size = Config.HIDDEN_SIZE
        self.input_shape = Config.FEATURE_DIM # should be [3, 360, 640]
        self.action_space = Config.ACTION_SPACE # list of action dimension sizes
        self.conv_dim = 512 * 4 * 8  # ResNet18 outputs 512 channels
        self.extr_feat = 1

        resnet = models.resnet18(weights='DEFAULT')

        # Freeze early ResNet layers before adding to Sequential
        # Freeze conv1, bn1, and layer1 (first two main layers)
        for param in resnet.conv1.parameters():
            param.requires_grad = False
        for param in resnet.bn1.parameters():
            param.requires_grad = False
        for param in resnet.layer1.parameters():
            param.requires_grad = False
        for param in resnet.layer2.parameters():
            param.requires_grad = False

        # Remove the classifier (fc) and avgpool layers - keep convolutional layers
        # ResNet18 outputs [batch, 512, H/32, W/32] before avgpool
        self.cnn_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((4, 8)),  # [512, H/32, W/32] -> [512, 16, 16]
            nn.ReLU()
        )

        self.feature_classifier = nn.Sequential(
            nn.Linear(self.conv_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512), # Corrected input dimension
            nn.ReLU(),
            nn.Dropout(0.2),
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
            pretrained = torch.load(pretrained, map_location=torch.device('cpu'))
            self.load_state_dict(pretrained)

    def forward(self, x, hidden):
        # x: tensor of shape [3, 360, 640]
        # pass through CNN
        device = next(self.parameters()).device # Get the device where the model is located
        x = x.unsqueeze(0).to(device)  # add batch dimension for CNN and move to device
        x = self.cnn_layers(x)  # [1, 128, 4, 8]
        x = x.view(-1)  # flatten to [conv_dim]

        extr_feat_logits = self.feature_classifier(x.unsqueeze(0)) # [conv_dim] -> [1, extr_feat]
        extr_feat_logits = extr_feat_logits.squeeze(0)

        # concatenate features to CNN output
        x = torch.cat([x, extr_feat_logits], dim=0)  # [conv_dim + extr_feat]

        # add batch and time dimensions for LSTM
        x = x.unsqueeze(0).unsqueeze(0) # [1, 1, conv_dim + extr_feat]
        # Ensure hidden state is on the same device as the model
        hidden = (hidden[0].to(device), hidden[1].to(device))
        x, hidden = self.lstm(x, hidden)

        x = x.squeeze(0).squeeze(0)  # remove batch and time dimensions: [hidden_size]

        # shared layers
        x = self.shared(x)

        # predict policy and value
        policy_logits = [head(x) for head in self.policy_heads]  # list of [action_dim_i]
        value = self.value_head(x)  # [1]

        return policy_logits, value, hidden, extr_feat_logits

    def init_hidden(self, device): # Modified to accept device
        # initialize LSTM hidden and cell states to zeros on the specified device
        h0 = torch.zeros(1, 1, Config.HIDDEN_SIZE).to(device)
        c0 = torch.zeros(1, 1, Config.HIDDEN_SIZE).to(device)
        return (h0, c0)

    def select_action(self, state, hidden):
        # state: [3, 360, 640]
        # Get the device from the model's parameters
        device = next(self.parameters()).device
        state = state.to(device) # Move state to the correct device
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
        # Get the device from the model's parameters
        device = next(self.parameters()).device
        states = states.to(device) # Ensure states are on the correct device
        actions = actions.to(device) # Ensure actions are on the correct device
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